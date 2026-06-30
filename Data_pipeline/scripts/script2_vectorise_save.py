"""
scripts/script2_vectorise_save.py
----------------------------------
Task 2 entry point.

Responsibilities:
  1. Download all per-tag Parquet files from Azure Blob (raw/ prefix).
  2. Concatenate into a single DataFrame, deduplicate.
  3. Encode post text with SentenceTransformer.
  4. Build a FAISS index (IndexFlatIP on L2-normalised vectors for cosine sim).
  5. Upload index + metadata pickle back to Blob (faiss/ prefix).

Airflow calls run() — also runnable directly for testing:
    python script2_vectorise_save.py
"""

from __future__ import annotations

import os
import pickle
import tempfile
from datetime import datetime
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AZURE_CONN_STR: str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_CONTAINER: str = os.getenv("AZURE_CONTAINER_NAME", "mh-etl-data")

BEYONDBLUE_TAGS: list[str] = [
    t.strip() for t in os.getenv(
        "BEYONDBLUE_TAGS",
        "anxiety,depression,ptsd-and-trauma,suicidal-thoughts-and-self-harm",
    ).split(",")
]

# Column whose text will be embedded (title + content combined)
TEXT_COL = "combined_text"

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH: int = 64

# Output blob names
INDEX_BLOB = "faiss/index.faiss"
META_BLOB  = "faiss/metadata.pkl"

# ---------------------------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------------------------

def _blob_client() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(AZURE_CONN_STR)


def _download_parquet(blob_name: str) -> pd.DataFrame | None:
    """Download *blob_name* from Blob and return as DataFrame (None if missing)."""
    try:
        data = (
            _blob_client()
            .get_blob_client(container=AZURE_CONTAINER, blob=blob_name)
            .download_blob()
            .readall()
        )
        return pd.read_parquet(BytesIO(data))
    except Exception as exc:
        print(f"  [Warning] Could not load {blob_name}: {exc}")
        return None


def _upload_bytes(data: bytes, blob_name: str) -> None:
    _blob_client().get_blob_client(
        container=AZURE_CONTAINER, blob=blob_name
    ).upload_blob(data, overwrite=True)
    print(f"  ✅ Uploaded → {blob_name}  ({len(data)/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create a single text column combining title and post content."""
    df = df.copy()
    df["Post Title"] = df["Post Title"].fillna("").astype(str)
    df["Post Content"] = df["Post Content"].fillna("").astype(str)
    df[TEXT_COL] = (df["Post Title"] + " " + df["Post Content"]).str.strip()
    # Drop rows with empty combined text
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def _build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product index (cosine similarity on normalised vecs)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)     # cosine sim = IP after L2 normalisation
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Main entry point called by Airflow Task 2."""
    print("=" * 60)
    print(f"Script 2 — Vectorise & Save  [{datetime.utcnow().isoformat()} UTC]")
    print(f"Model: {EMBED_MODEL}  |  Batch size: {EMBED_BATCH}")
    print("=" * 60)

    # ── 1. Load all tag Parquets from Blob ───────────────────────────────────
    frames: list[pd.DataFrame] = []
    for tag in BEYONDBLUE_TAGS:
        df = _download_parquet(f"raw/{tag}_posts.parquet")
        if df is not None and not df.empty:
            df["tag"] = tag          # preserve source tag
            frames.append(df)
            print(f"  Loaded {len(df):,} rows for tag '{tag}'")

    if not frames:
        raise RuntimeError("No data found in Blob. Run Task 1 first.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["Post ID"]).reset_index(drop=True)
    print(f"\n  Total unique posts: {len(full_df):,}")

    # ── 2. Prepare text column ───────────────────────────────────────────────
    full_df = _prepare_text(full_df)
    texts: list[str] = full_df[TEXT_COL].tolist()
    print(f"  Posts with non-empty text: {len(texts):,}")

    # ── 3. Encode ────────────────────────────────────────────────────────────
    print(f"\n  Encoding with {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # required for IndexFlatIP → cosine sim
    ).astype("float32")
    print(f"  Embedding shape: {embeddings.shape}")

    # ── 4. Build FAISS index ─────────────────────────────────────────────────
    index = _build_index(embeddings)
    print(f"  FAISS index built — {index.ntotal} vectors, dim={index.d}")

    # ── 5. Serialise ─────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        idx_path = f"{tmp}/index.faiss"
        meta_path = f"{tmp}/metadata.pkl"

        faiss.write_index(index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "texts": texts,
                    "post_ids": full_df["Post ID"].tolist(),
                    "tags": full_df["tag"].tolist(),
                    "df": full_df,
                    "model": EMBED_MODEL,
                    "created_at": datetime.utcnow().isoformat(),
                },
                f,
            )

        with open(idx_path, "rb") as f:
            _upload_bytes(f.read(), INDEX_BLOB)
        with open(meta_path, "rb") as f:
            _upload_bytes(f.read(), META_BLOB)

    print("\n✅ Script 2 complete.\n")


if __name__ == "__main__":
    run()