"""
scripts/script2_vectorise_save.py
----------------------------------
Task 2 entry point.

Responsibilities:
  1. Download all per-tag Parquet files from Azure Blob (raw/ prefix).
  2. Concatenate into a single DataFrame, deduplicate.
  3. Light text cleaning for downstream modeling (URL stripping, whitespace
     normalisation, drop too-short posts) — kept minimal since heavier
     cleaning (emoji/kaomoji conversion, lowercasing) already happened in
     script1, and transformer embeddings work best on close-to-natural text.
  4. Encode post text with SentenceTransformer.
  5. Save TWO output formats, sharing one canonical metadata file so nothing
     is duplicated across them:
       - npy/embeddings.npy        → raw embedding matrix, for BERTopic or
                                      whatever else you want downstream
       - npy/metadata.parquet      → post_id, tag, model_text, post_date —
                                      row order matches embeddings.npy 1:1
       - faiss/index.faiss         → IndexFlatIP built from the same
                                      embeddings, for RAG-style similarity
                                      search (use row index to look up
                                      npy/metadata.parquet for the matching
                                      record)

Airflow calls run() — also runnable directly for testing:
    python script2_vectorise_save.py
"""

from __future__ import annotations

import os
import re
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
MODEL_TEXT_COL = "model_text"

# Posts shorter than this (in words, after cleaning) are dropped — very
# short documents add little signal to BERTopic and tend to land as noise.
MIN_WORDS: int = int(os.getenv("MIN_WORDS", "3"))

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH: int = 64
# IndexFlatIP needs normalised vectors for cosine similarity. If you only
# care about the raw numpy embeddings for BERTopic (which handles its own
# distance metric internally), this can be set False via env var — but
# leaving it True is the safer default since FAISS depends on it.
NORMALIZE_EMBEDDINGS: bool = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

# Output blob names
EMBEDDINGS_BLOB = "npy/embeddings.npy"
METADATA_BLOB   = "npy/metadata.parquet"
INDEX_BLOB      = "faiss/index.faiss"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")

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


def _clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning on top of TEXT_COL, kept minimal on purpose.

    Heavier normalisation (emoji/kaomoji → text, lowercasing) already
    happened in script1. Transformer embeddings work best on text close to
    natural language, so this only strips things that are pure noise for
    both embeddings and BERTopic: URLs and the literal " ||| " comment
    separators inherited from how script1 joins multiple comments into one
    string. Posts that end up too short after cleaning are dropped — they
    add little signal to topic modeling and tend to land as outliers.
    """
    df = df.copy()
    text = df[TEXT_COL].astype(str)
    text = text.str.replace(_URL_RE, " ", regex=True)
    text = text.str.replace("|||", " ", regex=False)
    text = text.str.replace(_WHITESPACE_RE, " ", regex=True).str.strip()
    df[MODEL_TEXT_COL] = text

    word_counts = df[MODEL_TEXT_COL].str.split().str.len().fillna(0)
    before = len(df)
    df = df[word_counts >= MIN_WORDS].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} posts under {MIN_WORDS} words after cleaning")

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
    print(f"Model: {EMBED_MODEL}  |  Batch size: {EMBED_BATCH}  |  Min words: {MIN_WORDS}")
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

    # ── 2. Prepare + clean text ──────────────────────────────────────────────
    full_df = _prepare_text(full_df)
    full_df = _clean_for_modeling(full_df)
    texts: list[str] = full_df[MODEL_TEXT_COL].tolist()
    print(f"  Posts ready for embedding: {len(texts):,}")

    # ── 3. Encode ────────────────────────────────────────────────────────────
    print(f"\n  Encoding with {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    ).astype("float32")
    print(f"  Embedding shape: {embeddings.shape}")

    # ── 4. Build FAISS index (shares row order with embeddings/metadata) ────
    index = _build_index(embeddings)
    print(f"  FAISS index built — {index.ntotal} vectors, dim={index.d}")

    # ── 5. Save metadata — one canonical file, no duplication across
    #        outputs, and no pickle (the downstream notebook environment has
    #        its own pandas/numpy versions — a pickled DataFrame is a
    #        cross-environment compatibility risk that parquet avoids).
    #        Row order matches embeddings.npy and the FAISS index 1:1.
    metadata_df = full_df[["Post ID", "tag", MODEL_TEXT_COL, "Post Date"]].rename(
        columns={"Post ID": "post_id", "Post Date": "post_date", MODEL_TEXT_COL: "text"}
    )

    with tempfile.TemporaryDirectory() as tmp:
        idx_path = f"{tmp}/index.faiss"
        npy_path = f"{tmp}/embeddings.npy"
        meta_path = f"{tmp}/metadata.parquet"

        faiss.write_index(index, idx_path)
        np.save(npy_path, embeddings)
        metadata_df.to_parquet(meta_path, index=False)

        with open(idx_path, "rb") as f:
            _upload_bytes(f.read(), INDEX_BLOB)
        with open(npy_path, "rb") as f:
            _upload_bytes(f.read(), EMBEDDINGS_BLOB)
        with open(meta_path, "rb") as f:
            _upload_bytes(f.read(), METADATA_BLOB)

    print("\n✅ Script 2 complete.\n")


if __name__ == "__main__":
    run()