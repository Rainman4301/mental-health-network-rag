"""
scripts/script1_scrape_clean.py
--------------------------------
Task 1 entry point.

Responsibilities:
  1. Scrape Beyond Blue forum posts (Selenium + BeautifulSoup).
  2. Parse dates, convert emojis/emoticons/kaomoji to text.
  3. Skip posts already scraped (incremental load using existing Blob file).
  4. Upload cleaned DataFrame as Parquet to Azure Blob Storage → raw/ prefix.

Airflow calls run() — this file can also be executed directly for testing:
    python script1_scrape_clean.py
"""

from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import emoji
import emot
import pandas as pd
from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium_stealth import stealth
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config (read from environment variables set in .env / docker-compose)
# ---------------------------------------------------------------------------

AZURE_CONN_STR: str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_CONTAINER: str = os.getenv("AZURE_CONTAINER_NAME", "mh-etl-data")

# Comma-separated tags to scrape, e.g. "anxiety,depression"
BEYONDBLUE_TAGS: list[str] = [
    t.strip() for t in os.getenv(
        "BEYONDBLUE_TAGS",
        "anxiety,depression,ptsd-and-trauma,suicidal-thoughts-and-self-harm",
    ).split(",")
]

# Base URLs per tag — adjust if Beyond Blue changes their URL structure
TAG_URLS: dict[str, str] = {
    "anxiety": "https://forums.beyondblue.org.au/t5/anxiety/bd-p/anxiety",
    "depression": "https://forums.beyondblue.org.au/t5/depression/bd-p/depression",
    "ptsd-and-trauma": "https://forums.beyondblue.org.au/t5/ptsd-and-trauma/bd-p/ptsd-and-trauma",
    "suicidal-thoughts-and-self-harm": (
        "https://forums.beyondblue.org.au/t5/suicidal-thoughts-and-self-harm"
        "/bd-p/suicidal-thoughts-and-self-harm"
    ),
}

SCRAPE_PAGES: int = int(os.getenv("SCRAPE_PAGES", "2"))
CUTOFF_DATE: str = "2015-01-01"
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "3"))   # parallel threads for comment scraping

# Kaomoji mapping file — mounted alongside scripts
KAOMOJI_PATH = Path(__file__).parent / "kaomoji_to_text.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kaomoji() -> dict[str, str]:
    if KAOMOJI_PATH.exists():
        return json.loads(KAOMOJI_PATH.read_text(encoding="utf-8"))
    return {}


KAOMOJI: dict[str, str] = _load_kaomoji()
_WEEKDAYS = [d.lower() for d in list(calendar.day_name)]

# ---------------------------------------------------------------------------
# ChromeDriver resolution
# ---------------------------------------------------------------------------
# In Docker, this is baked into the image at build time (see Dockerfile) —
# no network lookup needed at runtime. For local (non-Docker) testing, falls
# back to whatever `chromedriver` is on PATH.
_CHROMEDRIVER_PATH: str | None = os.getenv("CHROMEDRIVER_PATH") or shutil.which("chromedriver")
if not _CHROMEDRIVER_PATH:
    raise RuntimeError(
        "Could not locate chromedriver. In Docker this should come from the "
        "CHROMEDRIVER_PATH env var baked in at build time — rebuild the image. "
        "For local testing, install chromedriver and ensure it's on PATH, or "
        "set CHROMEDRIVER_PATH yourself."
    )


def _make_driver() -> webdriver.Chrome:
    """Return a headless Chrome driver with stealth settings applied."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # "--disable-images" is not a real Chrome flag and was silently ignored —
    # this is the actual way to skip image loading (saves bandwidth/time
    # since we only ever scrape text).
    options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )

    driver = webdriver.Chrome(service=Service(_CHROMEDRIVER_PATH), options=options)
    driver.set_page_load_timeout(45)

    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    return driver


# ---------------------------------------------------------------------------
# Driver pool for comment scraping
# ---------------------------------------------------------------------------
# The original version launched a brand-new headless Chrome process for
# EVERY post's comment thread, even under ThreadPoolExecutor — with a couple
# hundred posts per run that's a couple hundred Chrome launches (~150-300MB
# and a few seconds each). Since ThreadPoolExecutor reuses the same N worker
# threads for the life of the pool, we instead give each worker thread ONE
# persistent driver via thread-local storage, reused across every post that
# thread handles. This drops total Chrome launches from "one per post" to
# "one per worker thread" (MAX_WORKERS total).
_thread_local = threading.local()
_pooled_drivers: list[webdriver.Chrome] = []
_pool_lock = threading.Lock()


def _get_pooled_driver() -> webdriver.Chrome:
    if not hasattr(_thread_local, "driver"):
        driver = _make_driver()
        _thread_local.driver = driver
        with _pool_lock:
            _pooled_drivers.append(driver)
    return _thread_local.driver


def _quit_pooled_drivers() -> None:
    with _pool_lock:
        for driver in _pooled_drivers:
            try:
                driver.quit()
            except Exception:
                pass
        _pooled_drivers.clear()


def _parse_post_date(raw: str) -> str | None:
    """Convert a Beyond Blue relative/absolute date string to YYYY-MM-DD."""
    today = datetime.now()
    raw = raw.strip().lower()

    # e.g. "3 hours ago", "45 minutes ago"
    if re.search(r"\d+\s*(hour|minute|min)", raw):
        return today.strftime("%Y-%m-%d")

    if raw in _WEEKDAYS:
        delta = (today.weekday() - _WEEKDAYS.index(raw)) % 7
        return (today - timedelta(days=delta)).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*week", raw)
    if m:
        return (today - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*month", raw)
    if m:
        return (today - timedelta(days=30 * int(m.group(1)))).strftime("%Y-%m-%d")

    try:
        return datetime.strptime(raw, "%d-%m-%Y").strftime("%Y-%m-%d")
    except ValueError:
        print(f"[Warning] Unparseable date: {raw!r}")
        return None


def _convert_emojis(text: str) -> str:
    """Replace kaomoji, emoticons, and Unicode emoji with text equivalents."""
    for kaomoji, label in KAOMOJI.items():
        if kaomoji in text:
            text = text.replace(kaomoji, f" {label} ")

    e = emot.core.emot()
    result = e.emoticons(text)
    for orig, meaning in zip(result["value"], result["mean"]):
        text = text.replace(orig, f" {meaning} ")

    return emoji.demojize(text).strip().lower()


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _scrape_comments(url: str, wait: int = 20) -> str:
    """Return all comments for a single post joined by ' ||| '."""
    driver = _get_pooled_driver()
    comments: list[str] = []
    for attempt in range(2):
        try:
            driver.get(url)
            WebDriverWait(driver, wait).until(
                EC.presence_of_element_located((By.CLASS_NAME, "linear-message-list"))
            )
            break
        except TimeoutException:
            if attempt == 1:
                return ""

    soup = BeautifulSoup(driver.page_source, "html.parser")
    section = soup.find("div", class_="lia-component-message-list-detail-with-inline-editors")
    if not section:
        return ""
    msg_list = section.find("div", class_="linear-message-list message-list")
    if not msg_list:
        return ""

    for comment_div in msg_list.find_all("div", recursive=False):
        main = comment_div.find("div", class_="lia-quilt-row lia-quilt-row-message-main")
        if not main:
            continue
        try:
            comments.append(_convert_emojis(main.get_text(separator=" ", strip=True)))
        except Exception as exc:
            print(f"[Warning] Comment parse error: {exc}")

    return " ||| ".join(comments)


def _scrape_tag(tag: str, base_url: str, pages: int, existing_ids: set[str]) -> list[dict]:
    """Scrape *pages* pages of forum listings for *tag* and return post records."""
    driver = _make_driver()
    url = base_url
    posts: list[dict] = []

    # One pool for the whole tag (not recreated per page) so the MAX_WORKERS
    # worker threads — and the pooled Chrome driver each one builds on first
    # use — get reused across every page of this tag, not just one page.
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for page in tqdm(range(1, pages + 1), desc=f"[{tag}] pages"):
                for attempt in range(2):
                    try:
                        driver.get(url)
                        WebDriverWait(driver, 8).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "custom-message-list"))
                        )
                        break
                    except TimeoutException:
                        if attempt == 1:
                            print(f"[Warning] Timeout on page {page}, skipping.")
                            break

                soup = BeautifulSoup(driver.page_source, "html.parser")
                block = soup.find("div", class_="custom-message-list all-discussions")
                if not block:
                    print(f"[Warning] No discussion list on page {page}.")
                    break

                post_links: list[tuple] = []
                for article in block.find_all("article"):
                    try:
                        href = article.find("h3").find_all("a")[1]["href"]
                        post_id = href.split("/")[-1]
                        if post_id in existing_ids:
                            continue
                        full_url = f"https://forums.beyondblue.org.au{href}"
                        post_links.append((article, full_url, post_id))
                    except Exception as exc:
                        print(f"[Error] Article parse: {exc}")

                # Parallel comment scraping (bounded thread pool, drivers pooled per worker)
                def _scrape_one(args):
                    article, full_url, post_id = args
                    try:
                        cat_aside = article.find("aside")
                        cat_info = (
                            cat_aside.find("div", class_="custom-tile-category-content")
                            if cat_aside else None
                        )
                        raw_date = (
                            cat_info.find("time").text.strip()
                            if cat_info and cat_info.find("time") else ""
                        )
                        post_date = _parse_post_date(raw_date)
                        if post_date and pd.to_datetime(post_date) < pd.to_datetime(CUTOFF_DATE):
                            return None

                        title_tag = article.find("h3").find_all("a")[1]
                        content_tag = article.find("p", class_="body-text")
                        author_info = (
                            article.find("aside")
                            .find("div", class_="custom-tile-author-info")
                            if article.find("aside") else None
                        )
                        reply_info = article.find("li", class_="custom-tile-replies")

                        return {
                            "Post ID": post_id,
                            "Post Title": _convert_emojis(title_tag.text.strip()) if title_tag else "",
                            "Post Content": _convert_emojis(content_tag.text.strip()) if content_tag else "",
                            "Post Author": (
                                author_info.find("a").find("span").text.strip()
                                if author_info and author_info.find("a") else ""
                            ),
                            "Post Date": post_date,
                            "Post Category": tag,
                            "Number of Comments": (
                                reply_info.find("b").text.strip()
                                if reply_info and reply_info.find("b") else "0"
                            ),
                            "Comments": _scrape_comments(full_url),
                        }
                    except Exception as exc:
                        print(f"[Error] Post {post_id}: {exc}")
                        return None

                futures = {pool.submit(_scrape_one, args): args for args in post_links}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        posts.append(result)

                next_page = soup.find("a", rel="next")
                if next_page and next_page.get("href", "").startswith("http"):
                    url = next_page["href"]
                    time.sleep(1)
                else:
                    break
    finally:
        driver.quit()
        _quit_pooled_drivers()

    return posts


# ---------------------------------------------------------------------------
# Azure Blob helpers
# ---------------------------------------------------------------------------

def _blob_client() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(AZURE_CONN_STR)


def _ensure_container() -> None:
    """Create the target container if it doesn't already exist (idempotent)."""
    client = _blob_client()
    try:
        client.create_container(AZURE_CONTAINER)
        print(f"  Created container '{AZURE_CONTAINER}'")
    except Exception:
        # Already exists (ResourceExistsError) — nothing to do.
        pass


def _load_existing_ids(tag: str) -> set[str]:
    """Pull the existing Parquet for *tag* from Blob and return its Post IDs."""
    client = _blob_client()
    blob_name = f"raw/{tag}_posts.parquet"
    try:
        data = (
            client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)
            .download_blob()
            .readall()
        )
        return set(pd.read_parquet(BytesIO(data))["Post ID"].astype(str))
    except Exception:
        return set()


def _upload_parquet(df: pd.DataFrame, blob_name: str) -> None:
    client = _blob_client()
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name).upload_blob(
        buf, overwrite=True
    )
    print(f"  ✅ Uploaded {len(df)} rows → {blob_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Main entry point called by Airflow Task 1."""
    print("=" * 60)
    print(f"Script 1 — Scrape & Clean  [{datetime.utcnow().isoformat()} UTC]")
    print(f"Tags: {BEYONDBLUE_TAGS}  |  Pages per tag: {SCRAPE_PAGES}")
    print("=" * 60)

    _ensure_container()

    for tag in BEYONDBLUE_TAGS:
        base_url = TAG_URLS.get(tag)
        if not base_url:
            print(f"[Warning] No URL configured for tag '{tag}', skipping.")
            continue

        print(f"\n→ Scraping tag: {tag}")
        existing_ids = _load_existing_ids(tag)
        print(f"  Existing post IDs in Blob: {len(existing_ids)}")

        new_posts = _scrape_tag(tag, base_url, SCRAPE_PAGES, existing_ids)
        if not new_posts:
            print(f"  No new posts found for '{tag}'.")
            continue

        new_df = pd.DataFrame(new_posts)
        new_df["Post Date"] = pd.to_datetime(new_df["Post Date"], errors="coerce")

        # Merge with existing data from Blob (if any)
        if existing_ids:
            existing_data = (
                _blob_client()
                .get_blob_client(container=AZURE_CONTAINER, blob=f"raw/{tag}_posts.parquet")
                .download_blob()
                .readall()
            )
            existing_df = pd.read_parquet(BytesIO(existing_data))
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.drop_duplicates(subset=["Post ID"]).reset_index(drop=True)
        _upload_parquet(combined, f"raw/{tag}_posts.parquet")

    print("\n✅ Script 1 complete.\n")


if __name__ == "__main__":
    run()