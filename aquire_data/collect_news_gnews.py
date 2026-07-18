"""
Multi-Company News Collector — Historical Data for Automated Trading
====================================================================
Collects news articles for multiple companies from GNews API (from 2020
to today) and saves each company to its own portable CSV file.

HOW TO RUN:
  1. pip install requests
  2. Set your API key in the CONFIG block below
  3. python collect_news.py

OUTPUT:
  nvidia_news.csv, apple_news.csv, ... — one file per company
"""

import csv
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv()

# ════════════════════════════════════════════════════════
#  CONFIG — edit these before running
# ════════════════════════════════════════════════════════

API_KEY = os.getenv('GNEWS_KEY', "API_KEY")       # paste your GNews key here

# Add or remove companies freely — each gets its own CSV + progress file
# Queries use GNews logical operators: AND, OR, NOT, "exact phrase"
COMPANIES = [
    {"name": "NVIDIA",        "query": "NVIDIA OR NVDA"},
    {"name": "Apple",         "query": "Apple AND (AAPL OR iPhone OR MacBook)"},
    {"name": "Microsoft",     "query": "Microsoft OR MSFT"},
    {"name": "Tesla",         "query": "Tesla OR TSLA"},
    {"name": "Google",        "query": "Google OR Alphabet OR GOOGL"},
    {"name": "CDProjektRed",  "query": '"CD Projekt" OR CDPR OR Cyberpunk OR Witcher'},
]

START_DATE = "2020-01-01"              # GNews student plan starts here
END_DATE   = datetime.today().strftime("%Y-%m-%d")   # up to today

LANGUAGE       = "en"
COUNTRY        = "us"
MAX_PER_PAGE   = 100   # maximum allowed by the API
MAX_PAGES      = 3     # pages per week chunk (1 page = 1 request, 3 pages = up to 300 articles/week)
                       # lower this if you want to save quota, raise to 10 for maximum coverage
CHUNK_DAYS     = 7     # one week per set of pages

# ════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger()

BASE_URL    = "https://gnews.io/api/v4/search"
DAILY_LIMIT = 1000

CSV_COLUMNS = [
    "published_at",
    "title",
    "description",
    "content",
    "url",
    "source_name",
    "source_url",
    "image_url",
    "query",
    "week_start",
    "week_end",
]


# ─── Quota tracker ────────────────────────────────────────────────────────────

def load_quota() -> dict:
    qfile = Path(".quota.json")
    if qfile.exists():
        data = json.loads(qfile.read_text())
        if data.get("date") == str(datetime.today().date()):
            return data
    return {"date": str(datetime.today().date()), "used": 0}

def save_quota(q: dict):
    Path(".quota.json").write_text(json.dumps(q))

def quota_remaining() -> int:
    return DAILY_LIMIT - load_quota()["used"]

def consume_quota():
    q = load_quota()
    q["used"] += 1
    save_quota(q)

def wait_until_tomorrow():
    """Sleep until just after midnight so the quota resets."""
    now       = datetime.now()
    tomorrow  = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    seconds   = (tomorrow - now).total_seconds()
    hours     = int(seconds // 3600)
    minutes   = int((seconds % 3600) // 60)
    log.info(f"Quota exhausted — sleeping {hours}h {minutes}m until {tomorrow.strftime('%Y-%m-%d %H:%M')}…")
    time.sleep(seconds)
    log.info("Woke up — quota reset, resuming.")


# ─── Progress / resume ────────────────────────────────────────────────────────

def load_progress(progress_file: str) -> set:
    """Returns a set of 'YYYY-MM-DD' week_start strings already collected."""
    p = Path(progress_file)
    if p.exists():
        return set(json.loads(p.read_text()).get("done", []))
    return set()

def save_progress(done: set, progress_file: str):
    Path(progress_file).write_text(json.dumps({"done": sorted(done)}))


# ─── API call ─────────────────────────────────────────────────────────────────

def fetch_week(from_date: str, to_date: str, query: str) -> list[dict]:
    """
    Fetch all pages for one week chunk.
    Each page = 1 API request. Returns deduplicated list of articles.
    Per docs: max 100 per page, up to page 10 (1000 articles max per query).
    """
    all_articles = []
    seen_urls    = set()

    for page in range(1, MAX_PAGES + 1):
        if quota_remaining() < 1:
            wait_until_tomorrow()

        params = {
            "q":        query,
            "lang":     LANGUAGE,
            "country":  COUNTRY,
            "max":      MAX_PER_PAGE,
            "in":       "title,description,content",   # search full text
            "nullable": "description,content,image",   # don't skip articles missing these
            "from":     f"{from_date}T00:00:00Z",
            "to":       f"{to_date}T23:59:59Z",
            "sortby":   "publishedAt",
            "page":     page,
            "apikey":   API_KEY,
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            data     = r.json()
            articles = data.get("articles", [])
            consume_quota()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            if code == 403:
                log.error("403 — invalid API key or server-side quota hit.")
            elif code == 429:
                log.warning("429 — rate limited. Waiting 60s…")
                time.sleep(60)
                continue
            else:
                log.error(f"HTTP {code}: {e}")
            break
        except requests.exceptions.ConnectionError:
            log.error("Network error. Check your connection.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            break

        if not articles:
            break   # no more results for this week

        # deduplicate within the week
        fresh = [a for a in articles if a.get("url") not in seen_urls]
        seen_urls.update(a["url"] for a in fresh if a.get("url"))
        all_articles.extend(fresh)

        if len(articles) < MAX_PER_PAGE:
            break   # last page — no need to fetch another

        time.sleep(0.3)   # small pause between pages

    return all_articles


# ─── Date range generator ─────────────────────────────────────────────────────

def date_chunks(start: str, end: str, step_days: int):
    """Yields (from_date, to_date) string pairs in YYYY-MM-DD format."""
    cursor = datetime.strptime(start, "%Y-%m-%d")
    dt_end = datetime.strptime(end,   "%Y-%m-%d")
    delta  = timedelta(days=step_days)
    while cursor <= dt_end:
        chunk_end = min(cursor + delta - timedelta(days=1), dt_end)
        yield cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cursor += delta


# ─── CSV helpers ──────────────────────────────────────────────────────────────

def article_to_row(art: dict, from_date: str, to_date: str, query: str) -> dict:
    def clean(val):
        return (val or "").replace("\n", " ")
    return {
        "published_at": art.get("publishedAt", ""),
        "title":        clean(art.get("title")),
        "description":  clean(art.get("description")),
        "content":      clean(art.get("content")),
        "url":          art.get("url",   ""),
        "source_name":  art.get("source", {}).get("name", ""),
        "source_url":   art.get("source", {}).get("url",  ""),
        "image_url":    art.get("image", ""),
        "query":        query,
        "week_start":   from_date,
        "week_end":     to_date,
    }

def init_csv(path: Path):
    """Write CSV header only if file doesn't exist yet."""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def append_rows(path: Path, rows: list[dict]):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


# ─── Per-company collection ───────────────────────────────────────────────────

def collect_company(company: dict) -> None:
    name          = company["name"]
    query         = company["query"]
    output_file   = f"{name.lower().replace(' ', '_')}_news.csv"
    progress_file = f".progress_{name.lower().replace(' ', '_')}.json"

    output = Path(output_file)
    init_csv(output)

    done      = load_progress(progress_file)
    chunks    = list(date_chunks(START_DATE, END_DATE, CHUNK_DAYS))
    total     = len(chunks)
    remaining = [c for c in chunks if c[0] not in done]
    new_articles = 0

    log.info(f"  Company  : {name}")
    log.info(f"  Query    : '{query}'")
    log.info(f"  Progress : {len(done)}/{total} weeks done  |  {len(remaining)} remaining")
    log.info(f"  Max articles per week: {MAX_PER_PAGE * MAX_PAGES} ({MAX_PAGES} pages × {MAX_PER_PAGE})")
    log.info(f"  Requests needed: ~{len(remaining) * MAX_PAGES} (quota remaining: {quota_remaining()})")
    log.info(f"  Output   : {output.resolve()}")

    if not remaining:
        log.info(f"  ✓ Already fully collected — skipping.\n")
        return

    for i, (from_date, to_date) in enumerate(remaining, 1):
        if quota_remaining() < 1:
            wait_until_tomorrow()

        pct = int((i / len(remaining)) * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r  [{bar}]  {i}/{len(remaining)}  {from_date}  quota:{quota_remaining()}", end="", flush=True)

        articles = fetch_week(from_date, to_date, query)  # consume_quota called inside

        rows = [article_to_row(a, from_date, to_date, query) for a in articles]
        append_rows(output, rows)
        new_articles += len(rows)

        done.add(from_date)
        save_progress(done, progress_file)

        time.sleep(0.5)

    print()
    log.info(f"  ✓ Done — {new_articles} new articles saved.\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n  ⚠  Open this file and paste your API key into API_KEY at the top.\n")
        return

    log.info("═" * 55)
    log.info(f"  News Collector — {len(COMPANIES)} companies")
    log.info(f"  Range : {START_DATE}  →  {END_DATE}")
    log.info(f"  Quota : {quota_remaining()} requests remaining today")
    log.info("═" * 55 + "\n")

    for company in COMPANIES:
        log.info(f"── {company['name']} {'─' * (47 - len(company['name']))}")
        collect_company(company)
        if quota_remaining() < 1:
            wait_until_tomorrow()

    log.info("═" * 55)
    log.info("All done! Load any file with:")
    log.info("  import pandas as pd")
    log.info("  df = pd.read_csv('nvidia_news.csv')")
    log.info("  df['published_at'] = pd.to_datetime(df['published_at'])")


if __name__ == "__main__":
    main()