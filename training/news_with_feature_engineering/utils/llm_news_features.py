"""
Per-calendar-day news feature extraction via LLM.

One API call per day with news: the LLM sees all of that day's articles
together and returns a single structured JSON record of features for
that day. Results are cached to disk (keyed by date) so re-running the
script doesn't re-pay for days already scored — important given this
step has a real, cumulative dollar cost, unlike everything else in the
pipeline so far.
"""

import json
import os
import time
from dataclasses import dataclass, asdict

import pandas as pd
from openai import OpenAI

MODEL = "gpt-5-mini"  # swap to "gpt-5-mini" if nano's judgments look too shallow on your validation sample
CACHE_PATH = "news_features_cache.jsonl"

SYSTEM_PROMPT = """You are a financial news analyst. You will be given one \
or more news articles about NVIDIA (NVDA) published on a single day. \
Analyze them TOGETHER as a set and return one summary of that day's news \
for NVDA specifically (not the sector or market in general, unless NVDA \
isn't otherwise mentioned).

If there is no meaningful NVDA-specific news in the provided text, \
return event_type "none", event_materiality 0.0, sentiment_score 0.0, \
sentiment_confidence 0.0, sentiment_disagreement 0.0, is_surprise false, \
contains_guidance_change false, time_horizon_of_impact "immediate"."""

# JSON Schema for OpenAI's Structured Outputs — the API guarantees the
# response matches this exactly, so no prompt-level "return only JSON"
# instruction or manual fence-stripping is needed (unlike the
# prompt-only approach used for the Anthropic version).
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "daily_news_features",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment_score": {"type": "number", "description": "-1 (very negative) to 1 (very positive), toward NVDA"},
                "sentiment_confidence": {"type": "number", "description": "0 (ambiguous/mixed) to 1 (very clear-cut)"},
                "sentiment_disagreement": {"type": "number", "description": "0 (articles agree) to 1 (articles conflict); 0 if only one article"},
                "event_type": {
                    "type": "string",
                    "enum": [
                        "earnings", "guidance", "product_launch", "regulatory_export_control",
                        "competitor_news", "supply_chain", "macro_fed", "mergers_acquisitions",
                        "leadership_change", "analyst_rating_change", "other", "none",
                    ],
                },
                "event_materiality": {"type": "number", "description": "0 (trivial/routine) to 1 (highly likely to move the stock)"},
                "is_surprise": {"type": "boolean"},
                "contains_guidance_change": {"type": "boolean"},
                "time_horizon_of_impact": {"type": "string", "enum": ["immediate", "short_term", "long_term"]},
            },
            "required": [
                "sentiment_score", "sentiment_confidence", "sentiment_disagreement",
                "event_type", "event_materiality", "is_surprise",
                "contains_guidance_change", "time_horizon_of_impact",
            ],
            "additionalProperties": False,
        },
    },
}


@dataclass
class DailyNewsFeatures:
    date: str
    sentiment_score: float
    sentiment_confidence: float
    sentiment_disagreement: float
    event_type: str
    event_materiality: float
    is_surprise: bool
    contains_guidance_change: bool
    time_horizon_of_impact: str
    article_count: int


def _load_cache() -> dict[str, dict]:
    """Load already-scored days from the cache file, keyed by date string."""
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    with open(CACHE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[record["date"]] = record
    return cache


def _append_to_cache(record: dict) -> None:
    """Append one day's result to the cache file (append-only, crash-safe)."""
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _build_prompt(articles_for_day: pd.DataFrame, text_col: str, title_col: str = "title") -> str:
    """Concatenate a day's articles into one prompt, capping length to control cost."""
    parts = []
    for i, row in articles_for_day.iterrows():
        title = row.get(title_col, "")
        text = row.get(text_col, "")
        if not isinstance(text, str):
            text = ""
        # cap each article's text to keep prompt size (and cost) bounded
        text = text[:1500]
        parts.append(f"Article {i + 1} — {title}\n{text}")
    return "\n\n---\n\n".join(parts)


def _call_llm(client: OpenAI, prompt: str, max_retries: int = 3) -> dict:
    """
    Call the LLM using OpenAI's Structured Outputs (response_format with
    a strict JSON schema) — the API enforces the schema server-side, so
    unlike a prompt-only "return JSON" instruction, malformed output
    isn't possible in normal operation. Retry is kept only as a safety
    net for transient API errors, not for parse failures.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=RESPONSE_SCHEMA,
            )
            raw_text = response.choices[0].message.content
            return json.loads(raw_text)  # safe: schema is enforced by the API
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    raise RuntimeError("unreachable")


def extract_daily_news_features(
    news_df: pd.DataFrame,
    date_col: str = "trading_day",
    text_col: str = "content",
    title_col: str = "title",
) -> pd.DataFrame:
    """
    Run the per-day LLM scoring pipeline over a news dataframe.

    Args:
        news_df: one row per article, with a date column and text column.
        date_col: column identifying which calendar/trading day each
            article belongs to (use the trading-day-aligned column from
            your mcal-based preprocessing, not the raw timestamp).
        text_col: column with article body text.
        title_col: column with article headline.

    Returns:
        DataFrame with one row per day, columns matching DailyNewsFeatures.
        Days already present in the cache are skipped (no API call made).
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    cache = _load_cache()
    news_df = news_df.copy()
    news_df[date_col] = pd.to_datetime(news_df[date_col]).dt.normalize()

    unique_days = sorted(news_df[date_col].unique())
    print(f"{len(unique_days)} unique days with news, {len(cache)} already cached")

    for day in unique_days:
        date_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        if date_str in cache:
            continue  # already scored, skip the API call entirely

        day_articles = news_df[news_df[date_col] == day]
        prompt = _build_prompt(day_articles, text_col=text_col, title_col=title_col)

        try:
            result = _call_llm(client, prompt)
        except Exception as e:
            print(f"  FAILED on {date_str}: {e} — skipping, will retry on next run")
            continue

        record = {
            "date": date_str,
            "article_count": len(day_articles),
            **result,
        }
        _append_to_cache(record)
        cache[date_str] = record
        print(f"  scored {date_str} ({len(day_articles)} articles): "
              f"sentiment={result.get('sentiment_score'):.2f} "
              f"event={result.get('event_type')} "
              f"materiality={result.get('event_materiality'):.2f}")

    daily_features = pd.DataFrame(cache.values())
    daily_features["date"] = pd.to_datetime(daily_features["date"])
    daily_features = daily_features.sort_values("date").reset_index(drop=True)
    return daily_features