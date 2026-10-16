"""
Turn per-day LLM-derived news features into rolling window features,
ready to merge onto the stock dataframe — same shape as the
NEWS_ARTICLE_COUNT_5D feature built earlier, just with richer inputs.
"""

import numpy as np
import pandas as pd


def build_rolling_news_features(
    daily_features: pd.DataFrame,
    news_start_date: pd.Timestamp,
    news_end_date: pd.Timestamp,
    window_days: int = 5,
) -> pd.DataFrame:
    """
    Reindex daily LLM features onto every calendar day in range, fill
    no-news days with neutral defaults, then compute rolling aggregates.

    Different features get different aggregation functions deliberately:
    - sentiment_score -> mean (average tone over the window)
    - sentiment_confidence -> mean
    - sentiment_disagreement -> max (the most-conflicted day dominates)
    - event_materiality -> max (the single biggest event matters most,
      not the average of it with several quiet days)
    - is_surprise -> any (a surprise anywhere in the window is a surprise)
    - contains_guidance_change -> any
    - article_count -> sum (total attention over the window)
    """
    daily_features = daily_features.copy()
    daily_features["date"] = pd.to_datetime(daily_features["date"]).dt.normalize()

    full_range = pd.DataFrame({
        "DATE": pd.date_range(news_start_date, news_end_date, freq="D")
    })
    df = full_range.merge(daily_features, left_on="DATE", right_on="date", how="left")
    df = df.drop(columns=["date"])

    # fill no-news days with neutral values
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    df["sentiment_confidence"] = df["sentiment_confidence"].fillna(0.0)
    df["sentiment_disagreement"] = df["sentiment_disagreement"].fillna(0.0)
    df["event_materiality"] = df["event_materiality"].fillna(0.0)
    df["is_surprise"] = df["is_surprise"].fillna(False).astype(bool)
    df["contains_guidance_change"] = df["contains_guidance_change"].fillna(False).astype(bool)
    df["article_count"] = df["article_count"].fillna(0)

    df = df.sort_values("DATE").reset_index(drop=True)

    roll = df.rolling(window=window_days, min_periods=1)
    out = pd.DataFrame({"DATE": df["DATE"]})
    out[f"NEWS_SENTIMENT_MEAN_{window_days}D"] = roll["sentiment_score"].mean()
    out[f"NEWS_CONFIDENCE_MEAN_{window_days}D"] = roll["sentiment_confidence"].mean()
    out[f"NEWS_DISAGREEMENT_MAX_{window_days}D"] = roll["sentiment_disagreement"].max()
    out[f"NEWS_MATERIALITY_MAX_{window_days}D"] = roll["event_materiality"].max()
    out[f"NEWS_HAS_SURPRISE_{window_days}D"] = (
        df["is_surprise"].rolling(window_days, min_periods=1).apply(lambda x: x.any()).astype(bool)
    )
    out[f"NEWS_HAS_GUIDANCE_CHANGE_{window_days}D"] = (
        df["contains_guidance_change"].rolling(window_days, min_periods=1).apply(lambda x: x.any()).astype(bool)
    )
    out[f"NEWS_ARTICLE_COUNT_{window_days}D"] = roll["article_count"].sum()

    # point-in-time safety: shift by 1 day so day t's feature only reflects
    # news through day t-1 (same rule as the count-only version earlier)
    feature_cols = [c for c in out.columns if c != "DATE"]
    out[feature_cols] = out[feature_cols].shift(1)
    out = out.dropna(subset=feature_cols).reset_index(drop=True)

    # convert booleans to 0/1 for model input
    out[f"NEWS_HAS_SURPRISE_{window_days}D"] = out[f"NEWS_HAS_SURPRISE_{window_days}D"].astype(float)
    out[f"NEWS_HAS_GUIDANCE_CHANGE_{window_days}D"] = out[f"NEWS_HAS_GUIDANCE_CHANGE_{window_days}D"].astype(float)

    return out
