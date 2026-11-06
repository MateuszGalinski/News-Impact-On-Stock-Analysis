"""Build each track's features, chronological splits and model windows."""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR.parent / "data"
STOCK_PATH = DATA_DIR / "data_stooq" / "daily" / "us" / "nasdaq stocks" / "2" / "nvda.us.txt"
NEWS_PATH = DATA_DIR / "data_gnews_api" / "nvidia_news.csv"

NO_NEWS_NO_FEATURES = "no_news_no_feature_engineering"
NO_NEWS_WITH_FEATURES = "no_news_with_feature_engineering"
NEWS_WITH_FEATURES = "news_with_feature_engineering"
TRACKS = (NO_NEWS_NO_FEATURES, NO_NEWS_WITH_FEATURES, NEWS_WITH_FEATURES)

TARGET_COLUMN = "LOG_CLOSE_RETURN"
CYCLICAL_COLUMNS = ["sin_doy", "cos_doy", "sin_month", "cos_month", "sin_dow", "cos_dow"]
TECHNICAL_COLUMNS = ["CLOSE_TO_TEMA_10", "EMA_5_20_SPREAD"]

NEWS_WINDOW_DAYS = 5
NEWS_COLUMNS = [
    f"NEWS_SENTIMENT_MEAN_{NEWS_WINDOW_DAYS}D",
    f"NEWS_CONFIDENCE_MEAN_{NEWS_WINDOW_DAYS}D",
    f"NEWS_DISAGREEMENT_MAX_{NEWS_WINDOW_DAYS}D",
    f"NEWS_MATERIALITY_MAX_{NEWS_WINDOW_DAYS}D",
    f"NEWS_HAS_SURPRISE_{NEWS_WINDOW_DAYS}D",
    f"NEWS_HAS_GUIDANCE_CHANGE_{NEWS_WINDOW_DAYS}D",
    f"NEWS_ARTICLE_COUNT_{NEWS_WINDOW_DAYS}D",
]
SCALED_NEWS_COLUMNS = [f"NEWS_ARTICLE_COUNT_{NEWS_WINDOW_DAYS}D"]


@dataclass(frozen=True)
class WindowedSplit:
    """Model inputs for one split. Dates are the first day of each target."""

    X: np.ndarray  # (samples, window_days, features)
    y: np.ndarray  # (samples, horizons)
    dates: np.ndarray


@dataclass(frozen=True)
class Dataset:
    """A track's windows. `y` is always in log-return units.

    standardized: inputs are already standardized, and `fit_model` trains on a
        standardized target and converts predictions back to log returns.
    """

    track: str
    feature_columns: tuple[str, ...]
    horizons: tuple[int, ...]
    train: WindowedSplit
    validation: WindowedSplit
    test: WindowedSplit
    standardized: bool = False


def load_stock(path: Path = STOCK_PATH) -> pd.DataFrame:
    """Load Stooq prices sorted by date, with the daily log return added."""
    stock = pd.read_csv(path)
    stock = stock.rename(columns=lambda column: column.strip("<>"))
    stock = stock.drop(columns=["OPENINT", "TICKER", "PER", "TIME"])
    stock["DATE"] = pd.to_datetime(stock["DATE"], format="%Y%m%d")
    stock = stock.sort_values("DATE").reset_index(drop=True)
    stock[TARGET_COLUMN] = np.log(stock["CLOSE"] / stock["CLOSE"].shift(1))
    return stock


def add_technical_features(stock: pd.DataFrame) -> pd.DataFrame:
    stock = stock.copy()
    close = stock["CLOSE"]
    ema_5 = close.ewm(span=5, adjust=False, min_periods=5).mean()
    ema_10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema_10_2 = ema_10.ewm(span=10, adjust=False, min_periods=10).mean()
    ema_10_3 = ema_10_2.ewm(span=10, adjust=False, min_periods=10).mean()
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()

    tema_10 = 3 * ema_10 - 3 * ema_10_2 + ema_10_3
    stock["CLOSE_TO_TEMA_10"] = close / tema_10 - 1
    stock["EMA_5_20_SPREAD"] = ema_5 / ema_20 - 1
    return stock


def add_cyclical_features(stock: pd.DataFrame) -> pd.DataFrame:
    stock = stock.copy()
    day_of_year = stock["DATE"].dt.dayofyear
    month = stock["DATE"].dt.month
    day_of_week = stock["DATE"].dt.dayofweek

    stock["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365)
    stock["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365)
    stock["sin_month"] = np.sin(2 * np.pi * month / 12)
    stock["cos_month"] = np.cos(2 * np.pi * month / 12)
    stock["sin_dow"] = np.sin(2 * np.pi * day_of_week / 7)
    stock["cos_dow"] = np.cos(2 * np.pi * day_of_week / 7)
    return stock


def load_aligned_news(path: Path = NEWS_PATH) -> pd.DataFrame:
    """Load news with `trading_day`: the NASDAQ session each article can first affect.

    Articles published at or after 16:00 US/Eastern roll to the next day, then
    any non-trading day rolls forward to the next session.
    """
    import pandas_market_calendars as mcal

    news = pd.read_csv(path)
    published_at = pd.to_datetime(news["published_at"], format="ISO8601")

    sessions = mcal.get_calendar("NASDAQ").schedule(
        start_date="1999-01-01", end_date="2027-01-01"
    ).index
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    trading_days = set(sessions.normalize())

    eastern = published_at.dt.tz_convert("America/New_York")
    is_after_close = (eastern.dt.time >= pd.Timestamp("16:00").time()).astype(int)
    day = eastern.dt.normalize().dt.tz_localize(None) + pd.to_timedelta(is_after_close, unit="D")

    news["trading_day"] = pd.to_datetime(
        [next(d for d in pd.date_range(start, periods=10) if d in trading_days) for start in day]
    )
    return news


def add_news_features(stock: pd.DataFrame) -> pd.DataFrame:
    """Merge rolling LLM news features onto the overlapping date range.

    Reads scores from the cache only, so this never calls the API or costs
    anything. Days missing from the cache are treated as no-news days, the
    same way the notebook treats days it failed to score.
    """
    if str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))
    from news_with_feature_engineering.utils.llm_news_features import (
        load_cached_daily_news_features,
    )
    from news_with_feature_engineering.utils.rolling_news_features import (
        build_rolling_news_features,
    )

    news = load_aligned_news()
    start = max(news["trading_day"].min(), stock["DATE"].min())
    end = min(news["trading_day"].max(), stock["DATE"].max())
    news = news[news["trading_day"].between(start, end)]
    stock = stock[stock["DATE"].between(start, end)]

    daily_features = load_cached_daily_news_features()
    missing_days = set(news["trading_day"]) - set(daily_features["date"])
    if missing_days:
        print(
            f"Warning: {len(missing_days)} news days have no cached LLM scores "
            "and count as no-news days"
        )

    rolling = build_rolling_news_features(
        daily_features,
        news_start_date=news["trading_day"].min(),
        news_end_date=news["trading_day"].max(),
        window_days=NEWS_WINDOW_DAYS,
    )
    return stock.merge(rolling, on="DATE", how="inner")


def build_track_frame(track: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return a track's rows, feature columns, and columns to scale."""
    stock = load_stock()
    if track == NO_NEWS_NO_FEATURES:
        features = [TARGET_COLUMN, *CYCLICAL_COLUMNS]
        scaled = []
    elif track == NO_NEWS_WITH_FEATURES:
        stock = add_technical_features(stock)
        features = [TARGET_COLUMN, *TECHNICAL_COLUMNS, *CYCLICAL_COLUMNS]
        scaled = []
    elif track == NEWS_WITH_FEATURES:
        features = [TARGET_COLUMN, *CYCLICAL_COLUMNS, *NEWS_COLUMNS]
        scaled = SCALED_NEWS_COLUMNS
    else:
        raise ValueError(f"Unknown track {track!r}, expected one of {TRACKS}")

    stock = stock.dropna().reset_index(drop=True)
    stock = add_cyclical_features(stock)
    if track == NEWS_WITH_FEATURES:
        stock = add_news_features(stock).sort_values("DATE").reset_index(drop=True)
    return stock, features, scaled


def _restrict_to_news_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the news track's dates, so every track gets the same rows."""
    news_dates = build_track_frame(NEWS_WITH_FEATURES)[0]["DATE"]
    missing = set(news_dates) - set(frame["DATE"])
    if missing:
        raise ValueError(f"{len(missing)} news track dates are missing, first is {min(missing)}")
    return frame[frame["DATE"].isin(news_dates)].reset_index(drop=True)


def _standardize_inputs(split: WindowedSplit, scaler: StandardScaler, indices: list[int]) -> WindowedSplit:
    X = split.X.copy()
    rows = X[:, :, indices].reshape(-1, len(indices))
    X[:, :, indices] = scaler.transform(rows).reshape(X.shape[0], X.shape[1], len(indices))
    return replace(split, X=X)


def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_days: int,
    horizons: tuple[int, ...],
) -> WindowedSplit:
    """Build one split's windows, using only that split's rows.

    Each target is the log return summed over the `horizon` days right after
    the window. Windows never cross a split, so training targets can't use
    prices from validation or test, and every horizon shares the same rows.
    """
    values = frame[feature_columns].to_numpy()
    returns = frame[TARGET_COLUMN].to_numpy()
    dates = frame["DATE"].to_numpy()

    sample_count = len(frame) - window_days - max(horizons) + 1
    if sample_count < 1:
        raise ValueError(
            f"{len(frame)} rows can't fit a {window_days}-day window and a "
            f"{max(horizons)}-day target"
        )

    X = np.stack([values[start:start + window_days] for start in range(sample_count)])
    y = np.array(
        [
            [returns[start + window_days:start + window_days + horizon].sum() for horizon in horizons]
            for start in range(sample_count)
        ]
    )
    return WindowedSplit(X=X, y=y, dates=dates[window_days:window_days + sample_count])


def prepare_dataset(
    track: str,
    *,
    horizons: tuple[int, ...] = (1,),
    window_days: int = 30,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    common_dates: bool = False,
    standardize: bool = False,
) -> Dataset:
    """Build a track's windows from a chronological train/validation/test split.

    common_dates: limit the track to the news track's dates, so all tracks share
        the same splits and targets. Indicators are calculated first, on the
        full price history.
    standardize: standardize the inputs using training rows only. Cyclical date
        features stay as they are because they're already between -1 and 1.
    """
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and leave rows for test")

    frame, features, scaled = build_track_frame(track)
    if common_dates and track != NEWS_WITH_FEATURES:
        frame = _restrict_to_news_dates(frame)

    train_size = int(len(frame) * train_fraction)
    validation_size = int(len(frame) * validation_fraction)
    train = frame.iloc[:train_size].copy()
    validation = frame.iloc[train_size:train_size + validation_size].copy()
    test = frame.iloc[train_size + validation_size:].copy()

    if scaled:
        scaler = StandardScaler().fit(train[scaled])
        for split in (train, validation, test):
            split[scaled] = scaler.transform(split[scaled])

    # Standardized after windowing because the return column also builds the targets.
    train_windows = make_windows(train, features, window_days, horizons)
    validation_windows = make_windows(validation, features, window_days, horizons)
    test_windows = make_windows(test, features, window_days, horizons)

    if standardize:
        columns = [c for c in features if c not in CYCLICAL_COLUMNS and c not in scaled]
        indices = [features.index(column) for column in columns]
        scaler = StandardScaler().fit(train[columns].to_numpy())
        train_windows = _standardize_inputs(train_windows, scaler, indices)
        validation_windows = _standardize_inputs(validation_windows, scaler, indices)
        test_windows = _standardize_inputs(test_windows, scaler, indices)

    return Dataset(
        track=track,
        feature_columns=tuple(features),
        horizons=tuple(horizons),
        train=train_windows,
        validation=validation_windows,
        test=test_windows,
        standardized=standardize,
    )
