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
NEWS_WITH_FEATURES_AND_TECH = "news_with_feature_engineering_and_tech"
TRACKS = (NO_NEWS_NO_FEATURES, NO_NEWS_WITH_FEATURES, NEWS_WITH_FEATURES)
UNIFIED_TRACKS = (*TRACKS, NEWS_WITH_FEATURES_AND_TECH)

TARGET_COLUMN = "LOG_CLOSE_RETURN"
RETURN_TARGET = "return"
MOVE_SIZE_TARGET = "move_size"
MOVE_INPUT_COLUMN = "ABS_LOG_RETURN"
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
    """A track's windows. `y` is always in log-return units, signed or absolute.

    standardized: inputs are already standardized, and `fit_model` trains on a
        standardized target and converts predictions back to those units.
    early_stopping: data for early stopping when it's separate from `validation`.
    """

    track: str
    feature_columns: tuple[str, ...]
    horizons: tuple[int, ...]
    train: WindowedSplit
    validation: WindowedSplit
    test: WindowedSplit
    standardized: bool = False
    early_stopping: WindowedSplit | None = None

    @property
    def stopping_split(self) -> WindowedSplit:
        """The split that early stopping scores against."""
        return self.validation if self.early_stopping is None else self.early_stopping


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
    elif track == NEWS_WITH_FEATURES_AND_TECH:
        stock = add_technical_features(stock)
        features = [TARGET_COLUMN, *TECHNICAL_COLUMNS, *CYCLICAL_COLUMNS, *NEWS_COLUMNS]
        scaled = SCALED_NEWS_COLUMNS
    else:
        raise ValueError(f"Unknown track {track!r}, expected one of {UNIFIED_TRACKS}")

    stock = stock.dropna().reset_index(drop=True)
    stock = add_cyclical_features(stock)
    if track in (NEWS_WITH_FEATURES, NEWS_WITH_FEATURES_AND_TECH):
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


def load_track_rows(
    track: str,
    *,
    common_dates: bool = False,
    target: str = RETURN_TARGET,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return a track's rows, feature columns, and columns to scale.

    common_dates limits the track to the news track's dates, after indicators
    have been calculated on the full price history.
    target: the move-size target adds each day's move as an input, so models
    don't have to work it out from signed returns.
    """
    frame, features, scaled = build_track_frame(track)
    if common_dates and track not in (NEWS_WITH_FEATURES, NEWS_WITH_FEATURES_AND_TECH):
        frame = _restrict_to_news_dates(frame)
    if target == MOVE_SIZE_TARGET:
        frame = frame.assign(**{MOVE_INPUT_COLUMN: frame[TARGET_COLUMN].abs()})
        features = [TARGET_COLUMN, MOVE_INPUT_COLUMN, *features[1:]]
    return frame, features, scaled


def split_sizes(row_count: int, train_fraction: float, validation_fraction: float) -> tuple[int, int]:
    """Return the train and validation row counts. The remaining rows are the test split."""
    return int(row_count * train_fraction), int(row_count * validation_fraction)


def _standardize_all(
    fit_rows: pd.DataFrame,
    features: list[str],
    scaled: list[str],
    splits: list[WindowedSplit],
) -> list[WindowedSplit]:
    """Standardize input windows using `fit_rows` only.

    Done after windowing because the return column also builds the targets.
    Cyclical date features stay as they are because they're already between -1
    and 1, and `scaled` columns have already been scaled.
    """
    columns = [c for c in features if c not in CYCLICAL_COLUMNS and c not in scaled]
    indices = [features.index(column) for column in columns]
    scaler = StandardScaler().fit(fit_rows[columns].to_numpy())
    return [_standardize_inputs(split, scaler, indices) for split in splits]


def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_days: int,
    horizons: tuple[int, ...],
    target: str = RETURN_TARGET,
) -> WindowedSplit:
    """Build one split's windows, using only that split's rows.

    Each target covers the `horizon` days right after the window. By default it's
    the summed log return. With the move-size target it's the average daily move,
    because summing first would let an up day and a down day cancel out.
    Windows never cross a split, so training targets can't use prices from
    validation or test, and every horizon shares the same rows.
    """
    if target not in (RETURN_TARGET, MOVE_SIZE_TARGET):
        raise ValueError(f"Unknown target {target!r}")
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
            [
                returns[start + window_days:start + window_days + horizon].sum()
                if target == RETURN_TARGET
                else np.abs(returns[start + window_days:start + window_days + horizon]).mean()
                for horizon in horizons
            ]
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
    early_stopping_fraction: float | None = None,
    target: str = RETURN_TARGET,
) -> Dataset:
    """Build a track's windows from a chronological train/validation/test split.

    common_dates: limit the track to the news track's dates, so all tracks share
        the same splits and targets. Indicators are calculated first, on the
        full price history.
    standardize: standardize the inputs using training rows only. Cyclical date
        features stay as they are because they're already between -1 and 1.
    early_stopping_fraction: keep this share of the end of the training rows for
        early stopping, so models don't stop on the data they're scored on. The
        slice's inputs can use the previous `window_days` rows. Models and
        scaling are fit on the rows before it.
    target: what the windows predict, as described in `make_windows`.
    """
    if early_stopping_fraction is not None and not 0 < early_stopping_fraction < 1:
        raise ValueError("early_stopping_fraction must be between 0 and 1")
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and leave rows for test")

    frame, features, scaled = load_track_rows(track, common_dates=common_dates, target=target)

    train_size, validation_size = split_sizes(len(frame), train_fraction, validation_fraction)
    train = frame.iloc[:train_size].copy()
    validation = frame.iloc[train_size:train_size + validation_size].copy()
    test = frame.iloc[train_size + validation_size:].copy()

    fit_size = train_size
    if early_stopping_fraction is not None:
        fit_size -= int(train_size * early_stopping_fraction)

    if scaled:
        scaler = StandardScaler().fit(train.iloc[:fit_size][scaled])
        for split in (train, validation, test):
            split[scaled] = scaler.transform(split[scaled])

    train_windows = make_windows(train.iloc[:fit_size], features, window_days, horizons, target)
    validation_windows = make_windows(validation, features, window_days, horizons, target)
    test_windows = make_windows(test, features, window_days, horizons, target)
    stopping_windows = None
    if early_stopping_fraction is not None:
        stopping_windows = make_windows(train.iloc[fit_size - window_days:], features, window_days, horizons, target)

    if standardize:
        fit_rows = train.iloc[:fit_size]
        train_windows, validation_windows, test_windows = _standardize_all(
            fit_rows, features, scaled, [train_windows, validation_windows, test_windows]
        )
        if stopping_windows is not None:
            stopping_windows = _standardize_all(fit_rows, features, scaled, [stopping_windows])[0]

    return Dataset(
        track=track,
        feature_columns=tuple(features),
        horizons=tuple(horizons),
        train=train_windows,
        validation=validation_windows,
        test=test_windows,
        standardized=standardize,
        early_stopping=stopping_windows,
    )


def prepare_walk_forward_round(
    frame: pd.DataFrame,
    features: list[str],
    scaled: list[str],
    *,
    cutoff: int,
    scored_end: int,
    track: str,
    horizons: tuple[int, ...] = (1,),
    window_days: int = 30,
    early_stopping_fraction: float = 0.1,
    standardize: bool = True,
    target: str = RETURN_TARGET,
) -> Dataset:
    """Build one walk-forward round. Rows are in time order.

    Rows before `cutoff` are known when the model trains, and `test` holds the
    scored rows from `cutoff` to `scored_end`. The end of the known rows becomes
    `validation`, the early stopping slice. Scaling is fit on the rows before it.

    A window's targets always stay inside its own range, but its inputs can use
    the previous `window_days` rows. Those days are known when the prediction
    is made, so every scored row gets a prediction.
    """
    validation_start = cutoff - int(cutoff * early_stopping_fraction)
    if validation_start < window_days or cutoff < window_days or scored_end <= cutoff:
        raise ValueError(f"Invalid round: cutoff {cutoff}, scored_end {scored_end}")

    rows = frame.iloc[:scored_end].copy()
    if scaled:
        scaler = StandardScaler().fit(rows.iloc[:validation_start][scaled])
        rows[scaled] = scaler.transform(rows[scaled])
    fit_rows = rows.iloc[:validation_start]

    splits = [
        make_windows(fit_rows, features, window_days, horizons, target),
        make_windows(rows.iloc[validation_start - window_days:cutoff], features, window_days, horizons, target),
        make_windows(rows.iloc[cutoff - window_days:scored_end], features, window_days, horizons, target),
    ]
    if standardize:
        splits = _standardize_all(fit_rows, features, scaled, splits)

    return Dataset(
        track=track,
        feature_columns=tuple(features),
        horizons=tuple(horizons),
        train=splits[0],
        validation=splits[1],
        test=splits[2],
        standardized=standardize,
    )
