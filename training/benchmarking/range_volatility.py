"""Range-based volatility test: predict the log of the next day's variance from prices.

Each day's variance is the overnight gap plus Garman–Klass, from the open, high,
low and close. It's much less noisy than the absolute return used in the
move-size test. Models train with MSE on the log, and forecasts are judged
with QLIKE and MSE after converting back to variance (Patton, 2011).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    NO_NEWS_NO_FEATURES,
    RANGE_TARGET,
    RANGE_VARIANCE_COLUMN,
    UNIFIED_TRACKS,
    load_stock,
)
from .experiment import ExperimentConfig, run_experiment
from .move_size import (
    HAR,
    HAR_WITH_NEWS,
    MOVE_MODELS,
    MOVE_SEEDS,
    NEWS_COMPARISONS,
    ROLLING_BASELINES,
    _models_for,
    _rank_correlation,
    _rank_correlations,
    _row_correlations,
)
from .walk_forward import WalkForwardConfig, _bootstrap_indices, load_development_rows, run_walk_forward

BAND_Z = 1.96


def run_range_walk_forward(results_path: Path) -> None:
    """Run every model through walk-forward. Saved fits are skipped."""
    for track in UNIFIED_TRACKS:
        run_walk_forward(
            WalkForwardConfig(track=track, models=_models_for(track), seeds=MOVE_SEEDS, target=RANGE_TARGET),
            results_path,
        )


def run_range_split(predictions_path: Path) -> None:
    """Run every model on the 80/10/10 split. Tracks already saved are skipped."""
    path = Path(predictions_path)
    done = set(pd.read_csv(path, usecols=["track"])["track"]) if path.exists() else set()
    for track in UNIFIED_TRACKS:
        if track in done:
            continue
        result = run_experiment(
            ExperimentConfig(
                track=track,
                models=_models_for(track),
                horizons=(1,),
                seeds=MOVE_SEEDS,
                common_dates=True,
                standardize=True,
                early_stopping_fraction=0.1,
                target=RANGE_TARGET,
            )
        )
        result["predictions"].to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"80/10/10: {track} done", flush=True)


def known_variances() -> pd.Series:
    """Return each day's variance, leaving out the locked test split."""
    config = WalkForwardConfig(track=NO_NEWS_NO_FEATURES, target=RANGE_TARGET)
    frame = load_development_rows(config)[0]
    return frame.set_index("DATE")[RANGE_VARIANCE_COLUMN]


def add_split_error_variance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add each 80/10/10 fit's error variance on its training rows, and keep its validation rows."""
    train = predictions[predictions["split"] == "train"]
    errors = (train["actual"] - train["predicted"]) ** 2
    variance = errors.groupby([train["track"], train["model"], train["seed"]]).mean().rename("known_error_variance")
    validation = predictions[predictions["split"] == "validation"]
    return validation.join(variance, on=["track", "model", "seed"])


def to_variance(predictions: pd.DataFrame, variances: pd.Series | None = None) -> pd.DataFrame:
    """Convert log forecasts back to variance, and add the train-mean baseline.

    A log forecast is converted with exp(forecast + s²/2), where s² is the fit's
    mean squared error on its known days, because exp of a log forecast alone
    is too low on average. The train mean is the average variance of the days
    known before each round or split.
    """
    variances = known_variances() if variances is None else variances
    rows = predictions.assign(day=pd.to_datetime(predictions["date"]))
    group = "round" if "round" in rows.columns else "split"
    first_days = rows.groupby(["track", "model", "seed", group])["day"].transform("min")
    baselines = {day: variances[variances.index < day].mean() for day in first_days.unique()}
    return rows.assign(
        actual=np.exp(rows["actual"]),
        predicted=np.exp(rows["predicted"] + rows["known_error_variance"] / 2),
        baseline=first_days.map(baselines),
    )


def rolling_baselines(converted: pd.DataFrame, variances: pd.Series | None = None) -> pd.DataFrame:
    """Add simple baselines that predict the average variance of recent days, with no training.

    Each one is shifted a day, so it only uses days before the scored day.
    """
    variances = known_variances() if variances is None else variances
    har = converted[converted["model"] == HAR.name]
    scored = har[har["seed"] == har["seed"].iloc[0]]
    frames = [
        scored.assign(model=name, predicted=variances.rolling(days).mean().shift(1).reindex(scored["day"]).to_numpy())
        for name, days in ROLLING_BASELINES.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _qlike(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1


def _daily(rows: pd.DataFrame) -> pd.DataFrame:
    """Average one model's losses and forecasts across seeds, per day."""
    return (
        rows.assign(
            error=(rows["actual"] - rows["predicted"]) ** 2,
            qlike=_qlike(rows["actual"], rows["predicted"]),
            baseline_error=(rows["actual"] - rows["baseline"]) ** 2,
            baseline_qlike=_qlike(rows["actual"], rows["baseline"]),
        )
        .groupby("day", sort=True)[["actual", "predicted", "error", "qlike", "baseline_error", "baseline_qlike"]]
        .mean()
    )


def range_intervals(
    converted: pd.DataFrame,
    *,
    mean_block_days: int = 20,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Score each model on the variance scale, with confidence intervals.

    qlike_skill and oos_r2 are relative to the train mean, so above 0 beats it.
    """
    rng = np.random.default_rng(seed)
    tail = 100 * (1 - confidence) / 2
    rows = []
    for (track, model), group in converted.groupby(["track", "model"], sort=False):
        daily = _daily(group)
        actual, predicted = daily["actual"].to_numpy(), daily["predicted"].to_numpy()
        qlike, baseline_qlike = daily["qlike"].to_numpy(), daily["baseline_qlike"].to_numpy()
        error, baseline_error = daily["error"].to_numpy(), daily["baseline_error"].to_numpy()
        indices = _bootstrap_indices(len(daily), samples, mean_block_days, rng)
        estimates = {
            "qlike_skill": (
                1 - qlike.mean() / baseline_qlike.mean(),
                1 - qlike[indices].mean(axis=1) / baseline_qlike[indices].mean(axis=1),
            ),
            "oos_r2": (
                1 - error.mean() / baseline_error.mean(),
                1 - error[indices].mean(axis=1) / baseline_error[indices].mean(axis=1),
            ),
            "rank_correlation": (
                _rank_correlation(actual, predicted),
                _rank_correlations(actual, predicted, indices),
            ),
        }
        row = {
            "track": track,
            "model": model,
            "days": len(daily),
            "qlike": qlike.mean(),
            "baseline_qlike": baseline_qlike.mean(),
            "correlation": np.corrcoef(actual, predicted)[0, 1],
        }
        for name, (estimate, resampled) in estimates.items():
            row[name] = estimate
            row[f"{name}_lower"] = np.nanpercentile(resampled, tail)
            row[f"{name}_upper"] = np.nanpercentile(resampled, 100 - tail)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_range(
    first_rows: pd.DataFrame,
    second_rows: pd.DataFrame,
    labels: dict,
    *,
    mean_block_days: int = 20,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Compare two sets of forecasts day by day. Positive differences favor the first."""
    first, second = _daily(first_rows), _daily(second_rows)
    if not first.index.equals(second.index):
        raise ValueError(f"{labels}: not scored on the same days")
    tail = 100 * (1 - confidence) / 2
    indices = _bootstrap_indices(len(first), samples, mean_block_days, np.random.default_rng(seed))
    actual = first["actual"].to_numpy()
    first_predicted, second_predicted = first["predicted"].to_numpy(), second["predicted"].to_numpy()

    qlike = second["qlike"].to_numpy() - first["qlike"].to_numpy()
    mse = second["error"].to_numpy() - first["error"].to_numpy()
    rank = _rank_correlations(actual, first_predicted, indices) - _rank_correlations(actual, second_predicted, indices)
    row = {
        **labels,
        "days": len(first),
        "qlike_first": first["qlike"].mean(),
        "qlike_second": second["qlike"].mean(),
        "rank_first": _rank_correlation(actual, first_predicted),
        "rank_second": _rank_correlation(actual, second_predicted),
    }
    for name, estimate, resampled in (
        ("qlike_improvement", qlike.mean(), qlike[indices].mean(axis=1)),
        ("mse_improvement", mse.mean(), mse[indices].mean(axis=1)),
        ("rank_gain", row["rank_first"] - row["rank_second"], rank),
    ):
        row[name] = estimate
        row[f"{name}_lower"] = np.nanpercentile(resampled, tail)
        row[f"{name}_upper"] = np.nanpercentile(resampled, 100 - tail)
    return row


def compare_to_har(converted: pd.DataFrame, **options) -> pd.DataFrame:
    """Compare every model with log-HAR, the main baseline. Positive differences favor the model."""
    har = converted[converted["model"] == HAR.name]
    rows = []
    for (track, model), group in converted.groupby(["track", "model"], sort=False):
        if model != HAR.name:
            rows.append(compare_range(group, har, {"track": track, "model": model}, **options))
    return pd.DataFrame(rows)


def compare_to_train_mean(converted: pd.DataFrame, **options) -> pd.DataFrame:
    """Compare every model with the train mean, day by day. Positive differences favor the model."""
    rows = []
    for (track, model), group in converted.groupby(["track", "model"], sort=False):
        mean = group.assign(predicted=group["baseline"])
        rows.append(compare_range(group, mean, {"track": track, "model": model}, **options))
    return pd.DataFrame(rows)


def compare_news(converted: pd.DataFrame, **options) -> pd.DataFrame:
    """Compare news with no news, using the same model. Positive differences favor news."""
    rows = []
    for label, first_track, second_track in NEWS_COMPARISONS:
        for spec in MOVE_MODELS:
            first = converted[(converted["track"] == first_track) & (converted["model"] == spec.name)]
            second = converted[(converted["track"] == second_track) & (converted["model"] == spec.name)]
            rows.append(compare_range(first, second, {"comparison": label, "model": spec.name}, **options))
    first = converted[converted["model"] == HAR_WITH_NEWS.name]
    second = converted[converted["model"] == HAR.name]
    rows.append(compare_range(first, second, {"comparison": "HAR + news vs HAR", "model": "HAR"}, **options))
    return pd.DataFrame(rows)


def price_bands(converted: pd.DataFrame, track: str, model: str) -> pd.DataFrame:
    """Turn one model's forecasts into a 95% band around the previous close.

    The band is previous close × exp(±1.96 σ), using the forecast averaged across
    seeds, and the train-mean band uses the baseline the same way. It's centred
    on the previous close, because the forecast has no direction.
    """
    stock = load_stock().set_index("DATE")
    previous_close = stock["CLOSE"].shift(1)
    rows = converted[(converted["track"] == track) & (converted["model"] == model)]
    daily = rows.groupby("day")[["predicted", "baseline"]].mean()
    bands = pd.DataFrame(
        {
            "day": daily.index,
            "previous_close": previous_close.reindex(daily.index).to_numpy(),
            "close": stock["CLOSE"].reindex(daily.index).to_numpy(),
        }
    )
    for prefix, column in (("", "predicted"), ("baseline_", "baseline")):
        spread = BAND_Z * np.sqrt(daily[column].to_numpy())
        bands[f"{prefix}lower"] = bands["previous_close"] * np.exp(-spread)
        bands[f"{prefix}upper"] = bands["previous_close"] * np.exp(spread)
        bands[f"{prefix}covered"] = bands["close"].between(bands[f"{prefix}lower"], bands[f"{prefix}upper"])
        bands[f"{prefix}width"] = (bands[f"{prefix}upper"] - bands[f"{prefix}lower"]) / bands["previous_close"]
    move = np.log(bands["close"] / bands["previous_close"]).abs()
    bands["miss_size"] = move / (BAND_Z * np.sqrt(daily["predicted"].to_numpy()))
    return bands


def band_scores(converted: pd.DataFrame) -> pd.DataFrame:
    """Summarize every model's bands. Coverage should be near 95%, and narrower is better at the same coverage."""
    rows = []
    for track, model in converted[["track", "model"]].drop_duplicates().itertuples(index=False):
        bands = price_bands(converted, track, model)
        rows.append(
            {
                "track": track,
                "model": model,
                "coverage": bands["covered"].mean(),
                "average_width": bands["width"].mean(),
                "baseline_coverage": bands["baseline_covered"].mean(),
                "baseline_average_width": bands["baseline_width"].mean(),
            }
        )
    return pd.DataFrame(rows)


def largest_misses(bands: pd.DataFrame, count: int = 15) -> pd.DataFrame:
    """Return the days when the close fell furthest outside the band, for checking against events by hand.

    miss_size is the day's move divided by the band's half-width, so above 1 is outside it.
    """
    misses = bands[~bands["covered"]].nlargest(count, "miss_size")
    move = np.log(misses["close"] / misses["previous_close"])
    return misses.assign(move=move)[["day", "previous_close", "close", "move", "lower", "upper", "miss_size"]]
