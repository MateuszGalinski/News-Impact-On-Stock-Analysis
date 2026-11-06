"""Move-size test: predict how big the next move is, ignoring its direction.

Next-day returns haven't beaten the baseline in any test so far. Big moves tend
to follow big moves, so this target gives news a pattern to add to. The questions,
in order: does HAR beat the train mean, does a trained model beat HAR, and does
news beat no news?
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import (
    MOVE_INPUT_COLUMN,
    MOVE_SIZE_TARGET,
    NEWS_WITH_FEATURES,
    NEWS_WITH_FEATURES_AND_TECH,
    NO_NEWS_NO_FEATURES,
    NO_NEWS_WITH_FEATURES,
    UNIFIED_TRACKS,
)
from .experiment import ExperimentConfig, run_experiment
from .models import UNIFIED_MODELS, ModelSpec
from .regularization import REGULARIZATION_MODELS, REGULARIZATION_SEEDS
from .walk_forward import WalkForwardConfig, _bootstrap_indices, load_development_rows, run_walk_forward

HAR = ModelSpec("HAR", "har")
HAR_WITH_NEWS = ModelSpec("HAR + news", "har", {"news": True})

_REGULARIZED = ("RNN | Dropout 0.5", "LSTM | Dropout 0.5", "XGBoost | Stronger")
MOVE_MODELS = (*UNIFIED_MODELS, *(spec for spec in REGULARIZATION_MODELS if spec.name in _REGULARIZED))
MOVE_SEEDS = REGULARIZATION_SEEDS

ROLLING_BASELINES = {"Yesterday": 1, "5-day average": 5, "20-day average": 20}
BIG_MOVE_QUANTILE = 0.8

# (label, first track, second track). A positive difference favors the first.
NEWS_COMPARISONS = (
    ("News vs no news, no features", NEWS_WITH_FEATURES, NO_NEWS_NO_FEATURES),
    ("News + indicators vs indicators", NEWS_WITH_FEATURES_AND_TECH, NO_NEWS_WITH_FEATURES),
)


def _models_for(track: str) -> tuple[ModelSpec, ...]:
    """HAR ignores the other inputs, so it's fitted once, without and with news."""
    extra = {NO_NEWS_NO_FEATURES: (HAR,), NEWS_WITH_FEATURES: (HAR_WITH_NEWS,)}
    return (*MOVE_MODELS, *extra.get(track, ()))


def run_move_walk_forward(results_path: Path, horizon: int = 1) -> None:
    """Run every model through walk-forward. Saved fits are skipped."""
    for track in UNIFIED_TRACKS:
        run_walk_forward(
            WalkForwardConfig(
                track=track,
                models=_models_for(track),
                horizons=(horizon,),
                seeds=MOVE_SEEDS,
                target=MOVE_SIZE_TARGET,
            ),
            results_path,
        )


def run_move_split(predictions_path: Path, horizon: int = 1) -> None:
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
                horizons=(horizon,),
                seeds=MOVE_SEEDS,
                common_dates=True,
                standardize=True,
                early_stopping_fraction=0.1,
                target=MOVE_SIZE_TARGET,
            )
        )
        result["predictions"].to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"80/10/10: {track} done", flush=True)


def known_moves() -> pd.Series:
    """Return each day's move size, leaving out the locked test split."""
    config = WalkForwardConfig(track=NO_NEWS_NO_FEATURES, target=MOVE_SIZE_TARGET)
    frame = load_development_rows(config)[0]
    return frame.set_index("DATE")[MOVE_INPUT_COLUMN]


def rolling_baselines(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add simple baselines that predict from recent moves, with no training.

    Each one is shifted a day, so it only uses moves from before the scored day.
    """
    moves = known_moves()
    har = predictions[predictions["model"] == HAR.name]
    scored = har[har["seed"] == har["seed"].iloc[0]]
    dates = pd.to_datetime(scored["date"])
    frames = [
        scored.assign(
            model=name,
            predicted=moves.rolling(days).mean().shift(1).reindex(dates).to_numpy(),
        )
        for name, days in ROLLING_BASELINES.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _daily(rows: pd.DataFrame) -> pd.DataFrame:
    """Average one model's errors and predictions across seeds, per day."""
    return (
        rows.assign(
            error=(rows["actual"] - rows["predicted"]) ** 2,
            absolute_error=(rows["actual"] - rows["predicted"]).abs(),
            baseline_error=(rows["actual"] - rows["baseline"]) ** 2,
            baseline_absolute_error=(rows["actual"] - rows["baseline"]).abs(),
        )
        .groupby("date", sort=True)[
            ["actual", "predicted", "error", "absolute_error", "baseline_error", "baseline_absolute_error"]
        ]
        .mean()
    )


def _row_correlations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the correlation between matching rows of two arrays."""
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    return (first * second).sum(axis=1) / np.sqrt((first**2).sum(axis=1) * (second**2).sum(axis=1))


def _rank_correlations(actual: np.ndarray, predicted: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return _row_correlations(rankdata(actual[indices], axis=1), rankdata(predicted[indices], axis=1))


def _rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(actual), rankdata(predicted))[0, 1])


def move_intervals(
    predictions: pd.DataFrame,
    *,
    mean_block_days: int = 20,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Score each model on all scored days, with confidence intervals.

    oos_r2 and mae_skill are relative to the train mean. Below 0 is worse than it.
    Rank correlation is included because a few huge days can dominate the others.
    """
    rng = np.random.default_rng(seed)
    tail = 100 * (1 - confidence) / 2
    rows = []
    for (track, model, horizon), group in predictions.groupby(["track", "model", "horizon"], sort=False):
        daily = _daily(group)
        actual, predicted = daily["actual"].to_numpy(), daily["predicted"].to_numpy()
        error, baseline_error = daily["error"].to_numpy(), daily["baseline_error"].to_numpy()
        absolute, baseline_absolute = daily["absolute_error"].to_numpy(), daily["baseline_absolute_error"].to_numpy()
        indices = _bootstrap_indices(len(daily), samples, mean_block_days, rng)
        estimates = {
            "oos_r2": (
                1 - error.mean() / baseline_error.mean(),
                1 - error[indices].mean(axis=1) / baseline_error[indices].mean(axis=1),
            ),
            "mae_skill": (
                1 - absolute.mean() / baseline_absolute.mean(),
                1 - absolute[indices].mean(axis=1) / baseline_absolute[indices].mean(axis=1),
            ),
            "correlation": (
                np.corrcoef(actual, predicted)[0, 1],
                _row_correlations(actual[indices], predicted[indices]),
            ),
            "rank_correlation": (
                _rank_correlation(actual, predicted),
                _rank_correlations(actual, predicted, indices),
            ),
        }
        row = {
            "track": track,
            "model": model,
            "horizon": horizon,
            "days": len(daily),
            "mse": error.mean(),
            "baseline_mse": baseline_error.mean(),
            "mae": absolute.mean(),
            "baseline_mae": baseline_absolute.mean(),
            "spread_ratio": predicted.std() / actual.std(),
        }
        for name, (estimate, resampled) in estimates.items():
            row[name] = estimate
            row[f"{name}_lower"] = np.nanpercentile(resampled, tail)
            row[f"{name}_upper"] = np.nanpercentile(resampled, 100 - tail)
        rows.append(row)
    return pd.DataFrame(rows)


def big_move_scores(predictions: pd.DataFrame, moves: pd.Series | None = None) -> pd.DataFrame:
    """Score how well each model flags the biggest moves.

    A big move is above the 80th percentile of the moves known before that round.
    A day is flagged when the prediction, averaged across seeds, is above that
    same cutoff. caught is the share of big moves flagged, and precision is the
    share of flagged days that were big. Compare precision with big_share, which
    is what flagging days at random would get.
    """
    moves = known_moves() if moves is None else moves
    rows = []
    for (track, model), group in predictions.groupby(["track", "model"], sort=False):
        scored = group.assign(day=pd.to_datetime(group["date"]))
        scored = scored.groupby(["round", "day"], as_index=False)[["actual", "predicted"]].mean()
        first_days = scored.groupby("round")["day"].transform("min")
        cutoffs = {day: moves[moves.index < day].quantile(BIG_MOVE_QUANTILE) for day in first_days.unique()}
        cutoff = first_days.map(cutoffs)
        big = scored["actual"] > cutoff
        flagged = scored["predicted"] > cutoff
        both = (big & flagged).sum()
        rows.append(
            {
                "track": track,
                "model": model,
                "big_share": big.mean(),
                "flagged_share": flagged.mean(),
                "caught": both / big.sum() if big.sum() else np.nan,
                "precision": both / flagged.sum() if flagged.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_moves(
    first_rows: pd.DataFrame,
    second_rows: pd.DataFrame,
    labels: dict,
    *,
    mean_block_days: int = 20,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Compare two sets of predictions day by day. Positive differences favor the first."""
    first, second = _daily(first_rows), _daily(second_rows)
    if not first.index.equals(second.index):
        raise ValueError(f"{labels}: not scored on the same days")
    tail = 100 * (1 - confidence) / 2
    indices = _bootstrap_indices(len(first), samples, mean_block_days, np.random.default_rng(seed))
    actual = first["actual"].to_numpy()
    first_predicted, second_predicted = first["predicted"].to_numpy(), second["predicted"].to_numpy()

    mse = second["error"].to_numpy() - first["error"].to_numpy()
    mae = second["absolute_error"].to_numpy() - first["absolute_error"].to_numpy()
    rank = _rank_correlations(actual, first_predicted, indices) - _rank_correlations(actual, second_predicted, indices)
    row = {
        **labels,
        "days": len(first),
        "mse_first": first["error"].mean(),
        "mse_second": second["error"].mean(),
        "mae_first": first["absolute_error"].mean(),
        "mae_second": second["absolute_error"].mean(),
        "rank_first": _rank_correlation(actual, first_predicted),
        "rank_second": _rank_correlation(actual, second_predicted),
    }
    for name, estimate, resampled in (
        ("mse_improvement", mse.mean(), mse[indices].mean(axis=1)),
        ("mae_improvement", mae.mean(), mae[indices].mean(axis=1)),
        ("rank_gain", row["rank_first"] - row["rank_second"], rank),
    ):
        row[name] = estimate
        row[f"{name}_lower"] = np.nanpercentile(resampled, tail)
        row[f"{name}_upper"] = np.nanpercentile(resampled, 100 - tail)
    return row


def compare_to_har(predictions: pd.DataFrame, **options) -> pd.DataFrame:
    """Compare every model with HAR, the main baseline. Positive differences favor the model."""
    har = predictions[predictions["model"] == HAR.name]
    rows = []
    for (track, model), group in predictions.groupby(["track", "model"], sort=False):
        if model != HAR.name:
            rows.append(compare_moves(group, har, {"track": track, "model": model}, **options))
    return pd.DataFrame(rows)


def compare_news(predictions: pd.DataFrame, **options) -> pd.DataFrame:
    """Compare news with no news, using the same model. Positive differences favor news."""
    rows = []
    for label, first_track, second_track in NEWS_COMPARISONS:
        for spec in MOVE_MODELS:
            first = predictions[(predictions["track"] == first_track) & (predictions["model"] == spec.name)]
            second = predictions[(predictions["track"] == second_track) & (predictions["model"] == spec.name)]
            rows.append(compare_moves(first, second, {"comparison": label, "model": spec.name}, **options))
    first = predictions[predictions["model"] == HAR_WITH_NEWS.name]
    second = predictions[predictions["model"] == HAR.name]
    rows.append(compare_moves(first, second, {"comparison": "HAR + news vs HAR", "model": "HAR"}, **options))
    return pd.DataFrame(rows)
