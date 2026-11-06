"""Walk-forward evaluation: train on the past, score the next stretch, and repeat.

Each round starts fresh, with its own scaling, model and train-mean baseline.
Only rows before the locked test split are ever loaded, so this evaluation can
be used for decisions. All predictions are then scored together.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_track_rows, prepare_walk_forward_round, split_sizes
from .metrics import score_predictions
from .models import NOTEBOOK_MODELS, ModelSpec, fit_model

PREDICTION_KEYS = ["track", "model", "seed", "round"]


@dataclass(frozen=True)
class WalkForwardConfig:
    """One walk-forward run. `models` defaults to the track's notebook models.

    start_days: rows known before the first round.
    step_days: rows scored per round. The last round takes whatever is left.
    train_fraction, validation_fraction: must match the 80/10/10 split, because
        they decide where the locked test split starts.
    early_stopping_fraction: share of each round's known rows kept for early
        stopping. It comes from the end of those rows, not from scored days.
    """

    track: str
    models: tuple[ModelSpec, ...] | None = None
    horizons: tuple[int, ...] = (1,)
    window_days: int = 30
    seeds: tuple[int, ...] = (42,)
    start_days: int = 600
    step_days: int = 100
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    early_stopping_fraction: float = 0.1
    common_dates: bool = True
    standardize: bool = True


def round_boundaries(row_count: int, start_days: int, step_days: int) -> list[tuple[int, int]]:
    """Return (cutoff, scored_end) for each round."""
    if start_days <= 0 or step_days <= 0 or start_days >= row_count:
        raise ValueError(f"Can't fit rounds: {row_count} rows, start {start_days}, step {step_days}")
    return [
        (cutoff, min(cutoff + step_days, row_count))
        for cutoff in range(start_days, row_count, step_days)
    ]


def load_development_rows(config: WalkForwardConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a track's rows, leaving out the locked test split."""
    frame, features, scaled = load_track_rows(config.track, common_dates=config.common_dates)
    train_size, validation_size = split_sizes(len(frame), config.train_fraction, config.validation_fraction)
    return frame.iloc[:train_size + validation_size].reset_index(drop=True), features, scaled


def _round_predictions(config, spec, seed, round_number, dataset) -> pd.DataFrame:
    fitted = fit_model(spec, dataset, seed)
    predicted = fitted.predict(dataset.test.X)
    known_targets = np.concatenate([dataset.train.y, dataset.validation.y])
    frames = [
        pd.DataFrame(
            {
                "track": config.track,
                "model": spec.name,
                "seed": seed,
                "round": round_number,
                "horizon": horizon,
                "date": dataset.test.dates,
                "actual": dataset.test.y[:, index],
                "predicted": predicted[:, index],
                # The baseline can use every target known when the round starts.
                "baseline": known_targets[:, index].mean(),
            }
        )
        for index, horizon in enumerate(config.horizons)
    ]
    if spec.kind in ("rnn", "lstm"):
        import keras

        keras.utils.clear_session()
    return pd.concat(frames, ignore_index=True)


def run_walk_forward(config: WalkForwardConfig, results_path: Path | None = None) -> pd.DataFrame:
    """Train and score every round, model and seed, and return the predictions.

    With `results_path`, each fit is appended to that CSV, and fits already in
    the file are skipped. An interrupted run can continue from where it stopped.
    Use a new file, or delete the old one, after changing any settings.
    """
    frame, features, scaled = load_development_rows(config)
    boundaries = round_boundaries(len(frame), config.start_days, config.step_days)
    specs = config.models if config.models is not None else NOTEBOOK_MODELS[config.track]

    done = set()
    if results_path is not None and Path(results_path).exists():
        saved = pd.read_csv(results_path, usecols=PREDICTION_KEYS)
        done = set(saved.drop_duplicates().itertuples(index=False, name=None))

    frames = []
    for round_number, (cutoff, scored_end) in enumerate(boundaries, start=1):
        pending = [
            (spec, seed)
            for spec in specs
            for seed in config.seeds
            if (config.track, spec.name, seed, round_number) not in done
        ]
        if not pending:
            continue
        dataset = prepare_walk_forward_round(
            frame,
            features,
            scaled,
            cutoff=cutoff,
            scored_end=scored_end,
            track=config.track,
            horizons=config.horizons,
            window_days=config.window_days,
            early_stopping_fraction=config.early_stopping_fraction,
            standardize=config.standardize,
        )
        for spec, seed in pending:
            predictions = _round_predictions(config, spec, seed, round_number, dataset)
            frames.append(predictions)
            if results_path is not None:
                path = Path(results_path)
                predictions.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"{config.track}: round {round_number} of {len(boundaries)} done", flush=True)

    if results_path is not None and Path(results_path).exists():
        return pd.read_csv(results_path)
    return pd.concat(frames, ignore_index=True)


def _score(rows: pd.DataFrame) -> dict:
    scores = score_predictions(rows["actual"], rows["predicted"], rows["baseline"].to_numpy())
    # R² against the train-mean baseline, unlike r2, which uses these rows' own mean.
    scores["oos_r2"] = 1 - scores["mse"] / scores["baseline_mse"]
    return scores


def score_walk_forward(predictions: pd.DataFrame, by_round: bool = False) -> pd.DataFrame:
    """Score predictions per seed, using all scored days or one round at a time."""
    keys = ["track", "model", "seed", "horizon"] + (["round"] if by_round else [])
    rows = [
        {**dict(zip(keys, group_key)), **_score(group)}
        for group_key, group in predictions.groupby(keys, sort=False)
    ]
    return pd.DataFrame(rows)


def _bootstrap_indices(day_count: int, samples: int, mean_block_days: int, rng) -> np.ndarray:
    """Resample days in blocks of random length (Politis & Romano, 1994).

    Blocks keep nearby days together, because returns and their volatility on
    nearby days are related.
    """
    restarts = rng.random((samples, day_count)) < 1 / mean_block_days
    starts = rng.integers(0, day_count, (samples, day_count))
    indices = np.empty((samples, day_count), dtype=int)
    indices[:, 0] = starts[:, 0]
    for day in range(1, day_count):
        indices[:, day] = np.where(restarts[:, day], starts[:, day], (indices[:, day - 1] + 1) % day_count)
    return indices


def walk_forward_intervals(
    predictions: pd.DataFrame,
    *,
    mean_block_days: int = 20,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate each model's edge over the baselines, with confidence intervals.

    Each day's error and direction are averaged across seeds first, since all
    seeds are scored on the same days and don't add evidence about the market.
    Days are then resampled, so the intervals reflect how much the result
    depends on which days happened to be scored.
    """
    rng = np.random.default_rng(seed)
    tail = 100 * (1 - confidence) / 2
    rows = []
    for (track, model, horizon), group in predictions.groupby(["track", "model", "horizon"], sort=False):
        daily = (
            group.assign(
                error=(group["actual"] - group["predicted"]) ** 2,
                baseline_error=(group["actual"] - group["baseline"]) ** 2,
                correct=np.sign(group["predicted"]) == np.sign(group["actual"]),
                up=group["actual"] > 0,
            )
            .groupby("date", sort=True)[["error", "baseline_error", "correct", "up"]]
            .mean()
        )
        error, baseline_error = daily["error"].to_numpy(), daily["baseline_error"].to_numpy()
        correct, up = daily["correct"].to_numpy(), daily["up"].to_numpy()

        indices = _bootstrap_indices(len(daily), samples, mean_block_days, rng)
        oos_r2 = 1 - error[indices].mean(axis=1) / baseline_error[indices].mean(axis=1)
        edge = 100 * (correct[indices].mean(axis=1) - up[indices].mean(axis=1))
        rows.append(
            {
                "track": track,
                "model": model,
                "horizon": horizon,
                "days": len(daily),
                "mse": error.mean(),
                "baseline_mse": baseline_error.mean(),
                "oos_r2": 1 - error.mean() / baseline_error.mean(),
                "oos_r2_lower": np.percentile(oos_r2, tail),
                "oos_r2_upper": np.percentile(oos_r2, 100 - tail),
                "directional_accuracy": correct.mean(),
                "always_up_accuracy": up.mean(),
                "direction_edge": 100 * (correct.mean() - up.mean()),
                "direction_edge_lower": np.percentile(edge, tail),
                "direction_edge_upper": np.percentile(edge, 100 - tail),
            }
        )
    return pd.DataFrame(rows)
