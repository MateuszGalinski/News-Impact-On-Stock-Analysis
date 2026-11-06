"""Diagnostics for finding out why a model isn't training."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIFIED_TRACKS, Dataset, WindowedSplit, prepare_dataset, prepare_walk_forward_round
from .metrics import score_predictions
from .models import RECURRENT_PARAMS, UNIFIED_RECURRENT_PARAMS, UNIFIED_XGBOOST_PARAMS, ModelSpec, fit_model
from .walk_forward import WalkForwardConfig, load_development_rows, round_boundaries

# Settings chosen to memorize as easily as possible, not to generalize.
_XGBOOST_MEMORIZE_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
    "n_jobs": -1,
}


# Same as the unified models, but with early stopping off, to see what happens after it would stop.
CURVE_MODELS = (
    (
        ModelSpec("RNN", "rnn", {k: v for k, v in UNIFIED_RECURRENT_PARAMS.items() if k != "early_stopping_patience"}),
        UNIFIED_RECURRENT_PARAMS["early_stopping_patience"],
    ),
    (
        ModelSpec("LSTM", "lstm", {k: v for k, v in UNIFIED_RECURRENT_PARAMS.items() if k != "early_stopping_patience"}),
        UNIFIED_RECURRENT_PARAMS["early_stopping_patience"],
    ),
    (
        ModelSpec("XGBoost", "xgboost", {k: v for k, v in UNIFIED_XGBOOST_PARAMS.items() if k != "early_stopping_rounds"}),
        UNIFIED_XGBOOST_PARAMS["early_stopping_rounds"],
    ),
)
CURVE_KEYS = ["track", "model", "seed", "round"]


def early_stopping_pick(losses, patience: int) -> int:
    """Return the 1-based step that early stopping would choose from these losses."""
    best_loss, best_step = np.inf, 0
    for step, loss in enumerate(losses, start=1):
        if loss < best_loss:
            best_loss, best_step = loss, step
        elif step - best_step >= patience:
            break
    return best_step


def learning_curves(
    tracks: tuple[str, ...] = UNIFIED_TRACKS,
    *,
    models: tuple = CURVE_MODELS,
    rounds: tuple[int, ...] = (1, 4, 8),
    seeds: tuple[int, ...] = (42, 43),
    results_path: Path | None = None,
) -> pd.DataFrame:
    """Record training and stopping-slice loss per step, on walk-forward rounds.

    Early stopping is off, so curves continue past where it would have stopped.
    Only each round's known days are used, never its scored days.

    Each loss is divided by the baseline's loss on the same data, the MSE of
    always predicting the training mean. Below 1 beats the baseline.

    With `results_path`, each fit is appended to that CSV, and fits already in
    the file are skipped. Use a new file after changing any settings.

    Args:
        models: (ModelSpec, patience) pairs. Patience only decides which step
            `pick` marks, the one early stopping would have chosen.
    """
    done = set()
    path = Path(results_path) if results_path is not None else None
    if path is not None and path.exists():
        saved = pd.read_csv(path, usecols=CURVE_KEYS)
        done = set(saved.drop_duplicates().itertuples(index=False, name=None))

    frames = []
    for track in tracks:
        config = WalkForwardConfig(track=track)
        frame, features, scaled = load_development_rows(config)
        boundaries = round_boundaries(len(frame), config.start_days, config.step_days)
        for round_number in rounds:
            pending = [
                (spec, patience, seed)
                for spec, patience in models
                for seed in seeds
                if (track, spec.name, seed, round_number) not in done
            ]
            if not pending:
                continue
            cutoff, scored_end = boundaries[round_number - 1]
            dataset = prepare_walk_forward_round(
                frame,
                features,
                scaled,
                cutoff=cutoff,
                scored_end=scored_end,
                track=track,
                early_stopping_fraction=config.early_stopping_fraction,
            )
            mean, std = dataset.train.y.mean(axis=0), dataset.train.y.std(axis=0)
            train_baseline = np.mean(((dataset.train.y - mean) / std) ** 2)
            stopping_baseline = np.mean(((dataset.validation.y - mean) / std) ** 2)

            for spec, patience, seed in pending:
                curve = fit_model(spec, dataset, seed).curves[0]
                steps = np.arange(1, len(curve["stopping"]) + 1)
                result = pd.DataFrame(
                    {
                        "track": track,
                        "model": spec.name,
                        "seed": seed,
                        "round": round_number,
                        "step": steps,
                        "train_loss": curve["train"] / train_baseline,
                        "stopping_loss": curve["stopping"] / stopping_baseline,
                        "pick": early_stopping_pick(curve["stopping"], patience),
                    }
                )
                frames.append(result)
                if path is not None:
                    result.to_csv(path, mode="a", header=not path.exists(), index=False)
                if spec.kind in ("rnn", "lstm"):
                    import keras

                    keras.utils.clear_session()
                print(f"{track}: {spec.name}, seed {seed}, round {round_number} done", flush=True)

    if path is not None and path.exists():
        return pd.read_csv(path)
    return pd.concat(frames, ignore_index=True)


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Return one row per fit, with losses relative to the baseline (below 1 beats it)."""
    rows = []
    for keys, fit in curves.groupby(CURVE_KEYS, sort=False):
        pick = fit["pick"].iloc[0]
        at_pick = fit[fit["step"] == pick].iloc[0]
        best = fit.loc[fit["stopping_loss"].idxmin()]
        rows.append(
            {
                **dict(zip(CURVE_KEYS, keys)),
                "pick": pick,
                "best_step": int(best["step"]),
                "stopping_at_pick": at_pick["stopping_loss"],
                "stopping_best": best["stopping_loss"],
                "stopping_final": fit["stopping_loss"].iloc[-1],
                "train_at_pick": at_pick["train_loss"],
                "train_final": fit["train_loss"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def _last_windows(split: WindowedSplit, count: int) -> WindowedSplit:
    return WindowedSplit(X=split.X[-count:], y=split.y[-count:], dates=split.dates[-count:])


def small_training_set(dataset: Dataset, window_count: int) -> Dataset:
    """Keep only the last `window_count` training windows.

    Validation is cut down too because it isn't used here and would otherwise
    be evaluated after every epoch.
    """
    if window_count > len(dataset.train.y):
        raise ValueError(f"window_count {window_count} exceeds {len(dataset.train.y)} training windows")
    return replace(
        dataset,
        train=_last_windows(dataset.train, window_count),
        validation=_last_windows(dataset.validation, 32),
    )


def overfit_check(
    track: str,
    *,
    window_count: int = 100,
    epochs: tuple[int, ...] = (20, 1000),
    seeds: tuple[int, ...] = (42, 43, 44),
    common_dates: bool = False,
    standardize: bool = False,
) -> pd.DataFrame:
    """Test whether each model can memorize a small training set.

    Returns one row per model, epoch count and seed, scored on the same windows
    the model trained on. An R² close to 1 means the setup can learn. If a model
    can't memorize 100 windows, more data or features won't fix it.

    XGBoost runs once per seed as a control, with its own memorizing settings.
    common_dates and standardize are described in `prepare_dataset`. Inputs are
    standardized using the full training split, and the target using these windows.
    """
    dataset = small_training_set(
        prepare_dataset(track, horizons=(1,), common_dates=common_dates, standardize=standardize),
        window_count,
    )
    specs = [
        (ModelSpec(name, kind, {**RECURRENT_PARAMS, "epochs": count}), count)
        for name, kind in (("RNN", "rnn"), ("LSTM", "lstm"))
        for count in epochs
    ]
    specs.append((ModelSpec("XGBoost", "xgboost", _XGBOOST_MEMORIZE_PARAMS), None))

    baseline = dataset.train.y[:, 0].mean()
    rows = []
    for spec, count in specs:
        for seed in seeds:
            predicted = fit_model(spec, dataset, seed).predict(dataset.train.X)
            rows.append(
                {
                    "track": track,
                    "model": spec.name,
                    "epochs": count,
                    "seed": seed,
                    **score_predictions(dataset.train.y[:, 0], predicted[:, 0], baseline),
                }
            )
    return pd.DataFrame(rows)
