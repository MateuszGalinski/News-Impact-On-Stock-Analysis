"""Diagnostics for finding out why a model isn't training."""

from dataclasses import replace

import pandas as pd

from .data import Dataset, WindowedSplit, prepare_dataset
from .metrics import score_predictions
from .models import RECURRENT_PARAMS, ModelSpec, fit_model

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
