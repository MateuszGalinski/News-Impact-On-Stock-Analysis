from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import prepare_dataset
from .metrics import score_predictions
from .models import NOTEBOOK_MODELS, ModelSpec, fit_model


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment. `models` defaults to the track's notebook models.

    common_dates, standardize and early_stopping_fraction are described in
    `prepare_dataset`.
    """

    track: str
    models: tuple[ModelSpec, ...] | None = None
    horizons: tuple[int, ...] = (1, 2, 3)
    window_days: int = 30
    seeds: tuple[int, ...] = (42,)
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    common_dates: bool = False
    standardize: bool = False
    early_stopping_fraction: float | None = None


def run_experiment(config: ExperimentConfig, *, evaluate_test: bool = False) -> dict:
    """Train every model and seed in `config`, then score them.

    The test split stays untouched unless `evaluate_test=True`, so tuning
    decisions can't be based on it. Keep that for the final comparison.

    Each target is the cumulative log return over `horizon` days. Each split
    builds its own windows, so no training target uses a later split's prices.

    Returns:
        metrics: one row per model, seed, split and horizon.
        predictions: one row per model, seed, split, horizon and date.
        models: fitted models keyed by (model name, seed).
        dataset: the windows used, without test results.
    """
    dataset = prepare_dataset(
        config.track,
        horizons=config.horizons,
        window_days=config.window_days,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        common_dates=config.common_dates,
        standardize=config.standardize,
        early_stopping_fraction=config.early_stopping_fraction,
    )
    specs = config.models if config.models is not None else NOTEBOOK_MODELS[config.track]
    known_targets = dataset.train.y
    if dataset.early_stopping is not None:
        known_targets = np.concatenate([dataset.train.y, dataset.early_stopping.y])

    splits = {"train": dataset.train, "validation": dataset.validation}
    if evaluate_test:
        splits["test"] = dataset.test

    metric_rows = []
    prediction_frames = []
    fitted_models = {}

    for spec in specs:
        for seed in config.seeds:
            fitted = fit_model(spec, dataset, seed)
            fitted_models[(spec.name, seed)] = fitted

            for split_name, split in splits.items():
                predicted = fitted.predict(split.X)
                for index, horizon in enumerate(config.horizons):
                    baseline = known_targets[:, index].mean()
                    labels = {
                        "track": config.track,
                        "model": spec.name,
                        "seed": seed,
                        "split": split_name,
                        "horizon": horizon,
                    }
                    metric_rows.append(
                        {
                            **labels,
                            **score_predictions(split.y[:, index], predicted[:, index], baseline),
                            "best_step": fitted.best_steps[index],
                            "steps_run": fitted.steps_run[index],
                        }
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                **labels,
                                "date": split.dates,
                                "actual": split.y[:, index],
                                "predicted": predicted[:, index],
                                "baseline": baseline,
                            }
                        )
                    )

    return {
        "config": config,
        "metrics": pd.DataFrame(metric_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "models": fitted_models,
        "dataset": dataset,
    }
