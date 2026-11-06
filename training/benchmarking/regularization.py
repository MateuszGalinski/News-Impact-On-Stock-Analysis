"""Regularization test: the unified models against fixed ways of limiting overfitting.

Model names are "<base model> | <candidate>". The control is the unified model,
read from its saved results using the same seeds, so it isn't retrained.
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .data import UNIFIED_TRACKS
from .diagnostics import learning_curves
from .experiment import ExperimentConfig, run_experiment
from .models import UNIFIED_RECURRENT_PARAMS, UNIFIED_XGBOOST_PARAMS, ModelSpec
from .walk_forward import WalkForwardConfig, compare_tracks, run_walk_forward

CONTROL = "Control"
REGULARIZATION_SEEDS = (42, 43, 44)

_RECURRENT_CANDIDATES = {
    "Dropout 0.2": {"dropout": 0.2},
    "Dropout 0.5": {"dropout": 0.5},
    "L2 0.0001": {"l2": 0.0001},
    "L2 0.001": {"l2": 0.001},
    "Smaller network": {"units": 16, "dense_units": 8},
}
REGULARIZATION_MODELS = (
    *(
        ModelSpec(f"{model} | {candidate}", kind, {**UNIFIED_RECURRENT_PARAMS, **changes})
        for model, kind in (("RNN", "rnn"), ("LSTM", "lstm"))
        for candidate, changes in _RECURRENT_CANDIDATES.items()
    ),
    ModelSpec("LSTM | 2-layer", "lstm", {**UNIFIED_RECURRENT_PARAMS, "units": (64, 32)}),
    ModelSpec(
        "XGBoost | Stronger",
        "xgboost",
        {
            **UNIFIED_XGBOOST_PARAMS,
            "max_depth": 2,
            "learning_rate": 0.02,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "reg_lambda": 10,
            "min_child_weight": 10,
        },
    ),
)


# The learning rate is 2.5 times lower than the control's, so patience is 2.5 times longer.
STRONGER_XGBOOST_LONG_PATIENCE = ModelSpec(
    "XGBoost | Stronger, patience 75",
    "xgboost",
    {**REGULARIZATION_MODELS[-1].params, "early_stopping_rounds": 75},
)


def _without_early_stopping(spec: ModelSpec) -> tuple[ModelSpec, int]:
    """Return the candidate with early stopping off, and the patience it would have used."""
    key = "early_stopping_rounds" if spec.kind == "xgboost" else "early_stopping_patience"
    params = {name: value for name, value in spec.params.items() if name != key}
    return replace(spec, params=params), spec.params[key]


REGULARIZATION_CURVE_MODELS = tuple(_without_early_stopping(spec) for spec in REGULARIZATION_MODELS)
REGULARIZATION_CURVE_SEEDS = (42,)


def run_regularization_curves(results_path: Path) -> pd.DataFrame:
    """Record every candidate's learning curves. Saved fits are skipped."""
    return learning_curves(
        models=REGULARIZATION_CURVE_MODELS, seeds=REGULARIZATION_CURVE_SEEDS, results_path=results_path
    )


def run_regularization_walk_forward(results_path: Path, models: tuple = REGULARIZATION_MODELS) -> None:
    """Run candidates through walk-forward. Saved fits are skipped."""
    for track in UNIFIED_TRACKS:
        run_walk_forward(
            WalkForwardConfig(track=track, models=models, seeds=REGULARIZATION_SEEDS),
            results_path,
        )


def run_regularization_split(metrics_path: Path, predictions_path: Path, models: tuple = REGULARIZATION_MODELS) -> None:
    """Run candidates on the 80/10/10 split. Tracks already saved are skipped."""
    done = set(pd.read_csv(metrics_path, usecols=["track"])["track"]) if Path(metrics_path).exists() else set()
    for track in UNIFIED_TRACKS:
        if track in done:
            continue
        result = run_experiment(
            ExperimentConfig(
                track=track,
                models=models,
                horizons=(1,),
                seeds=REGULARIZATION_SEEDS,
                common_dates=True,
                standardize=True,
                early_stopping_fraction=0.1,
            )
        )
        # Predictions first, so a finished track always has both files.
        result["predictions"].to_csv(
            predictions_path, mode="a", header=not Path(predictions_path).exists(), index=False
        )
        result["metrics"].to_csv(metrics_path, mode="a", header=not Path(metrics_path).exists(), index=False)
        print(f"80/10/10: {track} done", flush=True)


def with_control(candidates: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
    """Add the unified models as the control, using only the same seeds."""
    control = unified[unified["seed"].isin(set(candidates["seed"]))]
    control = control.assign(model=control["model"] + f" | {CONTROL}")
    return pd.concat([control, candidates], ignore_index=True)


def split_model_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Split "<base model> | <candidate>" into base_model and candidate columns."""
    parts = frame["model"].str.split(" | ", regex=False)
    return frame.assign(base_model=parts.str[0], candidate=parts.str[1])


def for_base_model(frame: pd.DataFrame, base_model: str) -> pd.DataFrame:
    """Keep one base model, with its candidates as the model names."""
    named = split_model_names(frame)
    named = named[named["base_model"] == base_model]
    return named.assign(model=named["candidate"]).drop(columns=["base_model", "candidate"])


def compare_to_control(predictions: pd.DataFrame, base_model: str, control: str = CONTROL, **options) -> pd.DataFrame:
    """Compare each candidate with a control, day by day, within each input set.

    Positive differences favor the candidate. Uses `compare_tracks` with the
    candidates in place of tracks and the input sets in place of models.
    """
    rows = for_base_model(predictions, base_model)
    swapped = rows.rename(columns={"track": "model", "model": "track"})
    candidates = [name for name in swapped["track"].unique() if name != control]
    label = "control" if control == CONTROL else control
    comparisons = tuple((f"{name} vs {label}", name, control) for name in candidates)
    return compare_tracks(swapped, comparisons, **options).rename(columns={"model": "track"})
