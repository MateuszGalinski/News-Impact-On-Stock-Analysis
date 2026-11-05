"""Check that run_experiment reproduces the training_exploration notebooks.

Run from the repo root:  python -m training.benchmarking.check_notebook_match

It runs each notebook's own code in memory (the notebook files aren't changed),
with the neural networks seeded, then requires the pipeline to produce the same
inputs, targets and predictions. OpenAI is stubbed, so nothing is sent or billed.

Metric differences are printed instead of metric values, because this check
uses the test split and shouldn't inform tuning decisions.
"""

import builtins
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import keras
import numpy as np
import openai

from .data import NEWS_WITH_FEATURES, NO_NEWS_NO_FEATURES, NO_NEWS_WITH_FEATURES, TRAINING_DIR
from .experiment import ExperimentConfig, run_experiment

SEED = 42
PREDICTION_TOLERANCE = 1e-6

# Cell ids so the check runs exactly the notebook code that affects the data and models.
CELLS = {
    NO_NEWS_NO_FEATURES: {
        "setup": ["95d200ef", "33dd2ba5", "8c0a48a2", "60d49dd3", "e5c96e26", "b2df08d6", "cf597c6b"],
        "build": ["3a2ecc0e", "6593ca65"],
        "fit": "1a488e91",
        "xgboost": "1a488e91",
        "xgboost_name": "XGBoost",
    },
    NO_NEWS_WITH_FEATURES: {
        "setup": ["78ee49b6", "cc4ba68d", "918d6f79", "0fc41720", "23335e83", "baa2d8f2", "45ab210c"],
        "build": ["89ab9b32", "c76f1841"],
        "fit": "bfca494e",
        "xgboost": "bfca494e",
        "xgboost_name": "XGBoost",
    },
    NEWS_WITH_FEATURES: {
        "setup": [
            "c6079f56", "9ab290f1", "250ea76a", "54c6dc10", "9ddce089",
            "86a312fa", "c1663b0d", "03bf1029", "98067ab2",
        ],
        "build": ["7c9419d1"],
        "fit": "c5f93cfb",
        "xgboost": "e4384670",
        "xgboost_name": "XGBoost (v2)",
    },
}

RNN_FIT = re.compile(r"rnn_history = RNN_model\.fit\(.*?\n\s*\)", re.DOTALL)
LSTM_FIT = re.compile(r"lstm_history = LSTM_model\.fit\(.*?\n\s*\)", re.DOTALL)


class _NoNetworkClient:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("OpenAI is stubbed in this check")


def _run_notebook(track: str) -> dict:
    """Return the notebook's arrays and predictions, with seeded neural networks."""
    notebook_dir = TRAINING_DIR / track / "notebooks"
    notebook = json.loads((notebook_dir / "training_exploration.ipynb").read_text(encoding="utf-8"))
    source = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    cells = CELLS[track]

    previous_dir = Path.cwd()
    os.chdir(notebook_dir)
    try:
        scope = {}
        for cell_id in cells["setup"]:
            exec(source[cell_id], scope)

        build = "\n".join(source[cell_id] for cell_id in cells["build"])
        rnn_build, lstm_build = build.split("LSTM_model = keras.Sequential(")
        lstm_build = "LSTM_model = keras.Sequential(" + lstm_build
        # The shared `n_features = ...` line sits in the RNN part of the news notebook.
        fit_source = source[cells["fit"]]

        keras.utils.set_random_seed(SEED)
        exec(rnn_build, scope)
        exec(RNN_FIT.search(fit_source).group(0), scope)

        keras.utils.set_random_seed(SEED)
        exec(lstm_build, scope)
        exec(LSTM_FIT.search(fit_source).group(0), scope)

        xgboost_source = source[cells["xgboost"]]
        xgboost_source = RNN_FIT.sub("", LSTM_FIT.sub("", xgboost_source))
        exec(xgboost_source, scope)
    finally:
        os.chdir(previous_dir)

    arrays = {
        "train": (scope["X_train"], scope["y_train"]),
        "validation": (scope["X_val"], scope["y_val"]),
        "test": (scope["X_test"], scope["y_test"]),
    }
    predictions = {}
    for split, (X, _) in arrays.items():
        predictions[("RNN", split)] = scope["RNN_model"].predict(X, verbose=0).ravel()
        predictions[("LSTM", split)] = scope["LSTM_model"].predict(X, verbose=0).ravel()

    if track == NEWS_WITH_FEATURES:
        xgboost_inputs = {
            "train": scope["X_train_df_v2"],
            "validation": scope["X_val_df_v2"],
            "test": scope["X_test_df_v2"],
        }
        xgboost_model = scope["xgb_model_v2"]
    else:
        xgboost_inputs = {
            "train": scope["X_train_xgb"],
            "validation": scope["X_val_xgb"],
            "test": scope["X_test_xgb"],
        }
        xgboost_model = scope["xgb_model"]
    for split, X in xgboost_inputs.items():
        predictions[(cells["xgboost_name"], split)] = xgboost_model.predict(X).ravel()

    return {
        "arrays": arrays,
        "predictions": predictions,
        "dates": scope["nvidia_stock"]["DATE"].iloc[len(scope["X_train"][0]):scope["train_size"]].to_numpy(),
    }


def _check_track(track: str) -> list[str]:
    print(f"\n=== {track}", flush=True)
    notebook = _run_notebook(track)
    result = run_experiment(
        ExperimentConfig(track=track, horizons=(1,), seeds=(SEED,)), evaluate_test=True
    )
    dataset = result["dataset"]
    failures = []

    def check(label: str, passed: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{f': {detail}' if detail else ''}", flush=True)
        if not passed:
            failures.append(f"{track}: {label}")

    splits = {"train": dataset.train, "validation": dataset.validation, "test": dataset.test}
    for name, split in splits.items():
        X, y = notebook["arrays"][name]
        check(f"{name} inputs identical", np.array_equal(X, split.X), f"shape {split.X.shape}")
        check(f"{name} targets identical", np.array_equal(y, split.y.ravel()))
    check("training dates identical", np.array_equal(notebook["dates"], dataset.train.dates))

    predictions = result["predictions"]
    metrics = result["metrics"]
    for (model, split), expected in notebook["predictions"].items():
        rows = predictions[(predictions["model"] == model) & (predictions["split"] == split)]
        difference = np.abs(rows["predicted"].to_numpy() - expected).max()
        check(
            f"{model} {split} predictions",
            difference <= PREDICTION_TOLERANCE,
            f"largest difference {difference:.2e}",
        )

    for model in metrics["model"].unique():
        row = metrics[(metrics["model"] == model) & (metrics["split"] == "test")].iloc[0]
        expected = notebook["predictions"][(model, "test")]
        actual = notebook["arrays"]["test"][1]
        difference = abs(row["mse"] - np.mean((actual - expected) ** 2))
        check(f"{model} test MSE", difference <= PREDICTION_TOLERANCE, f"difference {difference:.2e}")

    return failures


def main() -> int:
    openai.OpenAI = lambda *args, **kwargs: _NoNetworkClient()
    builtins.display = lambda *args, **kwargs: None

    failures = []
    for track in CELLS:
        failures.extend(_check_track(track))

    print()
    if failures:
        print(f"{len(failures)} CHECKS FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("ALL CHECKS PASSED: run_experiment matches the notebooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
