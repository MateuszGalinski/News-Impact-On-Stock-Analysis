"""Check the unified setup: the fourth track, early stopping and track comparisons.

Run from the repo root:  python -m training.benchmarking.check_unified_models

Uses cached news features only, so nothing is sent or billed. Trains small models.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from .data import (
    CYCLICAL_COLUMNS,
    NEWS_COLUMNS,
    NEWS_WITH_FEATURES,
    NEWS_WITH_FEATURES_AND_TECH,
    NO_NEWS_WITH_FEATURES,
    TARGET_COLUMN,
    TECHNICAL_COLUMNS,
    TRACKS,
    UNIFIED_TRACKS,
    load_track_rows,
    prepare_dataset,
)
from .diagnostics import CURVE_MODELS, early_stopping_pick, learning_curves, summarize_curves
from .experiment import ExperimentConfig, run_experiment
from .models import UNIFIED_MODELS, UNIFIED_RECURRENT_PARAMS, UNIFIED_XGBOOST_PARAMS, ModelSpec, fit_model
from .regularization import CONTROL, REGULARIZATION_MODELS, compare_to_control, split_model_names
from .walk_forward import WalkForwardConfig, compare_tracks, run_walk_forward, walk_forward_intervals

SPLITS = ("train", "validation", "test")
SHORT_RECURRENT = {**UNIFIED_RECURRENT_PARAMS, "epochs": 6, "early_stopping_patience": 2}


def check_new_track():
    assert len(TRACKS) == 3, "older notebook loops rely on TRACKS having 3 tracks"
    frame, features, scaled = load_track_rows(NEWS_WITH_FEATURES_AND_TECH, common_dates=True)
    assert set(features) == {TARGET_COLUMN, *TECHNICAL_COLUMNS, *CYCLICAL_COLUMNS, *NEWS_COLUMNS}
    assert len(features) == 16

    datasets = {t: prepare_dataset(t, common_dates=True) for t in UNIFIED_TRACKS}
    reference = datasets[NEWS_WITH_FEATURES]
    for track, dataset in datasets.items():
        for split in SPLITS:
            assert np.array_equal(getattr(dataset, split).dates, getattr(reference, split).dates), (track, split)
            assert np.allclose(getattr(dataset, split).y, getattr(reference, split).y, atol=1e-12, rtol=0)

    indicators, _, _ = load_track_rows(NO_NEWS_WITH_FEATURES, common_dates=True)
    news, _, _ = load_track_rows(NEWS_WITH_FEATURES)
    for column in TECHNICAL_COLUMNS:
        assert np.allclose(frame[column], indicators[column]), "indicators must keep full-history values"
    for column in NEWS_COLUMNS:
        assert np.allclose(frame[column], news[column]), "news values must match the news track"
    assert not frame[features].isna().any().any()
    print("[PASS] the new track has 16 inputs, and all 4 tracks share dates and targets")


def check_stopping_slice_on_split():
    track = NEWS_WITH_FEATURES_AND_TECH
    default = prepare_dataset(track, common_dates=True, standardize=True)
    carved = prepare_dataset(track, common_dates=True, standardize=True, early_stopping_fraction=0.1)
    assert default.early_stopping is None and default.stopping_split is default.validation
    assert carved.stopping_split is carved.early_stopping

    assert carved.train.dates[-1] < carved.early_stopping.dates[0]
    assert carved.early_stopping.dates[-1] < carved.validation.dates[0]
    assert np.array_equal(carved.validation.dates, default.validation.dates)
    assert np.array_equal(carved.validation.y, default.validation.y)
    assert np.array_equal(carved.test.dates, default.test.dates)

    frame, _, _ = load_track_rows(track, common_dates=True)
    train_size = int(len(frame) * 0.8)
    assert len(carved.train.y) + len(carved.early_stopping.y) == train_size - 30
    fit_rows = frame.iloc[:train_size - int(train_size * 0.1)]
    column = carved.feature_columns.index(TARGET_COLUMN)
    raw = prepare_dataset(track, common_dates=True, early_stopping_fraction=0.1)
    expected = (raw.validation.X[:, :, column] - fit_rows[TARGET_COLUMN].mean()) / fit_rows[TARGET_COLUMN].std(ddof=0)
    assert np.allclose(carved.validation.X[:, :, column], expected), "scaling must use rows before the slice"
    print("[PASS] the 80/10/10 stopping slice sits between training and validation, and scaling ignores it")


def _poison(split):
    return replace(split, X=np.full_like(split.X, 1e6), y=np.full_like(split.y, 1e6))


def check_scored_days_cant_affect_training():
    dataset = prepare_dataset(NEWS_WITH_FEATURES_AND_TECH, common_dates=True, standardize=True, early_stopping_fraction=0.1)
    poisoned = replace(dataset, validation=_poison(dataset.validation), test=_poison(dataset.test))
    for spec in (ModelSpec("LSTM", "lstm", SHORT_RECURRENT), ModelSpec("XGBoost", "xgboost", UNIFIED_XGBOOST_PARAMS)):
        clean = fit_model(spec, dataset, 42).predict(dataset.train.X)
        dirty = fit_model(spec, poisoned, 42).predict(dataset.train.X)
        assert np.array_equal(clean, dirty), f"{spec.name} used scored data while training"
    print("[PASS] changing scored days doesn't change what any model learns")


def check_best_point_is_used():
    dataset = prepare_dataset(NO_NEWS_WITH_FEATURES, common_dates=True, standardize=True, early_stopping_fraction=0.1)
    stopping = dataset.early_stopping
    target_mean, target_std = dataset.train.y.mean(axis=0), dataset.train.y.std(axis=0)
    scaled_y = (stopping.y - target_mean) / target_std

    # Short enough that training often reaches the limit without stopping early.
    spec = ModelSpec("LSTM", "lstm", {**UNIFIED_RECURRENT_PARAMS, "epochs": 12, "early_stopping_patience": 50})
    fitted = fit_model(spec, dataset, 42)
    assert fitted.steps_run == [12] and 1 <= fitted.best_steps[0] <= 12
    predicted = (fitted.predict(stopping.X) - target_mean) / target_std
    final_loss = np.mean((predicted - scaled_y) ** 2)

    import keras
    from keras import layers

    params = spec.params
    keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            layers.Input(shape=dataset.train.X.shape[1:]),
            layers.LSTM(params["units"]),
            layers.Dense(params["dense_units"], activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        dataset.train.X,
        (dataset.train.y - target_mean) / target_std,
        validation_data=(stopping.X, scaled_y),
        epochs=12,
        batch_size=params["batch_size"],
        verbose=0,
    )
    losses = history.history["val_loss"]
    assert np.isclose(final_loss, min(losses), atol=1e-4), (final_loss, min(losses))
    assert fitted.best_steps[0] == int(np.argmin(losses)) + 1

    spec = ModelSpec("XGBoost", "xgboost", UNIFIED_XGBOOST_PARAMS)
    fitted = fit_model(spec, dataset, 42)
    estimator = fitted.estimators[0]
    assert fitted.best_steps[0] < fitted.steps_run[0] <= UNIFIED_XGBOOST_PARAMS["n_estimators"]
    inputs = dataset.validation.X.reshape(len(dataset.validation.X), -1)
    best_only = estimator.predict(inputs, iteration_range=(0, fitted.best_steps[0]))
    assert np.array_equal(estimator.predict(inputs), best_only), "XGBoost must predict from its best tree count"
    print("[PASS] LSTM and XGBoost predict from their best point, not their last")


def check_defaults_unchanged():
    dataset = prepare_dataset(NO_NEWS_WITH_FEATURES, common_dates=True, standardize=True)
    assert dataset.early_stopping is None
    result = run_experiment(
        ExperimentConfig(
            track=NO_NEWS_WITH_FEATURES,
            models=(ModelSpec("XGBoost", "xgboost", {"n_estimators": 5}),),
            horizons=(1,),
            common_dates=True,
            standardize=True,
        )
    )
    assert np.isclose(result["predictions"]["baseline"].iloc[0], dataset.train.y.mean())
    assert (result["metrics"]["best_step"] == 5).all() and (result["metrics"]["steps_run"] == 5).all()

    carved = ExperimentConfig(
        track=NO_NEWS_WITH_FEATURES,
        models=(ModelSpec("XGBoost", "xgboost", {"n_estimators": 5}),),
        horizons=(1,),
        common_dates=True,
        standardize=True,
        early_stopping_fraction=0.1,
    )
    result = run_experiment(carved)
    data = result["dataset"]
    known = np.concatenate([data.train.y, data.early_stopping.y]).mean()
    assert np.isclose(result["predictions"]["baseline"].iloc[0], known), "the baseline uses all known targets"
    assert set(result["metrics"]["split"]) == {"train", "validation"}, "the test split must stay locked"
    print("[PASS] the default path is unchanged, and the test split stays locked")


def check_comparisons():
    dates = pd.date_range("2024-01-01", periods=120).astype(str)
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 0.02, len(dates))
    rows = []
    for track, noise in ((NEWS_WITH_FEATURES, 0.001), (TRACKS[0], 0.05)):
        for seed in (1, 2):
            rows.append(
                pd.DataFrame(
                    {
                        "track": track,
                        "model": "M",
                        "seed": seed,
                        "horizon": 1,
                        "date": dates,
                        "actual": actual,
                        "predicted": actual + np.random.default_rng(seed).normal(0, noise, len(dates)),
                        "baseline": 0.0,
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    result = compare_tracks(frame, comparisons=(("better vs worse", NEWS_WITH_FEATURES, TRACKS[0]),), samples=300)
    row = result.iloc[0]
    assert row["mse_improvement"] > 0 and row["mse_improvement_lower"] > 0, "a better first track must score positive"
    assert row["direction_gain"] > 0 and row["direction_gain_lower"] > 0

    swapped = compare_tracks(frame, comparisons=(("worse vs better", TRACKS[0], NEWS_WITH_FEATURES),), samples=300)
    assert np.isclose(swapped.iloc[0]["mse_improvement"], -row["mse_improvement"])

    mismatch = frame[~((frame["track"] == TRACKS[0]) & (frame["date"] == dates[0]))]
    try:
        compare_tracks(mismatch, comparisons=(("x", NEWS_WITH_FEATURES, TRACKS[0]),), samples=10)
    except ValueError:
        pass
    else:
        raise AssertionError("different days on each track must raise an error")
    print("[PASS] direct comparisons favor the better track and reject mismatched days")


def check_small_run():
    models = (
        ModelSpec("LSTM", "lstm", SHORT_RECURRENT),
        ModelSpec("XGBoost", "xgboost", UNIFIED_XGBOOST_PARAMS),
    )
    frames = []
    for track in (NEWS_WITH_FEATURES, NEWS_WITH_FEATURES_AND_TECH):
        config = WalkForwardConfig(track=track, models=models, seeds=(42,), start_days=1200, step_days=200)
        frames.append(run_walk_forward(config))
    predictions = pd.concat(frames, ignore_index=True)
    assert {"best_step", "steps_run"} <= set(predictions.columns)
    assert (predictions["best_step"] <= predictions["steps_run"]).all()
    assert len(walk_forward_intervals(predictions, samples=100)) == 4
    comparison = compare_tracks(
        predictions,
        comparisons=(("News + indicators vs news", NEWS_WITH_FEATURES_AND_TECH, NEWS_WITH_FEATURES),),
        samples=100,
    )
    assert len(comparison) == 2
    assert UNIFIED_MODELS[0].params["epochs"] == 200
    print("[PASS] a small walk-forward run works on the new track with the unified models")


def check_learning_curves():
    assert early_stopping_pick([5, 4, 3, 4, 4, 4, 1], patience=3) == 3, "stops before the later low"
    assert early_stopping_pick([5, 4, 3, 4, 4, 1], patience=3) == 6
    assert early_stopping_pick([3, 3, 3, 3], patience=2) == 1, "ties aren't improvements"

    # Early stopping's real choice must match the pick calculated from a full curve.
    dataset = prepare_dataset(NO_NEWS_WITH_FEATURES, common_dates=True, standardize=True, early_stopping_fraction=0.1)
    for spec, patience in CURVE_MODELS:
        params = {**spec.params, "epochs": 25} if spec.kind != "xgboost" else {**spec.params, "n_estimators": 150}
        full = fit_model(replace(spec, params=params), dataset, 42)
        pick = early_stopping_pick(full.curves[0]["stopping"], patience)
        stopped_params = (
            {**params, "early_stopping_patience": patience}
            if spec.kind != "xgboost"
            else {**params, "early_stopping_rounds": patience}
        )
        stopped = fit_model(replace(spec, params=stopped_params), dataset, 42)
        assert len(full.curves[0]["stopping"]) == params.get("epochs", params.get("n_estimators"))
        assert stopped.best_steps[0] == pick, (spec.name, stopped.best_steps[0], pick)
        if pick < len(full.curves[0]["stopping"]) - patience:
            assert stopped.steps_run[0] == pick + patience

    short = tuple(
        (replace(spec, params={**spec.params, "epochs": 3}), patience)
        if spec.kind != "xgboost"
        else (replace(spec, params={**spec.params, "n_estimators": 5}), patience)
        for spec, patience in CURVE_MODELS
    )
    curves = learning_curves((NEWS_WITH_FEATURES_AND_TECH,), models=short, rounds=(1,), seeds=(42,))
    assert set(curves["model"]) == {"RNN", "LSTM", "XGBoost"}
    assert curves[curves["model"] == "XGBoost"]["step"].max() == 5
    assert (curves["stopping_loss"] > 0).all() and (curves["train_loss"] > 0).all()
    summary = summarize_curves(curves)
    assert len(summary) == 3
    assert (summary["stopping_best"] <= summary["stopping_final"]).all()
    print("[PASS] curves record each step, and the pick matches early stopping's real choice")


def check_regularization_candidates():
    assert len(REGULARIZATION_MODELS) == 12, "5 candidates for RNN and LSTM, plus the 2-layer LSTM and XGBoost"
    assert len({spec.name for spec in REGULARIZATION_MODELS}) == 12
    dataset = prepare_dataset(NO_NEWS_WITH_FEATURES, common_dates=True, standardize=True, early_stopping_fraction=0.1)
    poisoned = replace(dataset, validation=_poison(dataset.validation), test=_poison(dataset.test))
    target_mean, target_std = dataset.train.y.mean(axis=0), dataset.train.y.std(axis=0)
    scaled_y = (dataset.early_stopping.y - target_mean) / target_std

    control = fit_model(ModelSpec("LSTM", "lstm", {**UNIFIED_RECURRENT_PARAMS, "epochs": 2}), dataset, 42)
    baseline_inputs = dataset.validation.X
    for spec in REGULARIZATION_MODELS:
        if spec.kind == "xgboost":
            params = {**spec.params, "n_estimators": 20}
        else:
            params = {**spec.params, "epochs": 2}
        short = replace(spec, params=params)
        fitted = fit_model(short, dataset, 42)
        assert np.array_equal(fitted.predict(baseline_inputs), fit_model(short, poisoned, 42).predict(baseline_inputs)), spec.name
        assert fitted.best_steps[0] <= fitted.steps_run[0]
        if spec.kind != "xgboost":
            assert not np.array_equal(fitted.predict(baseline_inputs), control.predict(baseline_inputs)), spec.name
        # The reported curve must be plain MSE, even with a weight penalty.
        predicted = (fitted.predict(dataset.early_stopping.X) - target_mean) / target_std
        assert np.isclose(np.mean((predicted - scaled_y) ** 2), fitted.curves[0]["stopping"].min(), atol=1e-3), spec.name

    names = split_model_names(pd.DataFrame({"model": ["LSTM | Dropout 0.2", f"RNN | {CONTROL}"]}))
    assert list(names["base_model"]) == ["LSTM", "RNN"] and list(names["candidate"]) == ["Dropout 0.2", CONTROL]

    dates = pd.date_range("2024-01-01", periods=60).astype(str)
    rows = []
    for name, offset in ((f"LSTM | {CONTROL}", 0.05), ("LSTM | Dropout 0.2", 0.0)):
        rows.append(
            pd.DataFrame(
                {
                    "track": TRACKS[0], "model": name, "seed": 1, "horizon": 1, "date": dates,
                    "actual": np.resize([0.01, -0.01], 60), "predicted": np.resize([0.01, -0.01], 60) + offset,
                    "baseline": 0.0,
                }
            )
        )
    comparison = compare_to_control(pd.concat(rows, ignore_index=True), "LSTM", samples=50)
    assert len(comparison) == 1 and comparison.iloc[0]["mse_improvement"] > 0
    print("[PASS] all 12 candidates train, ignore scored days, and report plain MSE")


def main():
    check_new_track()
    check_stopping_slice_on_split()
    check_scored_days_cant_affect_training()
    check_best_point_is_used()
    check_defaults_unchanged()
    check_comparisons()
    check_small_run()
    check_learning_curves()
    check_regularization_candidates()
    print("\nALL CHECKS PASSED: the unified setup is ready")


if __name__ == "__main__":
    main()
