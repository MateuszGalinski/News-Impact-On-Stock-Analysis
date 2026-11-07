"""Checks for the range-based volatility test. Run with: python -m training.benchmarking.check_range_volatility"""

import numpy as np
import pandas as pd

from .data import (
    NEWS_WITH_FEATURES,
    NO_NEWS_NO_FEATURES,
    RANGE_INPUT_COLUMN,
    RANGE_TARGET,
    RANGE_VARIANCE_COLUMN,
    TARGET_COLUMN,
    load_track_rows,
    make_windows,
    prepare_walk_forward_round,
    range_variance,
    split_sizes,
)
from .models import fit_model
from .move_size import HAR, HAR_WITH_NEWS, MOVE_MODELS
from .range_volatility import (
    BAND_Z,
    compare_range,
    known_variances,
    price_bands,
    range_intervals,
    rolling_baselines,
    to_variance,
)
from .walk_forward import WalkForwardConfig, load_development_rows, run_walk_forward


def check_formula():
    # The worked example: previous close 100, then open 102, high 105, low 101, close 104.
    stock = pd.DataFrame({"OPEN": [100.0, 102.0], "HIGH": [100.0, 105.0], "LOW": [100.0, 101.0], "CLOSE": [100.0, 104.0]})
    expected = np.log(1.02) ** 2 + 0.5 * np.log(105 / 101) ** 2 - (2 * np.log(2) - 1) * np.log(104 / 102) ** 2
    assert np.isclose(range_variance(stock).iloc[1], expected)
    assert np.isclose(expected, 0.001001, atol=2e-6)
    print("[PASS] the formula matches the worked example")


def check_target():
    frame, features, _ = load_track_rows(NO_NEWS_NO_FEATURES, common_dates=True, target=RANGE_TARGET)
    returns = load_track_rows(NO_NEWS_NO_FEATURES, common_dates=True)[0]
    assert RANGE_INPUT_COLUMN in features and RANGE_VARIANCE_COLUMN not in features
    assert RANGE_INPUT_COLUMN not in returns.columns, "the return target must stay unchanged"
    assert len(frame) == len(returns) and frame[RANGE_VARIANCE_COLUMN].notna().all()
    assert (frame[RANGE_VARIANCE_COLUMN] > 0).all()

    variance = frame[RANGE_VARIANCE_COLUMN].to_numpy()
    windows = make_windows(frame, features, 30, (1, 5), target=RANGE_TARGET)
    assert np.allclose(windows.y[:, 0], np.log(variance[30 : 30 + len(windows.y)]))
    assert np.isclose(windows.y[7, 1], np.log(variance[37:42].mean())), "5-day targets average the variance first"

    # A window's inputs end the day before its target, including the previous close.
    index = features.index(RANGE_INPUT_COLUMN)
    assert np.allclose(windows.X[:, -1, index], np.log(variance[29 : 29 + len(windows.y)]))
    news = load_track_rows(NEWS_WITH_FEATURES, target=RANGE_TARGET)[0]
    shared = frame.merge(news[["DATE", RANGE_VARIANCE_COLUMN]], on="DATE", suffixes=("", "_news"))
    assert np.allclose(shared[RANGE_VARIANCE_COLUMN], shared[f"{RANGE_VARIANCE_COLUMN}_news"]), "same values on every track"
    assert np.isclose(np.log(news["CLOSE"] / news["CLOSE"].shift(1)), news[TARGET_COLUMN], equal_nan=True).sum() > 0
    print("[PASS] the target is the log of the next day's variance, and inputs end the day before it")


def check_no_leakage():
    config = WalkForwardConfig(track=NO_NEWS_NO_FEATURES, target=RANGE_TARGET)
    frame, features, scaled = load_development_rows(config)
    full = load_track_rows(NO_NEWS_NO_FEATURES, common_dates=True, target=RANGE_TARGET)[0]
    train_size, validation_size = split_sizes(len(full), 0.8, 0.1)
    assert frame["DATE"].max() < full["DATE"].iloc[train_size + validation_size], "the test split stays locked"
    assert known_variances().index.max() == frame["DATE"].max()

    options = dict(cutoff=900, scored_end=1000, track=NO_NEWS_NO_FEATURES, target=RANGE_TARGET)
    clean = prepare_walk_forward_round(frame, features, scaled, **options)
    changed = frame.copy()
    changed.loc[900:, RANGE_VARIANCE_COLUMN] = 1.0
    changed.loc[900:, RANGE_INPUT_COLUMN] = 0.0
    changed.loc[900:, TARGET_COLUMN] = 1.0
    poisoned = prepare_walk_forward_round(changed, features, scaled, **options)
    for split in ("train", "validation"):
        assert np.array_equal(getattr(clean, split).X, getattr(poisoned, split).X), split
        assert np.array_equal(getattr(clean, split).y, getattr(poisoned, split).y), split
    assert np.array_equal(clean.test.X[0], poisoned.test.X[0]), "the first scored day only uses earlier inputs"
    assert not np.array_equal(clean.test.y, poisoned.test.y)
    print("[PASS] scored days can't reach training, and the test split stays locked")


def check_har():
    for track, spec, extra in ((NO_NEWS_NO_FEATURES, HAR, 0), (NEWS_WITH_FEATURES, HAR_WITH_NEWS, 7)):
        frame, features, scaled = load_development_rows(WalkForwardConfig(track=track, target=RANGE_TARGET))
        dataset = prepare_walk_forward_round(
            frame, features, scaled, cutoff=900, scored_end=1000, track=track, target=RANGE_TARGET
        )
        fitted = fit_model(spec, dataset, 42)
        assert len(fitted.estimators[0].coef_.ravel()) == 3 + extra, spec.name
        predicted = fitted.predict(dataset.test.X)[:, 0]
        # Log-HAR should be clearly better than the train mean on these days, in log units.
        assert np.mean((dataset.test.y[:, 0] - predicted) ** 2) < np.mean((dataset.test.y[:, 0] - dataset.train.y.mean()) ** 2)
    print("[PASS] log-HAR uses the log variance, and HAR + news adds the news features")


def check_conversion_and_scoring():
    dates = pd.date_range("2024-01-01", periods=120)
    actual = np.resize([0.0004, 0.0012, 0.0006, 0.002], 120)
    variances = pd.Series(np.full(600, 0.001), index=pd.date_range("2023-01-01", periods=600))

    def rows(model, predicted_log, error_variance):
        return pd.DataFrame(
            {
                "track": NO_NEWS_NO_FEATURES, "model": model, "seed": 42, "round": 1, "horizon": 1,
                "date": dates.astype(str), "actual": np.log(actual), "predicted": predicted_log,
                "baseline": 0.0, "known_error_variance": error_variance,
            }
        )

    good = to_variance(rows("good", np.log(actual), 0.0), variances)
    assert np.allclose(good["actual"], actual) and np.allclose(good["predicted"], actual)
    assert np.allclose(good["baseline"], 0.001), "the baseline is the average variance of earlier days"
    corrected = to_variance(rows("good", np.log(actual), 0.5), variances)
    assert np.allclose(corrected["predicted"], actual * np.exp(0.25)), "the bias correction is exp(s²/2)"

    flat = to_variance(rows(HAR.name, np.full(120, np.log(0.001)), 0.0), variances)
    comparison = compare_range(good, flat, {"model": "good"}, samples=100)
    assert comparison["qlike_improvement_lower"] > 0 and comparison["mse_improvement"] > 0
    intervals = range_intervals(pd.concat([good, flat]), samples=100).set_index("model")
    assert intervals.loc["good", "qlike_skill"] > 0.99 and abs(intervals.loc[HAR.name, "qlike_skill"]) < 1e-9

    baselines = rolling_baselines(flat, variances)
    assert np.allclose(baselines["predicted"], 0.001)
    print("[PASS] forecasts convert back with the bias correction, and QLIKE favors the better model")


def check_small_run():
    run = run_walk_forward(
        WalkForwardConfig(
            track=NO_NEWS_NO_FEATURES, models=(HAR, MOVE_MODELS[2]), seeds=(42,), start_days=1300, target=RANGE_TARGET
        )
    )
    assert (run["known_error_variance"] > 0).all()
    converted = to_variance(run)
    variances = known_variances()
    assert np.allclose(converted["actual"], variances.reindex(converted["day"]).to_numpy())

    bands = price_bands(converted, NO_NEWS_NO_FEATURES, HAR.name)
    assert (bands["lower"] < bands["previous_close"]).all() and (bands["upper"] > bands["previous_close"]).all()
    half_width = np.log(bands["upper"] / bands["previous_close"])
    daily = converted[converted["model"] == HAR.name].groupby("day")["predicted"].mean()
    assert np.allclose(half_width, BAND_Z * np.sqrt(daily.to_numpy()))
    inside = np.log(bands["close"] / bands["previous_close"]).abs() <= half_width
    assert (inside == bands["covered"]).all()
    print("[PASS] a small run trains, converts, and builds 95% price bands")


def main():
    check_formula()
    check_target()
    check_no_leakage()
    check_har()
    check_conversion_and_scoring()
    check_small_run()
    print("\nALL CHECKS PASSED: the range-based test can't see future rows or the locked test split")


if __name__ == "__main__":
    main()
