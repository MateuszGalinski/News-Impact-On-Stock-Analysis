"""Checks for the move-size test. Run with: python -m training.benchmarking.check_move_size"""

import numpy as np
import pandas as pd

from .data import (
    MOVE_INPUT_COLUMN,
    MOVE_SIZE_TARGET,
    NEWS_COLUMNS,
    NEWS_WITH_FEATURES,
    NO_NEWS_NO_FEATURES,
    TARGET_COLUMN,
    load_track_rows,
    make_windows,
    prepare_walk_forward_round,
    split_sizes,
)
from .models import fit_model
from .move_size import (
    HAR,
    HAR_WITH_NEWS,
    MOVE_MODELS,
    ROLLING_BASELINES,
    big_move_scores,
    compare_moves,
    compare_news,
    compare_to_har,
    known_moves,
    move_intervals,
    rolling_baselines,
)
from .walk_forward import WalkForwardConfig, load_development_rows, run_walk_forward


def check_target():
    frame, features, _ = load_track_rows(NO_NEWS_NO_FEATURES, common_dates=True, target=MOVE_SIZE_TARGET)
    returns, _, _ = load_track_rows(NO_NEWS_NO_FEATURES, common_dates=True)
    assert MOVE_INPUT_COLUMN in features
    assert MOVE_INPUT_COLUMN not in returns.columns, "the return target must stay unchanged"

    moves = frame[TARGET_COLUMN].abs().to_numpy()
    windows = make_windows(frame, features, 30, (1, 5), target=MOVE_SIZE_TARGET)
    assert np.allclose(windows.y[:, 0], moves[30 : 30 + len(windows.y)])
    assert np.allclose(windows.y[7, 1], moves[37:42].mean()), "5-day moves average the daily moves"
    assert (windows.y >= 0).all()

    # A window's inputs end the day before its target starts.
    index = features.index(MOVE_INPUT_COLUMN)
    assert np.allclose(windows.X[:, -1, index], moves[29 : 29 + len(windows.y)])
    assert (windows.dates == frame["DATE"].to_numpy()[30 : 30 + len(windows.y)]).all()

    summed = make_windows(returns, [TARGET_COLUMN], 30, (5,))
    assert np.allclose(summed.y[7, 0], returns[TARGET_COLUMN].to_numpy()[37:42].sum()), "returns still sum"
    print("[PASS] the target is the size of the next move, and inputs end the day before it")


def check_no_leakage():
    config = WalkForwardConfig(track=NO_NEWS_NO_FEATURES, target=MOVE_SIZE_TARGET)
    frame, features, scaled = load_development_rows(config)
    full = load_track_rows(NO_NEWS_NO_FEATURES, common_dates=True, target=MOVE_SIZE_TARGET)[0]
    train_size, validation_size = split_sizes(len(full), 0.8, 0.1)
    assert frame["DATE"].max() < full["DATE"].iloc[train_size + validation_size], "the test split stays locked"
    assert known_moves().index.max() == frame["DATE"].max()

    options = dict(cutoff=900, scored_end=1000, track=NO_NEWS_NO_FEATURES, target=MOVE_SIZE_TARGET)
    clean = prepare_walk_forward_round(frame, features, scaled, **options)
    changed = frame.copy()
    for column in (TARGET_COLUMN, MOVE_INPUT_COLUMN):
        changed.loc[900:, column] = 1.0
    poisoned = prepare_walk_forward_round(changed, features, scaled, **options)
    for split in ("train", "validation"):
        assert np.array_equal(getattr(clean, split).X, getattr(poisoned, split).X), split
        assert np.array_equal(getattr(clean, split).y, getattr(poisoned, split).y), split
    assert np.array_equal(clean.test.X[0], poisoned.test.X[0]), "the first scored day only uses earlier inputs"
    assert not np.array_equal(clean.test.y, poisoned.test.y)
    print("[PASS] scored days can't reach training, and the test split stays locked")


def check_har():
    results = {}
    for track, spec, extra in ((NO_NEWS_NO_FEATURES, HAR, 0), (NEWS_WITH_FEATURES, HAR_WITH_NEWS, len(NEWS_COLUMNS))):
        config = WalkForwardConfig(track=track, target=MOVE_SIZE_TARGET)
        frame, features, scaled = load_development_rows(config)
        dataset = prepare_walk_forward_round(
            frame, features, scaled, cutoff=900, scored_end=1000, track=track, target=MOVE_SIZE_TARGET
        )
        fitted = fit_model(spec, dataset, 42)
        assert len(fitted.estimators[0].coef_.ravel()) == 3 + extra, spec.name
        predicted = fitted.predict(dataset.test.X)
        assert predicted.shape == (len(dataset.test.y), 1)
        assert np.array_equal(predicted, fit_model(spec, dataset, 43).predict(dataset.test.X)), "HAR has no randomness"
        results[spec.name] = predicted
    assert not np.array_equal(results[HAR.name], results[HAR_WITH_NEWS.name])
    print("[PASS] HAR uses the 1, 5 and 22-day moves, and HAR + news adds the news features")


def check_scoring():
    dates = pd.date_range("2024-01-01", periods=120).astype(str)
    actual = np.resize([0.01, 0.03, 0.02, 0.05], 120)
    def rows(track, model, predicted):
        return pd.DataFrame(
            {
                "track": track, "model": model, "seed": 42, "round": 1, "horizon": 1, "date": dates,
                "actual": actual, "predicted": predicted, "baseline": actual.mean(),
            }
        )

    good = rows(NO_NEWS_NO_FEATURES, "good", actual + 0.001)
    flat = rows(NO_NEWS_NO_FEATURES, HAR.name, np.full(120, actual.mean()) + np.resize([0.0, 1e-6], 120))
    comparison = compare_moves(good, flat, {"model": "good"}, samples=100)
    assert comparison["mse_improvement"] > 0 and comparison["mae_improvement"] > 0
    assert comparison["mse_improvement_lower"] > 0
    assert len(compare_to_har(pd.concat([good, flat]), samples=50)) == 1

    intervals = move_intervals(pd.concat([good, flat]), samples=100).set_index("model")
    assert intervals.loc["good", "oos_r2"] > 0.9
    assert intervals.loc["good", "rank_correlation"] > 0.99
    assert abs(intervals.loc[HAR.name, "oos_r2"]) < 1e-3

    moves = pd.Series(np.full(120, 0.02), index=pd.date_range("2023-01-01", periods=120))
    scores = big_move_scores(good, moves).iloc[0]
    assert scores["caught"] == 1.0 and scores["flagged_share"] >= scores["big_share"]
    print("[PASS] comparisons favor the better model, and scoring handles known answers")


def check_small_run():
    moves = known_moves()
    runs = []
    for track, specs in (
        (NO_NEWS_NO_FEATURES, (HAR, MOVE_MODELS[2])),
        (NEWS_WITH_FEATURES, (HAR_WITH_NEWS, MOVE_MODELS[2])),
    ):
        runs.append(
            run_walk_forward(
                WalkForwardConfig(track=track, models=specs, seeds=(42,), start_days=1300, target=MOVE_SIZE_TARGET)
            )
        )
    predictions = pd.concat(runs, ignore_index=True)
    assert np.allclose(predictions["actual"], moves.reindex(pd.to_datetime(predictions["date"])).to_numpy())

    baselines = rolling_baselines(predictions)
    assert set(baselines["model"]) == set(ROLLING_BASELINES)
    yesterday = baselines[baselines["model"] == "Yesterday"]
    dates = pd.to_datetime(yesterday["date"])
    assert np.allclose(yesterday["predicted"], moves.shift(1).reindex(dates).to_numpy())

    assert len(move_intervals(pd.concat([predictions, baselines]), samples=50)) == 7
    assert (predictions["baseline"] > 0).all()
    print("[PASS] a small run trains, scores and adds the simple baselines")


def check_news_comparison_labels():
    dates = pd.date_range("2024-01-01", periods=60).astype(str)
    frames = []
    from .data import NEWS_WITH_FEATURES_AND_TECH, NO_NEWS_WITH_FEATURES

    tracks = (NO_NEWS_NO_FEATURES, NO_NEWS_WITH_FEATURES, NEWS_WITH_FEATURES, NEWS_WITH_FEATURES_AND_TECH)
    for track in tracks:
        for spec in (*MOVE_MODELS, HAR, HAR_WITH_NEWS):
            frames.append(
                pd.DataFrame(
                    {
                        "track": track, "model": spec.name, "seed": 42, "round": 1, "horizon": 1, "date": dates,
                        "actual": np.resize([0.01, 0.03], 60), "predicted": np.resize([0.012, 0.02], 60),
                        "baseline": 0.02,
                    }
                )
            )
    predictions = pd.concat(frames, ignore_index=True)
    # HAR only exists once in a real run, so keep one row set for each.
    predictions = predictions[
        ((predictions["model"] != HAR.name) | (predictions["track"] == NO_NEWS_NO_FEATURES))
        & ((predictions["model"] != HAR_WITH_NEWS.name) | (predictions["track"] == NEWS_WITH_FEATURES))
    ]
    assert len(compare_news(predictions, samples=20)) == 2 * len(MOVE_MODELS) + 1
    print("[PASS] the news comparisons cover every model, plus HAR with and without news")


def main():
    assert len(MOVE_MODELS) == 6
    check_target()
    check_no_leakage()
    check_har()
    check_scoring()
    check_news_comparison_labels()
    check_small_run()
    print("\nALL CHECKS PASSED: the move-size test can't see future rows or the locked test split")


if __name__ == "__main__":
    main()
