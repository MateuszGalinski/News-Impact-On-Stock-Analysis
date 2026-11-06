"""Check that walk-forward evaluation can't see the future or the locked test split.

Run from the repo root:  python -m training.benchmarking.check_walk_forward

Uses cached news features only, so nothing is sent or billed. The final check
trains small models on two rounds.
"""

import numpy as np
import pandas as pd

from .data import (
    NEWS_WITH_FEATURES,
    NO_NEWS_WITH_FEATURES,
    TARGET_COLUMN,
    TRACKS,
    load_track_rows,
    prepare_walk_forward_round,
    split_sizes,
)
from .models import RECURRENT_PARAMS, XGBOOST_V2_PARAMS, ModelSpec
from .walk_forward import (
    WalkForwardConfig,
    load_development_rows,
    round_boundaries,
    run_walk_forward,
    score_walk_forward,
    walk_forward_intervals,
)


def _round(frame, features, scaled, track, cutoff, scored_end, horizons=(1,)):
    return prepare_walk_forward_round(
        frame, features, scaled, cutoff=cutoff, scored_end=scored_end, track=track, horizons=horizons
    )


def check_locked_test_split():
    for track in TRACKS:
        full, _, _ = load_track_rows(track, common_dates=True)
        rows, _, _ = load_development_rows(WalkForwardConfig(track=track))
        train_size, validation_size = split_sizes(len(full), 0.8, 0.1)
        assert len(rows) == train_size + validation_size
        assert rows["DATE"].max() < full["DATE"].iloc[train_size + validation_size]
    print("[PASS] locked test rows are never loaded")


def check_rounds_cover_every_day():
    config = WalkForwardConfig(track=NO_NEWS_WITH_FEATURES)
    frame, features, scaled = load_development_rows(config)
    boundaries = round_boundaries(len(frame), config.start_days, config.step_days)

    scored = []
    for cutoff, scored_end in boundaries:
        dataset = _round(frame, features, scaled, config.track, cutoff, scored_end)
        assert len(dataset.test.y) == scored_end - cutoff, "every scored row needs a prediction"
        scored.append(dataset.test.dates)
    scored = np.concatenate(scored)
    assert np.array_equal(scored, frame["DATE"].to_numpy()[config.start_days:]), "days skipped or repeated"
    print(f"[PASS] {len(boundaries)} rounds score {len(scored)} days once each, with no gaps")


def check_targets_stay_in_their_range():
    config = WalkForwardConfig(track=NO_NEWS_WITH_FEATURES)
    frame, features, scaled = load_development_rows(config)
    dates = frame["DATE"].to_numpy()
    returns = frame[TARGET_COLUMN].to_numpy()

    for horizons in ((1,), (1, 3)):
        for cutoff, scored_end in round_boundaries(len(frame), config.start_days, config.step_days):
            if scored_end - cutoff <= max(horizons):
                continue
            dataset = _round(frame, features, scaled, config.track, cutoff, scored_end, horizons)
            early_start = cutoff - int(cutoff * 0.1)
            last_train = np.searchsorted(dates, dataset.train.dates[-1]) + max(horizons) - 1
            last_validation = np.searchsorted(dates, dataset.validation.dates[-1]) + max(horizons) - 1
            last_scored = np.searchsorted(dates, dataset.test.dates[-1]) + max(horizons) - 1
            assert last_train < early_start, "a training target reaches the early stopping slice"
            assert np.searchsorted(dates, dataset.validation.dates[0]) >= early_start
            assert last_validation < cutoff, "an early stopping target reaches the scored days"
            assert np.searchsorted(dates, dataset.test.dates[0]) >= cutoff
            assert last_scored < scored_end, "a scored target reaches past its round"

            start = np.searchsorted(dates, dataset.test.dates[0])
            for index, horizon in enumerate(horizons):
                assert np.isclose(dataset.test.y[0, index], returns[start:start + horizon].sum())
    print("[PASS] train, early stopping and scored targets stay in their own ranges")


def check_future_rows_change_nothing():
    """Change future rows and require that everything known earlier stays identical."""
    for track in (NO_NEWS_WITH_FEATURES, NEWS_WITH_FEATURES):
        frame, features, scaled = load_development_rows(WalkForwardConfig(track=track))
        cutoff, scored_end = 700, 800
        original = _round(frame, features, scaled, track, cutoff, scored_end)

        numeric = [c for c in features]
        changed = frame.copy()
        changed.loc[changed.index[cutoff]:, numeric] = changed.loc[changed.index[cutoff]:, numeric] + 1000.0
        altered = _round(changed, features, scaled, track, cutoff, scored_end)
        for split in ("train", "validation"):
            assert np.array_equal(getattr(original, split).X, getattr(altered, split).X)
            assert np.array_equal(getattr(original, split).y, getattr(altered, split).y)
        assert np.array_equal(original.test.X[0], altered.test.X[0]), "first scored inputs must predate its target"

        # A scored window sees rows up to the day before its target, and no later.
        target_row = cutoff + 40
        changed = frame.copy()
        changed.loc[changed.index[target_row]:, numeric] = changed.loc[changed.index[target_row]:, numeric] + 1000.0
        altered = _round(changed, features, scaled, track, cutoff, scored_end)
        assert np.array_equal(original.test.X[:41], altered.test.X[:41])
        assert np.array_equal(original.test.y[:40], altered.test.y[:40])
        assert not np.array_equal(original.test.X[41], altered.test.X[41]), "the test should detect a change"
    print("[PASS] changing future rows doesn't change anything known earlier")


def check_scaling_uses_known_rows_only():
    track = NO_NEWS_WITH_FEATURES
    frame, features, scaled = load_development_rows(WalkForwardConfig(track=track))
    cutoff, scored_end = 700, 800
    raw = prepare_walk_forward_round(
        frame, features, scaled, cutoff=cutoff, scored_end=scored_end, track=track, standardize=False
    )
    standardized = _round(frame, features, scaled, track, cutoff, scored_end)

    fit_rows = frame.iloc[:cutoff - int(cutoff * 0.1)]
    column = features.index(TARGET_COLUMN)
    expected = (raw.test.X[:, :, column] - fit_rows[TARGET_COLUMN].mean()) / fit_rows[TARGET_COLUMN].std(ddof=0)
    assert np.allclose(standardized.test.X[:, :, column], expected)
    assert np.array_equal(raw.test.y, standardized.test.y), "targets must stay in log returns"
    print("[PASS] scaling is fit on rows before the early stopping slice, and targets stay in log returns")


def check_end_to_end():
    models = (
        ModelSpec("LSTM", "lstm", {**RECURRENT_PARAMS, "epochs": 1}),
        ModelSpec("XGBoost (v2)", "xgboost", XGBOOST_V2_PARAMS, news_last_only=True),
    )
    config = WalkForwardConfig(track=NEWS_WITH_FEATURES, models=models, seeds=(42, 43), start_days=1100, step_days=150)
    predictions = run_walk_forward(config)
    frame, _, _ = load_development_rows(config)
    scored_days = len(frame) - config.start_days
    assert len(predictions) == len(models) * len(config.seeds) * scored_days
    assert predictions["round"].nunique() == 2

    metrics = score_walk_forward(predictions)
    assert np.allclose(metrics["oos_r2"], 1 - metrics["mse"] / metrics["baseline_mse"])
    assert len(score_walk_forward(predictions, by_round=True)) == 2 * len(metrics)

    # Baselines are fixed within a round, and come from before it.
    baselines = predictions.groupby("round")["baseline"].nunique()
    assert (baselines == 1).all()
    frame_returns = frame[TARGET_COLUMN]
    first_round_days = predictions[predictions["round"] == 1]["date"]
    assert pd.Timestamp(first_round_days.min()) == frame["DATE"].iloc[config.start_days]
    known = frame_returns.iloc[30:config.start_days].mean()
    assert abs(predictions[predictions["round"] == 1]["baseline"].iloc[0] - known) < 1e-3

    intervals = walk_forward_intervals(predictions, samples=200)
    assert len(intervals) == len(models)
    assert (intervals["oos_r2_lower"] <= intervals["oos_r2_upper"]).all()
    assert (intervals["direction_edge_lower"] <= intervals["direction_edge_upper"]).all()
    print("[PASS] a small run trains, scores and builds intervals")


def main():
    check_locked_test_split()
    check_rounds_cover_every_day()
    check_targets_stay_in_their_range()
    check_future_rows_change_nothing()
    check_scaling_uses_known_rows_only()
    check_end_to_end()
    print("\nALL CHECKS PASSED: walk-forward can't see future rows or the locked test split")


if __name__ == "__main__":
    main()
