import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SUMMARY_GROUPS = ["track", "model", "split", "horizon"]
SUMMARY_METRICS = ["r2", "correlation", "spread_ratio", "directional_accuracy"]


def summarize_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Combine a metrics table's seeds into one row per track, model, split and horizon.

    share_beating_baseline is the fraction of seeds with a lower MSE than the
    train-mean baseline. always_up_accuracy doesn't depend on the seed.
    """
    grouped = metrics.groupby(SUMMARY_GROUPS, sort=False)
    summary = grouped[SUMMARY_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["always_up_accuracy"] = grouped["always_up_accuracy"].first()
    summary["share_beating_baseline"] = (
        (metrics["mse"] < metrics["baseline_mse"]).groupby(
            [metrics[column] for column in SUMMARY_GROUPS], sort=False
        ).mean()
    )
    summary["seeds"] = grouped["seed"].nunique()
    return summary.reset_index()


def score_predictions(actual, predicted, baseline: float) -> dict[str, float]:
    """Score predictions against a constant baseline such as the train mean.

    directional_accuracy compares signs, so a flat day only counts as correct
    when the prediction is exactly zero. Compare it with always_up_accuracy,
    not with 50%.
    """
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    if len(actual) == 0:
        raise ValueError("actual contains no rows")
    if len(predicted) != len(actual):
        raise ValueError(
            f"got {len(predicted)} predictions for {len(actual)} actual values"
        )

    constant = np.full(len(actual), baseline)
    has_spread = actual.std() > 0 and predicted.std() > 0
    return {
        "mse": mean_squared_error(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "baseline_mse": mean_squared_error(actual, constant),
        "baseline_mae": mean_absolute_error(actual, constant),
        "r2": r2_score(actual, predicted),
        "correlation": np.corrcoef(actual, predicted)[0, 1] if has_spread else np.nan,
        "spread_ratio": predicted.std() / actual.std() if actual.std() > 0 else np.nan,
        "directional_accuracy": np.mean(np.sign(predicted) == np.sign(actual)),
        "always_up_accuracy": np.mean(actual > 0),
    }
