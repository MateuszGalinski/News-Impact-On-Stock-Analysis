"""Sanity check: how closely models fit the data they were trained on.

How to read the summary table:
  r2: 0 means no better than the training mean; negative means worse.
  correlation: whether predictions move with the actual returns at all.
  spread_ratio: prediction std / actual std. Near 0 means an almost flat line.
  directional_accuracy: compare with always_up_accuracy, not with 50%.

A model that can't beat these on its own training data has learned nothing
usable. A near-perfect fit points to overfitting or leakage instead, since
daily returns are mostly noise.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import score_predictions

ACTUAL_COLOR = "#2a78d6"
FITTED_COLOR = "#eb6834"
SURFACE_COLOR = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
CHART_STYLE = {
    "figure.facecolor": SURFACE_COLOR,
    "axes.facecolor": SURFACE_COLOR,
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": SECONDARY_INK,
    "axes.titlecolor": PRIMARY_INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
}


def training_fit_summary(actual, predictions: Mapping, baseline: float | None = None) -> pd.DataFrame:
    """Return one row of fit metrics per model.

    The baseline defaults to the mean of `actual`, which is the train mean when
    scoring training data. Pass the train mean when scoring any other split.
    """
    actual = np.ravel(actual)
    if len(actual) == 0:
        raise ValueError("actual contains no rows")

    rows = {}
    for name, predicted in predictions.items():
        try:
            rows[name] = score_predictions(actual, predicted, actual.mean() if baseline is None else baseline)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_training_fit(
    actual,
    predictions: Mapping,
    dates=None,
    title: str = "Fit on training data",
    zoom_days: int = 100,
    split_name: str = "training",
    baseline: float | None = None,
) -> pd.DataFrame:
    """Plot actual vs predicted values and return the summary table.

    Each model gets a row with the whole period on the left and the last
    `zoom_days` on the right, where individual days are readable.

    Args:
        actual: targets, in time order.
        predictions: model name -> predictions on the matching inputs.
        dates: optional date for each target. Rows are numbered without it.
        split_name: the split being plotted, used in the labels.
        baseline: optional train mean, drawn as a flat line and used for the
            baseline metrics.
    """
    summary = training_fit_summary(actual, predictions, baseline)
    actual = np.ravel(actual)
    predicted_label = "Fitted" if split_name == "training" else "Predicted"

    if dates is None:
        x = np.arange(len(actual))
        x_label = f"{split_name.capitalize()} row"
    else:
        x = pd.to_datetime(pd.Series(dates)).to_numpy()
        x_label = None
        if len(x) != len(actual):
            raise ValueError(f"got {len(x)} dates for {len(actual)} actual values")

    zoom_start = max(0, len(actual) - zoom_days)
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            len(predictions),
            2,
            figsize=(13, 3 * len(predictions) + 0.8),
            squeeze=False,
            layout="constrained",
            width_ratios=[3, 2],
        )
        for row, (name, predicted) in zip(axes, predictions.items()):
            predicted = np.ravel(predicted)
            full_ax, zoom_ax = row
            full_ax.plot(x, actual, color=ACTUAL_COLOR, linewidth=0.7, label="Actual")
            full_ax.plot(x, predicted, color=FITTED_COLOR, linewidth=1.2, label=predicted_label)
            zoom_ax.plot(x[zoom_start:], actual[zoom_start:], color=ACTUAL_COLOR, linewidth=1.5)
            zoom_ax.plot(x[zoom_start:], predicted[zoom_start:], color=FITTED_COLOR, linewidth=2)
            if baseline is not None:
                full_ax.axhline(
                    baseline, color=PRIMARY_INK, linewidth=1.2, linestyle="--", label="Train-mean baseline"
                )
                zoom_ax.axhline(baseline, color=PRIMARY_INK, linewidth=1.2, linestyle="--")

            stats = summary.loc[name]
            full_ax.set_title(
                f"{name}: R² {stats['r2']:.3f}, correlation {stats['correlation']:.2f}",
                loc="left",
                fontsize=11,
            )
            zoom_ax.set_title(
                f"{name}: last {len(actual) - zoom_start} {split_name} days",
                loc="left",
                fontsize=11,
            )
            full_ax.set_ylabel("Log return")

            for ax in row:
                ax.axhline(0, color=AXIS_COLOR, linewidth=0.8, zorder=0)
                ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(length=0, labelsize=9)
                if x_label:
                    ax.set_xlabel(x_label)
                else:
                    ax.tick_params(axis="x", labelrotation=0)
                    ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=7))
                    ax.xaxis.set_major_formatter(
                        plt.matplotlib.dates.ConciseDateFormatter(ax.xaxis.get_major_locator())
                    )

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(handles, labels, loc="upper right", ncols=len(labels), frameon=False, fontsize=10)
        plt.show()

    return summary


def plot_validation_fit(predictions: pd.DataFrame, seed: int | None = None, zoom_days: int = 60, title: str | None = None) -> pd.DataFrame:
    """Plot actual vs predicted validation returns, one figure per horizon.

    Takes the `predictions` table from `run_experiment`, which can be read back
    from a saved file. Uses the first seed unless `seed` is given.
    """
    rows = predictions[predictions["split"] == "validation"]
    seed = rows["seed"].iloc[0] if seed is None and not rows.empty else seed
    rows = rows[rows["seed"] == seed]
    if rows.empty:
        raise ValueError(f"No validation predictions for seed {seed}")

    summaries = []
    for horizon, by_horizon in rows.groupby("horizon"):
        by_model = dict(tuple(by_horizon.groupby("model", sort=False)))
        first = next(iter(by_model.values()))
        heading = title or "Fit on validation data"
        summary = plot_training_fit(
            first["actual"],
            {name: frame["predicted"] for name, frame in by_model.items()},
            dates=first["date"],
            title=f"{heading}: {horizon}-day cumulative log return, seed {seed}",
            zoom_days=zoom_days,
            split_name="validation",
            baseline=first["baseline"].iloc[0],
        )
        summaries.append(summary.assign(horizon=horizon))
    return pd.concat(summaries)


def plot_saved_training_fit(
    predictions: pd.DataFrame,
    seed: int | None = None,
    zoom_days: int = 100,
    title: str | None = None,
) -> pd.DataFrame:
    """Plot actual vs fitted training returns from a `predictions` table, one figure per horizon.

    The table comes from `run_experiment` and can be read back from a saved
    file. Uses the first seed unless `seed` is given.
    """
    rows = predictions[predictions["split"] == "train"]
    seed = rows["seed"].iloc[0] if seed is None and not rows.empty else seed
    rows = rows[rows["seed"] == seed]
    if rows.empty:
        raise ValueError(f"No training predictions for seed {seed}")

    summaries = []
    for horizon, by_horizon in rows.groupby("horizon"):
        by_model = dict(tuple(by_horizon.groupby("model", sort=False)))
        first = next(iter(by_model.values()))
        heading = title or "Fit on training data"
        summary = plot_training_fit(
            first["actual"],
            {name: frame["predicted"] for name, frame in by_model.items()},
            dates=first["date"],
            title=f"{heading}: {horizon}-day cumulative log return",
            zoom_days=zoom_days,
        )
        summaries.append(summary.assign(horizon=horizon))
    return pd.concat(summaries)


def plot_experiment_training_fit(result: dict, seed: int | None = None, zoom_days: int = 100) -> pd.DataFrame:
    """Run the training-fit check on a `run_experiment` result, one figure per horizon.

    Uses the first seed in the experiment unless `seed` is given.
    """
    seed = result["config"].seeds[0] if seed is None else seed
    return plot_saved_training_fit(result["predictions"], seed=seed, zoom_days=zoom_days)
