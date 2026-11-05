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


def training_fit_summary(actual, predictions: Mapping) -> pd.DataFrame:
    """Return one row of training-fit metrics per model."""
    actual = np.ravel(actual)
    if len(actual) == 0:
        raise ValueError("actual contains no rows")

    rows = {}
    for name, predicted in predictions.items():
        try:
            rows[name] = score_predictions(actual, predicted, actual.mean())
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_training_fit(
    actual,
    predictions: Mapping,
    dates=None,
    title: str = "Fit on training data",
    zoom_days: int = 100,
) -> pd.DataFrame:
    """Plot actual vs fitted training values and return the summary table.

    Each model gets a row with the whole training period on the left and the
    last `zoom_days` on the right, where individual days are readable.

    Args:
        actual: training targets, in time order.
        predictions: model name -> predictions on the training inputs.
        dates: optional date for each target. Rows are numbered without it.
    """
    summary = training_fit_summary(actual, predictions)
    actual = np.ravel(actual)

    if dates is None:
        x = np.arange(len(actual))
        x_label = "Training row"
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
            full_ax.plot(x, predicted, color=FITTED_COLOR, linewidth=1.2, label="Fitted")
            zoom_ax.plot(x[zoom_start:], actual[zoom_start:], color=ACTUAL_COLOR, linewidth=1.5)
            zoom_ax.plot(x[zoom_start:], predicted[zoom_start:], color=FITTED_COLOR, linewidth=2)

            stats = summary.loc[name]
            full_ax.set_title(
                f"{name}: R² {stats['r2']:.3f}, correlation {stats['correlation']:.2f}",
                loc="left",
                fontsize=11,
            )
            zoom_ax.set_title(
                f"{name}: last {len(actual) - zoom_start} training days",
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
        fig.legend(handles, labels, loc="upper right", ncols=2, frameon=False, fontsize=10)
        plt.show()

    return summary


def plot_experiment_training_fit(result: dict, seed: int | None = None, zoom_days: int = 100) -> pd.DataFrame:
    """Run the training-fit check on a `run_experiment` result, one figure per horizon.

    Uses the first seed in the experiment unless `seed` is given.
    """
    config = result["config"]
    seed = config.seeds[0] if seed is None else seed
    predictions = result["predictions"]
    rows = predictions[(predictions["split"] == "train") & (predictions["seed"] == seed)]
    if rows.empty:
        raise ValueError(f"No training predictions for seed {seed}")

    summaries = []
    for horizon, by_horizon in rows.groupby("horizon"):
        by_model = dict(tuple(by_horizon.groupby("model", sort=False)))
        first = next(iter(by_model.values()))
        summary = plot_training_fit(
            first["actual"],
            {name: frame["predicted"] for name, frame in by_model.items()},
            dates=first["date"],
            title=f"Fit on training data: {horizon}-day cumulative log return",
            zoom_days=zoom_days,
        )
        summaries.append(summary.assign(horizon=horizon))
    return pd.concat(summaries)
