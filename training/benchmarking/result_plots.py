"""Charts for comparing experiment results across tracks, models and seeds."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import (
    NEWS_WITH_FEATURES,
    NEWS_WITH_FEATURES_AND_TECH,
    NO_NEWS_NO_FEATURES,
    NO_NEWS_WITH_FEATURES,
    UNIFIED_TRACKS,
)
from .fit_check import (
    ACTUAL_COLOR,
    AXIS_COLOR,
    CHART_STYLE,
    FITTED_COLOR,
    GRID_COLOR,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE_COLOR,
)

TRACK_TITLES = {
    NO_NEWS_NO_FEATURES: "No news, no features",
    NO_NEWS_WITH_FEATURES: "No news, with indicators",
    NEWS_WITH_FEATURES: "News, no indicators",
    NEWS_WITH_FEATURES_AND_TECH: "News and indicators",
}
SPLIT_COLORS = {"train": ACTUAL_COLOR, "validation": FITTED_COLOR}
SPLIT_LABELS = {"train": "Train", "validation": "Validation"}


def _tracks_in(frame: pd.DataFrame) -> list[str]:
    return [track for track in UNIFIED_TRACKS if track in set(frame["track"])]


def _draw_row(ax, y: float, values, color: str, limits: tuple[float, float] | None) -> None:
    """Draw one dot per seed and a tick at their mean.

    Values outside `limits` are pinned to the axis edge as arrows, and the row
    gets a label with their true mean.
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(1)

    if limits is None:
        inside = np.ones(len(values), dtype=bool)
    else:
        inside = (values >= limits[0]) & (values <= limits[1])

    ax.scatter(
        values[inside],
        y + jitter[inside],
        s=42,
        color=color,
        edgecolors=SURFACE_COLOR,
        linewidths=1,
        zorder=3,
    )
    if limits is not None and not inside.all():
        for edge, marker, is_off_scale in (
            (limits[0], "<", values < limits[0]),
            (limits[1], ">", values > limits[1]),
        ):
            ax.scatter(
                np.full(is_off_scale.sum(), edge),
                y + jitter[is_off_scale],
                s=42,
                marker=marker,
                color=color,
                edgecolors=SURFACE_COLOR,
                linewidths=1,
                zorder=3,
            )
    if limits is None or limits[0] <= mean <= limits[1]:
        ax.plot([mean, mean], [y - 0.2, y + 0.2], color=PRIMARY_INK, linewidth=1.5, zorder=4)
    else:
        ax.annotate(
            f"mean {mean:.2f}, off scale",
            xy=(limits[0] if mean < limits[0] else limits[1], y),
            xytext=(0, -13),
            textcoords="offset points",
            ha="left" if mean < limits[0] else "right",
            va="center",
            fontsize=9,
            color=SECONDARY_INK,
        )


def _style_panel(ax, row_labels: list[str], reference_lines=(0,)) -> None:
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    for position in reference_lines:
        ax.axvline(position, color=AXIS_COLOR, linewidth=1, zorder=1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=9)


def _set_limits_with_margin(ax, limits: tuple[float, float]) -> None:
    """Leave room for markers pinned to the axis limits."""
    margin = 0.03 * (limits[1] - limits[0])
    ax.set_xlim(limits[0] - margin, limits[1] + margin)


def _legend_handles(splits: list[str]) -> list:
    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=7, color=SPLIT_COLORS[split],
            label=f"{SPLIT_LABELS[split]} (one dot per seed)",
        )
        for split in splits
    ]
    handles.append(plt.Line2D([], [], color=PRIMARY_INK, linewidth=1.5, label="Mean"))
    return handles


def _forward_panels(frame: pd.DataFrame, height: float = 6.6):
    tracks = _tracks_in(frame)
    fig, axes = plt.subplots(
        2, len(tracks), figsize=(max(13, 4 * len(tracks)), height), sharex="row", squeeze=False, layout="constrained"
    )
    return fig, axes, tracks


def plot_split_results(
    metrics: pd.DataFrame,
    horizon: int = 1,
    r2_limits: tuple[float, float] = (-0.4, 0.5),
    title: str | None = None,
) -> None:
    """Plot `run_experiment` metrics for one horizon, with one dot per seed.

    The top row shows R² against the train-mean baseline. The bottom row shows
    directional accuracy in percent, with always-up marked per split. Right of
    the line beats the baseline in both rows.
    """
    rows = metrics[metrics["horizon"] == horizon].copy()
    if rows.empty:
        raise ValueError(f"No metrics for horizon {horizon}")
    rows["oos_r2"] = 1 - rows["mse"] / rows["baseline_mse"]
    rows["accuracy"] = 100 * rows["directional_accuracy"]

    splits = [split for split in SPLIT_COLORS if split in set(rows["split"])]
    offsets = dict(zip(splits, np.linspace(-0.2, 0.2, len(splits))))
    always_up = {split: 100 * rows[rows["split"] == split]["always_up_accuracy"].iloc[0] for split in splits}

    with plt.rc_context(CHART_STYLE):
        fig, axes, tracks = _forward_panels(rows)
        for column, track in zip(axes.T, tracks):
            track_rows = rows[rows["track"] == track]
            models = list(track_rows["model"].unique())
            for ax, metric, limits in ((column[0], "oos_r2", r2_limits), (column[1], "accuracy", None)):
                for y, model in enumerate(models):
                    for split in splits:
                        values = track_rows[(track_rows["model"] == model) & (track_rows["split"] == split)][metric]
                        _draw_row(ax, y + offsets[split], values, SPLIT_COLORS[split], limits)
            _style_panel(column[0], models)
            _style_panel(column[1], models, reference_lines=())
            for split in splits:
                column[1].axvline(always_up[split], color=SPLIT_COLORS[split], linewidth=1.2, linestyle="--", zorder=1)
            _set_limits_with_margin(column[0], r2_limits)
            column[0].set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
            column[0].set_xlabel("R² (0 = train-mean baseline)")
            column[1].set_xlabel("Directional accuracy, %")

        fig.suptitle(
            title or f"Full training across seeds, {horizon}-day horizon",
            x=0.01, ha="left", fontsize=13, color=PRIMARY_INK,
        )
        handles = _legend_handles(splits) + [
            plt.Line2D(
                [], [], color=SPLIT_COLORS[split], linewidth=1.2, linestyle="--",
                label=f"{SPLIT_LABELS[split]} always-up, {always_up[split]:.1f}%",
            )
            for split in splits
        ]
        fig.legend(handles=handles, loc="outside upper right", ncols=len(handles), frameon=False, fontsize=9)
        plt.show()


def plot_track_comparisons(comparisons: pd.DataFrame, title: str = "Direct comparisons between tracks") -> None:
    """Plot `compare_tracks`, with one column per model and a 95% interval around each estimate.

    Right of zero favors the first track named in each row. An interval
    crossing zero means the two tracks can't be told apart.
    """
    labels = list(comparisons["comparison"].unique())
    models = list(comparisons["model"].unique())
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            2, len(models), figsize=(13, 5.8), sharex="row", sharey=True, squeeze=False, layout="constrained"
        )
        for column, model in zip(axes.T, models):
            rows = comparisons[comparisons["model"] == model].set_index("comparison").loc[labels]
            for ax, estimate, lower, upper, scale, color in (
                (column[0], "mse_improvement", "mse_improvement_lower", "mse_improvement_upper", 1e4, ACTUAL_COLOR),
                (column[1], "direction_gain", "direction_gain_lower", "direction_gain_upper", 1, FITTED_COLOR),
            ):
                for y, row in enumerate(rows.itertuples()):
                    ax.plot(
                        [scale * getattr(row, lower), scale * getattr(row, upper)],
                        [y, y], color=color, linewidth=2, zorder=2,
                    )
                    ax.scatter(
                        scale * getattr(row, estimate), y, s=56, color=color,
                        edgecolors=SURFACE_COLOR, linewidths=1, zorder=3,
                    )
                _style_panel(ax, labels)
            column[0].set_title(model, loc="left", fontsize=11)
            column[0].set_xlabel("MSE improvement × 10,000")
            column[1].set_xlabel("Direction gain (points)")

        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=SECONDARY_INK, marker="o", markersize=7, linewidth=2, label="Estimate with 95% interval"),
            ],
            loc="upper right", frameon=False, fontsize=10,
        )
        plt.show()


def plot_walk_forward_intervals(intervals: pd.DataFrame, title: str = "Walk-forward results, all scored days") -> None:
    """Plot `walk_forward_intervals`, with a 95% interval around each estimate.

    The top row shows out-of-sample R² against the train-mean baseline. The
    bottom row shows directional accuracy against always-up. Right of the line
    beats the baseline in both rows. An interval crossing the line means the
    result can't be told apart from the baseline.
    """
    with plt.rc_context(CHART_STYLE):
        fig, axes, tracks = _forward_panels(intervals)
        always_up = 100 * intervals["always_up_accuracy"].iloc[0]
        for column, track in zip(axes.T, tracks):
            rows = intervals[intervals["track"] == track]
            models = list(rows["model"])
            for y, row in enumerate(rows.itertuples()):
                column[0].plot(
                    [row.oos_r2_lower, row.oos_r2_upper], [y, y], color=ACTUAL_COLOR, linewidth=2, zorder=2
                )
                column[0].scatter(
                    row.oos_r2, y, s=56, color=ACTUAL_COLOR, edgecolors=SURFACE_COLOR, linewidths=1, zorder=3
                )
                column[1].plot(
                    [always_up + row.direction_edge_lower, always_up + row.direction_edge_upper],
                    [y, y],
                    color=FITTED_COLOR,
                    linewidth=2,
                    zorder=2,
                )
                column[1].scatter(
                    100 * row.directional_accuracy,
                    y,
                    s=56,
                    color=FITTED_COLOR,
                    edgecolors=SURFACE_COLOR,
                    linewidths=1,
                    zorder=3,
                )
            _style_panel(column[0], models)
            _style_panel(column[1], models, reference_lines=(always_up,))
            column[0].set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
            column[0].set_xlabel("Out-of-sample R² (0 = train-mean baseline)")
            column[1].set_xlabel(f"Directional accuracy, % (line = always-up, {always_up:.1f}%)")

        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=SECONDARY_INK, marker="o", markersize=7, linewidth=2, label="Estimate with 95% interval"),
            ],
            loc="upper right", frameon=False, fontsize=10,
        )
        plt.show()


def plot_walk_forward_rounds(round_metrics: pd.DataFrame, title: str = "Walk-forward results, round by round") -> None:
    """Plot each round's score, averaged across seeds, to show how consistent a model is."""
    rows = round_metrics.copy()
    rows["direction_edge"] = 100 * (rows["directional_accuracy"] - rows["always_up_accuracy"])
    rows = rows.groupby(["track", "model", "round"], sort=False, as_index=False)[["oos_r2", "direction_edge"]].mean()

    with plt.rc_context(CHART_STYLE):
        fig, axes, tracks = _forward_panels(rows)
        for column, track in zip(axes.T, tracks):
            track_rows = rows[rows["track"] == track]
            models = list(track_rows["model"].unique())
            for ax, metric, color in (
                (column[0], "oos_r2", ACTUAL_COLOR),
                (column[1], "direction_edge", FITTED_COLOR),
            ):
                for y, model in enumerate(models):
                    _draw_row(ax, y, track_rows[track_rows["model"] == model][metric], color, None)
                _style_panel(ax, models)
            column[0].set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
            column[0].set_xlabel("Out-of-sample R² (0 = train-mean baseline)")
            column[1].set_xlabel("Directional accuracy minus always-up (points)")

        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], linestyle="none", marker="o", markersize=7, color=SECONDARY_INK, label="One dot per round"),
                plt.Line2D([], [], color=PRIMARY_INK, linewidth=1.5, label="Mean"),
            ],
            loc="upper right", ncols=2, frameon=False, fontsize=10,
        )
        plt.show()


MOVE_INTERVAL_METRICS = (
    ("oos_r2", "Out-of-sample R² (0 = train mean)"),
    ("rank_correlation", "Rank correlation with actual moves"),
)


def plot_move_intervals(
    intervals: pd.DataFrame,
    title: str = "Move size: all scored days",
    r2_limits: tuple[float, float] | None = None,
    metrics: tuple = MOVE_INTERVAL_METRICS,
) -> None:
    """Plot `move_intervals`, with a 95% interval around each estimate.

    Each row shows one of `metrics`, by default out-of-sample R² against the
    train mean and rank correlation with actual moves. The dashed lines mark HAR, the
    main baseline, so a model has to be right of them to beat it. With
    `r2_limits`, an R² below the limit gets a label with its value in place of a dot.
    """
    har = intervals[intervals["model"] == "HAR"].iloc[0]
    with plt.rc_context(CHART_STYLE):
        fig, axes, tracks = _forward_panels(intervals)
        for column, track in zip(axes.T, tracks):
            rows = intervals[intervals["track"] == track]
            models = list(rows["model"])
            for ax, (metric, _), color in zip(column, metrics, (ACTUAL_COLOR, FITTED_COLOR)):
                for y, row in enumerate(rows.to_dict("records")):
                    ax.plot([row[f"{metric}_lower"], row[f"{metric}_upper"]], [y, y], color=color, linewidth=2, zorder=2)
                    ax.scatter(row[metric], y, s=56, color=color, edgecolors=SURFACE_COLOR, linewidths=1, zorder=3)
                _style_panel(ax, models)
                ax.axvline(har[metric], color=PRIMARY_INK, linewidth=1, linestyle="--", zorder=1)
            if r2_limits is not None:
                column[0].set_xlim(*r2_limits)
                first = metrics[0][0]
                for y, row in enumerate(rows.to_dict("records")):
                    if row[first] < r2_limits[0]:
                        column[0].annotate(
                            f"\u2190 {row[first]:.2f}", (r2_limits[0], y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=9, color=SECONDARY_INK,
                        )
            column[0].set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
            for ax, (_, label) in zip(column, metrics):
                ax.set_xlabel(label)

        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=SECONDARY_INK, marker="o", markersize=7, linewidth=2, label="Estimate with 95% interval"),
                plt.Line2D([], [], color=PRIMARY_INK, linewidth=1, linestyle="--", label="HAR"),
            ],
            loc="upper right", ncols=2, frameon=False, fontsize=10,
        )
        plt.show()


MOVE_COMPARISON_METRICS = (
    ("mse_improvement", 1e4, "MSE improvement × 10,000"),
    ("mae_improvement", 1e4, "MAE improvement × 10,000"),
    ("rank_gain", 1, "Rank correlation gain"),
)


def plot_move_comparisons(
    comparisons: pd.DataFrame, column: str, title: str, metrics: tuple = MOVE_COMPARISON_METRICS
) -> None:
    """Plot `compare_moves` rows, with one panel per value of `column` and a 95% interval.

    Right of zero favors the first set of predictions named in the comparison.
    An interval crossing zero means the two can't be told apart.
    """
    panels = list(comparisons[column].unique())
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            len(metrics), len(panels), figsize=(max(13, 4 * len(panels)), 2.8 * len(metrics)), sharex="row",
            squeeze=False, layout="constrained",
        )
        for panel_axes, panel in zip(axes.T, panels):
            rows = comparisons[comparisons[column] == panel]
            models = list(rows["model"])
            for index, (ax, (metric, scale, label)) in enumerate(zip(panel_axes, metrics)):
                color = FITTED_COLOR if index == len(metrics) - 1 else ACTUAL_COLOR
                for y, row in enumerate(rows.to_dict("records")):
                    ax.plot(
                        [scale * row[f"{metric}_lower"], scale * row[f"{metric}_upper"]],
                        [y, y], color=color, linewidth=2, zorder=2,
                    )
                    ax.scatter(scale * row[metric], y, s=56, color=color, edgecolors=SURFACE_COLOR, linewidths=1, zorder=3)
                _style_panel(ax, models)
                ax.set_xlabel(label)
            panel_axes[0].set_title(TRACK_TITLES.get(panel, panel), loc="left", fontsize=11)

        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=SECONDARY_INK, marker="o", markersize=7, linewidth=2, label="Estimate with 95% interval"),
            ],
            loc="upper right", frameon=False, fontsize=10,
        )
        plt.show()


def plot_move_predictions(predictions: pd.DataFrame, models: dict, zoom_days: int = 120, title: str = "Predicted and actual move size") -> None:
    """Plot actual moves against each model's predictions, averaged across seeds.

    `models` maps a chart label to (track, model). The top panel shows all scored
    days, and the bottom panel shows the last `zoom_days`.
    """
    colors = [FITTED_COLOR, PRIMARY_INK]
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), layout="constrained")
        for ax, days in zip(axes, (None, zoom_days)):
            for index, (label, (track, model)) in enumerate(models.items()):
                rows = predictions[(predictions["track"] == track) & (predictions["model"] == model)]
                daily = rows.assign(day=pd.to_datetime(rows["date"])).groupby("day")[["actual", "predicted"]].mean()
                daily = daily if days is None else daily.iloc[-days:]
                if index == 0:
                    ax.plot(daily.index, daily["actual"], color=ACTUAL_COLOR, linewidth=0.9, alpha=0.75, label="Actual", zorder=1)
                ax.plot(daily.index, daily["predicted"], color=colors[index], linewidth=1.8, label=label, zorder=2 + index)
            ax.set_ylabel("Move size (absolute log return)")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=0, labelsize=9)
        axes[0].set_title("All scored days", loc="left", fontsize=11)
        axes[1].set_title(f"Last {zoom_days} scored days", loc="left", fontsize=11)
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncols=len(labels), frameon=False, fontsize=10)
        plt.show()


def plot_price_bands(bands: pd.DataFrame, label: str, days: int | None = None, title: str = "Price bands") -> None:
    """Plot the real close against a model's 95% band and the train-mean band.

    `bands` comes from `price_bands`. Dots mark days when the close fell outside
    the model's band. With `days`, only the last that many scored days are shown.
    """
    rows = bands.sort_values("day")
    rows = rows if days is None else rows.iloc[-days:]
    misses = rows[~rows["covered"]]
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.2), layout="constrained")
        ax.fill_between(rows["day"], rows["lower"], rows["upper"], color=FITTED_COLOR, alpha=0.22, linewidth=0, label=f"{label}, 95% band")
        ax.plot(rows["day"], rows["baseline_lower"], color=SECONDARY_INK, linewidth=0.9, linestyle="--", label="Train-mean band")
        ax.plot(rows["day"], rows["baseline_upper"], color=SECONDARY_INK, linewidth=0.9, linestyle="--")
        ax.plot(rows["day"], rows["close"], color=ACTUAL_COLOR, linewidth=1.3, label="Actual close")
        ax.scatter(misses["day"], misses["close"], s=22, color=PRIMARY_INK, zorder=4, label="Outside the model's band")
        ax.set_ylabel("Price (split-adjusted)")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, labelsize=9)
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(loc="outside lower center", ncols=4, frameon=False, fontsize=10)
        plt.show()


def plot_candidate_curves(
    curves: pd.DataFrame,
    control: str,
    y_limits: tuple[float, float] = (0, 2.5),
    panels_per_row: int = 4,
    title: str | None = None,
) -> None:
    """Compare each candidate's learning curves with the control's, with one panel per candidate.

    `curves` has the candidates as its model names. Thin lines are single fits
    on the stopping slice, and thick lines are averages across fits. Losses are
    relative to the baseline, so 1 means no better than always predicting the
    training mean. Values above the limit run off the chart.
    """
    def average(rows: pd.DataFrame, column: str) -> pd.Series:
        return rows.groupby("step")[column].mean()

    control_stopping = average(curves[curves["model"] == control], "stopping_loss")
    candidates = [name for name in curves["model"].unique() if name != control]
    panels_per_row = min(panels_per_row, len(candidates))
    row_count = int(np.ceil(len(candidates) / panels_per_row))
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            row_count,
            panels_per_row,
            figsize=(13, 2.9 * row_count + 0.9),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for ax in axes.flat[len(candidates):]:
            ax.set_visible(False)
        for ax, candidate in zip(axes.flat, candidates):
            rows = curves[curves["model"] == candidate]
            for _, fit in rows.groupby(["track", "round", "seed"]):
                ax.plot(fit["step"], fit["stopping_loss"], color=FITTED_COLOR, linewidth=0.6, alpha=0.3, zorder=1)
            ax.plot(control_stopping.index, control_stopping, color=PRIMARY_INK, linewidth=1.5, zorder=3)
            train = average(rows, "train_loss")
            stopping = average(rows, "stopping_loss")
            ax.plot(train.index, train, color=ACTUAL_COLOR, linewidth=2, zorder=3)
            ax.plot(stopping.index, stopping, color=FITTED_COLOR, linewidth=2, zorder=4)
            ax.axhline(1, color=PRIMARY_INK, linewidth=1, linestyle="--", zorder=2)
            ax.set_ylim(*y_limits)
            ax.set_title(candidate, loc="left", fontsize=11)
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=0, labelsize=9, labelbottom=True)
            ax.set_xlabel("Trees" if curves["step"].max() > 200 else "Epochs")
        for row_axes in axes:
            row_axes[0].set_ylabel("Loss vs baseline")

        fig.suptitle(title or "Learning curves by candidate", x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=ACTUAL_COLOR, linewidth=2, label="Training data, average"),
                plt.Line2D([], [], color=FITTED_COLOR, linewidth=2, label="Stopping slice, average"),
                plt.Line2D([], [], color=FITTED_COLOR, linewidth=0.6, alpha=0.5, label="Stopping slice, each fit"),
                plt.Line2D([], [], color=PRIMARY_INK, linewidth=1.5, label="Control's stopping slice, average"),
                plt.Line2D([], [], color=PRIMARY_INK, linewidth=1, linestyle="--", label="Baseline"),
            ],
            loc="outside lower center",
            ncols=5,
            frameon=False,
            fontsize=9,
        )
        plt.show()


def plot_learning_curves(
    curves: pd.DataFrame,
    model: str,
    y_limits: tuple[float, float] = (0, 2),
    title: str | None = None,
) -> None:
    """Plot one model's `learning_curves`, with a row per round and a column per track.

    Losses are relative to the baseline, so 1 means no better than always
    predicting the training mean. Values above the limit run off the chart.
    """
    rows = curves[curves["model"] == model]
    tracks = _tracks_in(rows)
    rounds = sorted(rows["round"].unique())
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            len(rounds),
            len(tracks),
            figsize=(max(13, 4 * len(tracks)), 2.6 * len(rounds) + 0.9),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for row_axes, round_number in zip(axes, rounds):
            for ax, track in zip(row_axes, tracks):
                panel = rows[(rows["track"] == track) & (rows["round"] == round_number)]
                for _, fit in panel.groupby("seed"):
                    ax.plot(fit["step"], fit["train_loss"], color=ACTUAL_COLOR, linewidth=1.2, zorder=2)
                    ax.plot(fit["step"], fit["stopping_loss"], color=FITTED_COLOR, linewidth=1.2, zorder=2)
                    at_pick = fit[fit["step"] == fit["pick"].iloc[0]]
                    ax.scatter(
                        at_pick["step"],
                        at_pick["stopping_loss"].clip(upper=y_limits[1]),
                        s=42,
                        color=PRIMARY_INK,
                        edgecolors=SURFACE_COLOR,
                        linewidths=1,
                        zorder=4,
                    )
                ax.axhline(1, color=PRIMARY_INK, linewidth=1, linestyle="--", zorder=3)
                ax.set_ylim(*y_limits)
                ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(length=0, labelsize=9)
            row_axes[0].set_ylabel(f"Round {round_number}\nLoss vs baseline")
        for ax, track in zip(axes[0], tracks):
            ax.set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
        step_name = "Trees" if model.startswith("XGBoost") else "Epochs"
        for ax in axes[-1]:
            ax.set_xlabel(step_name)

        fig.suptitle(title or f"{model}: learning curves", x=0.01, ha="left", fontsize=13, color=PRIMARY_INK)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=ACTUAL_COLOR, linewidth=1.5, label="Training data"),
                plt.Line2D([], [], color=FITTED_COLOR, linewidth=1.5, label="Stopping slice"),
                plt.Line2D([], [], color=PRIMARY_INK, linewidth=1, linestyle="--", label="Baseline"),
                plt.Line2D(
                    [], [], linestyle="none", marker="o", markersize=7, color=PRIMARY_INK,
                    label="Early stopping's pick",
                ),
            ],
            loc="outside upper right",
            ncols=4,
            frameon=False,
            fontsize=10,
        )
        plt.show()


def plot_overfit_results(
    overfit: pd.DataFrame,
    r2_limits: tuple[float, float] = (-1, 1.05),
    title: str = "Can each model memorize 100 training windows?",
) -> None:
    """Plot `overfit_check` results: can each model memorize a small training set?"""
    tracks = _tracks_in(overfit)
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            1, len(tracks), figsize=(13, 3.9), sharex=True, squeeze=False, layout="constrained"
        )
        for ax, track in zip(axes[0], tracks):
            rows = overfit[overfit["track"] == track]
            groups = list(rows.groupby(["model", "epochs"], sort=False, dropna=False))
            labels = []
            for y, ((model, epochs), group) in enumerate(groups):
                labels.append(f"{model}, {int(epochs):,} epochs" if pd.notna(epochs) else model)
                _draw_row(ax, y, group["r2"], ACTUAL_COLOR, r2_limits)
            _style_panel(ax, labels, reference_lines=(0, 1))
            _set_limits_with_margin(ax, r2_limits)
            ax.set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
            ax.set_xlabel("R² on the windows it trained on (1 = memorized)")

        fig.suptitle(
            title,
            x=0.01, ha="left", fontsize=13, color=PRIMARY_INK,
        )
        fig.legend(
            handles=[
                plt.Line2D([], [], linestyle="none", marker="o", markersize=7, color=ACTUAL_COLOR, label="One dot per seed"),
                plt.Line2D([], [], color=PRIMARY_INK, linewidth=1.5, label="Mean"),
            ],
            loc="upper right", ncols=2, frameon=False, fontsize=10,
        )
        plt.show()


def plot_seed_results(
    metrics: pd.DataFrame,
    horizon: int = 1,
    r2_limits: tuple[float, float] = (-0.4, 0.35),
    title: str | None = None,
) -> None:
    """Plot `run_experiment` metrics for one horizon, with one dot per seed.

    The top row shows R² against the train-mean baseline. The bottom row shows
    directional accuracy minus always-up accuracy. Right of the zero line beats
    the baseline in both rows.
    """
    rows = metrics[metrics["horizon"] == horizon].copy()
    if rows.empty:
        raise ValueError(f"No metrics for horizon {horizon}")
    rows["direction_edge"] = 100 * (rows["directional_accuracy"] - rows["always_up_accuracy"])

    tracks = _tracks_in(rows)
    splits = [split for split in SPLIT_COLORS if split in set(rows["split"])]
    offsets = dict(zip(splits, np.linspace(-0.2, 0.2, len(splits))))

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            2, len(tracks), figsize=(13, 6.6), sharex="row", squeeze=False, layout="constrained"
        )
        for column, track in zip(axes.T, tracks):
            track_rows = rows[rows["track"] == track]
            models = list(track_rows["model"].unique())
            for ax, metric, limits in (
                (column[0], "r2", r2_limits),
                (column[1], "direction_edge", None),
            ):
                for y, model in enumerate(models):
                    for split in splits:
                        values = track_rows[
                            (track_rows["model"] == model) & (track_rows["split"] == split)
                        ][metric]
                        _draw_row(ax, y + offsets[split], values, SPLIT_COLORS[split], limits)
                _style_panel(ax, models)

            _set_limits_with_margin(column[0], r2_limits)
            column[0].set_title(TRACK_TITLES.get(track, track), loc="left", fontsize=11)
            column[0].set_xlabel("R² (0 = train-mean baseline)")
            column[1].set_xlabel("Direction edge over always-up (percentage points)")

        fig.suptitle(
            title or f"Full training across seeds, {horizon}-day horizon",
            x=0.01, ha="left", fontsize=13, color=PRIMARY_INK,
        )
        fig.legend(handles=_legend_handles(splits), loc="upper right", ncols=3, frameon=False, fontsize=10)
        plt.show()
