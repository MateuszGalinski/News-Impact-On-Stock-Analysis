"""Charts for comparing experiment results across tracks, models and seeds."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import NEWS_WITH_FEATURES, NO_NEWS_NO_FEATURES, NO_NEWS_WITH_FEATURES, TRACKS
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
    NO_NEWS_WITH_FEATURES: "No news, with features",
    NEWS_WITH_FEATURES: "News, with features",
}
SPLIT_COLORS = {"train": ACTUAL_COLOR, "validation": FITTED_COLOR}
SPLIT_LABELS = {"train": "Train", "validation": "Validation"}


def _tracks_in(frame: pd.DataFrame) -> list[str]:
    return [track for track in TRACKS if track in set(frame["track"])]


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
        2, len(tracks), figsize=(13, height), sharex="row", squeeze=False, layout="constrained"
    )
    return fig, axes, tracks


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
