"""Figures: one metric panel, and the paper's three-by-three grid.

Each panel plots a metric against the number of speakers the attacker must
search, on a log x-axis, with one curve per attacker model and a reference line
at chance level. Higher is always worse for privacy, so the three metrics can be
read off the same axis.

The series palette is fixed and validated for colour-vision deficiency, and every
series also carries its own marker and dash pattern, so the figures stay readable
in greyscale print and for colourblind readers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from legal_eval.metrics.singling_out import TRIVIAL_SINGLING_OUT
from legal_eval.sweeps import SweepResult

#: Attacker models in fixed order. Colour follows the attacker, never its rank,
#: so adding or dropping a series never repaints the others.
ATTACKER_STYLE: dict[str, dict[str, object]] = {
    "original": {"color": "#2a78d6", "marker": "o", "linestyle": "-", "label": "Original"},
    "informed": {"color": "#eb6834", "marker": "s", "linestyle": "--", "label": "Informed"},
    "semi_informed": {"color": "#1baf7a", "marker": "^", "linestyle": "-.", "label": "Semi-Informed"},
    "ignorant": {"color": "#4a3aa7", "marker": "D", "linestyle": ":", "label": "Ignorant"},
}

_CHANCE_COLOR = "#52514e"

METRIC_LABEL = {
    "singling_out": "Singling Out",
    "linkability": "Linkability",
    "eer": "1 $-$ EER",
}


def chance_level(metric: str, speaker_counts: Sequence[int]) -> np.ndarray:
    """Chance-level performance of a trivial attacker.

    Args:
        metric: ``"singling_out"``, ``"linkability"`` or ``"eer"``.
        speaker_counts: The x-axis values.

    Returns:
        The chance level at each speaker count: ``exp(-1)`` for Singling Out,
        ``1/N'`` for Linkability, and ``0.5`` for ``1 - EER``.

    Raises:
        ValueError: If ``metric`` is unknown.
    """
    counts = np.asarray(speaker_counts, dtype=np.float64)
    if metric == "singling_out":
        return np.full_like(counts, TRIVIAL_SINGLING_OUT)
    if metric == "linkability":
        return 1.0 / counts
    if metric == "eer":
        return np.full_like(counts, 0.5)
    raise ValueError(f"unknown metric {metric!r}")


def _compact_count(value: float, _position: int = 0) -> str:
    """Format a speaker count compactly: 20, 500, 2k, 20k.

    Spelling out five digits makes the labels collide once the axis spans three
    decades, which it does for the paper's 20 to 22,024 range.
    """
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _style_axis(ax) -> None:
    """Apply the shared panel styling: log x, 0-1 y, plain tick labels.

    Matplotlib's default log formatter labels decades as powers of ten and adds
    minor labels like ``2 x 10^0``, which collide on a narrow range. Speaker
    counts read better as plain integers at the 1/2/5 steps.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_count))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_metric_panel(
    ax,
    results: Mapping[str, SweepResult],
    metric: str,
    show_chance: bool = True,
    show_error_bars: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Draw one metric's curves onto an existing axis.

    Args:
        ax: A matplotlib axis.
        results: Mapping from attacker key (see :data:`ATTACKER_STYLE`) to that
            attacker's sweep. Unknown keys are drawn in the next free style.
        metric: Which metric these results hold. ``"eer"`` is plotted as
            ``1 - value``, the quantity the paper shows.
        show_chance: Draw the trivial-attacker reference line.
        show_error_bars: Show the spread across runs as error bars.
        xlabel: X-axis label. Omitted when ``None``.
        ylabel: Y-axis label. Omitted when ``None``.
        title: Panel title.
    """
    [key for key in results if key not in ATTACKER_STYLE]
    spare = [style for key, style in ATTACKER_STYLE.items() if key not in results]

    all_counts: set[int] = set()
    for key, result in results.items():
        style = dict(ATTACKER_STYLE.get(key) or (spare.pop(0) if spare else {}))
        style.setdefault("label", key.replace("_", " ").title())
        style.setdefault("color", _CHANCE_COLOR)
        style.setdefault("marker", "o")
        style.setdefault("linestyle", "-")

        counts = result.speaker_counts
        all_counts.update(counts)
        means = result.mean()
        y = np.array([means[n] for n in counts])
        if metric == "eer":
            y = 1.0 - y

        # Markers only help when the grid is coarse; on the paper's 220-point
        # grid they would swamp the line.
        marker = style["marker"] if len(counts) <= 24 else None
        if show_error_bars:
            stds = result.std()
            ax.errorbar(
                counts,
                y,
                yerr=[stds[n] for n in counts],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=marker,
                markersize=5,
                linewidth=2.0,
                elinewidth=1.0,
                capsize=2,
                label=style["label"],
            )
        else:
            ax.plot(
                counts,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=marker,
                markersize=5,
                linewidth=2.0,
                label=style["label"],
            )

    if show_chance and all_counts:
        counts = sorted(all_counts)
        ax.plot(
            counts,
            chance_level(metric, counts),
            color=_CHANCE_COLOR,
            linestyle=(0, (1, 2)),
            linewidth=1.5,
            label="Trivial (chance)",
        )

    _style_axis(ax)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)


def plot_paper_figure(
    results: Mapping[str, Mapping[int, Mapping[str, SweepResult]]],
    path: str | Path,
    conversation_lengths: Sequence[int] = (1, 3, 30),
    metrics: Sequence[str] = ("singling_out", "linkability", "eer"),
    figsize: tuple = (11.0, 8.5),
) -> None:
    """Draw the paper's grid: one row per metric, one column per ``L``.

    Args:
        results: Nested mapping ``metric -> L -> attacker -> SweepResult``.
            Missing combinations leave their panel empty and annotated, so a
            partial reproduction still produces a readable figure.
        path: Output path. The extension picks the format; use ``.pdf`` for
            inclusion in a paper.
        conversation_lengths: Column order.
        metrics: Row order.
        figsize: Figure size in inches.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(metrics), len(conversation_lengths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    # subplots squeezes singleton dimensions away; restore the full grid shape so
    # indexing works for a single row or a single column too.
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    handles: list = []
    labels: list[str] = []
    for row, metric in enumerate(metrics):
        for col, length in enumerate(conversation_lengths):
            ax = axes[row, col]
            panel = (results.get(metric) or {}).get(length) or {}
            if not panel:
                ax.text(
                    0.5,
                    0.5,
                    "not computed",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=_CHANCE_COLOR,
                    fontsize=9,
                )
                _style_axis(ax)
            else:
                plot_metric_panel(ax, panel, metric=metric)
                if not handles:
                    handles, labels = ax.get_legend_handles_labels()

            # Titles and axis labels describe the grid position, so they are set
            # whether or not that combination was computed.
            if row == 0:
                ax.set_title(f"$L = {length}$", fontsize=10)
            if col == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric))
            if row == n_rows - 1:
                ax.set_xlabel("Number of speakers")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_metric(
    results: Mapping[str, SweepResult],
    metric: str,
    path: str | Path,
    title: str | None = None,
    figsize: tuple = (6.0, 4.0),
) -> None:
    """Draw one metric for one conversation length to its own file.

    Args:
        results: Mapping from attacker key to that attacker's sweep.
        metric: Which metric these results hold.
        path: Output path.
        title: Figure title.
        figsize: Figure size in inches.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    plot_metric_panel(
        ax,
        results,
        metric=metric,
        xlabel="Number of speakers",
        ylabel=METRIC_LABEL.get(metric, metric),
        title=title,
    )
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
