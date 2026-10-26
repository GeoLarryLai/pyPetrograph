"""Legend check figure, grain-size distribution figure, and figure saving."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from pyPetrograph.common.constants import PathLike

RGB = Tuple[int, int, int]


def save_figure(fig, path: PathLike, dpi: int = 150) -> Path:
    """``fig.savefig(path, dpi=dpi, bbox_inches="tight")`` at the caller's dpi (no 150 cap). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    return path


def plot_legend(legend: Dict[str, RGB], skip_names: Sequence[str] = ()):
    """
    Redraw the legend from the RGB values: one row per entry with a color square
    (black edge), the name, and ``(r, g, b)``. Names in ``skip_names`` are gray italic.
    Returns the Figure.
    """
    from pyPetrograph.amics.legend_read import name_key

    skip = {name_key(item) for item in skip_names}
    n = len(legend)
    if n == 0:
        fig, ax = plt.subplots(figsize=(7.0, 1.2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "(empty legend)", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig_w = 7.0
    fig_h = max(1.2, 0.32 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(n, 0.0)
    side_y = 0.62
    side_x = side_y * (fig_h / n) / fig_w
    for i, (name, rgb) in enumerate(legend.items()):
        color = (float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0)
        y0 = i + (1.0 - side_y) / 2.0
        ax.add_patch(
            Rectangle(
                (0.02, y0),
                side_x,
                side_y,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
            )
        )
        skipped = all(name_key(part) in skip for part in str(name).split(" / "))
        style = "italic" if skipped else "normal"
        text_color = "gray" if skipped else "black"
        ax.text(
            0.02 + side_x + 0.03,
            i + 0.5,
            str(name),
            va="center",
            ha="left",
            fontsize=10,
            fontstyle=style,
            color=text_color,
        )
        ax.text(
            0.98,
            i + 0.5,
            f"({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})",
            va="center",
            ha="right",
            fontsize=10,
            fontstyle=style,
            color=text_color,
        )
    ax.set_axis_off()
    ax.set_title(f"Legend ({n} entries)")
    fig.tight_layout()
    return fig


def _column_values(table: pd.DataFrame, column: str) -> np.ndarray:
    if len(table) == 0 or column not in table.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)


def _log_bins(positive: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """Log-spaced edges from the smallest to the largest value, so every grain is in a bin."""
    if positive.size < 2:
        return None
    lo, hi = float(np.min(positive)), float(np.max(positive))
    if not np.isfinite(lo) or not np.isfinite(hi) or not (hi > lo) or lo <= 0.0:
        return None
    pad = (hi / lo) ** (0.01)  # 1% of the log range on each side so the end grains sit inside
    return np.logspace(np.log10(lo / pad), np.log10(hi * pad), int(n_bins) + 1)


def _cdf_line(sizes: np.ndarray, x_lo: float, x_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative % finer as a line: one vertex per grain at plotting position (i - 0.5) / n,
    joined by straight lines, from 0 at ``x_lo`` to 100 at ``x_hi``.
    """
    ordered = np.sort(np.asarray(sizes, dtype=float))
    n = ordered.size
    x = np.concatenate([[x_lo], ordered, [x_hi]])
    y = np.concatenate([[0.0], 100.0 * (np.arange(1, n + 1) - 0.5) / float(n), [100.0]])
    return x, y


def _minerals_in_order(table: pd.DataFrame) -> list:
    if "mineral" not in table.columns:
        return []
    seen: list = []
    for value in table["mineral"].tolist():
        if not any(_same_label(value, previous) for previous in seen):
            seen.append(value)
    return seen


def _same_label(left, right) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if bool(left_missing) and bool(right_missing):
        return True
    return left == right


def _mineral_rgb(mineral, colors: Optional[Dict[str, RGB]], index: int) -> Tuple[float, float, float]:
    if colors is not None and mineral in colors:
        red, green, blue = colors[mineral]
        return (float(red) / 255.0, float(green) / 255.0, float(blue) / 255.0)
    sample = plt.get_cmap("tab10")(index % 10)
    return (float(sample[0]), float(sample[1]), float(sample[2]))


def _darker(rgb: Tuple[float, float, float], factor: float = 0.55) -> Tuple[float, float, float]:
    return (rgb[0] * factor, rgb[1] * factor, rgb[2] * factor)


ALL_GRAINS_D_RGB = (0.20, 0.45, 0.62)  # steel blue: D in the all-grains figure
ECD_RGB = (0.29, 0.25, 0.22)  # dark brownish gray: ECD in the all-grains figure


def plot_gsd(
    table: pd.DataFrame,
    *,
    unit: str,
    color: Optional[Tuple[float, float, float]] = None,
    title: Optional[str] = None,
    n_bins: int = 30,
):
    """
    2x2 grain-size figure. Top row: histogram (% of grains per log bin). Bottom row: cumulative % finer.

    ``unit`` is ``"um"`` (columns ``a_um`` / ``b_um`` / ``D_um`` / ``ECD_um``) or ``"px"``.
    Column 1: D as a half-transparent filled histogram and a solid line, b dashed, a dotted
    (outline histograms and lines). Column 2: ECD, filled histogram and solid line.
    ``color`` (0-1 RGB) is the D / ECD color; default is steel blue for D and dark brownish gray
    for ECD. With a ``color`` (one mineral) D and ECD both use it and a / b use a darker shade.
    Bins and both x axes span the smallest to the largest value of all four sizes. Cumulative
    curves are lines only (no fill), from 0 to 100 across the axis. Returns the Figure.
    """
    if unit not in ("um", "px"):
        raise ValueError(f"unit must be 'um' or 'px', got {unit!r}")
    unit_label = "µm" if unit == "um" else "px"
    series = {name: _column_values(table, f"{name}_{unit}") for name in ("a", "b", "D", "ECD")}
    series = {name: v[np.isfinite(v) & (v > 0.0)] for name, v in series.items()}
    every = np.concatenate(list(series.values()))

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    bins = _log_bins(every, n_bins)
    if bins is None or min(v.size for v in series.values()) < 2:
        for ax in axes.ravel():
            _mark_not_enough(ax)
        return fig
    x_lo, x_hi = float(bins[0]), float(bins[-1])

    d_rgb = color if color is not None else ALL_GRAINS_D_RGB
    ecd_rgb = color if color is not None else ECD_RGB
    line_rgb = _darker(d_rgb)

    def percent(values: np.ndarray) -> np.ndarray:
        counts, _edges = np.histogram(values, bins=bins)
        return 100.0 * counts / values.size

    # column 1: a (dotted), b (dashed), D (shade + solid)
    ax_h, ax_c = axes[0, 0], axes[1, 0]
    ax_h.stairs(percent(series["D"]), bins, fill=True, color=d_rgb, alpha=0.5, linewidth=0.0, zorder=1)
    ax_h.stairs(percent(series["b"]), bins, fill=False, color=line_rgb, linestyle="--", linewidth=1.3, zorder=3)
    ax_h.stairs(percent(series["a"]), bins, fill=False, color=line_rgb, linestyle=":", linewidth=1.6, zorder=3)
    styles = (("D", d_rgb, "-", 1.8), ("b", line_rgb, "--", 1.4), ("a", line_rgb, ":", 1.8))
    for name, rgb, style, width in styles:
        ax_c.plot(*_cdf_line(series[name], x_lo, x_hi), color=rgb, linestyle=style, linewidth=width)
    ax_c.legend(
        handles=[
            Line2D([0], [0], color=line_rgb, linestyle=":", linewidth=1.8, label="a (long axis)"),
            Line2D([0], [0], color=line_rgb, linestyle="--", linewidth=1.4, label="b (short axis)"),
            Line2D([0], [0], color=d_rgb, linestyle="-", linewidth=1.8, label="D = sqrt(a b)"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )

    # column 2: ECD
    ax_h2, ax_c2 = axes[0, 1], axes[1, 1]
    ax_h2.stairs(percent(series["ECD"]), bins, fill=True, color=ecd_rgb, alpha=0.6, linewidth=0.0)
    ax_c2.plot(*_cdf_line(series["ECD"], x_lo, x_hi), color=ecd_rgb, linewidth=1.8)
    ax_c2.legend(
        handles=[Line2D([0], [0], color=ecd_rgb, linewidth=1.8, label="ECD")],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )

    for column_axes, xlabel in (((ax_h, ax_c), f"a, b, D ({unit_label})"), ((ax_h2, ax_c2), f"ECD ({unit_label})")):
        for ax in column_axes:
            ax.set_xscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(xlabel)
    ax_c.set_ylim(-2.0, 102.0)
    ax_c2.set_ylim(-2.0, 102.0)
    ax_h.set_ylabel("grains (%)")
    ax_c.set_ylabel("cumulative % finer")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def _mark_not_enough(ax) -> None:
    ax.clear()
    ax.set_axis_off()
    ax.text(0.5, 0.5, "not enough grains", ha="center", va="center", transform=ax.transAxes)


def plot_gsd_by_mineral(
    table: pd.DataFrame,
    *,
    unit: str,
    colors: Optional[Dict[str, RGB]] = None,
    min_grains: int = 5,
    n_bins: int = 30,
) -> list:
    """
    One ``plot_gsd`` figure per mineral with at least ``min_grains`` grains (legend color from
    ``colors``, 0-255 RGB). Returns ``[(mineral, Figure), ...]`` in order of first appearance.
    """
    figures = []
    labels = table["mineral"].to_numpy() if "mineral" in table.columns else np.array([])
    for index, mineral in enumerate(_minerals_in_order(table)):
        if pd.isna(mineral):
            chosen = np.array([pd.isna(label) for label in labels], dtype=bool)
        else:
            chosen = labels == mineral
        part = table.loc[chosen]
        size_col = f"D_{unit}"
        n_group = int(np.count_nonzero(np.isfinite(_column_values(part, size_col))))
        if n_group < int(min_grains):
            continue
        name = "Unknown" if pd.isna(mineral) else str(mineral)
        figures.append(
            (
                mineral,
                plot_gsd(
                    part,
                    unit=unit,
                    color=_mineral_rgb(mineral, colors, index),
                    title=f"{name}  n={n_group}",
                    n_bins=n_bins,
                ),
            )
        )
    return figures
