"""Mineral map on the BSE grid: color snapping, per-grain vote, overlays, and counts."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from pyPetrograph.amics.legend_read import name_key
from pyPetrograph.common.constants import PathLike
from pyPetrograph.mineral_map.counts import count_grains
from pyPetrograph.mineral_map.legend import read_rgba, snap_to_legend

RGB = Tuple[int, int, int]
Legend = Dict[str, RGB]

DEFAULT_SKIP_NAMES: Tuple[str, ...] = ("Unknown", "Low_Counts", "Un_x_rayed", "Shadows", "Pores")

VOTE_COLUMNS = (
    "grain_id",
    "mineral",
    "purity",
    "second_mineral",
    "second_frac",
    "mixed",
    "known_px",
    "map_bg_frac",
)

UNMAPPED = "Unmapped"


def _skipped_codes(legend: Legend, skip_names: Sequence[str]) -> Set[int]:
    """1-based legend codes whose name (every ``" / "`` part for merged keys) is in ``skip_names``."""
    skip = {name_key(s) for s in skip_names}
    out: Set[int] = set()
    for code, key in enumerate(legend.keys(), start=1):
        parts = str(key).split(" / ")
        if all(name_key(p) in skip for p in parts):
            out.add(code)
    return out


def load_mineral_map(
    path: PathLike,
    legend: Legend,
    *,
    footer_row: int,
    color_tol: float = 10.0,
    white_tol: float = 3.0,
    recolor: Optional[Dict[RGB, RGB]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    ``(map_rgb uint8 HxWx3, cmap uint8 HxW, valid bool HxW, missing_df)``.

    ``cmap`` codes: 0 background, 1..K legend order (``snap_to_legend``).
    Rows >= ``footer_row`` are invalid. ``missing_df`` = map colors beyond ``color_tol``
    (counted as Unknown) with pixel counts and the closest legend name.
    ``recolor`` ``{map color: legend color}`` repaints those map colors before snapping
    (from ``match_map_colors``); the returned ``map_rgb`` has the repainted colors.
    """
    rgb, alpha = read_rgba(path)
    if recolor:
        rgb = np.array(rgb, dtype=np.uint8, copy=True)
        for src, dst in recolor.items():
            rgb[np.all(rgb == np.asarray(src, dtype=np.uint8), axis=-1)] = np.asarray(dst, dtype=np.uint8)
    if alpha is None:
        alpha2 = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    else:
        alpha2 = np.array(alpha, dtype=np.uint8, copy=True)
    alpha2[int(footer_row):] = 0
    cmap, valid, missing_df = snap_to_legend(rgb, alpha2, legend, tol=color_tol, white_tol=white_tol)
    return rgb, cmap, valid, missing_df


_GRAY_CHROMA = 10.0


def _lab(rgb: Sequence[int]) -> np.ndarray:
    from skimage.color import rgb2lab

    return rgb2lab(np.asarray([[rgb]], dtype=np.float64) / 255.0)[0, 0]


def _color_difference(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """CIEDE2000 color difference (about 2 = just visible). Grays only pair with grays (``inf`` otherwise)."""
    from skimage.color import deltaE_ciede2000

    gray_a = float(np.hypot(lab_a[1], lab_a[2])) < _GRAY_CHROMA
    gray_b = float(np.hypot(lab_b[1], lab_b[2])) < _GRAY_CHROMA
    if gray_a != gray_b:
        return float("inf")
    return float(deltaE_ciede2000(lab_a, lab_b))


MATCH_COLUMNS = (
    "map_red",
    "map_green",
    "map_blue",
    "pixels",
    "assigned",
    "distance",
    "nearest",
    "nearest_distance",
    "nearest_taken",
    "status",
)


def match_map_colors(
    missing: pd.DataFrame,
    legend: Legend,
    cmap: np.ndarray,
    skip_names: Sequence[str] = DEFAULT_SKIP_NAMES,
    *,
    max_dist: float = 25.0,
    min_pixels: int = 20,
) -> pd.DataFrame:
    """
    Name the map colors that matched no legend swatch, looking at the whole set of colors at once.

    1. Legend names whose swatch is already in the map (``cmap`` from ``load_mineral_map``) are taken and
       keep their exact color.
    2. The leftover map colors in ``missing`` (at least ``min_pixels`` pixels) are paired with the free
       legend names, one map color per name, choosing the pairing with the smallest total CIEDE2000
       difference weighted by pixel count (a big color decides before a small one); pairs further apart than ``max_dist`` are not made (status ``free name``).
    3. A colored leftover (not gray) then takes the nearest mineral name even if it is already used, since
       a colored map pixel is some mineral (status ``shared name``). ``skip_names`` and gray names are
       never used here.
    Grays only pair with grays; a gray with no gray partner stays unnamed (status ``unnamed``).

    Returns ``MATCH_COLUMNS``, one row per color of ``missing`` in its order. ``assigned`` is ``""`` when
    unnamed. ``nearest`` is the closest legend name overall and ``nearest_taken`` says whether an exact map
    color already holds it.
    """
    from scipy.optimize import linear_sum_assignment

    from pyPetrograph.mineral_map.legend import unknown_code

    names = list(legend.keys())
    counts = np.bincount(np.asarray(cmap, dtype=np.int64).ravel(), minlength=len(names) + 1)
    # far colors are coded as Unknown too: Unknown is taken only if more pixels carry it than ``missing`` holds
    k_unk = int(unknown_code(legend))
    n_far = int(missing["pixels"].sum()) if len(missing) else 0
    taken = {
        names[code - 1]
        for code in range(1, len(names) + 1)
        if counts[code] > (n_far if code == k_unk else 0)
    }
    free = [n for n in names if n not in taken]

    colors = [(int(r.map_red), int(r.map_green), int(r.map_blue)) for r in missing.itertuples(index=False)]
    pixels = missing["pixels"].astype(int).to_numpy() if len(missing) else np.zeros(0, int)
    legend_lab = {n: _lab(legend[n]) for n in names}
    color_lab = [_lab(c) for c in colors]
    diff_all = np.array([[_color_difference(cl, legend_lab[n]) for n in names] for cl in color_lab]).reshape(
        len(colors), len(names)
    )

    assigned = [""] * len(colors)
    used = [i for i in range(len(colors)) if pixels[i] >= int(min_pixels)]
    if used and free:
        free_idx = [names.index(n) for n in free]
        cost = diff_all[np.ix_(used, free_idx)]
        ok = np.isfinite(cost) & (cost <= float(max_dist))
        weight = (pixels[used] / float(pixels[used].sum()))[:, None]  # a big color decides before a small one
        cost = np.where(ok, cost * weight, 1e6)
        stay = np.where(np.eye(len(used)) > 0, float(max_dist) * weight, 1e6)  # "stay unnamed" per color
        rows, cols = linear_sum_assignment(np.hstack([cost, stay]))
        for r, c in zip(rows, cols):
            if c < len(free_idx) and ok[r, c]:
                assigned[used[r]] = free[c]

    status = ["free name" if name else "unnamed" for name in assigned]
    skipped = _skipped_codes(legend, skip_names)
    minerals = [
        j
        for j, n in enumerate(names)
        if (j + 1) not in skipped and float(np.hypot(*legend_lab[n][1:])) >= _GRAY_CHROMA
    ]
    for i, cl in enumerate(color_lab):
        if assigned[i] or float(np.hypot(cl[1], cl[2])) < _GRAY_CHROMA or not minerals:
            continue
        j = min(minerals, key=lambda m: diff_all[i, m])
        assigned[i], status[i] = names[j], "shared name"

    out = []
    for i, color in enumerate(colors):
        k = int(np.argmin(diff_all[i])) if len(names) else -1
        nearest = names[k] if k >= 0 and np.isfinite(diff_all[i, k]) else ""
        nearest_d = float(diff_all[i, k]) if nearest else float("nan")
        name = assigned[i]
        dist = float(diff_all[i, names.index(name)]) if name else float("nan")
        out.append((*color, int(pixels[i]), name, dist, nearest, nearest_d, nearest in taken, status[i]))
    return pd.DataFrame(out, columns=list(MATCH_COLUMNS))


def recolor_from_matches(matches: pd.DataFrame, legend: Legend) -> Dict[RGB, RGB]:
    """``{map color: legend color}`` for the assigned rows of ``match_map_colors`` (for ``load_mineral_map``)."""
    out: Dict[RGB, RGB] = {}
    for _, row in matches.iterrows():
        if row["assigned"]:
            key = (int(row["map_red"]), int(row["map_green"]), int(row["map_blue"]))
            out[key] = tuple(int(v) for v in legend[row["assigned"]])
    return out


def plot_missing_colors(missing_df: pd.DataFrame, top: int = 15):
    """One row per missing map color: patch, ``(r, g, b)``, pixels, closest legend name. Returns the Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    total = len(missing_df)
    if total == 0:
        fig, ax = plt.subplots(figsize=(7.0, 1.2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "(no missing colors)", ha="center", va="center", transform=ax.transAxes)
        return fig

    df = missing_df.head(int(top))
    n = len(df)
    fig_w = 7.0
    fig_h = max(1.2, 0.32 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(n, 0.0)
    ax.set_axis_off()
    side_y = 0.62
    side_x = side_y * (fig_h / n) / fig_w
    for i, row in enumerate(df.itertuples(index=False)):
        r, g, b = int(row.map_red), int(row.map_green), int(row.map_blue)
        y0 = i + (1.0 - side_y) / 2.0
        ax.add_patch(
            Rectangle(
                (0.02, y0),
                side_x,
                side_y,
                facecolor=(r / 255.0, g / 255.0, b / 255.0),
                edgecolor="black",
                linewidth=0.8,
            )
        )
        ax.text(
            0.02 + side_x + 0.03,
            i + 0.5,
            f"({r}, {g}, {b})  {int(row.pixels)} px  closest: {row.closest_legend_color}",
            va="center",
            ha="left",
            fontsize=10,
        )
    ax.set_title(f"Map colors not in the legend ({n} shown of {total})")
    fig.tight_layout()
    return fig


def _polygon_parts(geom) -> List:
    """Non-empty Polygon parts of a geometry (Polygon, Multi*, GeometryCollection)."""
    if geom is None or getattr(geom, "is_empty", True):
        return []
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out: List = []
        for g in geom.geoms:
            out.extend(_polygon_parts(g))
        return out
    return []


def vote_grains(
    polys: Sequence,
    cmap: np.ndarray,
    legend: Legend,
    skip_names: Sequence[str] = DEFAULT_SKIP_NAMES,
    mixed_frac: float = 0.30,
    unmapped_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Majority mineral inside each polygon, ignoring background and ``skip_names``.

    Returns ``VOTE_COLUMNS``; ``grain_id`` 1..N in input order. ``map_bg_frac`` = share of the
    grain's pixels where the map has no color (white background or outside the map). When it is
    above ``unmapped_frac`` the grain is ``"Unmapped"`` (AMICS measured nothing there). Otherwise,
    no known pixels -> ``"Unknown"``, purity NaN, mixed False. ``mixed`` = ``second_frac >= mixed_frac``.
    """
    from skimage.draw import polygon as draw_polygon

    keys = list(legend.keys())
    K = len(keys)
    skipped = _skipped_codes(legend, skip_names)
    drop = np.array([0, *sorted(skipped)], dtype=np.int64)
    drop = drop[drop <= K]

    rows = []
    for gid, geom in enumerate(polys, start=1):
        chunks = []
        for part in _polygon_parts(geom):
            xy = np.asarray(part.exterior.coords, dtype=np.float64)
            if len(xy) < 3:
                continue
            # SEG polygons come from find_contours: integer coordinates are pixel CENTERS,
            # the same convention as skimage.draw.polygon, so no half-pixel shift.
            rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=cmap.shape[:2])
            if len(rr):
                chunks.append(np.asarray(cmap[rr, cc], dtype=np.int64))
        if chunks:
            counts = np.bincount(np.concatenate(chunks), minlength=K + 1)[: K + 1]
        else:
            counts = np.zeros(K + 1, dtype=np.int64)
        counts = counts.copy()
        total = int(counts.sum())
        bg_frac = float(counts[0]) / total if total else 0.0
        counts[drop] = 0
        known = int(counts.sum())
        if total and bg_frac > unmapped_frac:
            rows.append((gid, UNMAPPED, float("nan"), "", 0.0, False, known, bg_frac))
            continue
        if known == 0:
            rows.append((gid, "Unknown", float("nan"), "", 0.0, False, 0, bg_frac))
            continue
        order = sorted(np.nonzero(counts)[0].tolist(), key=lambda c: (-int(counts[c]), c))
        top = order[0]
        purity = float(counts[top]) / known
        if len(order) > 1:
            second_mineral = keys[order[1] - 1]
            second_frac = float(counts[order[1]]) / known
        else:
            second_mineral, second_frac = "", 0.0
        rows.append(
            (gid, keys[top - 1], purity, second_mineral, second_frac, bool(second_frac >= mixed_frac), known, bg_frac)
        )

    df = pd.DataFrame(rows, columns=list(VOTE_COLUMNS))
    df["grain_id"] = df["grain_id"].astype(int)
    df["mineral"] = df["mineral"].astype(str)
    df["second_mineral"] = df["second_mineral"].astype(str)
    df["purity"] = df["purity"].astype(float)
    df["second_frac"] = df["second_frac"].astype(float)
    df["mixed"] = df["mixed"].astype(bool)
    df["known_px"] = df["known_px"].astype(int)
    df["map_bg_frac"] = df["map_bg_frac"].astype(float)
    return df


def _blend(gray: np.ndarray, map_rgb: np.ndarray, map_valid: np.ndarray, alpha: float) -> np.ndarray:
    """Opaque gray base; mineral colors blended at ``alpha`` where valid and not near-white."""
    g = np.asarray(gray, dtype=np.uint8)
    base = np.repeat(g[..., None], 3, axis=-1).astype(np.float32)
    rgb = np.asarray(map_rgb)
    show = np.asarray(map_valid, dtype=bool) & ~np.all(rgb >= 253, axis=-1)
    out = base.copy()
    out[show] = (1.0 - alpha) * base[show] + alpha * rgb[show].astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _outline_segments(polys: Sequence) -> List[np.ndarray]:
    segs: List[np.ndarray] = []
    for geom in polys:
        for part in _polygon_parts(geom):
            xy = np.asarray(part.exterior.coords, dtype=np.float64)
            if len(xy) >= 2:
                segs.append(xy[:, :2])
    return segs


def _overlay_figure(gray, map_rgb, map_valid, polys, alpha):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    img = _blend(gray, map_rgb, map_valid, alpha)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10.0, 10.0 * h / w))
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    segs = _outline_segments(polys)
    if segs:
        ax.add_collection(LineCollection(segs, colors="white", linewidths=0.3))
    fig.tight_layout(pad=0)
    return fig, ax


def plot_overlay(gray: np.ndarray, map_rgb: np.ndarray, map_valid: np.ndarray, polys: Sequence, alpha: float = 0.5):
    """Gray BSE (opaque) + mineral map at ``alpha`` (valid, non-white) + thin white grain outlines. Returns the Figure."""
    fig, _ = _overlay_figure(gray, map_rgb, map_valid, polys, alpha)
    return fig


def plot_mixed(
    gray: np.ndarray,
    map_rgb: np.ndarray,
    map_valid: np.ndarray,
    polys: Sequence,
    table: pd.DataFrame,
    alpha: float = 0.5,
):
    """Overlay + red circles (radius = long axis / 2) on grains with ``mixed`` True. Returns the Figure."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    fig, ax = _overlay_figure(gray, map_rgb, map_valid, polys, alpha)
    mixed = table["mixed"].to_numpy(dtype=bool)
    cx = pd.to_numeric(table["centroid_x"], errors="coerce").to_numpy(dtype=float)
    cy = pd.to_numeric(table["centroid_y"], errors="coerce").to_numpy(dtype=float)
    lp = pd.to_numeric(table["a_px"], errors="coerce").to_numpy(dtype=float)
    circles = [
        Circle((float(cx[i]), float(cy[i])), float(lp[i]) / 2.0 + 3.0)
        for i in range(len(table))
        if mixed[i] and np.isfinite(cx[i]) and np.isfinite(cy[i]) and np.isfinite(lp[i])
    ]
    if circles:
        ax.add_collection(PatchCollection(circles, facecolor="none", edgecolor="red", linewidths=0.8))
    ax.set_title(f"mixed grains: {int(mixed.sum())} of {len(table)}")
    return fig


def plot_unmapped(
    gray: np.ndarray,
    map_rgb: np.ndarray,
    map_valid: np.ndarray,
    polys: Sequence,
    table: pd.DataFrame,
    alpha: float = 0.5,
):
    """Overlay + thick cyan outlines on ``"Unmapped"`` grains (no map color under them). Returns the Figure."""
    from matplotlib.collections import LineCollection

    fig, ax = _overlay_figure(gray, map_rgb, map_valid, polys, alpha)
    unmapped = (table["mineral"].astype(str) == UNMAPPED).to_numpy()
    segs = _outline_segments([p for p, u in zip(polys, unmapped) if u])
    if segs:
        ax.add_collection(LineCollection(segs, colors="cyan", linewidths=1.2))
    ax.set_title(f"unmapped grains (cyan): {int(unmapped.sum())} of {len(table)}")
    return fig


def _finite_median(values) -> float:
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.nanmedian(v)) if len(v) else float("nan")


def count_table(
    table: pd.DataFrame,
    legend: Legend,
    skip_names: Sequence[str],
    *,
    sample: str,
    min_grain_px: float,
    unit: str,
) -> pd.DataFrame:
    """
    Notebook-2 style counts (``count_grains`` with the same table as split and no-split)
    over non-skipped minerals + ``"Unknown"``, plus ``median_a_{unit}``, ``median_b_{unit}``,
    ``median_D_{unit}``, ``median_ECD_{unit}`` and ``n_mixed`` per class.
    """
    skipped = _skipped_codes(legend, skip_names)
    count_legend: Legend = {
        name: tuple(int(v) for v in rgb)
        for code, (name, rgb) in enumerate(legend.items(), start=1)
        if code not in skipped
    }
    if "Unknown" not in count_legend:
        count_legend["Unknown"] = (0, 0, 0)

    t = table.copy()
    t["class"] = t["mineral"]
    counts = count_grains(t, t, count_legend, sample, min_grain_px=min_grain_px)

    size_names = ("a", "b", "D", "ECD")
    area = pd.to_numeric(t["area_px"], errors="coerce")
    big = t[area >= float(min_grain_px)]
    mixed_big = big["mixed"].fillna(False).astype(bool)

    medians = {name: [] for name in size_names}
    n_mixed = []
    for cls in counts["class"]:
        sel = big if cls == "TOTAL" else big[big["class"] == cls]
        for name in size_names:
            medians[name].append(_finite_median(sel[f"{name}_{unit}"]))
        sel_mixed = mixed_big if cls == "TOTAL" else mixed_big[big["class"] == cls]
        n_mixed.append(int(sel_mixed.sum()))
    for name in size_names:
        counts[f"median_{name}_{unit}"] = medians[name]
    counts["n_mixed"] = n_mixed
    return counts


def _class_color(name: str, legend: Legend) -> RGB:
    if name in legend:
        return tuple(int(v) for v in legend[name])
    if name == "Unknown":
        return (0, 0, 0)
    return (128, 128, 128)


def plot_composition(
    polys: Sequence,
    table: pd.DataFrame,
    legend: Legend,
    counts: pd.DataFrame,
    *,
    sample: str,
    shape: Tuple[int, int],
):
    """
    Left: each grain painted with its mineral color (white background, gray edges).
    Right: one stacked bar, percent of grains per mineral.

    Same idea as notebook 2: the class-colored grain picture, plus the ``share_pct`` bar.
    ``polys`` lines up with ``table`` (needs ``mineral``). ``counts`` is ``count_table`` output;
    the TOTAL row and classes with ``n`` of 0 are left off the bar. ``shape`` is (height, width)
    of the image the polygons sit on. Returns the Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from skimage.draw import polygon as draw_polygon
    from skimage.segmentation import find_boundaries

    if len(polys) != len(table):
        raise ValueError(f"polys ({len(polys)}) and table ({len(table)}) length mismatch")
    height, width = int(shape[0]), int(shape[1])
    labels = np.zeros((height, width), dtype=np.int32)
    minerals = table["mineral"].astype(str).tolist()
    for i, geom in enumerate(polys, start=1):
        for part in _polygon_parts(geom):
            xy = np.asarray(part.exterior.coords, dtype=np.float64)
            if len(xy) < 3:
                continue
            rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=labels.shape)
            if len(rr):
                labels[rr, cc] = i
    lut = np.full((len(polys) + 1, 3), 255, dtype=np.uint8)
    for i, name in enumerate(minerals, start=1):
        lut[i] = _class_color(name, legend)
    painted = lut[labels]
    if int(labels.max()) > 0:
        painted[find_boundaries(labels, mode="inner", connectivity=1)] = (60, 60, 60)

    share = counts.loc[(counts["class"] != "TOTAL") & (counts["n"] > 0)].copy()
    fig_h = max(4.2, min(9.0, 8.0 * height / max(width, 1)))
    fig_w = fig_h * width / max(height, 1) + 4.2
    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[4.4, 0.55, 1.7])
    ax_img = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1])
    ax_key = fig.add_subplot(grid[2])
    ax_img.imshow(painted, interpolation="nearest")
    ax_img.set_axis_off()
    ax_img.set_title(f"{sample}: grains by mineral")

    bottom = 0.0
    handles = []
    for _, row in share.iterrows():
        name = str(row["class"])
        pct = float(row["share_pct"])
        color = tuple(v / 255.0 for v in _class_color(name, legend))
        ax_bar.bar([0], [pct], bottom=bottom, color=color, edgecolor="k", linewidth=0.3, width=0.62)
        handles.append(Patch(facecolor=color, edgecolor="k", label=f"{name}  {pct:.1f}%"))
        bottom += pct
    ax_bar.set_xlim(-0.7, 0.7)
    ax_bar.set_ylim(0.0, 100.0)
    ax_bar.set_xticks([0])
    ax_bar.set_xticklabels([sample])
    ax_bar.set_ylabel("share of grains (%)")
    ax_key.set_axis_off()
    if handles:
        ax_key.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    else:
        ax_bar.set_axis_off()
        ax_bar.text(0.5, 0.5, "no counted grains", ha="center", va="center", transform=ax_bar.transAxes)
    return fig
