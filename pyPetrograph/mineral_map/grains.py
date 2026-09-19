"""
Grains from a class map: absorb specks → colour regions → split → table → polygons.

A grain is one connected blob of **one legend colour**; a colour change is a
grain boundary. Unknown / black pixels are gaps (they do not glue grains). A
black patch surrounded by a single grain is filled into that grain so the
outline keeps the outer shape; a black blob that touches no coloured pixel is
its own Unknown particle.

Touching same-colour grains are split three ways, in order: thin bridges are
broken (morphological opening, ``bridge_px``), necks are cut where the distance
map dips at least ``h_min`` below both peaks (h-maxima watershed), and concave
V / L / needle shapes are cut between two deep notches when both halves come
out compact (``notch_solidity``, ``notch_halves``).

Class = largest known colour inside the grain. Unknown only when Unknown pixels
are at least ``unknown_frac`` of the grain (default 0.95).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import shapes as rasterio_shapes
from scipy import ndimage as ndi
from shapely.geometry import mapping, shape
from skimage.draw import line as draw_line
from skimage.measure import label as cc_label, regionprops_table
from skimage.morphology import binary_opening, convex_hull_image, disk, h_maxima, remove_small_holes
from skimage.segmentation import find_boundaries, watershed

from pyPetrograph.common.constants import COORD_DECIMALS, PathLike
from pyPetrograph.mineral_map.legend import UNLISTED_RGB, Legend, RGB, class_names, unknown_code

_PROPS = (
    "label",
    "area",
    "centroid",
    "bbox",
    "equivalent_diameter_area",
    "axis_major_length",
    "axis_minor_length",
    "solidity",
)
_BOUNDARY_RGB = (60, 60, 60)


def absorb_specks(cmap: np.ndarray, valid: np.ndarray, speck_px: int = 30) -> np.ndarray:
    """
    Replace same-color blobs smaller than ``speck_px`` with the class of the
    nearest non-speck pixel (grey / black / off-legend specks inside a grain,
    small holes). Pixels outside ``valid`` are never specks and never sources.
    """
    if speck_px <= 1:
        return cmap
    work = cmap.astype(np.int32)
    work[~valid] = -1
    lab = cc_label(work, connectivity=1, background=-2)
    areas = np.bincount(lab.ravel())
    small = areas < int(speck_px)
    small[0] = False
    speck = small[lab] & valid
    if not speck.any():
        return cmap
    idx = ndi.distance_transform_edt(speck, return_distances=False, return_indices=True)
    out = cmap.copy()
    out[speck] = cmap[idx[0][speck], idx[1][speck]]
    return out


def absorb_class_specks(
    cmap: np.ndarray,
    valid: np.ndarray,
    class_id: int,
    speck_px: int = 50,
) -> np.ndarray:
    """Absorb blobs of one class (e.g. Unknown/black) smaller than ``speck_px``."""
    if speck_px <= 1:
        return cmap
    mask = (cmap == int(class_id)) & valid
    if not mask.any():
        return cmap
    lab, _ = ndi.label(mask)
    areas = np.bincount(lab.ravel())
    small = areas < int(speck_px)
    small[0] = False
    speck = small[lab]
    if not speck.any():
        return cmap
    idx = ndi.distance_transform_edt(speck, return_distances=False, return_indices=True)
    out = cmap.copy()
    out[speck] = cmap[idx[0][speck], idx[1][speck]]
    return out


def _relabel_min_area(labels: np.ndarray, min_px: int) -> np.ndarray:
    """Drop regions smaller than ``min_px`` and renumber 1..n."""
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_px)
    keep[0] = False
    lut = np.zeros(len(areas), dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[labels]


def _colour_regions(cmap: np.ndarray, colored: np.ndarray) -> np.ndarray:
    """Connected same-colour blobs, int32 1..n. A colour change is a boundary."""
    out = np.zeros(cmap.shape, dtype=np.int32)
    off = 0
    for k in np.unique(cmap[colored]):
        lab, n = ndi.label((cmap == int(k)) & colored)
        if n:
            out = np.where(lab > 0, lab.astype(np.int32) + off, out)
            off += int(n)
    return out


def _intersect(lab: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Split ``lab`` wherever it crosses a ``regions`` boundary; renumber 1..n."""
    n = int(regions.max()) + 1
    comb = lab.astype(np.int64) * n + regions.astype(np.int64)
    comb[lab == 0] = -1
    _, inv = np.unique(comb, return_inverse=True)
    out = (inv.reshape(lab.shape) + 1).astype(np.int32)
    out[lab == 0] = 0
    return _relabel_min_area(out, 1)


def _absorb_small_pieces(lab: np.ndarray, min_px: int) -> np.ndarray:
    """A piece smaller than ``min_px`` that touches exactly one bigger piece joins it."""
    n = int(lab.max())
    if n == 0:
        return lab
    area = np.bincount(lab.ravel(), minlength=n + 1)
    small = area < int(min_px)
    small[0] = False
    if not small.any():
        return lab
    pairs = []
    for a, b in ((lab[:-1, :], lab[1:, :]), (lab[:, :-1], lab[:, 1:])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack([a[m], b[m]], axis=1))
        pairs.append(np.stack([b[m], a[m]], axis=1))
    p = np.unique(np.concatenate(pairs), axis=0)
    p = p[small[p[:, 0]] & ~small[p[:, 1]]]
    if len(p) == 0:
        return lab
    cnt = np.bincount(p[:, 0], minlength=n + 1)
    single = p[cnt[p[:, 0]] == 1]
    lut = np.arange(n + 1, dtype=np.int32)
    lut[single[:, 0]] = single[:, 1]
    return lut[lab]


def _add_black(cmap: np.ndarray, valid: np.ndarray, unk: int, pieces: np.ndarray) -> np.ndarray:
    """
    Black / Unknown pixels: a blob surrounded by one piece is filled into it; a
    blob touching no coloured pixel becomes its own particle; the rest (glue
    between grains) stays background.
    """
    black = (cmap == int(unk)) & valid
    out = pieces.copy()
    if not black.any():
        return out
    colored = pieces > 0
    black_lab, n_black = ndi.label(black)
    ids = np.arange(1, n_black + 1)
    areas = np.bincount(black_lab.ravel(), minlength=n_black + 1)
    if colored.any():
        holes = ndi.binary_fill_holes(colored) & ~colored & valid
        hole_hits = np.bincount(black_lab[holes].ravel(), minlength=n_black + 1)
        in_holes = (areas > 0) & (hole_hits == areas)
        idx = ndi.distance_transform_edt(~colored, return_distances=False, return_indices=True)
        nearest = pieces[idx[0], idx[1]]
        del idx
        mn = np.zeros(n_black + 1, dtype=np.int32)
        mx = np.zeros(n_black + 1, dtype=np.int32)
        mn[1:] = ndi.minimum(nearest, labels=black_lab, index=ids)
        mx[1:] = ndi.maximum(nearest, labels=black_lab, index=ids)
        enclosed = in_holes & (mn == mx) & (mn > 0)
        enclosed[0] = False
        px = enclosed[black_lab]
        out[px] = nearest[px]
    touch_hits = np.bincount(black_lab[ndi.binary_dilation(colored) & black].ravel(), minlength=n_black + 1)
    standalone = (areas > 0) & (touch_hits == 0)
    standalone[0] = False
    sb = standalone[black_lab]
    if sb.any():
        extra, _ = ndi.label(sb)
        out = np.where(sb, extra.astype(np.int32) + int(out.max()), out)
    return out


def _break_bridges(colored: np.ndarray, regions: np.ndarray, bridge_px: int, min_grain_px: int) -> np.ndarray:
    """
    Pieces = colour regions, additionally cut where an opening with a disk of
    radius ``bridge_px`` disconnects them (bridges thinner than ~2r+1 px). Thin
    grains shaved off entirely keep their own piece if at least ``min_grain_px``.
    """
    if bridge_px <= 0 or not colored.any():
        return regions.copy()
    opened = binary_opening(colored, disk(int(bridge_px)))
    markers, n = ndi.label(opened)
    markers = markers.astype(np.int32)
    rem = colored & ~opened
    del opened
    if rem.any():
        rem_lab, n_rem = ndi.label(rem)
        rem_area = np.bincount(rem_lab.ravel(), minlength=n_rem + 1)
        big = rem_area >= int(min_grain_px)
        big[0] = False
        keep = big[rem_lab]
        markers = np.where(keep, rem_lab.astype(np.int32) + int(n), markers)
        del rem_lab
    del rem
    markers = _intersect(markers, regions)  # a marker never spans two colours
    dist = ndi.distance_transform_edt(colored).astype(np.float32)
    lab = watershed(-dist, markers, mask=colored, connectivity=1).astype(np.int32)
    del dist, markers
    left = colored & (lab == 0)
    if left.any():
        extra, _ = ndi.label(left)
        lab = np.where(left, extra.astype(np.int32) + int(lab.max()), lab)
    return _intersect(lab, regions)


_HOLE_FILL_PX = int(np.pi * 15.0 ** 2)


def _split_hmax(pieces: np.ndarray, h_min: float) -> np.ndarray:
    """
    Cut pieces at necks: markers are h-maxima of the distance map (two peaks
    count as two grains only if the neck dips ≥ ``h_min`` below the lower one).
    Labels never cross a piece boundary.
    """
    mask = pieces > 0
    if not mask.any():
        return pieces
    # small holes are filled for the distance map only, so a dust patch inside
    # a grain does not turn its ridge into a ring
    filled = remove_small_holes(mask, area_threshold=max(1, _HOLE_FILL_PX))
    dist = ndi.distance_transform_edt(filled).astype(np.float32)
    del filled
    peak = h_maxima(dist, float(h_min)).astype(bool) & mask
    markers, n_peaks = ndi.label(peak, structure=np.ones((3, 3), dtype=bool))
    markers = markers.astype(np.int32)
    # every piece needs at least one marker
    n_parts = int(pieces.max())
    has = np.zeros(n_parts + 1, dtype=bool)
    has[np.unique(pieces[peak])] = True
    del peak
    missing = np.where(~has[1:])[0] + 1
    if len(missing):
        pos = ndi.maximum_position(dist, labels=pieces, index=missing)
        for k, (r, c) in enumerate(pos, start=n_peaks + 1):
            markers[int(r), int(c)] = k
    lab = watershed(-dist, markers, mask=mask, connectivity=1).astype(np.int32)
    del dist, markers
    return _intersect(lab, pieces)


def _solidity(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 0.0
    hull_area = int(convex_hull_image(mask).sum())
    return area / float(hull_area) if hull_area else 0.0


def _try_notch_cut(
    mask: np.ndarray,
    *,
    min_px: int,
    sol_max: float,
    halves_min: float,
    depth_min: float = 4.0,
    gain: float = 0.05,
    cut_ratio: float = 1.0,
    neck_ratio: float = 1.0,
    n_try: int = 6,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    One straight cut between the two deepest notches of a concave grain.
    Returns the two halves, or None when no cut passes every gate:
    solidity < ``sol_max``; both notches ≥ ``depth_min`` px deep; cut length ≤
    ``cut_ratio`` × (depth₁ + depth₂); cut runs inside the grain; both halves ≥
    ``min_px`` and at least ``halves_min`` (and ``gain`` above the whole) in
    solidity; cut ≤ ``neck_ratio`` × width of the narrower half.
    """
    area = int(mask.sum())
    if area < 2 * int(min_px):
        return None
    hull = convex_hull_image(mask)
    hull_area = int(hull.sum())
    if hull_area == 0:
        return None
    sol = area / float(hull_area)
    if sol >= float(sol_max):
        return None
    bay_lab, n_bay = ndi.label(hull & ~mask)
    if n_bay < 2:
        return None
    hull_d = ndi.distance_transform_edt(hull)
    ids = np.arange(1, n_bay + 1)
    depth = np.asarray(ndi.maximum(hull_d, labels=bay_lab, index=ids), dtype=float)
    good = np.where(depth >= float(depth_min))[0]
    if len(good) < 2:
        return None
    tips = ndi.maximum_position(hull_d, labels=bay_lab, index=(good + 1).tolist())
    tips = [(int(r), int(c)) for r, c in tips]
    cands = []
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            L = float(np.hypot(tips[i][0] - tips[j][0], tips[i][1] - tips[j][1]))
            cands.append((L / (depth[good[i]] + depth[good[j]]), L, i, j))
    cands.sort()
    dist = None
    for score, L, i, j in cands[: int(n_try)]:
        if score > float(cut_ratio):
            break
        rr, cc = draw_line(tips[i][0], tips[i][1], tips[j][0], tips[j][1])
        if len(rr) <= 2 or not mask[rr, cc][1:-1].all():
            continue
        cut = mask.copy()
        cut[rr, cc] = False
        parts, n_parts = ndi.label(cut)
        if n_parts < 2:
            continue
        sizes = np.bincount(parts.ravel(), minlength=n_parts + 1)
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]
        if sizes[order[1]] < int(min_px):
            continue
        a = parts == order[0]
        b = parts == order[1]
        rest = mask & ~a & ~b  # the cut line and any slivers
        if rest.any():
            two = np.where(a, 1, np.where(b, 2, 0))
            ind = ndi.distance_transform_edt(two == 0, return_distances=False, return_indices=True)
            nn = two[ind[0], ind[1]]
            a = a | (rest & (nn == 1))
            b = b | (rest & (nn == 2))
        sa, sb = _solidity(a), _solidity(b)
        if min(sa, sb) < max(float(halves_min), sol + float(gain)):
            continue
        if dist is None:
            dist = ndi.distance_transform_edt(mask)
        wa = 2.0 * float(dist[a].max())
        wb = 2.0 * float(dist[b].max())
        if L > float(neck_ratio) * min(wa, wb):
            continue
        return a, b
    return None


def _notch_split(
    labels: np.ndarray,
    *,
    min_px: int,
    sol_max: float,
    halves_min: float,
    max_rounds: int = 3,
) -> np.ndarray:
    """Apply ``_try_notch_cut`` to every concave grain; halves are re-tried."""
    out = labels.copy()
    next_id = int(out.max()) + 1
    objs = ndi.find_objects(out)
    queue = [(i + 1, sl) for i, sl in enumerate(objs) if sl is not None]
    for _ in range(int(max_rounds)):
        if not queue:
            break
        again = []
        for lid, sl in queue:
            sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
            view = out[sl]
            res = _try_notch_cut(view == lid, min_px=min_px, sol_max=sol_max, halves_min=halves_min)
            if res is None:
                continue
            _, b = res
            view[b] = next_id
            again.append((lid, sl))
            again.append((next_id, sl))
            next_id += 1
        queue = again
    return out


def label_grains(
    cmap: np.ndarray,
    valid: np.ndarray,
    unknown_code: int,
    *,
    min_grain_px: int = 100,
    split: bool = True,
    bridge_px: int = 4,
    h_min: float = 3.0,
    notch_solidity: float = 0.85,
    notch_halves: float = 0.88,
    floor_px: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grains = connected same-colour pixels (a colour change is a boundary;
    Unknown / black is a gap). Enclosed black is filled into the surrounding
    grain; standalone black is its own particle. With ``split`` the same-colour
    pieces are cut at thin bridges (``bridge_px``), at necks of the distance
    map (``h_min``), and between two deep notches when both halves are compact
    (``notch_solidity`` = only grains less compact than this are tried, 0 =
    off; ``notch_halves`` = smallest solidity of each half).

    ``min_grain_px`` drives the absorb / notch size rules. ``floor_px`` (default
    ``min_grain_px``) is the smallest region kept in the returned labels; pass a
    smaller value to keep sub-threshold grains for a method range.

    Returns ``(labels, labels_nosplit)`` as int32 HxW, each renumbered 1..n.
    """
    unk = int(unknown_code)
    floor = int(min_grain_px if floor_px is None else floor_px)
    colored = (cmap > 0) & (cmap != unk) & valid
    regions = _colour_regions(cmap, colored)
    nosplit = _relabel_min_area(_add_black(cmap, valid, unk, regions), floor)
    if not split:
        return nosplit, nosplit
    pieces = _break_bridges(colored, regions, int(bridge_px), int(min_grain_px))
    del regions, colored
    pieces = _absorb_small_pieces(pieces, int(min_grain_px))
    pieces = _relabel_min_area(_add_black(cmap, valid, unk, pieces), floor)
    labels = _split_hmax(pieces, float(h_min))
    del pieces
    labels = _absorb_small_pieces(labels, int(min_grain_px))
    labels = _relabel_min_area(labels, floor)
    if float(notch_solidity) > 0:
        labels = _notch_split(
            labels, min_px=int(min_grain_px), sol_max=float(notch_solidity), halves_min=float(notch_halves)
        )
        labels = _relabel_min_area(labels, floor)
    return labels, nosplit


def grain_table(
    labels: np.ndarray,
    cmap: np.ndarray,
    legend: Legend,
    *,
    unknown_frac: float = 0.95,
) -> pd.DataFrame:
    """
    One row per grain: class, area, shape, and ``frac_{name}`` columns.

    Class is the largest known color. Unknown only when Unknown pixels are at
    least ``unknown_frac`` of the grain, or there is no known color.
    """
    names = class_names(legend)
    n_cls = len(names)
    n = int(labels.max())
    cols = [
        "grain_id", "class", "class_frac", "area_px", "equiv_diam_px",
        "centroid_row", "centroid_col", "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1",
        "major_px", "minor_px", "solidity",
    ] + [f"frac_{nm}" for nm in names[1:]]
    if n == 0:
        return pd.DataFrame(columns=cols)
    pair = labels.astype(np.int64) * n_cls + cmap.astype(np.int64)
    comp = np.bincount(pair.ravel(), minlength=(n + 1) * n_cls).reshape(n + 1, n_cls)[1:]
    area = comp.sum(axis=1).astype(np.float64)
    area[area == 0] = np.nan
    frac = comp / area[:, None]
    unk = unknown_code(legend)
    scored = frac.copy()
    scored[:, 0] = -1.0  # never pick background
    scored[:, unk] = -1.0
    has_known = np.clip(scored, 0.0, None).sum(axis=1) > 0
    too_unknown = frac[:, unk] >= float(unknown_frac)
    dom = np.where(has_known & ~too_unknown, scored.argmax(axis=1), unk)
    props = regionprops_table(labels, properties=_PROPS)
    order = np.argsort(props["label"])
    df = pd.DataFrame(
        {
            "grain_id": props["label"][order].astype(int),
            "class": [names[int(k)] for k in dom[props["label"][order] - 1]],
            "class_frac": frac[props["label"][order] - 1, dom[props["label"][order] - 1]],
            "area_px": props["area"][order].astype(int),
            "equiv_diam_px": props["equivalent_diameter_area"][order],
            "centroid_row": props["centroid-0"][order],
            "centroid_col": props["centroid-1"][order],
            "bbox_r0": props["bbox-0"][order].astype(int),
            "bbox_c0": props["bbox-1"][order].astype(int),
            "bbox_r1": props["bbox-2"][order].astype(int),
            "bbox_c1": props["bbox-3"][order].astype(int),
            "major_px": props["axis_major_length"][order],
            "minor_px": props["axis_minor_length"][order],
            "solidity": props["solidity"][order],
        }
    )
    ids = df["grain_id"].to_numpy() - 1
    for k, nm in enumerate(names[1:], start=1):
        df[f"frac_{nm}"] = frac[ids, k]
    return df[cols]


def _round_coords(obj, ndigits: int):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), ndigits) for v in obj]
        return [_round_coords(o, ndigits) for o in obj]
    return obj


def grains_to_geojson(
    labels: np.ndarray,
    table: pd.DataFrame,
    path: PathLike,
    *,
    simplify_px: float = 0.0,
    ndigits: int = COORD_DECIMALS,
) -> Path:
    """
    Write one polygon per grain (pixel coords, x = col, y = row) with properties
    ``grain_id``, ``class``, ``area_px``.
    """
    path = Path(path)
    meta = table.set_index("grain_id")[["class", "area_px"]]
    lab = np.ascontiguousarray(labels.astype(np.int32))
    mask = lab > 0
    features = []
    for geom, val in rasterio_shapes(lab, mask=mask, transform=Affine.identity(), connectivity=4):
        gid = int(val)
        if gid not in meta.index:
            continue
        g = shape(geom)
        if simplify_px > 0:
            g = g.simplify(float(simplify_px), preserve_topology=True)
        if g.is_empty:
            continue
        m = mapping(g)
        m["coordinates"] = _round_coords(m["coordinates"], ndigits)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "grain_id": gid,
                    "class": str(meta.at[gid, "class"]),
                    "area_px": int(meta.at[gid, "area_px"]),
                },
                "geometry": m,
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, separators=(",", ":")))
    return path


def class_rgb(
    labels: np.ndarray,
    table: pd.DataFrame,
    legend: Legend,
    *,
    boundaries: bool = True,
    display_rgb: Optional[Dict[str, RGB]] = None,
) -> np.ndarray:
    """Paint each grain with its class color (white background, dark grain edges).

    ``display_rgb`` optionally overrides legend colors for the overlay only
    (e.g. a dark class that is hard to see on black).
    """
    colors: Dict[str, Tuple[int, int, int]] = {n: tuple(int(v) for v in c) for n, c in legend.items()}
    if display_rgb:
        colors.update({n: tuple(int(v) for v in c) for n, c in display_rgb.items()})
    n = int(labels.max())
    lut = np.full((n + 1, 3), 255, dtype=np.uint8)
    for gid, cls in zip(table["grain_id"].to_numpy(), table["class"].to_numpy()):
        lut[int(gid)] = colors.get(str(cls), UNLISTED_RGB)
    out = lut[labels]
    if boundaries and n > 0:
        edge = find_boundaries(labels, mode="inner", connectivity=1)
        out[edge] = _BOUNDARY_RGB
    return out
