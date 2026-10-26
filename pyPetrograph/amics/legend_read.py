"""
Read an AMICS legend image: swatch colors by row + mineral names by OCR (rapidocr).

Names that share one color are merged into ``"A / B"``. ``rapidocr_onnxruntime``
is imported lazily; without it the names fall back to ``swatch_{i}``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pyPetrograph.common.constants import PathLike
from pyPetrograph.mineral_map.legend import read_rgba

RGB = Tuple[int, int, int]
Legend = Dict[str, RGB]

KNOWN_TOKENS: Tuple[str, ...] = (
    "Unknown", "Counts", "Shadows", "Pores", "Garnet", "Spessartine", "Almandine",
    "Pyrope", "Quartz", "Zircon", "Rutile", "Ilmenite", "Hematite", "Magnetite",
    "Monazite", "Xenotime", "Biotite", "Muscovite", "Chlorite", "Chromite",
    "Orthoclase", "Sanidine", "Plagioclase", "Albite", "Anorthite", "Microcline",
    "Calcite", "Dolomite", "Apatite", "Epidote", "Tourmaline", "Topaz", "Illite",
    "Kaolinite", "Pyrite", "Titanite", "Sphene", "Barite", "Gypsum", "Magnesite",
    "Siderite", "Baddeleyite", "Manganite", "Pseudobrookite", "Carpholite",
    "Calderite", "Axinite", "Staurolite", "Kyanite", "Sillimanite", "Andalusite",
    "Amphibole", "Hornblende", "Pyroxene", "Olivine", "Oxides", "Iron",
    "Cassiterite", "Corundum", "Spinel", "Fluorite", "Halite",
)

_SCALE = 3
_PAD = 30
_ENGINE: Any = None
_ENGINE_FAILED = False
_LAST_FIXES: List[Tuple[str, str]] = []


def name_key(name: str) -> str:
    """Lower-case ``name`` and drop every non-letter, non-digit character."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _load_rgb(path: PathLike) -> np.ndarray:
    """RGB uint8 HxWx3; pixels with alpha == 0 become white."""
    rgb, alpha = read_rgba(path)
    rgb = np.array(rgb, dtype=np.uint8)
    if alpha is not None:
        rgb[alpha == 0] = 255
    return rgb


def _runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    """(start, end inclusive) of each run of True."""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask_1d) - 1))
    return out


def read_legend_rows(path: PathLike) -> List[Dict[str, Any]]:
    """
    One dict per legend entry, top to bottom:
    ``{"y0": int, "y1": int, "rgb": (r, g, b), "text_x0": int}``.
    """
    rgb = _load_rgb(path)
    nonwhite = ~np.all(rgb == 255, axis=2)
    cols = np.where(nonwhite.any(axis=0))[0]
    if cols.size == 0:
        return []
    x_left = int(cols[0])
    # Box corners are clipped at x_left, so also use the next column (its corners are filled).
    edge = nonwhite[:, x_left].copy()
    if x_left + 1 < nonwhite.shape[1]:
        edge |= nonwhite[:, x_left + 1]
    runs = [(a, b) for a, b in _runs(edge) if b - a + 1 >= 6]
    if not runs:
        return []

    a0, b0 = runs[0]
    mid = (a0 + b0) // 2
    x_right = x_left
    w = rgb.shape[1]
    while x_right + 1 < w and nonwhite[mid, x_right + 1]:
        x_right += 1
    text_x0 = x_right + 2

    rows: List[Dict[str, Any]] = []
    for y0, y1 in runs:
        xa, xb = x_left + 3, x_right - 3
        ya, yb = y0 + 3, y1 - 3
        if xb >= xa and yb >= ya:
            block = rgb[ya:yb + 1, xa:xb + 1].reshape(-1, 3)
        else:
            block = rgb[(y0 + y1) // 2, (x_left + x_right) // 2].reshape(1, 3)
        vals, counts = np.unique(block, axis=0, return_counts=True)
        color = vals[int(np.argmax(counts))]
        rows.append({
            "y0": int(y0),
            "y1": int(y1),
            "rgb": (int(color[0]), int(color[1]), int(color[2])),
            "text_x0": int(text_x0),
        })
    return rows


def _get_engine() -> Any:
    """Cached RapidOCR engine, or None when unavailable."""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    except Exception:  # ImportError or engine creation failure
        _ENGINE_FAILED = True
        return None
    return _ENGINE


def _fix_tokens(name: str) -> Tuple[str, bool]:
    """Repair single-letter OCR mistakes against ``KNOWN_TOKENS``."""
    known_lower = {k.lower() for k in KNOWN_TOKENS}
    parts = re.split(r"([^A-Za-z]+)", name)
    changed = False
    for i, tok in enumerate(parts):
        if i % 2 == 1 or len(tok) < 5 or tok.lower() in known_lower:
            continue
        low = tok.lower()
        cands = [
            k for k in KNOWN_TOKENS
            if len(k) == len(tok) and sum(a != b for a, b in zip(k.lower(), low)) == 1
        ]
        if len(cands) == 1:
            new = cands[0] if tok[0].isupper() else cands[0].lower()
            parts[i] = new
            changed = True
    return "".join(parts), changed


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".,;:").strip()


def _run_ocr(engine: Any, crop_rgb: np.ndarray, pad: int) -> List[Tuple[float, float, float, str]]:
    """OCR one crop (x3, padded). Returns ``(center_y, left_x, top_y, text)`` in crop pixels."""
    im = Image.fromarray(crop_rgb)
    im = im.resize((im.width * _SCALE, im.height * _SCALE), Image.LANCZOS)
    canvas = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), (255, 255, 255))
    canvas.paste(im, (pad, pad))
    result, _ = engine(np.asarray(canvas))
    out: List[Tuple[float, float, float, str]] = []
    for item in result or []:
        box, text = item[0], item[1]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        cy = ((min(ys) + max(ys)) / 2 - pad) / _SCALE
        lx = (min(xs) - pad) / _SCALE
        out.append((cy, lx, 0.0, str(text)))
    return out


def ocr_names(path: PathLike, rows: List[Dict[str, Any]]) -> List[str]:
    """One OCR name per row (``swatch_{i}`` when OCR finds nothing or rapidocr is missing)."""
    global _LAST_FIXES
    _LAST_FIXES = []
    n = len(rows)
    if n == 0:
        return []
    engine = _get_engine()
    if engine is None:
        print("rapidocr_onnxruntime is not installed; using swatch_1..N names")
        return [f"swatch_{i}" for i in range(1, n + 1)]

    rgb = _load_rgb(path)
    h, w = rgb.shape[:2]
    text_x0 = int(rows[0]["text_x0"])
    found = _run_ocr(engine, rgb[:, text_x0:w], _PAD)

    mids = [(r["y0"] + r["y1"]) / 2 for r in rows]
    pitches = np.diff(mids)
    pitch = float(np.median(pitches)) if len(pitches) else float(rows[0]["y1"] - rows[0]["y0"] + 1)

    per_row: List[List[Tuple[float, str]]] = [[] for _ in rows]
    for cy, lx, _, text in found:
        idx: Optional[int] = None
        for i, r in enumerate(rows):
            if r["y0"] <= cy <= r["y1"]:
                idx = i
                break
        if idx is None:
            dists = [abs(cy - m) for m in mids]
            j = int(np.argmin(dists))
            if dists[j] <= pitch / 2:
                idx = j
        if idx is not None:
            per_row[idx].append((lx, text))

    names: List[str] = []
    for i, r in enumerate(rows):
        texts = [t for _, t in sorted(per_row[i], key=lambda p: p[0])]
        name = _clean_text("_".join(t.strip() for t in texts))
        if not name:
            ya, yb = max(0, r["y0"] - 3), min(h, r["y1"] + 5)
            sub = _run_ocr(engine, rgb[ya:yb, text_x0:w], 20)
            texts = [t for _, _, _, t in sorted(sub, key=lambda p: p[1])]
            name = _clean_text("_".join(t.strip() for t in texts))
        if not name:
            names.append(f"swatch_{i + 1}")
            continue
        fixed, changed = _fix_tokens(name)
        if changed:
            _LAST_FIXES.append((name, fixed))
        names.append(fixed)
    return names


def read_legend(path: PathLike) -> Tuple[Legend, List[str]]:
    """
    ``({name: (r, g, b)}, report_lines)`` in row order; shared colors merged into
    ``"A / B"``; ``"Unknown": (0, 0, 0)`` appended if no Unknown / black entry exists.
    """
    global _LAST_FIXES
    rows = read_legend_rows(path)
    if not rows:
        return {"Unknown": (0, 0, 0)}, ["no legend swatches found"]
    names = ocr_names(path, rows)
    fixes = list(_LAST_FIXES)
    _LAST_FIXES = []

    groups: Dict[RGB, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(tuple(r["rgb"]), []).append(i)

    legend: Legend = {}
    merged_lines: List[str] = []
    for color, idxs in groups.items():
        uniq: List[str] = []
        for i in idxs:
            if names[i] not in uniq:
                uniq.append(names[i])
        key = " / ".join(uniq)
        if key in legend:
            k = 2
            while f"{key} ({k})" in legend:
                k += 1
            key = f"{key} ({k})"
        legend[key] = color
        if len(idxs) > 1:
            merged_lines.append(f"merged (same color {color}): {key}")

    if not any(name_key(k) == "unknown" for k in legend) and not any(c == (0, 0, 0) for c in legend.values()):
        legend["Unknown"] = (0, 0, 0)

    report = [f"{len(rows)} legend rows -> {len(legend)} classes"]
    report += [f"OCR fix: '{old}' -> '{new}'" for old, new in fixes]
    report += merged_lines
    for i, nm in enumerate(names, 1):
        if nm == f"swatch_{i}":
            report.append(f"row {i}: no OCR text; named swatch_{i}")
    report.append(
        "Check the names: OCR drops underscore suffixes like _3 and can misread letters. "
        "Edit LEGEND_EDIT if needed."
    )
    return legend, report


def legend_to_text(legend: Legend) -> str:
    """Dict literal, one entry per line, ready to paste into ``LEGEND_EDIT = {...}``."""
    if not legend:
        return "{}"
    lines = ["{"]
    for name, (r, g, b) in legend.items():
        lines.append(f"    {json.dumps(name, ensure_ascii=False)}: ({int(r)}, {int(g)}, {int(b)}),")
    lines.append("}")
    return "\n".join(lines)
