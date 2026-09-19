"""
Stage 6: point-count style stats from the object table (area %, QFL, Folk & Ward).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.paths import stem_paths

_COLS = ("name", "n", "area_px", "area_pct", "count_pct", "note")
_PORE = ("pore", "holes", "void")
_CEMENT = ("cement",)
_MATRIX = ("matrix",)
_QUARTZ = ("quartz", "qtz")
_FELDSPAR = ("feldspar", "kfs", "plag")
_LITHIC = ("lithic", "rock fragment")
_LITHIC_RF = r"(?:^|[^a-z])rf(?:[^a-z]|$)"
_FW_PS = (5, 16, 25, 50, 75, 84, 95)


def folk_ward_percentiles(
    phi: np.ndarray,
    weights: np.ndarray,
    ps: Sequence[float] = _FW_PS,
) -> dict:
    """Area-weighted φ percentiles. ``φp`` is where cumulative area reaches p% of grain area."""
    phi = np.asarray(phi, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    n = min(int(phi.size), int(weights.size))
    phi = phi[:n]
    weights = weights[:n]
    out: dict = {}
    for p in ps:
        key = int(p) if float(p) == int(float(p)) else float(p)
        out[key] = np.nan
    ok = np.isfinite(phi) & np.isfinite(weights) & (weights > 0)
    phi = phi[ok]
    weights = weights[ok]
    if phi.size == 0:
        return out
    order = np.argsort(phi, kind="mergesort")
    phi = phi[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    total = float(cum[-1])
    if total <= 0:
        return out
    frac = np.concatenate(([0.0], cum / total))
    phi_c = np.concatenate(([phi[0]], phi))
    for p in ps:
        key = int(p) if float(p) == int(float(p)) else float(p)
        out[key] = float(np.interp(float(p) / 100.0, frac, phi_c))
    return out


def _row(
    name: str,
    n=np.nan,
    area_px=np.nan,
    area_pct=np.nan,
    count_pct=np.nan,
    note: str = "",
) -> dict:
    return {
        "name": name,
        "n": n,
        "area_px": area_px,
        "area_pct": area_pct,
        "count_pct": count_pct,
        "note": note,
    }


def _match(names: pd.Series, needles: Sequence[str]) -> pd.Series:
    n = names.fillna("").astype(str).str.lower()
    m = pd.Series(False, index=names.index)
    for needle in needles:
        m = m | n.str.contains(str(needle), regex=False)
    return m


def _pct(part: float, total: float):
    if total <= 0 or not np.isfinite(part):
        return np.nan
    return 100.0 * float(part) / float(total)


def _count_pct(n: int, n_total: int):
    if n_total <= 0:
        return np.nan
    return 100.0 * float(n) / float(n_total)


def _fill_names(df: pd.DataFrame, class_names: Optional[dict]) -> pd.Series:
    names_map: Dict[int, str] = {}
    if class_names:
        names_map = {int(k): str(v) for k, v in class_names.items()}
    if "class_id" in df.columns:
        mapped = pd.to_numeric(df["class_id"], errors="coerce").fillna(0).astype(int).map(
            lambda i: names_map.get(int(i), "")
        )
    else:
        mapped = pd.Series([""] * len(df), index=df.index)
    if "class_name" not in df.columns:
        return mapped.fillna("").astype(str)
    raw = df["class_name"]
    blank = raw.isna() | (raw.astype(str).str.strip() == "") | (raw.astype(str).str.lower() == "nan")
    return pd.Series(np.where(blank, mapped, raw.astype(str)), index=df.index)


def compute_object_stats(
    table: pd.DataFrame,
    class_names: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Per-class area rows, then summary rows (porosity, cement, matrix, IGV, QFL, Folk & Ward).

    Returns columns ``name, n, area_px, area_pct, count_pct, note``. Unused summary
    stats are NaN with a note. Folk & Ward numbers sit in ``area_pct``.
    """
    df = table.copy()
    rows = []
    if "area_px" in df.columns:
        area = pd.to_numeric(df["area_px"], errors="coerce").fillna(0.0)
    else:
        area = pd.Series(0.0, index=df.index)
    total_area = float(area[area > 0].sum())
    n_total = int((area > 0).sum())
    names = _fill_names(df, class_names)
    if "class_id" in df.columns:
        class_id = pd.to_numeric(df["class_id"], errors="coerce").fillna(0).astype(int)
    else:
        class_id = pd.Series(0, index=df.index, dtype=int)
    if "kind" in df.columns:
        kind = df["kind"].fillna("").astype(str).str.lower()
    else:
        kind = pd.Series("", index=df.index)

    classified = class_id > 0
    for cid in sorted(int(v) for v in class_id[classified].unique()):
        sel = classified & (class_id == cid)
        n = int((sel & (area > 0)).sum())
        a = float(area[sel].sum())
        label = ""
        vals = names[sel].astype(str)
        nonempty = vals[vals.str.strip() != ""]
        if len(nonempty):
            label = str(nonempty.iloc[0])
        if not label.strip():
            label = f"class_{cid}"
        note = "" if total_area > 0 else "total_area is 0"
        rows.append(
            _row(
                label,
                n=n,
                area_px=a,
                area_pct=_pct(a, total_area),
                count_pct=_count_pct(n, n_total),
                note=note,
            )
        )

    pore_m = _match(names, _PORE)
    cement_m = _match(names, _CEMENT)
    matrix_m = _match(names, _MATRIX)
    q_m = _match(names, _QUARTZ)
    f_m = _match(names, _FELDSPAR)
    l_m = _match(names, _LITHIC) | names.fillna("").astype(str).str.lower().str.contains(
        _LITHIC_RF, regex=True
    )
    leftover_pore_m = _match(names, ("pore",)) & (kind == "leftover")

    def _comp(mask: pd.Series, label: str, missing_note: str) -> dict:
        if not bool(mask.any()):
            return _row(label, note=missing_note)
        n = int((mask & (area > 0)).sum())
        a = float(area[mask].sum())
        note = "" if total_area > 0 else "total_area is 0"
        return _row(
            label,
            n=n,
            area_px=a,
            area_pct=_pct(a, total_area),
            count_pct=_count_pct(n, n_total),
            note=note,
        )

    por_row = _comp(pore_m, "porosity", "no pore/holes/void class")
    cem_row = _comp(cement_m, "cement", "no cement class")
    mat_row = _comp(matrix_m, "matrix", "no matrix class")
    rows.extend([por_row, cem_row, mat_row])

    pvig = np.nan
    lp_area = float(area[leftover_pore_m].sum())
    pore_area = float(area[pore_m].sum())
    if lp_area > 0:
        pvig = lp_area
    elif bool(pore_m.any()):
        pvig = pore_area

    cem_area = float(area[cement_m].sum()) if bool(cement_m.any()) else np.nan
    mat_area = float(area[matrix_m].sum()) if bool(matrix_m.any()) else np.nan
    if not np.isfinite(pvig) and not np.isfinite(cem_area) and not np.isfinite(mat_area):
        rows.append(_row("IGV", note="no pore, matrix, or cement class"))
    else:
        igv_area = float(np.nansum([pvig, cem_area, mat_area]))
        rows.append(
            _row(
                "IGV",
                area_px=igv_area,
                area_pct=_pct(igv_area, total_area),
                note="" if total_area > 0 else "total_area is 0",
            )
        )

    q_area = float(area[q_m].sum()) if bool(q_m.any()) else 0.0
    f_area = float(area[f_m].sum()) if bool(f_m.any()) else 0.0
    l_area = float(area[l_m].sum()) if bool(l_m.any()) else 0.0
    qfl = q_area + f_area + l_area
    has_qfl = bool(q_m.any() or f_m.any() or l_m.any())
    if (not has_qfl) or qfl <= 0:
        qfl_note = "Q+F+L is 0" if has_qfl else "no quartz/feldspar/lithic class"
        for lab in ("Q", "F", "L", "Q_norm", "F_norm", "L_norm"):
            rows.append(_row(lab, note=qfl_note))
    else:
        tot_note = "" if total_area > 0 else "total_area is 0"
        for lab, mask, a in (("Q", q_m, q_area), ("F", f_m, f_area), ("L", l_m, l_area)):
            n = int((mask & (area > 0)).sum())
            rows.append(
                _row(
                    lab,
                    n=n,
                    area_px=a,
                    area_pct=_pct(a, total_area),
                    count_pct=_count_pct(n, n_total),
                    note=tot_note,
                )
            )
        rows.append(_row("Q_norm", area_pct=100.0 * q_area / qfl))
        rows.append(_row("F_norm", area_pct=100.0 * f_area / qfl))
        rows.append(_row("L_norm", area_pct=100.0 * l_area / qfl))

    if "phi" not in df.columns:
        for lab in ("Mz", "sigma_I", "SkI", "KG"):
            rows.append(_row(lab, note="px_per_um not set"))
    else:
        fw_m = (kind == "grain") & ~pore_m & ~cement_m & ~matrix_m
        phi = pd.to_numeric(df.loc[fw_m, "phi"], errors="coerce").to_numpy(dtype=np.float64)
        w = area[fw_m].to_numpy(dtype=np.float64)
        n_fw = int(np.sum(np.isfinite(phi) & np.isfinite(w) & (w > 0)))
        a_fw = float(np.nansum(w[np.isfinite(w) & (w > 0) & np.isfinite(phi)]))
        pcts = folk_ward_percentiles(phi, w, ps=_FW_PS)
        p5, p16, p25, p50, p75, p84, p95 = (pcts[k] for k in _FW_PS)
        if n_fw <= 0 or not np.isfinite(p16):
            note = "no grain rows for Folk & Ward" if int(fw_m.sum()) == 0 else "no finite phi"
            for lab in ("Mz", "sigma_I", "SkI", "KG"):
                rows.append(_row(lab, n=n_fw, area_px=a_fw, note=note))
        else:
            d84_16 = p84 - p16
            d95_5 = p95 - p5
            d75_25 = p75 - p25
            mz = (p16 + p50 + p84) / 3.0
            sigma_i = (d84_16) / 4.0 + (d95_5) / 6.6
            if d84_16 != 0 and d95_5 != 0:
                ski = (p16 + p84 - 2.0 * p50) / (2.0 * d84_16) + (p5 + p95 - 2.0 * p50) / (2.0 * d95_5)
                ski_note = ""
            else:
                ski = np.nan
                ski_note = "phi84-phi16 or phi95-phi5 is 0"
            if d75_25 != 0:
                kg = (d95_5) / (2.44 * d75_25)
                kg_note = ""
            else:
                kg = np.nan
                kg_note = "phi75-phi25 is 0"
            rows.append(_row("Mz", n=n_fw, area_px=a_fw, area_pct=mz))
            rows.append(_row("sigma_I", n=n_fw, area_px=a_fw, area_pct=sigma_i))
            rows.append(_row("SkI", n=n_fw, area_px=a_fw, area_pct=ski, note=ski_note))
            rows.append(_row("KG", n=n_fw, area_px=a_fw, area_pct=kg, note=kg_note))

    return pd.DataFrame(rows, columns=list(_COLS))


def save_object_stats(stats: pd.DataFrame, image_path: PathLike) -> Path:
    """Write ``objects/{stem}_stats.csv``."""
    sp = stem_paths(image_path)
    sp.objects_dir.mkdir(parents=True, exist_ok=True)
    p = sp.stats_csv
    stats.to_csv(p, index=False)
    return p


def load_object_stats(image_path: PathLike) -> Optional[pd.DataFrame]:
    """Reload ``objects/{stem}_stats.csv``, or ``None`` if missing."""
    p = stem_paths(image_path).stats_csv
    if not p.exists():
        return None
    return pd.read_csv(p)
