"""
Per-class grain counts with uncertainty.

Statistical: Poisson ``sqrt(n)`` on the count and a Wilson 95% interval on the
count share. Method: the no-split count and the counts with the minimum grain
size halved / doubled give ``n_lo`` .. ``n_hi``.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from pyPetrograph.mineral_map.legend import Legend, class_names

COUNT_COLS = (
    "image",
    "class",
    "n",
    "poisson_sd",
    "n_lo",
    "n_hi",
    "n_nosplit",
    "n_min_half",
    "n_min_double",
    "share_pct",
    "share_lo95",
    "share_hi95",
    "area_px",
    "area_pct",
)
_Z95 = 1.959963984540054


def wilson_interval(k: np.ndarray, n: int, z: float = _Z95):
    """Wilson score interval for ``k`` successes in ``n`` trials (fractions)."""
    k = np.asarray(k, dtype=np.float64)
    if n <= 0:
        nan = np.full(k.shape, np.nan)
        return nan, nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def _counts_by_class(table: pd.DataFrame, names: Sequence[str], min_px: float) -> np.ndarray:
    if len(table) == 0:
        return np.zeros(len(names), dtype=np.int64)
    sel = table[pd.to_numeric(table["area_px"], errors="coerce") >= float(min_px)]
    vc = sel["class"].value_counts()
    return np.asarray([int(vc.get(nm, 0)) for nm in names], dtype=np.int64)


def count_grains(
    table: pd.DataFrame,
    table_nosplit: pd.DataFrame,
    legend: Legend,
    image_name: str,
    *,
    min_grain_px: int = 100,
) -> pd.DataFrame:
    """
    Count grains per class for one image.

    Parameters
    ----------
    table : grain table (split), must contain grains down to ``min_grain_px / 2``
    table_nosplit : grain table of the unsplit particles
    legend : name → rgb (defines the class rows; far colors already sit in Unknown)
    image_name : written to the ``image`` column
    min_grain_px : main minimum grain size; the method range uses half and double

    Returns
    -------
    DataFrame with ``COUNT_COLS``; one row per class plus a final ``TOTAL`` row.
    Shares are over all counted grains (including Unknown).
    """
    names: List[str] = class_names(legend)[1:]  # drop background
    n = _counts_by_class(table, names, min_grain_px)
    n_half = _counts_by_class(table, names, min_grain_px / 2.0)
    n_double = _counts_by_class(table, names, min_grain_px * 2.0)
    n_nosplit = _counts_by_class(table_nosplit, names, min_grain_px)
    stack = np.vstack([n, n_half, n_double, n_nosplit])
    n_lo = stack.min(axis=0)
    n_hi = stack.max(axis=0)
    total = int(n.sum())
    lo, hi = wilson_interval(n, total)
    share = (n / total) if total > 0 else np.full(len(names), np.nan)

    main = table[pd.to_numeric(table["area_px"], errors="coerce") >= float(min_grain_px)]
    area = np.asarray(
        [float(pd.to_numeric(main.loc[main["class"] == nm, "area_px"], errors="coerce").sum()) for nm in names]
    )
    total_area = float(area.sum())
    area_pct = (100.0 * area / total_area) if total_area > 0 else np.full(len(names), np.nan)

    df = pd.DataFrame(
        {
            "image": image_name,
            "class": names,
            "n": n,
            "poisson_sd": np.sqrt(n.astype(np.float64)),
            "n_lo": n_lo,
            "n_hi": n_hi,
            "n_nosplit": n_nosplit,
            "n_min_half": n_half,
            "n_min_double": n_double,
            "share_pct": 100.0 * share,
            "share_lo95": 100.0 * lo,
            "share_hi95": 100.0 * hi,
            "area_px": area.astype(np.int64),
            "area_pct": area_pct,
        }
    )
    tot = pd.DataFrame(
        [
            {
                "image": image_name,
                "class": "TOTAL",
                "n": total,
                "poisson_sd": float(np.sqrt(total)),
                "n_lo": int(stack.sum(axis=1).min()),
                "n_hi": int(stack.sum(axis=1).max()),
                "n_nosplit": int(n_nosplit.sum()),
                "n_min_half": int(n_half.sum()),
                "n_min_double": int(n_double.sum()),
                "share_pct": 100.0 if total > 0 else np.nan,
                "share_lo95": np.nan,
                "share_hi95": np.nan,
                "area_px": int(total_area),
                "area_pct": 100.0 if total_area > 0 else np.nan,
            }
        ]
    )
    return pd.concat([df, tot], ignore_index=True)[list(COUNT_COLS)]


def pivot_counts(all_counts: pd.DataFrame, value: str = "n") -> pd.DataFrame:
    """Class × image table of one column (e.g. ``n`` or ``share_pct``)."""
    return all_counts.pivot_table(index="class", columns="image", values=value, aggfunc="first")


def format_counts(all_counts: pd.DataFrame) -> pd.DataFrame:
    """Class × image table of ``n ± poisson_sd [n_lo–n_hi]`` strings."""
    df = all_counts.copy()
    df["cell"] = [
        f"{int(n)} ± {sd:.0f} [{int(lo)}–{int(hi)}]"
        for n, sd, lo, hi in zip(df["n"], df["poisson_sd"], df["n_lo"], df["n_hi"])
    ]
    return df.pivot_table(index="class", columns="image", values="cell", aggfunc="first")
