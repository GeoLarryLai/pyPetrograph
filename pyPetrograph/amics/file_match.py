"""
Find the BSE, mineral map, and legend files that belong together.

Standard library only, so both the notebook side and ``window_worker.py`` (which must not
import ``pyPetrograph``) can use it as a sibling file.

Naming rule: ``{sample}_BSE_Image.tiff`` and ``{sample}_Mineral_Image.tiff`` share a sample
prefix (text before the first ``_``). A legend is any image with ``legend`` in its name; the
one whose name shares the longest prefix with the sample wins (``10-Mineral_Legend.tiff`` for 10B).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
KEYS = ("legend", "bse", "mineral")


def sample_prefix(path: PathLike) -> str:
    """Text before the first ``_`` of the file stem (``10B_BSE_Image.tiff`` -> ``10B``)."""
    return Path(path).stem.split("_", 1)[0]


def _images(folder: Path) -> List[Path]:
    try:
        return sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
    except OSError:
        return []


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.lower(), b.lower()):
        if x != y:
            break
        n += 1
    return n


def find_companions(path: PathLike, kind: str) -> Dict[str, Optional[Path]]:
    """
    Files that go with ``path`` (a ``bse`` or ``mineral`` file) from the same folder.

    Returns ``{"legend", "bse", "mineral"}``; ``kind`` itself and anything not found are None.
    A legend file gives no companions (its name does not carry the sample).
    """
    out: Dict[str, Optional[Path]] = {k: None for k in KEYS}
    if kind not in ("bse", "mineral"):
        return out
    path = Path(path)
    files = _images(path.parent)
    sample = sample_prefix(path).lower()

    def find(word: str, exclude: Optional[str] = None) -> Optional[Path]:
        for f in files:
            name = f.name.lower()
            if name.startswith(sample + "_") and word in name and (exclude is None or exclude not in name):
                return f
        return None

    out["bse"] = find("bse")
    out["mineral"] = find("mineral", exclude="legend")
    legends = [f for f in files if "legend" in f.name.lower()]
    best, best_score = None, 0
    for f in legends:
        score = _common_prefix_len(sample, f.name)
        if score > best_score:
            best, best_score = f, score
    if best is None and len(legends) == 1:
        best = legends[0]
    out["legend"] = best
    out[kind] = None
    return out


def complete(
    legend: Optional[PathLike] = None,
    bse: Optional[PathLike] = None,
    mineral: Optional[PathLike] = None,
) -> Tuple[Dict[str, Optional[Path]], List[str]]:
    """
    Fill the missing entries from the ones given (BSE first, then the map as the anchor).

    Returns ``(paths, auto_keys)``; ``auto_keys`` lists the entries that were filled in.
    """
    paths: Dict[str, Optional[Path]] = {
        "legend": Path(legend).expanduser() if legend else None,
        "bse": Path(bse).expanduser() if bse else None,
        "mineral": Path(mineral).expanduser() if mineral else None,
    }
    auto: List[str] = []
    for kind in ("bse", "mineral"):
        anchor = paths[kind]
        if anchor is None or not anchor.exists():
            continue
        for key, found in find_companions(anchor, kind).items():
            if paths[key] is None and found is not None:
                paths[key] = found
                auto.append(key)
        break
    return paths, auto
