"""Sample name, file-safe mineral names, and output paths for the AMICS workflow."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.paths import get_output_dir

# key -> file name suffix; every file is ``{sample}_{suffix}`` in the sample folder
PATH_SUFFIXES: Dict[str, str] = {
    "legend_json": "legend.json",
    "legend_png": "legend.png",
    "threshold_json": "threshold.json",
    "threshold_png": "threshold.png",
    "scale_json": "scale.json",
    "grains": "grains.geojson",
    "gsd": "gsd.png",
    "overlay": "overlay.png",
    "mixed": "mixed.png",
    "unmapped": "unmapped.png",
    "composition": "composition.png",
    "minerals": "minerals.geojson",
    "counts": "counts.csv",
    "knolling": "knolling.png",
}


def sample_name(path: PathLike) -> str:
    """Text before the first ``_`` of the file stem (``Con-1_BSE_Image.tiff`` -> ``Con-1``)."""
    return Path(path).stem.split("_", 1)[0]


def safe_name(name: str) -> str:
    """File-safe mineral name (``"Carpholite? / Zircon / Baddeleyite"`` -> ``Carpholite-Zircon-Baddeleyite``)."""
    text = name.replace(" / ", "-")
    text = text.replace("?", "")
    text = text.replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9_-]", "", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip("-")
    return text or "unnamed"


def amics_paths(sample: str, image_path: PathLike) -> Dict[str, Path]:
    """
    Output paths for one sample; creates the folder.

    Folder = ``(get_output_dir() or Path(image_path).parent) / "amics" / sample``.
    Returns ``{"folder": folder, key: folder / f"{sample}_{suffix}"}`` for every
    key in ``PATH_SUFFIXES``.
    """
    base = get_output_dir() or Path(image_path).parent
    folder = base / "amics" / str(sample)
    folder.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {"folder": folder}
    for key, suffix in PATH_SUFFIXES.items():
        paths[key] = folder / f"{sample}_{suffix}"
    return paths


def mineral_csv(folder: PathLike, sample: str, mineral: str) -> Path:
    """``folder / f"{sample}_{safe_name(mineral)}.csv"``."""
    return Path(folder) / f"{sample}_{safe_name(mineral)}.csv"


def mineral_gsd_png(folder: PathLike, sample: str, mineral: str) -> Path:
    """``folder / f"{sample}_gsd_{safe_name(mineral)}.png"``."""
    return Path(folder) / f"{sample}_gsd_{safe_name(mineral)}.png"
