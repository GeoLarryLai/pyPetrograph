"""Path helpers, stem companions, and package install check."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pyPetrograph.common.constants import (
    CACHE_SUBDIR,
    LABELS_SUBDIR,
    MODELS_SUBDIR,
    PathLike,
    PREDICTIONS_SUBDIR,
    REQUIRED_PACKAGES,
)

def project_root() -> Path:
    """Folder that contains the notebook + ``models/`` (parent of ``pyPetrograph/``)."""
    return Path(__file__).resolve().parent.parent.parent


def project_models_dir() -> Path:
    """Notebook-level ``models/`` for universal + per-image joblibs."""
    d = project_root() / MODELS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d

@dataclass
class StemPaths:
    """Companion paths under labels/ cache/ models/ predictions/ beside the image folder."""

    directory: Path  # image parent folder
    stem: str
    models_dir: Optional[Path] = None
    predictions_dir: Optional[Path] = None

    @property
    def labels_dir(self) -> Path:
        return self.directory / LABELS_SUBDIR

    @property
    def cache_dir(self) -> Path:
        return self.directory / CACHE_SUBDIR

    @property
    def labels(self) -> Path:
        """Label-summary figure export (`print_label_summary`)."""
        return self.labels_dir / f"{self.stem}_labels.png"

    @property
    def polygons(self) -> Path:
        return self.labels_dir / f"{self.stem}_polygons.geojson"

    @property
    def session(self) -> Path:
        return self.labels_dir / f"{self.stem}_session.json"

    @property
    def model(self) -> Path:
        base = self.models_dir if self.models_dir is not None else (self.directory / MODELS_SUBDIR)
        return base / f"{self.stem}_model.joblib"

    @property
    def rgb_cache(self) -> Path:
        return self.cache_dir / f"{self.stem}_rgb_cache.npy"

    @property
    def rgb_cache_meta(self) -> Path:
        return self.cache_dir / f"{self.stem}_rgb_cache.json"

    @property
    def pred(self) -> Path:
        """RGB class map (HSV colors) — human-viewable."""
        base = (
            self.predictions_dir
            if self.predictions_dir is not None
            else (self.directory / PREDICTIONS_SUBDIR)
        )
        return base / f"{self.stem}_pred.png"

    @property
    def pred_rgb(self) -> Path:
        """Original image with class-color overlay."""
        base = (
            self.predictions_dir
            if self.predictions_dir is not None
            else (self.directory / PREDICTIONS_SUBDIR)
        )
        return base / f"{self.stem}_pred_rgb.png"

    @property
    def fractions_csv(self) -> Path:
        base = (
            self.predictions_dir
            if self.predictions_dir is not None
            else (self.directory / PREDICTIONS_SUBDIR)
        )
        return base / f"{self.stem}_fractions.csv"

    @property
    def confidence(self) -> Path:
        base = (
            self.predictions_dir
            if self.predictions_dir is not None
            else (self.directory / PREDICTIONS_SUBDIR)
        )
        return base / f"{self.stem}_confidence.npy"

    def ensure_dirs(self) -> None:
        """Create only labels/ and predictions/ beside the image (not cache/ or models/)."""
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        (self.directory / PREDICTIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
        if self.models_dir is not None:
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        elif self.predictions_dir is not None:
            Path(self.predictions_dir).mkdir(parents=True, exist_ok=True)


def stem_paths(
    image_path: PathLike,
    stem: Optional[str] = None,
    models_dir: Optional[PathLike] = None,
    predictions_dir: Optional[PathLike] = None,
) -> StemPaths:
    """Build companion paths under subfolders of ``image_path``'s parent."""
    p = Path(image_path).resolve()
    return StemPaths(
        directory=p.parent,
        stem=stem or p.stem,
        models_dir=Path(models_dir) if models_dir else None,
        predictions_dir=Path(predictions_dir) if predictions_dir else None,
    )


def relpath_display(path: PathLike, base: Optional[PathLike] = None) -> str:
    """Short path relative to ``base`` (default: cwd / notebook folder)."""
    p = Path(path).resolve()
    root = Path(base).resolve() if base is not None else Path(".").resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return p.name


def _subdir_rel(path: PathLike, image_dir: Path) -> str:
    """e.g. labels/foo_labels.png relative to the image folder."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(image_dir.resolve()))
    except ValueError:
        return p.name


def cleanup_legacy_outputs(image_dir: PathLike) -> List[str]:
    """
    Delete obsolete beside-image companions in ``image_dir`` (not under subfolders).

    Removes: *_labels.png, *_polygons.geojson, *_session.json, *_rgb_cache.*,
    *_image_small.png, *_model.joblib, labels_image.png, polygons.geojson
    that sit directly in the folder (legacy layout).
    """
    d = Path(image_dir).resolve()
    if not d.is_dir():
        return []
    patterns = [
        "*_labels.png",
        "*_polygons.geojson",
        "*_session.json",
        "*_rgb_cache.npy",
        "*_rgb_cache.json",
        "*_image_small.png",
        "*_model.joblib",
        "labels_image.png",
        "polygons.geojson",
    ]
    removed: List[str] = []
    for pat in patterns:
        for f in d.glob(pat):
            if not f.is_file():
                continue
            # Skip if somehow inside a subdir (glob is only top-level)
            try:
                f.unlink()
                removed.append(f.name)
            except OSError:
                pass
    return removed


def delete_legacy_image_small(image_path: PathLike, stem: Optional[str] = None) -> None:
    """No-op kept for call sites; legacy flat companions cleaned by cleanup_legacy_outputs."""
    return


def check_and_install_packages(
    packages: Optional[Sequence[Tuple[str, str]]] = None,
    install: bool = True,
) -> Dict[str, str]:
    """
    Check packages needed by this toolbox; optionally pip-install missing ones
    into the current kernel env (Cursor/VS Code Jupyter: select the env first).

    Returns
    -------
    dict
        import_name → "ok" | "installed" | "failed: …"
    """
    pkgs = list(packages) if packages is not None else list(REQUIRED_PACKAGES)
    results: Dict[str, str] = {}
    missing: List[Tuple[str, str]] = []

    for import_name, pip_name in pkgs:
        try:
            __import__(import_name)
            results[import_name] = "ok"
        except ImportError:
            missing.append((import_name, pip_name))
            results[import_name] = "missing"

    # Print the required list once (status at check time).
    print("Package check")
    print("-" * 40)
    for name, status in results.items():
        print(f"  {name:16s}  {status}")
    print("-" * 40)

    if not missing:
        print("All required packages are available.")
        return results

    if not install:
        print("Pass install=True to pip-install missing packages into this env.")
        return results

    n_miss = len(missing)
    for i, (import_name, pip_name) in enumerate(missing, 1):
        print(f"Installing {pip_name} ({i}/{n_miss}) …")
        try:
            # Stream pip progress to the notebook (do not swallow stdout).
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            __import__(import_name)
            results[import_name] = "installed"
            print(f"  {import_name}: installed")
        except Exception as exc:  # noqa: BLE001
            results[import_name] = f"failed: {exc}"
            print(f"  {import_name}: failed")

    failed = [k for k, v in results.items() if v.startswith("failed") or v == "missing"]
    if failed:
        print("Still missing:", ", ".join(failed))
        print("Fix: activate your env and pip/conda install those packages.")
        if "PyQt5" in failed:
            print("  PyQt5 (label window): conda install -c conda-forge pyqt")
    else:
        print("Package check: done.")
    return results
