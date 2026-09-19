"""Path helpers, stem companions, and package install check."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pyPetrograph.common.constants import (
    CACHE_SUBDIR,
    CHANNELS_SUBDIR,
    GRAINS_GEOJSON_SUFFIX,
    GRAINS_MASK_SUFFIX,
    LABELS_SUBDIR,
    MODELS_SUBDIR,
    OBJECTS_SUBDIR,
    PathLike,
    PREDICTIONS_SUBDIR,
    REQUIRED_PACKAGES,
    SAM2_PACKAGES,
    SEG_PACKAGES,
)

_OUTPUT_DIR: Optional[Path] = None


def set_output_dir(path: Optional[PathLike]) -> Optional[Path]:
    """
    Optional folder for labels / predictions / models / align / channels /
    objects / mineralmap. ``None`` restores the default (beside each image;
    models at the project root). Notebooks call this once in Settings.
    """
    global _OUTPUT_DIR
    if path is None:
        _OUTPUT_DIR = None
        return None
    d = Path(path).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    _OUTPUT_DIR = d
    return _OUTPUT_DIR


def get_output_dir() -> Optional[Path]:
    """Current ``set_output_dir`` folder, or ``None`` (default beside-image layout)."""
    return _OUTPUT_DIR


def project_root() -> Path:
    """Folder that contains the notebook + ``models/`` (parent of ``pyPetrograph/``)."""
    return Path(__file__).resolve().parent.parent.parent


def project_models_dir() -> Path:
    """``models/`` under ``get_output_dir()`` if set, else the project root."""
    base = get_output_dir() or project_root()
    d = base / MODELS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d

@dataclass
class StemPaths:
    """
    Companion paths under labels/ cache/ models/ predictions/ channels/ objects/
    beside the image folder.

    ``channels/`` holds the stage-2 physics channel stack (npz + json);
    ``objects/`` holds the object table, labels, classes and stats (stages 4–6).
    """

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
    def grains(self) -> Path:
        return self.labels_dir / f"{self.stem}{GRAINS_GEOJSON_SUFFIX}"

    @property
    def grains_mask(self) -> Path:
        return self.labels_dir / f"{self.stem}{GRAINS_MASK_SUFFIX}"

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

    # --- stage 2: channels/ ---

    @property
    def channels_dir(self) -> Path:
        return self.directory / CHANNELS_SUBDIR

    @property
    def channels(self) -> Path:
        """Channel stack (``stack`` + ``valid``) as npz."""
        return self.channels_dir / f"{self.stem}_channels.npz"

    @property
    def channels_meta(self) -> Path:
        """Channel names / kinds / channel_set_id as json."""
        return self.channels_dir / f"{self.stem}_channels.json"

    # --- stages 4–6: objects/ ---

    @property
    def objects_dir(self) -> Path:
        return self.directory / OBJECTS_SUBDIR

    @property
    def objects_csv(self) -> Path:
        return self.objects_dir / f"{self.stem}_objects.csv"

    @property
    def objects_mask(self) -> Path:
        return self.objects_dir / f"{self.stem}_objects_mask.png"

    @property
    def object_labels(self) -> Path:
        return self.objects_dir / f"{self.stem}_object_labels.json"

    @property
    def classes_csv(self) -> Path:
        return self.objects_dir / f"{self.stem}_classes.csv"

    @property
    def classes_rgb(self) -> Path:
        return self.objects_dir / f"{self.stem}_classes_rgb.png"

    @property
    def stats_csv(self) -> Path:
        return self.objects_dir / f"{self.stem}_stats.csv"

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
    """
    Companion paths under subfolders of ``image_path``'s parent, or under
    ``get_output_dir()`` when that is set.
    """
    p = Path(image_path).resolve()
    base = get_output_dir() or p.parent
    return StemPaths(
        directory=base,
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


def _spec_present(import_name: str) -> bool:
    """True if the module can be found. Does not import it (safe for TensorFlow)."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _print_package_status(title: str, results: Dict[str, str]) -> None:
    print(title)
    print("-" * 40)
    for name, status in results.items():
        print(f"  {name:20s}  {status}")
    print("-" * 40)


def _install_missing(
    missing: List[Tuple[str, str]],
    results: Dict[str, str],
    *,
    import_after: bool,
) -> Dict[str, str]:
    n_miss = len(missing)
    for i, (import_name, pip_name) in enumerate(missing, 1):
        print(f"Installing {pip_name} ({i}/{n_miss}) …")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            if import_after:
                __import__(import_name)
            elif not _spec_present(import_name):
                raise ImportError(f"{import_name} still missing after pip")
            results[import_name] = "installed"
            print(f"  {import_name}: installed")
        except Exception as exc:  # noqa: BLE001
            results[import_name] = f"failed: {exc}"
            print(f"  {import_name}: failed")
    return results


def check_and_install_packages(
    packages: Optional[Sequence[Tuple[str, str]]] = None,
    install: bool = True,
    *,
    include_seg: bool = False,
    include_sam2: bool = False,
) -> Dict[str, str]:
    """
    Check packages needed by this toolbox; optionally pip-install missing ones
    into the current kernel env (Cursor/VS Code Jupyter: select the env first).

    Kernel-safe extras: ``include_seg`` / ``include_sam2`` use ``find_spec``
    only — they never ``import tensorflow`` in this process.

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

    _print_package_status("Package check", results)

    if missing and install:
        results = _install_missing(missing, results, import_after=True)
    elif missing:
        print("Pass install=True to pip-install missing packages into this env.")

    extra: List[Tuple[str, str]] = []
    if include_seg:
        extra.extend(SEG_PACKAGES)
    if include_sam2:
        extra.extend(SAM2_PACKAGES)
    if extra:
        extra_results: Dict[str, str] = {}
        extra_missing: List[Tuple[str, str]] = []
        for import_name, pip_name in extra:
            if _spec_present(import_name):
                extra_results[import_name] = "ok"
            else:
                extra_missing.append((import_name, pip_name))
                extra_results[import_name] = "missing"
        _print_package_status(
            "SEG / SAM2 check (not imported in this kernel)", extra_results
        )
        if extra_missing and install:
            extra_results = _install_missing(
                extra_missing, extra_results, import_after=False
            )
        elif extra_missing:
            print("Or: pip install 'pyPetrograph[seg]' / 'pyPetrograph[all]'")
        results.update(extra_results)

    failed = [k for k, v in results.items() if str(v).startswith("failed") or v == "missing"]
    if failed:
        print("Still missing:", ", ".join(failed))
        print("Fix: conda env create -f environment.yml")
        print("  or: pip install 'pyPetrograph[all]'")
        if "PyQt5" in failed:
            print("  PyQt5 (label window): conda install -c conda-forge pyqt")
    elif not missing:
        print("All required packages are available.")
    else:
        print("Package check: done.")
    return results
