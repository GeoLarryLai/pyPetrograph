"""Session state for multi-image labeling / train / predict."""
from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

from pyPetrograph.common.constants import (
    CACHE_SUBDIR,
    DEFAULT_CLASSES,
    DEFAULT_FIGURE_DPI,
    DEFAULT_LBP_P,
    DEFAULT_LBP_R,
    DEFAULT_LOAD_TARGET_MP,
    DEFAULT_MEDIAN_SIZE,
    DEFAULT_METHOD,
    DEFAULT_N_JOBS,
    DEFAULT_PREVIEW_DPI,
    DEFAULT_SURE_PROBA,
    DEFAULT_TEXTURE_WINDOW,
    DEFAULT_WAND_SIMPLIFY,
    DEFAULT_WAND_THRESHOLD,
    DEFAULT_Y_TARGET,
    LABELS_SUBDIR,
    MODELS_SUBDIR,
    PathLike,
    PREDICTIONS_SUBDIR,
    UNIVERSAL_MODEL_NAME,
)
from pyPetrograph.common.image_io import (
    _to_rgb_uint8,
    ensure_rgb_cache,
    save_rgb_cache,
    save_rgb_png,
)
from pyPetrograph.common.paths import (
    StemPaths,
    _subdir_rel,
    cleanup_legacy_outputs,
    get_output_dir,
    project_models_dir,
    relpath_display,
    stem_paths,
)
from pyPetrograph.image_processing.features import default_feature_toggles, resolve_view_target_mp
from pyPetrograph.image_processing.fractions import compute_fractions, compute_fractions_uncertain
from pyPetrograph.image_processing.viz import (
    colorize_labels_hsv,
    colorize_prediction,
    overlay_class_map,
    try_load_legacy_label_ids,
)
from pyPetrograph.labeling_ml.model_io import load_model_bundle, save_model_bundle
from pyPetrograph.labeling_ml.polygons import (
    load_polygons_geojson,
    magic_wand_to_polygon,
    rasterize_polygons,
    save_polygons_geojson,
)
from pyPetrograph.labeling_ml.predict import (
    confidence_for_labels,
    predict_image_proba,
    smooth_prediction,
)
from pyPetrograph.labeling_ml.train import (
    _format_accuracy_pct,
    train_classifier,
    train_classifier_from_xy,
)
from pyPetrograph.image_processing.features import extract_training_table

class Session:
    """
    Project state for a queue of images: labels, polygons, models, params.

    Drive the notebook / Qt UI through this object.
    """

    def __init__(self) -> None:
        self.image_path: Optional[Path] = None
        self.stem: str = "untitled"
        self.image: Optional[np.ndarray] = None  # uint8 RGB
        self.labels: Optional[np.ndarray] = None  # uint8
        self.polygons: List[Polygon] = []
        self.class_ids: List[int] = []
        self.class_names: Dict[int, str] = dict(DEFAULT_CLASSES)
        self.default_class_names: Dict[int, str] = dict(DEFAULT_CLASSES)
        self.current_class: int = 1
        self.y_target: float = DEFAULT_Y_TARGET
        self.apply_norm: bool = True
        self.method: str = DEFAULT_METHOD
        self.n_jobs: int = DEFAULT_N_JOBS
        self.median_size: int = DEFAULT_MEDIAN_SIZE
        self.scale: float = 1.0
        self.max_side: Optional[int] = None
        self.load_target_mp: Optional[float] = DEFAULT_LOAD_TARGET_MP
        self.model_bundle: Optional[Dict[str, Any]] = None
        self.pred: Optional[np.ndarray] = None
        self.pred_smooth: Optional[np.ndarray] = None
        self.pred_rgb: Optional[np.ndarray] = None
        self.pred_confidence: Optional[np.ndarray] = None
        self.status: str = "Ready."
        from pyPetrograph.common.ui import UndoStack
        self._undo = UndoStack()
        self._listeners: List[Callable[[], None]] = []

        # Multi-image queue
        self.image_queue: List[Path] = []
        self.queue_index: int = -1
        self.models_dir: Optional[Path] = None
        self.predictions_dir: Optional[Path] = None
        self.wand_threshold: float = DEFAULT_WAND_THRESHOLD
        self.draw_mode: str = "wand"  # "wand" | "polygon"
        self.editing_active: bool = True
        # Browse start for “Select images…”. None or "." → folder where the notebook is run
        # (typically the folder that contains the .ipynb). Models/predictions are always
        # <image folder>/models and <image folder>/predictions — not set from the notebook.
        self.image_dir: Optional[Path] = None
        self.feature_toggles: Dict[str, bool] = default_feature_toggles()
        self.preview_target_mp: Optional[float] = None  # None → size from screen
        self.preview_dpi: int = DEFAULT_PREVIEW_DPI
        self.figure_dpi: int = DEFAULT_FIGURE_DPI
        self.texture_window: int = DEFAULT_TEXTURE_WINDOW
        self.lbp_p: int = DEFAULT_LBP_P
        self.lbp_r: float = DEFAULT_LBP_R

    def remember_settings_classes(self) -> None:
        """Snapshot current class_names as Settings defaults (kept for launch config)."""
        self.default_class_names = {
            int(k): str(v) for k, v in self.class_names.items()
        }

    def on_change(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"UI listener error: {exc}")

    def set_status(self, msg: str) -> None:
        self.status = msg
        self._notify()

    def paths(self) -> StemPaths:
        """Companion paths for the current image (beside that image's folder)."""
        if self.image_path is None:
            return StemPaths(
                directory=Path("."),
                stem=self.stem,
            )
        return self.paths_for(self.image_path, stem=self.stem)

    def work_directory(self) -> Path:
        """Folder for shared prints when the queue has a single parent.

        Mixed folders: current image parent (else first queue parent).
        Per-image outputs always use ``paths_for(image)`` (beside that image).
        """
        if self.image_queue:
            parents = {p.parent.resolve() for p in self.image_queue}
            if len(parents) == 1:
                return next(iter(parents))
            if self.image_path is not None:
                return self.image_path.parent.resolve()
            return self.image_queue[0].parent.resolve()
        if self.image_path is not None:
            return self.image_path.parent.resolve()
        return Path(".").resolve()

    def queue_parent_dirs(self) -> List[Path]:
        """Unique parent folders of queued images (sorted)."""
        if not self.image_queue:
            return [self.work_directory()]
        return sorted({p.parent.resolve() for p in self.image_queue})

    def ensure_output_dirs(self) -> Tuple[Path, Path]:
        """Create labels/ + predictions/; models/ via ``project_models_dir``."""
        self.models_dir = project_models_dir()
        out = get_output_dir()
        if out is not None:
            (out / LABELS_SUBDIR).mkdir(parents=True, exist_ok=True)
            (out / PREDICTIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
            self.predictions_dir = out / PREDICTIONS_SUBDIR
            return self.models_dir, self.predictions_dir
        bases: List[Path] = (
            self.queue_parent_dirs() if self.image_queue else [self.work_directory()]
        )
        for base in bases:
            (base / LABELS_SUBDIR).mkdir(parents=True, exist_ok=True)
            (base / PREDICTIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.work_directory() / PREDICTIONS_SUBDIR
        return self.models_dir, self.predictions_dir

    def paths_for(
        self, image_path: PathLike, stem: Optional[str] = None
    ) -> StemPaths:
        """Labels/predictions beside the image; models under notebook ``models/``."""
        self.models_dir = project_models_dir()
        return stem_paths(
            Path(image_path),
            stem=stem,
            models_dir=self.models_dir,
        )

    def model_path_for_stem(
        self, stem: str, image_path: Optional[PathLike] = None
    ) -> Path:
        """Per-image model under notebook-level ``models/{stem}_model.joblib``."""
        del image_path  # stem is the key; all models live in project models/
        self.ensure_output_dirs()
        assert self.models_dir is not None
        return self.models_dir / f"{stem}_model.joblib"

    def universal_model_path(self, image_path: Optional[PathLike] = None) -> Path:
        """``models/universal_model.joblib`` next to the notebook."""
        del image_path
        self.ensure_output_dirs()
        assert self.models_dir is not None
        return self.models_dir / UNIVERSAL_MODEL_NAME

    def save_universal_model(self, bundle: Dict[str, Any]) -> List[Path]:
        """Write universal model once under notebook-level ``models/``."""
        out = self.universal_model_path()
        save_model_bundle(out, bundle)
        return [out]

    def cleanup_unused_image_subfolders(self) -> List[str]:
        """Remove leftover image-folder ``cache/`` and ``models/`` (keep labels/)."""
        import shutil

        removed: List[str] = []
        for parent in self.queue_parent_dirs():
            for name in (CACHE_SUBDIR, MODELS_SUBDIR):
                d = parent / name
                if d.is_dir():
                    # Migrate any .joblib into project models/ before delete
                    if name == MODELS_SUBDIR:
                        dest = project_models_dir()
                        for f in d.glob("*.joblib"):
                            target = dest / f.name
                            if not target.exists():
                                try:
                                    shutil.copy2(f, target)
                                except OSError:
                                    pass
                    try:
                        shutil.rmtree(d)
                        removed.append(str(d))
                    except OSError:
                        pass
        return removed

    def _snapshot(self):
        from pyPetrograph.common.ui import UndoState
        return UndoState(
            polygons=[Polygon(p.exterior.coords) for p in self.polygons],
            class_ids=list(self.class_ids),
            labels=None if self.labels is None else self.labels.copy(),
            class_names=dict(self.class_names),
        )

    def push_undo(self) -> None:
        self._undo.push(self._snapshot())

    def undo(self) -> bool:
        state = self._undo.pop()
        if state is None:
            self.set_status("Nothing to undo.")
            return False
        self.polygons = state.polygons
        self.class_ids = state.class_ids
        self.labels = state.labels
        self.class_names = state.class_names
        self.set_status("Undid last step.")
        return True

    def set_image_queue(self, paths: Sequence[PathLike], open_first: bool = True) -> None:
        """Set ordered image queue from multi-select; optionally open the first."""
        resolved = [Path(p).resolve() for p in paths]
        self.image_queue = resolved
        self.queue_index = -1
        self.ensure_output_dirs()
        # Drop flat/legacy companions once per image folder so redo starts clean
        cleaned_dirs: set = set()
        for path in resolved:
            parent = path.parent.resolve()
            if parent in cleaned_dirs:
                continue
            cleaned_dirs.add(parent)
            removed = cleanup_legacy_outputs(parent)
            if removed:
                print(f"Removed legacy files in {relpath_display(parent)}:")
                for name in removed:
                    print(f"  - {name}")
        gone = self.cleanup_unused_image_subfolders()
        if gone:
            print("Removed unused image subfolders (cache/, models/ → notebook models/):")
            for p in gone:
                print(f"  - {relpath_display(p)}")
        self.cache_queue_images()
        if open_first and resolved:
            self.switch_image(0, autosave=False)
        else:
            self.set_status(f"Queued {len(resolved)} image(s).")

    def cache_queue_images(self) -> None:
        """Warm RAM RGB cache for every queued image (no disk writes)."""
        n = len(self.image_queue)
        for i, path in enumerate(self.image_queue, 1):
            ensure_rgb_cache(
                path,
                scale=self.scale,
                max_side=self.max_side,
                load_target_mp=self.load_target_mp,
            )
            print(f"RAM cache [{i}/{n}] {relpath_display(path)}")
        if n:
            self.set_status(f"RAM-cached {n} image(s) (until kernel restart).")

    def switch_image(
        self,
        index_or_path: Union[int, PathLike],
        autosave: bool = False,
        stem: Optional[str] = None,
    ) -> None:
        """Open another queue image. Disk write only if autosave=True (default off)."""
        if autosave and self.image is not None:
            self.save_all()

        if isinstance(index_or_path, int):
            idx = int(index_or_path)
            if idx < 0 or idx >= len(self.image_queue):
                raise IndexError(
                    f"Image index {idx} is out of range "
                    f"(queue has {len(self.image_queue)} image(s))."
                )
            path = self.image_queue[idx]
            self.queue_index = idx
        else:
            path = Path(index_or_path).resolve()
            if path in self.image_queue:
                self.queue_index = self.image_queue.index(path)
            else:
                self.image_queue.append(path)
                self.queue_index = len(self.image_queue) - 1

        self.open_image(path, scale=self.scale, stem=stem, auto_load=True)
        self.editing_active = True

    def view_target_mp(self) -> float:
        """MP for label overview / feature preview (screen-based if unset)."""
        return resolve_view_target_mp(self.preview_target_mp)

    def open_image(
        self,
        path: PathLike,
        scale: Optional[float] = None,
        max_side: Optional[int] = None,
        target_mp: Optional[float] = None,
        stem: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        """Load image (prefer rgb_cache), init blank labels, optionally auto-load companions."""
        path = Path(path).resolve()
        if scale is not None:
            self.scale = float(scale)
        if max_side is not None:
            self.max_side = max_side
        if target_mp is not None:
            self.load_target_mp = target_mp

        self.image = ensure_rgb_cache(
            path,
            scale=self.scale,
            max_side=self.max_side,
            load_target_mp=self.load_target_mp,
            stem=stem,
        )
        self.image_path = path
        self.stem = stem or path.stem
        if path not in self.image_queue:
            self.image_queue.append(path)
            self.queue_index = len(self.image_queue) - 1
        else:
            self.queue_index = self.image_queue.index(path)

        self.labels = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.polygons = []
        self.class_ids = []
        self.pred = None
        self.pred_smooth = None
        self.pred_rgb = None
        # Keep loaded model_bundle only if user loaded via Train cell; clear per-image
        self.model_bundle = None
        self._undo.clear()
        self.ensure_output_dirs()

        found = []
        if auto_load:
            found = self._auto_load_companions()
        msg = f"Loaded {relpath_display(path)} {self.image.shape[1]}×{self.image.shape[0]}"
        if found:
            msg += " | " + ", ".join(found)
        self.set_status(msg)

    def _auto_load_companions(self) -> List[str]:
        p = self.paths()
        p.ensure_dirs()
        found: List[str] = []

        if p.session.exists():
            self.load_session_file(p.session)
            found.append(f"{LABELS_SUBDIR}/{p.session.name}")
            p = self.paths()

        if p.polygons.exists():
            polys, ids = load_polygons_geojson(p.polygons)
            self.polygons = polys
            self.class_ids = ids
            if polys and self.image is not None:
                self.labels = rasterize_polygons(
                    self.polygons, self.class_ids, self.image.shape[:2]
                )
                found.append(f"{LABELS_SUBDIR}/{p.polygons.name}→labels")
            else:
                found.append(f"{LABELS_SUBDIR}/{p.polygons.name}")
        elif self.image is not None:
            # Legacy: uint8 class-id PNG only (RGB summary figures are ignored).
            lab = try_load_legacy_label_ids(p.labels, self.image.shape[:2])
            if lab is not None and np.any(lab > 0):
                self.labels = lab
                found.append(f"{LABELS_SUBDIR}/{p.labels.name} (legacy ids)")

        if p.model.exists():
            self.model_bundle = load_model_bundle(p.model)
            found.append(f"{MODELS_SUBDIR}/{p.model.name}")

        return found

    def save_all(self, *, allow_empty: bool = False) -> List[str]:
        """Save polygons + session under labels/ (no class-id PNG).

        By default, refuse to overwrite non-empty disk polygons with empty memory
        (common after ``launch_app``: notebook session is stale while the Qt
        child already saved real labels).
        """
        if self.image is None or self.labels is None:
            self.set_status("Nothing to save — open an image first.")
            return []
        p = self.paths()
        p.ensure_dirs()
        mem_empty = (not self.polygons) and (not np.any(self.labels > 0))
        if mem_empty and not allow_empty:
            disk_has = False
            if p.polygons.exists():
                try:
                    polys, _ = load_polygons_geojson(p.polygons)
                    disk_has = bool(polys)
                except Exception:
                    disk_has = p.polygons.stat().st_size > 50
            if disk_has:
                self.set_status(
                    "Skip save: in-memory labels empty; keeping disk files "
                    f"under {LABELS_SUBDIR}/ (re-open image or use Label Save)."
                )
                return []
        saved = []
        if self.polygons and not np.any(self.labels > 0):
            self.rebuild_labels_from_polygons()
        save_polygons_geojson(p.polygons, self.polygons, self.class_ids)
        saved.append(_subdir_rel(p.polygons, p.directory))
        if self.image_path is not None and self.image is not None:
            # Keep RAM warm; do not write disk cache/
            save_rgb_cache(
                self.image_path,
                self.image,
                scale=self.scale,
                max_side=self.max_side,
                load_target_mp=self.load_target_mp,
                stem=self.stem,
            )
        self.save_session_file(p.session)
        saved.append(_subdir_rel(p.session, p.directory))
        self.set_status("Saved: " + ", ".join(saved))
        return saved

    def reload_current_from_disk(self) -> None:
        """Re-load labels/polygons/session for the current image from disk."""
        if self.image_path is None:
            return
        path = self.image_path
        idx = self.queue_index
        self.open_image(path, scale=self.scale, stem=self.stem, auto_load=True)
        if 0 <= idx < len(self.image_queue):
            self.queue_index = idx

    def load_labels_only(self) -> None:
        """Reload polygons from disk into memory (pushes undo; no write)."""
        p = self.paths()
        if not p.polygons.exists():
            # Legacy uint8 class-id PNG only
            lab = (
                try_load_legacy_label_ids(p.labels, self.labels.shape)
                if self.labels is not None
                else None
            )
            if lab is None:
                self.set_status(f"No polygons file under {LABELS_SUBDIR}/.")
                return
            self.push_undo()
            self.labels = lab
            self.polygons = []
            self.class_ids = []
            self.set_status(
                f"Reloaded legacy {LABELS_SUBDIR}/{p.labels.name} (Undo restores prior scene)"
            )
            return
        self.push_undo()
        self.polygons, self.class_ids = load_polygons_geojson(p.polygons)
        if self.image is not None:
            self.labels = rasterize_polygons(
                self.polygons, self.class_ids, self.image.shape[:2]
            )
        self.set_status(
            f"Reloaded {LABELS_SUBDIR}/{p.polygons.name} (Undo restores prior scene)"
        )

    def rebuild_labels_from_polygons(self) -> None:
        if self.image is None:
            return
        self.labels = rasterize_polygons(
            self.polygons, self.class_ids, self.image.shape[:2]
        )

    def add_polygon(
        self, verts: Sequence[Tuple[float, float]], class_id: Optional[int] = None
    ) -> None:
        """Add one training polygon for the current (or given) class."""
        if len(verts) < 3:
            return
        cid = int(class_id if class_id is not None else self.current_class)
        self.push_undo()
        poly = Polygon([(float(x), float(y)) for x, y in verts])
        if not poly.is_valid or poly.is_empty:
            self._undo.pop()
            self.set_status("Ignored invalid polygon.")
            return
        self.polygons.append(poly)
        self.class_ids.append(cid)
        if self.labels is None and self.image is not None:
            self.labels = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if self.labels is not None:
            burn = rasterize_polygons([poly], [cid], self.labels.shape)
            self.labels[burn > 0] = cid
        self.set_status(
            f"Added polygon for class {cid} ({self.class_names.get(cid, '?')})."
        )

    def add_wand_at(self, row: int, col: int) -> bool:
        """Magic-wand click: grow similar color → polygon for current class."""
        if self.image is None:
            self.set_status("Open an image first.")
            return False
        verts = magic_wand_to_polygon(
            self.image,
            int(row),
            int(col),
            threshold=self.wand_threshold,
            simplify_tol=DEFAULT_WAND_SIMPLIFY,
        )
        if verts is None:
            self.set_status(
                f"Wand found no region at ({col}, {row}). "
                f"Try a higher threshold (now {self.wand_threshold})."
            )
            return False
        self.add_polygon(verts, class_id=self.current_class)
        return True

    def restart_labels(self) -> None:
        """Clear current class labels in memory only (no disk write)."""
        if self.image is None:
            self.set_status("Open an image first.")
            return
        cid = int(self.current_class)
        self.push_undo()
        keep = [(p, c) for p, c in zip(self.polygons, self.class_ids) if int(c) != cid]
        self.polygons = [p for p, _ in keep]
        self.class_ids = [c for _, c in keep]
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.uint8).copy()
            self.labels[self.labels == cid] = 0
        name = self.class_names.get(cid, "?")
        self.set_status(f"Cleared class {cid} ({name}) from scene (not saved).")

    def set_class_name(self, class_id: int, name: str) -> None:
        self.push_undo()
        self.class_names[int(class_id)] = name.strip() or f"class_{class_id}"
        self.set_status(f"Renamed class {class_id} → {self.class_names[int(class_id)]}")

    def add_class(self, name: str) -> int:
        self.push_undo()
        new_id = (max(self.class_names.keys()) + 1) if self.class_names else 1
        self.class_names[new_id] = name.strip() or f"class_{new_id}"
        self.current_class = new_id
        self.set_status(f"Added class {new_id}: {self.class_names[new_id]}")
        return new_id

    def remove_class(self, class_id: int) -> None:
        class_id = int(class_id)
        if class_id not in self.class_names:
            self.set_status(f"Class {class_id} not found.")
            return
        self.push_undo()
        keep = [(p, c) for p, c in zip(self.polygons, self.class_ids) if c != class_id]
        self.polygons = [p for p, _ in keep]
        self.class_ids = [c for _, c in keep]
        del self.class_names[class_id]
        self.rebuild_labels_from_polygons()
        if self.current_class == class_id and self.class_names:
            self.current_class = sorted(self.class_names.keys())[0]
        self.set_status(f"Removed class {class_id}.")

    def has_training_labels(self) -> bool:
        if self.polygons:
            return True
        if self.labels is not None and np.any(self.labels > 0):
            return True
        return False

    def label_stats(self) -> Dict[str, Any]:
        classes_present: List[Dict[str, Any]] = []
        n_pix = 0
        if self.labels is not None:
            flat = self.labels.ravel()
            n_pix = int(np.sum(flat > 0))
            for cid in sorted(int(c) for c in np.unique(flat) if c > 0):
                classes_present.append(
                    {
                        "id": cid,
                        "name": self.class_names.get(cid, f"class_{cid}"),
                        "count": int(np.sum(flat == cid)),
                    }
                )
        return {
            "stem": self.stem,
            "path": str(self.image_path) if self.image_path else None,
            "n_polygons": len(self.polygons),
            "n_labeled_pixels": n_pix,
            "classes_present": classes_present,
        }

    def save_session_file(self, path: Optional[PathLike] = None) -> Path:
        """Persist only class id→name map (Settings knobs stay in the notebook cell)."""
        path = Path(path) if path else self.paths().session
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "current_class": int(self.current_class),
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def load_session_file(self, path: PathLike) -> None:
        """Restore class names only; never overwrite Settings-cell parameters."""
        data = json.loads(Path(path).read_text())
        if data.get("class_names"):
            self.class_names = {
                int(k): str(v) for k, v in data["class_names"].items()
            }
        if "current_class" in data:
            cid = int(data["current_class"])
            if cid in self.class_names:
                self.current_class = cid
            elif self.class_names:
                self.current_class = min(self.class_names.keys())

    def train(self) -> Dict[str, Any]:
        if self.image is None or self.labels is None:
            raise RuntimeError("Open an image and create labels first.")
        bundle = train_classifier(
            self.image,
            self.labels,
            method=self.method,
            y_target=self.y_target,
            n_jobs=self.n_jobs,
            apply_norm=self.apply_norm,
            toggles=self.feature_toggles,
            texture_window=self.texture_window,
            lbp_p=self.lbp_p,
            lbp_r=self.lbp_r,
            polygons=self.polygons,
            class_names=self.class_names,
        )
        self.model_bundle = bundle
        self.set_status(
            f"Trained {bundle['method']} | accuracy={_format_accuracy_pct(bundle['accuracy'])} | "
            f"feats={bundle.get('feature_names')}"
        )
        return bundle

    def predict(self) -> np.ndarray:
        if self.image is None or self.model_bundle is None:
            raise RuntimeError("Need an image and a trained/loaded model.")
        out = predict_image_proba(
            self.image, self.model_bundle, polygons=self.polygons
        )
        self.pred = out["labels"]
        self.pred_smooth = smooth_prediction(self.pred, size=self.median_size)
        self.pred_confidence = confidence_for_labels(
            self.pred_smooth, out["proba"], out["class_ids"]
        )
        self.pred_confidence[self.pred_smooth == 0] = 0.0
        palette = self.model_bundle.get("color_palette", {0: [255, 255, 255]})
        self.pred_rgb = colorize_prediction(self.pred_smooth, palette)
        self.set_status("Prediction complete.")
        return self.pred_smooth

    def fractions(self) -> Dict[int, Dict[str, Any]]:
        if self.pred_smooth is None:
            raise RuntimeError("Run predict first.")
        return compute_fractions(self.pred_smooth, self.class_names)

    def load_model_from_path(self, path: PathLike) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Pick an existing .joblib model, or switch Train mode to Retrain."
            )
        if path.suffix.lower() != ".joblib":
            raise ValueError(
                f"Expected a .joblib model file, got: {path.name}\n"
                "Use the file picker filtered to trained models (.joblib)."
            )
        bundle = load_model_bundle(path)
        self.model_bundle = bundle
        self.set_status(f"Loaded model: {path}")
        return bundle

    def _load_labels_for_path(
        self, path: Path
    ) -> Tuple[np.ndarray, np.ndarray, str, List[Polygon], List[int]]:
        """Return (image, labels, stem, polygons, class_ids) for a queue path."""
        img = ensure_rgb_cache(
            path,
            scale=self.scale,
            max_side=self.max_side,
            load_target_mp=self.load_target_mp,
        )
        stem = path.stem
        sp = self.paths_for(path, stem=stem)
        labels = np.zeros(img.shape[:2], dtype=np.uint8)
        polys: List[Polygon] = []
        ids: List[int] = []
        if sp.polygons.exists():
            polys, ids = load_polygons_geojson(sp.polygons)
        if polys:
            labels = rasterize_polygons(polys, ids, img.shape[:2])
        else:
            lab = try_load_legacy_label_ids(sp.labels, labels.shape)
            if lab is not None:
                labels = lab
        return img, labels, stem, polys, ids

    def train_queue(self, scope: str = "per_image") -> Dict[str, Any]:
        """
        Retrain models for the queue.

        scope:
          - "per_image": one model per labeled image → models/{stem}_model.joblib
          - "universal": pool all labeled pixels → models/universal_model.joblib
        Images with no labels/polygons are skipped.
        """
        if not self.image_queue:
            raise RuntimeError(
                "No images in the queue.\n"
                "Select images with select_images(session) first."
            )
        scope = scope.lower().strip()
        if scope not in {"per_image", "universal"}:
            raise ValueError('scope must be "per_image" or "universal".')

        self.ensure_output_dirs()
        if self.image is not None:
            self.save_all()

        results: Dict[str, Any] = {
            "scope": scope,
            "trained": [],
            "skipped": [],
            "models_dir": str(self.models_dir),
        }

        if scope == "per_image":
            for path in self.image_queue:
                img, labels, stem, polys, _ids = self._load_labels_for_path(path)
                if not np.any(labels > 0):
                    results["skipped"].append(path.name)
                    continue
                bundle = train_classifier(
                    img,
                    labels,
                    method=self.method,
                    y_target=self.y_target,
                    n_jobs=self.n_jobs,
                    apply_norm=self.apply_norm,
                    toggles=self.feature_toggles,
                    texture_window=self.texture_window,
                    lbp_p=self.lbp_p,
                    lbp_r=self.lbp_r,
                    polygons=polys,
                    class_names=self.class_names,
                )
                out = self.model_path_for_stem(stem, image_path=path)
                save_model_bundle(out, bundle)
                results["trained"].append(
                    {
                        "image": path.name,
                        "stem": stem,
                        "path": str(out),
                        "accuracy": bundle["accuracy"],
                        "report": bundle.get("report"),
                        "cv_folds": bundle.get("cv_folds"),
                        "n_pixels": int(np.sum(labels > 0)),
                        "features": bundle.get("feature_names"),
                    }
                )
            if not results["trained"]:
                raise RuntimeError(
                    "No labeled images to train.\n"
                    "Draw polygons or use the magic wand on at least one image, "
                    "then run Train again."
                )
            # Keep last trained in memory if it matches current
            if results["trained"]:
                last = results["trained"][-1]
                self.model_bundle = load_model_bundle(last["path"])
        else:
            Xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            feat_names: Optional[List[str]] = None
            palette: Dict[int, List[int]] = {0: [255, 255, 255]}
            for path in self.image_queue:
                img, labels, stem, polys, _ids = self._load_labels_for_path(path)
                if not np.any(labels > 0):
                    results["skipped"].append(path.name)
                    continue
                X, y, names = extract_training_table(
                    img,
                    labels,
                    y_target=self.y_target,
                    apply_norm=self.apply_norm,
                    toggles=self.feature_toggles,
                    texture_window=self.texture_window,
                    lbp_p=self.lbp_p,
                    lbp_r=self.lbp_r,
                    polygons=polys,
                )
                if feat_names is None:
                    feat_names = names
                elif names != feat_names:
                    raise RuntimeError(
                        f"Feature names mismatch for {path.name}: {names} vs {feat_names}"
                    )
                Xs.append(X)
                ys.append(y)
                img_u8 = _to_rgb_uint8(img)
                flat_u8 = img_u8.reshape(-1, 3)
                lab = labels.reshape(-1)
                for cid in np.unique(lab[lab > 0]):
                    pix = flat_u8[lab == cid]
                    palette[int(cid)] = [int(round(x)) for x in pix.mean(axis=0)]
                results["trained"].append(
                    {"image": path.name, "stem": stem, "n_pixels": int(len(y))}
                )
            if not Xs:
                raise RuntimeError(
                    "No labeled images to train a universal model.\n"
                    "Label at least one image, then run Train again."
                )
            X_all = np.concatenate(Xs, axis=0)
            y_all = np.concatenate(ys, axis=0)
            bundle = train_classifier_from_xy(
                X_all,
                y_all,
                method=self.method,
                y_target=self.y_target,
                n_jobs=self.n_jobs,
                apply_norm=self.apply_norm,
                color_palette=palette,
                feature_names=feat_names,
                toggles=self.feature_toggles,
                texture_window=self.texture_window,
                lbp_p=self.lbp_p,
                lbp_r=self.lbp_r,
                class_names=self.class_names,
            )
            out_paths = self.save_universal_model(bundle)
            out = out_paths[0]
            self.model_bundle = bundle
            results["universal_path"] = str(out)
            results["universal_paths"] = [str(p) for p in out_paths]
            results["accuracy"] = bundle["accuracy"]
            results["report"] = bundle.get("report")
            results["cv_folds"] = bundle.get("cv_folds")
            results["n_pixels"] = int(len(y_all))
            results["features"] = feat_names

        self.set_status(
            f"Train done ({scope}): {len(results['trained'])} used, "
            f"{len(results['skipped'])} skipped."
        )
        return results

    def predict_queue(self, model_source: str = "universal") -> Dict[str, Any]:
        """
        Predict every queued image. Writes under that image's ``predictions/``.

        model_source:
          - "universal": use models/universal_model.joblib (searched per folder)
          - "per_image": use models/{stem}_model.joblib for each image

        Missing model → RuntimeError (stop); plain message says what to do.
        """
        if not self.image_queue:
            raise RuntimeError(
                "No images in the queue.\n"
                "Select images with select_images(session) first."
            )
        model_source = model_source.lower().strip()
        if model_source not in {"universal", "per_image"}:
            raise ValueError('model_source must be "universal" or "per_image".')

        self.ensure_output_dirs()

        if model_source == "universal":
            uni = self.universal_model_path()
            if not uni.exists():
                raise RuntimeError(
                    "Universal model not found.\n"
                    f"Looked for: {relpath_display(uni)}\n"
                    "What to do: run the Train cell with Retrain + universal, "
                    "or Load a .joblib model into notebook-level models/."
                )
            universal_bundle = load_model_bundle(uni)
        else:
            universal_bundle = None
            uni = None

        results: Dict[str, Any] = {
            "model_source": model_source,
            "models_dir": str(project_models_dir()),
            "predictions_dirs": [
                str(p / PREDICTIONS_SUBDIR) for p in self.queue_parent_dirs()
            ],
            "images": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for path in self.image_queue:
            img, _labels, stem, polys, _ids = self._load_labels_for_path(path)
            if model_source == "universal":
                bundle = universal_bundle
                model_used = str(uni)
            else:
                mpath = self.model_path_for_stem(stem, image_path=path)
                if not mpath.exists():
                    raise RuntimeError(
                        f"Per-image model missing for '{path.name}'.\n"
                        f"Looked for: {relpath_display(mpath)}\n"
                        "What to do: run Train with Retrain + per_image for this "
                        "image, switch Predict to universal, or Load a model."
                    )
                model_used = str(mpath)
                bundle = load_model_bundle(model_used)

            out = predict_image_proba(img, bundle, polygons=polys)
            pred = out["labels"]
            pred_s = smooth_prediction(pred, size=self.median_size)
            conf = confidence_for_labels(pred_s, out["proba"], out["class_ids"])
            conf[pred_s == 0] = 0.0
            pred_ids = np.asarray(pred_s, dtype=np.uint8)
            pred_vis = colorize_labels_hsv(pred_ids)
            pred_overlay = overlay_class_map(img, pred_ids)
            fr = compute_fractions_uncertain(
                pred_ids,
                class_names=self.class_names,
                proba=out["proba"],
                class_ids=out["class_ids"],
                confidence=conf,
                sure_threshold=DEFAULT_SURE_PROBA,
            )

            sp = self.paths_for(path, stem=stem)
            sp.ensure_dirs()
            save_rgb_png(sp.pred, pred_vis)
            save_rgb_png(sp.pred_rgb, pred_overlay)
            np.save(sp.confidence, conf.astype(np.float32))
            with sp.fractions_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "class_id",
                        "name",
                        "count",
                        "area_pct",
                        "area_pct_low",
                        "area_pct_high",
                        "area_pct_plus",
                        "area_pct_minus",
                        "sure_threshold",
                        "model_used",
                    ]
                )
                for cid, info in sorted(fr.items()):
                    writer.writerow(
                        [
                            int(cid),
                            info["name"],
                            int(info["count"]),
                            round(float(info["area_pct"]), 2),
                            round(float(info["area_pct_low"]), 2),
                            round(float(info["area_pct_high"]), 2),
                            round(float(info["area_pct_plus"]), 2),
                            round(float(info["area_pct_minus"]), 2),
                            float(info["sure_threshold"]),
                            model_used,
                        ]
                    )

            entry = {
                "image": relpath_display(path),
                "stem": stem,
                "model_used": model_used,
                "pred": str(sp.pred),
                "pred_rgb": str(sp.pred_rgb),
                "confidence": str(sp.confidence),
                "fractions_csv": str(sp.fractions_csv),
                "output_dir": str(sp.directory / PREDICTIONS_SUBDIR),
                "fractions": {
                    str(cid): {
                        "name": info["name"],
                        "count": int(info["count"]),
                        "area_pct": round(float(info["area_pct"]), 2),
                        "area_pct_low": round(float(info["area_pct_low"]), 2),
                        "area_pct_high": round(float(info["area_pct_high"]), 2),
                        "area_pct_plus": round(float(info["area_pct_plus"]), 2),
                        "area_pct_minus": round(float(info["area_pct_minus"]), 2),
                    }
                    for cid, info in sorted(fr.items())
                },
            }
            results["images"].append(entry)

            if self.image_path is not None and path.resolve() == self.image_path.resolve():
                self.pred = pred
                self.pred_smooth = pred_ids
                self.pred_rgb = pred_overlay
                self.pred_confidence = conf
                self.model_bundle = bundle

        self.set_status(
            f"Predicted {len(results['images'])} image(s) → each image's {PREDICTIONS_SUBDIR}/"
        )
        return results
