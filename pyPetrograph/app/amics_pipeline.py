# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
AMICS workflow state for the browser app (``pyPetrograph.webapp``): the ``pyPetrograph_4_AMICSworkflow``
notebook, step by step, calling the same ``pyPetrograph.amics`` functions with
the same arguments. Qt-free (runs headless; the app calls it from a job thread).
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Log = Callable[[str], None]


# app-only file (not written by notebook 4): hand-named map colours, EXTRA_LEGEND
EXTRA_LEGEND_JSON = "extra_legend.json"
LEGEND_EDIT_JSON = "{sample}_legend_edit.json"  # app only: skip ticks + OCR origin of the legend rows
# Session files the app reloads (threshold / scale / grains / legend / extra legend) live in
# this hidden subfolder of the sample's export folder; figures and tables are kept in a temp
# cache until the user exports them. Export copies them byte for byte under notebook 4 names.
STATE_DIR = ".pypetrograph"
DATA_KEYS = ("legend_json", "threshold_json", "scale_json", "grains")   # session files, exported with the CSVs


@dataclass
class AmicsSettings:
    """Notebook ``Settings`` cell defaults."""

    fig_dpi: int = 150
    skip_names: List[str] = field(
        default_factory=lambda: ["Unknown", "Low_Counts", "Un_x_rayed", "Shadows", "Pores"]
    )
    color_tol: int = 10
    white_tol: int = 3
    auto_match_colors: bool = True
    match_max_dist: float = 25
    footer_row: Optional[int] = None
    thr_radius: int = 5
    thr_pct: float = 99
    scale_length: float = 500
    scale_unit: str = "µm"
    min_grain_px: float = 10
    dark_drop_frac: float = 0.5
    use_sam2: bool = True
    unmapped_frac: float = 0.5
    mixed_frac: float = 0.30
    overlay_alpha: float = 0.5
    knoll_alpha: float = 0.5
    knoll_width_px: Optional[int] = None
    min_grains_curve: int = 5


def shared_models_dir() -> Path:
    """One model cache for every output folder: ``$PYPETROGRAPH_MODELS`` or ``~/.cache/pyPetrograph/models``."""
    import os

    d = Path(os.environ.get("PYPETROGRAPH_MODELS") or Path.home() / ".cache" / "pyPetrograph" / "models")
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dirs() -> List[Path]:
    """Where the app looks for weights, in order (the notebooks' folders first, so nothing is fetched twice)."""
    from pyPetrograph.common.paths import get_output_dir, project_root

    out = []
    for d in (
        (get_output_dir() / "models") if get_output_dir() else None,
        project_root() / "outputs" / "models",
        project_root() / "models",
        shared_models_dir(),
    ):
        if d is not None and Path(d) not in out:
            out.append(Path(d))
    return out


def _rel(p) -> str:
    """Path as the notebook records it: relative to the project folder when inside it."""
    from pyPetrograph.common.paths import project_root

    for base in (project_root(), Path.cwd()):
        try:
            return str(Path(p).resolve().relative_to(Path(base).resolve()))
        except ValueError:
            pass
    return str(Path(p))


class AmicsSession:
    """One sample. Each method is one notebook step; attributes match notebook names."""

    def __init__(self, settings: Optional[AmicsSettings] = None, log: Optional[Log] = None) -> None:
        self.s = settings or AmicsSettings()
        import matplotlib.pyplot as plt

        # notebook 4 setup cell: figure sizes / tight bounding boxes depend on these
        plt.rcParams["figure.dpi"] = self.s.fig_dpi
        plt.rcParams["savefig.dpi"] = self.s.fig_dpi
        self.log: Log = log or print
        self.legend_file = self.bse_file = self.mineral_file = None
        self.sample: Optional[str] = None
        self.legend_auto: Dict[str, Tuple[int, int, int]] = {}
        self.legend_report: List[str] = []
        self.legend: Dict[str, Tuple[int, int, int]] = {}
        self.P: Dict[str, Path] = {}
        self.bse_gray = self.bse_rgba = self.bse_valid = self.bse_stretch = None
        self.footer: int = 0
        self.points: List[Tuple[float, float]] = []
        self.threshold: Optional[float] = None
        self.dark = None
        self.scale: Dict[str, Any] = {}
        self.grain_polys: List[Any] = []
        self.grains: Optional[pd.DataFrame] = None
        self.vote_legend: Dict[str, Tuple[int, int, int]] = {}
        self.extra_legend: Dict[str, Tuple[int, int, int]] = {}
        self.map_rgb = self.cmap = self.map_valid = None
        self.missing = None
        self.color_matches = None
        self.mapped = None
        self.counts = None
        self.knoll = None
        self.written: List[Path] = []
        self.S: Dict[str, Path] = {}   # session files (reload): <folder>/.pypetrograph/
        self.C: Dict[str, Path] = {}   # figures / tables before export: temp cache
        self._cache: Optional[Path] = None
        self._raw_map = None

    # -- where files go ----------------------------------------------------------
    def _set_paths(self) -> None:
        """``P`` = export names (notebook 4), ``S`` = session files, ``C`` = cache (same names)."""
        from pyPetrograph.amics import amics_paths

        self.P = amics_paths(self.sample, self.bse_file)
        sd = self.P["folder"] / STATE_DIR
        if self._cache is None or not self._cache.is_dir():
            self._cache = Path(tempfile.mkdtemp(prefix=f"pp_cache_{self.sample}_"))
        self.S = {k: sd / Path(v).name for k, v in self.P.items() if k != "folder"}
        self.S["folder"] = sd
        self.C = {k: self._cache / Path(v).name for k, v in self.P.items() if k != "folder"}
        self.C["folder"] = self._cache

    def _state(self, key: str) -> Path:
        """Session file path (folder created on first write)."""
        self.S["folder"].mkdir(parents=True, exist_ok=True)
        return self.S[key]

    def saved_path(self, key: str) -> Optional[Path]:
        """Session file if present, else the notebook's file in the export folder, else None."""
        for d in (self.S, self.P):
            if d.get(key) is not None and Path(d[key]).exists():
                return Path(d[key])
        return None

    # -- 1. files -------------------------------------------------------------
    def set_files(self, legend, bse, mineral, sample: Optional[str] = None) -> None:
        from pyPetrograph.amics import sample_name
        from pyPetrograph.amics.windows import save_last_files

        self.legend_file, self.bse_file, self.mineral_file = Path(legend), Path(bse), Path(mineral)
        for k, p in (("legend", legend), ("BSE", bse), ("mineral map", mineral)):
            if not Path(p).is_file():
                raise FileNotFoundError(f"{k} file not found: {p}")
        self.sample = sample or sample_name(self.bse_file)
        save_last_files({"legend": self.legend_file, "bse": self.bse_file, "mineral": self.mineral_file})

    def saved(self) -> Dict[str, Any]:  # noqa: C901
        """What notebook 4 / the app saved for this sample in the current output folder.

        ``threshold`` + ``points`` (threshold.json), ``scale`` (scale.json, else the
        grains GeoJSON header), ``legend`` (legend.json), ``extra_legend``
        (extra_legend.json, app only). Missing items are left out."""
        from pyPetrograph.amics import amics_paths

        if self.sample is None or self.bse_file is None:
            return {}
        if not self.P or self.P.get("folder") != amics_paths(self.sample, self.bse_file)["folder"]:
            self._set_paths()
        P = self.P
        out: Dict[str, Any] = {"folder": P["folder"]}

        def js(path):
            try:
                return json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        def first(key):
            return self.saved_path(key) or P[key]

        thr = js(first("threshold_json"))
        if isinstance(thr, dict) and thr.get("threshold") is not None:
            out["threshold"] = float(thr["threshold"])
            out["points"] = [tuple(p) for p in thr.get("points") or []]
        sc = js(first("scale_json"))
        if not (isinstance(sc, dict) and sc.get("um_per_px")):
            g = js(first("grains")) if first("grains").exists() else None
            sc = (g or {}).get("scale") if isinstance(g, dict) else None
        if isinstance(sc, dict) and sc.get("um_per_px"):
            out["scale"] = sc
        leg = js(first("legend_json"))
        if isinstance(leg, dict) and leg:
            out["legend"] = {str(k): tuple(int(c) for c in v) for k, v in leg.items()}
        le = self.S["folder"] / LEGEND_EDIT_JSON.format(sample=self.sample)
        ed = js(le) if le.exists() else None
        if isinstance(ed, dict):
            out["legend_edit"] = ed
        ex = self.S["folder"] / EXTRA_LEGEND_JSON
        extra = js(ex if ex.exists() else P["folder"] / EXTRA_LEGEND_JSON)
        if isinstance(extra, dict) and extra:
            out["extra_legend"] = {str(k): tuple(int(c) for c in v) for k, v in extra.items()}
        return out

    def save_legend_edit(self, legend, skip, ocr) -> Path:
        """App-only record of the Legend step: unticked names and the OCR rows the edits started from.
        (legend.json itself stays in the notebook's name -> [r, g, b] form.)"""
        if not self.P and self.sample and self.bse_file:
            self._set_paths()
        self.S["folder"].mkdir(parents=True, exist_ok=True)
        p = self.S["folder"] / LEGEND_EDIT_JSON.format(sample=self.sample)
        p.write_text(json.dumps({"legend": {n: [int(c) for c in v] for n, v in dict(legend).items()},
                                 "skip": list(skip),
                                 "ocr": [[n, [int(c) for c in v]] for n, v in list(ocr)]}, indent=1))
        return p

    def set_extra_legend(self, extra: Optional[Dict[str, Sequence[int]]]) -> None:
        """``EXTRA_LEGEND``: map colours with no legend swatch, named by hand."""
        self.extra_legend = {str(k): tuple(int(c) for c in v) for k, v in dict(extra or {}).items()}

    # -- 2. legend ------------------------------------------------------------
    def read_legend(self) -> Dict[str, Tuple[int, int, int]]:
        from pyPetrograph.amics import read_legend

        self.legend_auto, self.legend_report = read_legend(self.legend_file)
        self.legend = dict(self.legend_auto)
        return self.legend

    def set_legend(self, legend: Optional[Dict[str, Sequence[int]]]) -> None:
        """``LEGEND_EDIT`` (None keeps the OCR legend)."""
        src = legend if legend else self.legend_auto
        self.legend = {str(k): tuple(int(c) for c in v) for k, v in dict(src).items()}

    # -- 3. BSE + legend files --------------------------------------------------
    def prepare_bse(self) -> None:
        import matplotlib.pyplot as plt

        from pyPetrograph.amics import load_bse, plot_legend, save_figure

        self._set_paths()
        self._state("legend_json").write_text(
            json.dumps({name: [int(c) for c in rgb] for name, rgb in self.legend.items()}, indent=1),
            encoding="utf-8",
        )
        fig = plot_legend(self.legend, self.s.skip_names)
        save_figure(fig, self.C["legend_png"], self.s.fig_dpi)
        plt.close(fig)
        self.bse_gray, self.bse_rgba = load_bse(self.bse_file)
        self.set_footer(self.s.footer_row, log=False)
        self.log(f"sample {self.sample}; image {self.bse_gray.shape[1]} x {self.bse_gray.shape[0]} px; data rows end at {self.footer}")

    def set_footer(self, row: Optional[int] = None, log: bool = True) -> int:
        """``FOOTER_ROW`` (None = detected): first row of the scale-bar strip; BSE_VALID / BSE_STRETCH follow."""
        from pyPetrograph.amics import detect_footer, stretch, valid_mask

        self.s.footer_row = None if row is None else int(row)
        self.footer = int(row) if row is not None else int(detect_footer(self.bse_rgba))
        self.bse_valid = valid_mask(self.bse_rgba, self.footer)
        self.bse_stretch = stretch(self.bse_gray, self.bse_valid)
        if log:
            self.log(f"data rows end at {self.footer}")
        return self.footer

    def detected_footer(self) -> int:
        from pyPetrograph.amics import detect_footer

        return int(detect_footer(self.bse_rgba))

    def clear_saved(self, kind: str) -> List[Path]:
        """Delete saved definitions for this sample: ``threshold``, ``scale``, ``grains``, ``legend``, ``extra``."""
        names = {"threshold": ["threshold_json", "threshold_png"], "scale": ["scale_json"],
                 "grains": ["grains"], "legend": ["legend_json"]}
        if not self.P and self.sample and self.bse_file:
            self._set_paths()
        paths = [d[k] for d in (self.S, self.P) for k in names.get(kind, []) if k in d]
        if kind == "legend" and self.S.get("folder"):
            paths.append(self.S["folder"] / LEGEND_EDIT_JSON.format(sample=self.sample))
        if kind == "extra" and self.P.get("folder"):
            paths = [self.S["folder"] / EXTRA_LEGEND_JSON, self.P["folder"] / EXTRA_LEGEND_JSON]
        gone = []
        for p in paths:
            if Path(p).exists():
                Path(p).unlink()
                gone.append(Path(p))
        if gone:
            self.log("deleted " + ", ".join(p.name for p in gone))
        return gone

    # -- 4. threshold -----------------------------------------------------------
    def threshold_from(self, points: Sequence[Tuple[float, float]]) -> float:
        from pyPetrograph.amics import threshold_from_points

        return threshold_from_points(
            self.bse_stretch, list(points), self.bse_valid, radius=self.s.thr_radius, pct=self.s.thr_pct
        )

    def preview_dark(self, thr: float) -> np.ndarray:
        from pyPetrograph.amics import dark_mask

        return dark_mask(self.bse_stretch, float(thr), self.bse_valid)

    def set_threshold(self, value: Optional[float] = None, points: Sequence[Tuple[float, float]] = ()) -> float:
        """``BSE_THRESHOLD`` when ``value`` is given, else from the clicked ``points``."""
        import matplotlib.pyplot as plt

        from pyPetrograph.amics import plot_threshold_preview, save_figure

        if value is not None:
            self.threshold, self.points = float(value), []
        else:
            if not points:
                raise ValueError("Click at least one epoxy / hole pixel, or type a threshold.")
            self.points = [(float(x), float(y)) for x, y in points]
            self.threshold = self.threshold_from(self.points)
        self.dark = self.preview_dark(self.threshold)
        self._state("threshold_json").write_text(
            json.dumps({"threshold": self.threshold, "points": [[float(x), float(y)] for x, y in self.points]}, indent=1),
            encoding="utf-8",
        )
        fig = plot_threshold_preview(self.bse_stretch, self.dark, self.footer, self.points)
        save_figure(fig, self.C["threshold_png"], self.s.fig_dpi)
        plt.close(fig)
        self.log(f"threshold {self.threshold:.1f}; non-grain or outside the data {100 * float(self.dark.mean()):.1f}% of the image")
        return self.threshold

    # -- 5. scale ---------------------------------------------------------------
    def set_scale(self, um_per_px: Optional[float] = None, drawn: Optional[Dict[str, Any]] = None) -> None:
        """``UM_PER_PX`` (not saved, as in the notebook) or a drawn line (saved to scale.json)."""
        if um_per_px is not None:
            self.scale = {"um_per_px": float(um_per_px)}
        elif drawn:
            self.scale = dict(drawn)
            self._state("scale_json").write_text(json.dumps(self.scale, indent=1), encoding="utf-8")
        else:
            self.scale = {}

    @staticmethod
    def scale_from_line(p1, p2, length: float, unit: str) -> Dict[str, Any]:
        """Same dict as the notebook scale window."""
        import math

        from pyPetrograph.amics import UNIT_TO_UM

        px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if px == 0 or not (length > 0) or unit not in UNIT_TO_UM:
            raise ValueError("invalid scale line, length or unit")
        return {
            "p1": [p1[0], p1[1]], "p2": [p2[0], p2[1]], "px_length": px,
            "length": float(length), "unit": unit, "um_per_px": float(length) * UNIT_TO_UM[unit] / px,
        }

    def footer_ocr(self) -> Optional[float]:
        from pyPetrograph.amics import read_footer_pixelsize

        return read_footer_pixelsize(self.bse_rgba, self.footer)

    @property
    def um_per_px(self) -> Optional[float]:
        return self.scale.get("um_per_px")

    @property
    def size_unit(self) -> str:
        return "um" if self.um_per_px else "px"

    # -- 6. segmentation --------------------------------------------------------
    def _seg_inputs(self, tmp: Path) -> Tuple[Path, Path]:
        """Same RGB PNG + dark .npy that ``run_seg_bse`` hands its worker."""
        from PIL import Image

        st = np.asarray(self.bse_stretch)
        rgb = np.repeat(st[..., None], 3, axis=-1) if st.ndim == 2 else st[..., :3]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        png, npy = tmp / "bse.png", tmp / "dark.npy"
        Image.fromarray(rgb).save(png)
        np.save(npy, np.asarray(self.dark, dtype=bool))
        return png, npy

    def seg_image_png(self) -> Path:
        """Persistent copy of the SEG input (for SAM click-to-add)."""
        tmp = Path(tempfile.gettempdir()) / f"pp_app_{self.sample}"
        tmp.mkdir(exist_ok=True)
        return self._seg_inputs(tmp)[0]

    def weights(self, sam2: Optional[bool] = None) -> Dict[str, Path]:
        """Model files from the shared cache (see :func:`model_dirs`); download only when none has them."""
        from pyPetrograph.objects.qc import SEG_SAM_NAME, SEG_UNET_NAME, ensure_seg_weights

        sam2 = bool(self.s.use_sam2) if sam2 is None else bool(sam2)
        need = {"unet": SEG_UNET_NAME, **({"sam": SEG_SAM_NAME} if sam2 else {})}
        found = {}
        for key, name in need.items():
            for d in model_dirs():
                if (d / name).is_file():
                    found[key] = d / name
                    break
        if len(found) < len(need):
            got = ensure_seg_weights(sam2=sam2, models_dir=shared_models_dir())
            found = {k: found.get(k, got[k]) for k in need}
        return found

    def segment(self, backend, progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Any]:
        """``run_seg_bse(..., edit=False, min_area=MIN_GRAIN_PX / 2, drop_frac=DARK_DROP_FRAC)``.

        ``progress`` gets the server's progress dicts (stage, done, total, frac)."""
        from pyPetrograph.amics.seg_run import read_polygons_geojson

        w = self.weights()
        with tempfile.TemporaryDirectory(prefix="amics_seg_") as tmp:
            png, npy = self._seg_inputs(Path(tmp))
            kw = {"progress": progress} if progress is not None else {}
            backend.request(
                "segment", image=str(png), dark=str(npy), unet=str(w["unet"]),
                sam=str(w["sam"]) if (self.s.use_sam2 and "sam" in w) else "",
                min_area=float(self.s.min_grain_px / 2), drop_frac=float(self.s.dark_drop_frac),
                out=str(self._state("grains")), **kw,
            )
        self.grain_polys = read_polygons_geojson(self.S["grains"])
        return self.grain_polys

    def load_saved_grains(self) -> List[Any]:
        from pyPetrograph.amics import read_grains_geojson

        self.grain_polys, _t, _m = read_grains_geojson(self.saved_path("grains"))
        return self.grain_polys

    def save_edits(self, backend, polys: Sequence[Any]) -> List[Any]:
        """Edited grains go through the GrainPlot save path (``polygons_to_grains`` + ``save_grains``)."""
        from shapely.geometry import mapping

        from pyPetrograph.amics.seg_run import read_polygons_geojson

        with tempfile.TemporaryDirectory(prefix="amics_edit_") as tmp:
            src = Path(tmp) / "edited.geojson"
            src.write_text(json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {}, "geometry": mapping(p)} for p in polys]}), encoding="utf-8")
            png, _ = self._seg_inputs(Path(tmp))
            backend.request("normalize", src=str(src), image=str(png), out=str(self._state("grains")))
        self.grain_polys = read_polygons_geojson(self.S["grains"])
        return self.grain_polys

    def _meta(self) -> Dict[str, Any]:
        return {
            "sample": self.sample, "um_per_px": self.um_per_px, "scale": self.scale,
            "threshold": self.threshold, "source_image": _rel(self.bse_file),
        }

    def finalize_grains(self) -> pd.DataFrame:
        """Notebook steps 'grains' + 'sizes' + size curve."""
        import matplotlib.pyplot as plt

        from pyPetrograph.amics import add_size_metrics, measure_grains, plot_gsd, save_figure, write_grains_geojson

        s = self.s
        grains = measure_grains(self.grain_polys, um_per_px=self.um_per_px)
        keep = grains["area_px"].fillna(0) >= (s.min_grain_px / 2)
        self.grain_polys = [p for p, ok in zip(self.grain_polys, keep) if ok]
        grains = grains.loc[keep].reset_index(drop=True)
        grains["grain_id"] = np.arange(1, len(grains) + 1)
        write_grains_geojson(self._state("grains"), self.grain_polys, grains, self._meta())
        grains = add_size_metrics(grains, self.um_per_px)
        write_grains_geojson(self._state("grains"), self.grain_polys, grains, self._meta())
        self.grains = self._measured = grains
        big = grains["area_px"] >= s.min_grain_px
        fig = plot_gsd(grains.loc[big], unit=self.size_unit)
        save_figure(fig, self.C["gsd"], s.fig_dpi)
        plt.close(fig)
        self.log(f"kept {len(self.grain_polys)} grains with area >= {s.min_grain_px / 2:g} px; {int(big.sum())} >= {s.min_grain_px:g} px")
        return grains

    # -- 7. minerals (notebook section 3), vote, outputs (section 4) ---------------
    def _fig_to(self, fig, path, out: List[Path]) -> None:
        import matplotlib.pyplot as plt

        from pyPetrograph.amics import save_figure

        save_figure(fig, path, self.s.fig_dpi)
        plt.close(fig)
        out.append(Path(path))

    def missing_colors(self):
        """Cell 'Minerals': map colours the legend (alone) does not name. Nothing is written."""
        from pyPetrograph.amics import load_mineral_map

        kw = dict(footer_row=self.footer, color_tol=self.s.color_tol, white_tol=self.s.white_tol)
        map_rgb, _cmap, _valid, missing = load_mineral_map(self.mineral_file, self.legend, **kw)
        if map_rgb.shape[:2] != self.bse_gray.shape[:2]:
            raise ValueError(f"mineral map is {map_rgb.shape[1]}x{map_rgb.shape[0]}, BSE is {self.bse_gray.shape[1]}x{self.bse_gray.shape[0]}")
        self.missing_legend_only = missing
        return missing

    def minerals(self) -> List[Path]:
        """EXTRA_LEGEND + colour matching + the overlay figure."""
        from pyPetrograph.amics import load_mineral_map, match_map_colors, plot_overlay, recolor_from_matches

        s, P, F = self.s, self.P, self.footer
        out: List[Path] = []
        # EXTRA_LEGEND rows are saved only when there are some (app only), so a run
        # without them leaves exactly the notebook's files in the folder.
        extra_path = self.S["folder"] / EXTRA_LEGEND_JSON
        if self.extra_legend:
            self.S["folder"].mkdir(parents=True, exist_ok=True)
            extra_path.write_text(
                json.dumps({n: [int(c) for c in rgb] for n, rgb in self.extra_legend.items()}, indent=1),
                encoding="utf-8",
            )
        elif extra_path.exists():
            extra_path.unlink()
        vote_legend = {**self.legend, **self.extra_legend}
        kw = dict(footer_row=F, color_tol=s.color_tol, white_tol=s.white_tol)
        map_rgb, cmap, map_valid, missing = load_mineral_map(self.mineral_file, vote_legend, **kw)
        if map_rgb.shape[:2] != self.bse_gray.shape[:2]:
            raise ValueError(f"mineral map is {map_rgb.shape[1]}x{map_rgb.shape[0]}, BSE is {self.bse_gray.shape[1]}x{self.bse_gray.shape[0]}")
        self.color_matches = None
        self._raw_map = (map_rgb, map_valid, missing)
        if s.auto_match_colors and len(missing):
            self.color_matches = match_map_colors(missing, vote_legend, cmap, s.skip_names, max_dist=s.match_max_dist)
            recolor = recolor_from_matches(self.color_matches, vote_legend)
            if recolor:
                map_rgb, cmap, map_valid, missing = load_mineral_map(self.mineral_file, vote_legend, recolor=recolor, **kw)
        self.vote_legend, self.map_rgb, self.cmap, self.map_valid, self.missing = vote_legend, map_rgb, cmap, map_valid, missing
        st = self.bse_stretch[:F]
        self._fig_to(plot_overlay(st, map_rgb[:F], map_valid[:F], self.grain_polys, alpha=s.overlay_alpha), self.C["overlay"], out)
        return out

    def vote(self) -> List[Path]:
        """Mineral vote per grain, unmapped / mixed figures, counts, composition, size curves per mineral."""
        from pyPetrograph.amics import (
            MEASURE_COLUMNS, count_table, mineral_gsd_png, plot_composition, plot_gsd_by_mineral, plot_mixed,
            plot_unmapped, vote_grains, write_grains_geojson,
        )

        s, P, F = self.s, self.C, self.footer
        out: List[Path] = []
        fig_to = lambda fig, path: self._fig_to(fig, path, out)  # noqa: E731
        for old in P["folder"].glob(f"{self.sample}_gsd_*.png"):  # curves of an earlier vote
            old.unlink()
        vote_legend, map_rgb, cmap, map_valid = self.vote_legend, self.map_rgb, self.cmap, self.map_valid
        st = self.bse_stretch[:F]
        self.grains = getattr(self, "_measured", self.grains)  # vote again from the measured table
        votes = vote_grains(self.grain_polys, cmap, vote_legend, s.skip_names, mixed_frac=s.mixed_frac, unmapped_frac=s.unmapped_frac)
        base_cols = [c for c in MEASURE_COLUMNS if c in self.grains.columns]
        grains = pd.concat(
            [self.grains.reindex(columns=base_cols).reset_index(drop=True), votes.drop(columns=["grain_id"]).reset_index(drop=True)],
            axis=1,
        )
        self.grains = grains
        fig_to(plot_unmapped(st, map_rgb[:F], map_valid[:F], self.grain_polys, grains, alpha=s.overlay_alpha), P["unmapped"])
        write_grains_geojson(P["minerals"], self.grain_polys, grains,
                             {"sample": self.sample, "um_per_px": self.um_per_px, "source_image": _rel(self.mineral_file)})
        out.append(P["minerals"])
        is_mapped = (grains["mineral"] != "Unmapped").to_numpy()
        mapped = grains.loc[is_mapped].reset_index(drop=True)
        mapped_polys = [p for p, ok in zip(self.grain_polys, is_mapped) if ok]
        fig_to(plot_mixed(st, map_rgb[:F], map_valid[:F], mapped_polys, mapped, alpha=s.overlay_alpha), P["mixed"])
        counts = count_table(mapped, vote_legend, s.skip_names, sample=self.sample, min_grain_px=s.min_grain_px, unit=self.size_unit)
        keep = (mapped["area_px"] >= s.min_grain_px).to_numpy()
        fig_to(plot_composition([p for p, ok in zip(mapped_polys, keep) if ok], mapped.loc[keep].reset_index(drop=True),
                                vote_legend, counts, sample=self.sample, shape=st.shape[:2]), P["composition"])
        self.mapped, self.mapped_polys, self.counts = mapped, mapped_polys, counts
        out += self.replot_mineral_gsd()
        self.log(f"{len(mapped)} mapped grains ({int((~is_mapped).sum())} unmapped dropped)")
        return out

    def replot_mineral_gsd(self) -> List[Path]:
        """Size curve per mineral (cache)."""
        from pyPetrograph.amics import mineral_gsd_png, plot_gsd_by_mineral

        s, C = self.s, self.C
        out: List[Path] = []
        for old in C["folder"].glob(f"{self.sample}_gsd_*.png"):
            old.unlink()
        mapped = self.mapped
        big = mapped["area_px"] >= s.min_grain_px
        for mineral, fig in plot_gsd_by_mineral(mapped.loc[big], unit=self.size_unit, colors=self.vote_legend, min_grains=s.min_grains_curve):
            self._fig_to(fig, mineral_gsd_png(C["folder"], self.sample, mineral), out)
        return out

    def replot_gsd(self) -> List[Path]:
        """All-grain size curve + one per mineral, again (cache)."""
        from pyPetrograph.amics import plot_gsd

        out: List[Path] = []
        g = self._measured
        self._fig_to(plot_gsd(g.loc[g["area_px"] >= self.s.min_grain_px], unit=self.size_unit), self.C["gsd"], out)
        if self.mapped is not None:
            out += self.replot_mineral_gsd()
        return out

    def gsd_figures(self) -> Dict[str, Path]:
        """'All grains' + one entry per mineral with a curve (cache paths)."""
        from pyPetrograph.amics import mineral_gsd_png

        out = {"All grains": self.C["gsd"]} if self.C.get("gsd") and self.C["gsd"].exists() else {}
        for name in self.vote_legend:
            p = mineral_gsd_png(self.C["folder"], self.sample, name)
            if p.exists():
                out[name] = p
        return out

    def outputs(self) -> List[Path]:
        """Counts table, one CSV per mineral, knolling image (cache until exported)."""
        from pyPetrograph.amics import write_mineral_csvs

        s, P = self.s, self.C
        out: List[Path] = []
        for old in P["folder"].glob("*.csv"):  # tables of an earlier run
            old.unlink()
        mapped, mapped_polys, counts = self.mapped, self.mapped_polys, self.counts
        keep = (mapped["area_px"] >= s.min_grain_px).to_numpy()
        counted = mapped.loc[keep].reset_index(drop=True)
        counts.to_csv(P["counts"], index=False)
        out.append(P["counts"])
        out += [Path(p) for p in write_mineral_csvs(counted, self.sample, P["folder"])]
        out += self.replot_knolling()
        self.written = out
        return out

    def replot_knolling(self) -> List[Path]:
        from pyPetrograph.amics import make_knolling, save_knolling

        s, mapped = self.s, self.mapped
        keep = (mapped["area_px"] >= s.min_grain_px).to_numpy()
        counted = mapped.loc[keep].reset_index(drop=True)
        counted_polys = [p for p, ok in zip(self.mapped_polys, keep) if ok]
        self.knoll = make_knolling(self.bse_stretch, counted_polys, counted, self.vote_legend, alpha=s.knoll_alpha, width_px=s.knoll_width_px)
        save_knolling(self.knoll, self.C["knolling"], s.fig_dpi)
        return [self.C["knolling"]]

    def compute_downstream(self) -> List[Path]:
        """After 'Use this grain map': colours, vote, counts, figures, tables, knolling (cache only)."""
        out = self.minerals() + self.vote() + self.outputs()
        self.log(f"{len(self.mapped)} mapped grains; figures and tables ready (not exported yet)")
        return out

    # -- export (the only writes into the export folder) ----------------------------
    def _copy_out(self, srcs) -> List[Path]:
        import shutil

        self.P["folder"].mkdir(parents=True, exist_ok=True)
        out = []
        for src in srcs:
            if src is not None and Path(src).exists():
                dst = self.P["folder"] / Path(src).name
                shutil.copyfile(src, dst)
                out.append(dst)
        return out

    def export_data(self) -> List[Path]:
        """CSV data sheets + the notebook's JSON / GeoJSON files, under notebook 4 names."""
        C = self.C
        srcs = [C["counts"]] + sorted(p for p in C["folder"].glob("*.csv") if p != C["counts"]) + [C["minerals"]]
        srcs += [self.S[k] for k in DATA_KEYS if self.S.get(k) is not None and self.S[k].exists()]
        out = self._copy_out(srcs)
        self.log(f"exported {len(out)} data files to {self.P['folder']}")
        return out

    def export_figures(self) -> List[Path]:
        out = self._copy_out(sorted(self.C["folder"].glob("*.png")))
        self.log(f"exported {len(out)} figures to {self.P['folder']}")
        return out

    def results(self) -> List[Path]:
        """Sections 3 + 4 in one go, then both exports (headless runs)."""
        self.compute_downstream()
        return self.export_data() + self.export_figures()

    # -- minerals table -------------------------------------------------------------
    def mineral_table(self) -> List[Dict[str, Any]]:
        """One row per legend mineral (+ unlisted map colours) for the Minerals step.

        status ``ok`` (map colour identical to the legend), ``reassigned`` (near colour within
        COLOR_TOL or matched by AUTO_MATCH_COLORS), ``missing`` (legend mineral not in the map),
        ``unlisted`` (map colour no legend name could take: counts as Unknown)."""
        if self._raw_map is None:
            return []
        rgb, valid, missing = self._raw_map
        F = self.footer
        a = np.asarray(rgb[:F]).reshape(-1, 3).astype(np.int64)
        v = np.asarray(valid[:F]).ravel()
        packed = (a[v, 0] << 16) | (a[v, 1] << 8) | a[v, 2]
        u, cnt = np.unique(packed, return_counts=True)
        cols = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], axis=1).astype(float)
        white = np.sqrt(((cols - 255.0) ** 2).sum(1)) <= self.s.white_tol
        cols, cnt = cols[~white], cnt[~white]
        total = float(cnt.sum()) or 1.0
        names = list(self.vote_legend)
        lut = np.asarray([self.vote_legend[n] for n in names], dtype=float)
        d = np.sqrt(((cols[:, None, :] - lut[None]) ** 2).sum(-1)) if len(names) else np.zeros((len(cols), 0))
        near = d.argmin(1) if len(names) else np.zeros(len(cols), int)
        within = d.min(1) <= self.s.color_tol if len(names) else np.zeros(len(cols), bool)
        cm = self.color_matches
        rows = []
        for k, name in enumerate(names):
            leg = tuple(int(c) for c in self.vote_legend[name])
            idx = np.where(within & (near == k))[0]
            exact = [i for i in idx if tuple(int(c) for c in cols[i]) == leg]
            approx = [(tuple(int(c) for c in cols[i]), int(cnt[i])) for i in idx if i not in exact]
            if cm is not None and len(cm):
                for _, m in cm.loc[cm["assigned"] == name].iterrows():
                    approx.append(((int(m.map_red), int(m.map_green), int(m.map_blue)), int(m.pixels)))
            approx.sort(key=lambda t: -t[1])
            n_exact = int(sum(cnt[i] for i in exact))
            px = n_exact + sum(n for _, n in approx)
            status = "reassigned" if approx else ("ok" if n_exact else "missing")
            rows.append({"status": status, "mineral": name, "legend_rgb": leg,
                         "map_rgbs": ([leg] if n_exact else []) + [c for c, _ in approx],
                         "pixels": px, "pct": 100.0 * px / total, "extra": name in self.extra_legend,
                         "skipped": name in self.s.skip_names})
        if cm is not None and len(cm):
            left = cm.loc[cm["assigned"].astype(str) == ""]
            unl = [((int(m.map_red), int(m.map_green), int(m.map_blue)), int(m.pixels)) for _, m in left.iterrows()]
        else:
            unl = [((int(m.map_red), int(m.map_green), int(m.map_blue)), int(m.pixels)) for _, m in missing.iterrows()] \
                if missing is not None and len(missing) else []
        for c, n in unl:
            rows.append({"status": "unlisted", "mineral": "(unlisted map colour)", "legend_rgb": None,
                         "map_rgbs": [c], "pixels": n, "pct": 100.0 * n / total, "extra": False, "skipped": False})
        return rows
