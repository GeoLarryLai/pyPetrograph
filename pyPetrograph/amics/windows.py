"""
Kernel-side launchers for AMICS pop-ups.

Every window runs ``window_worker.py`` in a subprocess so the kernel never imports Qt.
With no screen (Colab, SSH, Linux without a display) the launchers fall back to the
Settings values and say which one to type.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyPetrograph.amics.file_match import complete
from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.ui import _pick_files_native

IMAGE_FILTER = "Images (*.tif *.tiff *.TIF *.TIFF *.png *.PNG *.jpg *.JPG *.jpeg *.JPEG);;All files (*)"
UNIT_TO_UM: Dict[str, float] = {"µm": 1.0, "um": 1.0, "mm": 1000.0, "nm": 0.001}

_WORKER = Path(__file__).resolve().parent / "window_worker.py"


class WindowUnavailable(RuntimeError):
    """The pop-up process could not open a window (no display, missing PyQt5, crash)."""


def has_screen() -> bool:
    """False on Colab or on Linux with no DISPLAY / WAYLAND_DISPLAY; True on Windows and macOS."""
    if "google.colab" in sys.modules:
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def pick_file(caption: str, start_dir: PathLike = ".", name_filter: str = IMAGE_FILTER) -> Optional[Path]:
    """One-file pop-up (``_pick_files_native(..., multi=False)``); None if cancelled."""
    paths = _pick_files_native(str(start_dir), multi=False, caption=caption, name_filter=name_filter)
    return Path(paths[0]) if paths else None


def _run_worker(
    mode: str,
    image: Optional[np.ndarray] = None,
    title: str = "",
    extra: Optional[Dict[str, Any]] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Run ``window_worker.py`` in a subprocess (blocks until the window closes); return its result dict.

    ``arrays`` are saved as .npy and passed by path (config key = array name). Raises
    ``WindowUnavailable`` if the process fails or writes no result.
    """
    tmp = Path(tempfile.mkdtemp(prefix="amics_win_"))
    try:
        out_path = tmp / "result.json"
        cfg_path = tmp / "config.json"
        cfg: Dict[str, Any] = {"mode": mode, "title": title, "out": str(out_path), **(extra or {})}
        all_arrays = dict(arrays or {})
        if image is not None:
            all_arrays["image"] = image
        for name, arr in all_arrays.items():
            p = tmp / f"{name}.npy"
            np.save(p, np.asarray(arr))
            cfg[name] = str(p)
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["MPLBACKEND"] = "QtAgg"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, str(_WORKER), "--config", str(cfg_path)],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").splitlines()[-8:])
            raise WindowUnavailable(f"AMICS window failed (exit {proc.returncode}):\n{tail}")
        if not out_path.exists():
            raise WindowUnavailable("AMICS window closed without a result")
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_FILE_KEYS = ("legend", "bse", "mineral")


def last_files_path() -> Path:
    """``<output dir>/amics/last_files.json`` — the three paths last chosen in the file window."""
    from pyPetrograph.common.paths import get_output_dir

    return (get_output_dir() or Path(".")) / "amics" / "last_files.json"


def load_last_files() -> Dict[str, str]:
    """Saved ``legend`` / ``bse`` / ``mineral`` / ``start_dir`` strings, or ``{}``."""
    path = last_files_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, str)}


def save_last_files(paths: Dict[str, Path]) -> None:
    """Remember the chosen files and the folder they live in, for the next pop-up."""
    path = last_files_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    anchor = paths.get("bse") or paths.get("mineral") or paths.get("legend")
    doc = {k: str(paths[k]) for k in _FILE_KEYS if paths.get(k)}
    if anchor is not None:
        doc["start_dir"] = str(Path(anchor).parent)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def pick_amics_files(
    legend: Optional[PathLike] = None,
    bse: Optional[PathLike] = None,
    mineral: Optional[PathLike] = None,
    start_dir: PathLike = ".",
    use_saved: bool = True,
) -> Dict[str, Path]:
    """
    One window with three labeled rows (Mineral legend / BSE image / Mineral map).

    A path you pass fills its row. Rows you leave empty are filled from the last choice
    (``last_files.json``) when ``use_saved`` and that file still exists. Picking the BSE image
    or the map fills the other rows by sample name. If you pass all three and they exist, the
    window is skipped. The choice is saved for next time. Returns
    ``{"legend": Path, "bse": Path, "mineral": Path}``. Raises ``RuntimeError`` (naming the
    Settings values to type) if there is no screen or the window is cancelled.
    """
    given = {
        "legend": Path(legend).expanduser() if legend else None,
        "bse": Path(bse).expanduser() if bse else None,
        "mineral": Path(mineral).expanduser() if mineral else None,
    }
    if all(p is not None and p.is_file() for p in given.values()):
        save_last_files(given)  # type: ignore[arg-type]
        return {k: Path(v) for k, v in given.items()}  # type: ignore[arg-type]

    saved = load_last_files() if use_saved else {}
    merged: Dict[str, Optional[Path]] = dict(given)
    remembered = []
    # A path typed in Settings wins. The last choice fills the rows only when none were typed.
    if not any(given.values()):
        for key in _FILE_KEYS:
            if saved.get(key) and Path(saved[key]).expanduser().is_file():
                merged[key] = Path(saved[key]).expanduser()
                remembered.append(key)
    if remembered:
        print(f"file window starts from the last choice ({', '.join(remembered)})")
    paths, auto = complete(merged["legend"], merged["bse"], merged["mineral"])

    start = Path(start_dir).expanduser()
    anchor = next((paths[k] for k in ("bse", "mineral", "legend") if paths[k] is not None), None)
    if anchor is not None and Path(anchor).is_file():
        start = Path(anchor).parent
    elif saved.get("start_dir") and Path(saved["start_dir"]).expanduser().is_dir():
        start = Path(saved["start_dir"]).expanduser()

    missing = [k for k, p in paths.items() if p is None or not Path(p).is_file()]
    hint = "set LEGEND_PATH, BSE_PATH and MINERAL_PATH in Settings"
    if not has_screen():
        raise RuntimeError(f"No screen for the file window ({', '.join(missing)} missing): {hint}.")
    try:
        res = _run_worker(
            "files",
            title="",
            extra={
                "start_dir": str(start),
                "legend": str(paths["legend"]) if paths["legend"] else None,
                "bse": str(paths["bse"]) if paths["bse"] else None,
                "mineral": str(paths["mineral"]) if paths["mineral"] else None,
                "auto": auto,
            },
        )
    except WindowUnavailable as exc:
        raise RuntimeError(f"The file window could not open ({exc}). Please {hint}.") from exc
    if res.get("cancelled"):
        raise RuntimeError(f"File selection cancelled. Choose the three files, or {hint}.")
    chosen = {k: Path(res[k]) for k in _FILE_KEYS}
    save_last_files(chosen)
    return chosen


def click_points(
    image: np.ndarray,
    valid: Optional[np.ndarray] = None,
    points: Sequence[Tuple[float, float]] = (),
    radius: int = 5,
    pct: float = 99.0,
    footer_row: Optional[int] = None,
    title: str = "Dark-pixel threshold: click the epoxy / holes",
) -> List[Tuple[float, float]]:
    """
    Pop-up on the stretched BSE image; starts with the saved ``points``. Left click adds, right
    click removes the nearest, C clears all; a blue tint and the threshold update live. Enter or
    closing the window finishes. Returns the final points (maybe empty). With no screen it returns
    the saved ``points`` (or raises, asking for ``BSE_THRESHOLD``).
    """
    saved = [(float(x), float(y)) for x, y in points]

    def fallback(why: str) -> List[Tuple[float, float]]:
        if saved:
            print(f"{why}; using the saved threshold points ({len(saved)}).")
            return saved
        raise RuntimeError(f"{why}: set BSE_THRESHOLD in Settings (a number such as 62).")

    if not has_screen():
        return fallback("No screen for the threshold window")
    arrays = {"valid": np.asarray(valid, dtype=bool)} if valid is not None else {}
    try:
        res = _run_worker(
            "points",
            image,
            title,
            extra={
                "points": [list(p) for p in saved],
                "radius": int(radius),
                "pct": float(pct),
                "footer_row": footer_row,
                "window_title": "AMICS: BSE dark-pixel threshold",
            },
            arrays=arrays,
        )
    except WindowUnavailable as exc:
        print(exc)
        return fallback("The threshold window could not open")
    return [(float(x), float(y)) for x, y in res.get("points", [])]


def draw_scale(
    image: np.ndarray,
    saved: Optional[Dict[str, Any]] = None,
    default_length: float = 500.0,
    default_unit: str = "µm",
) -> Optional[Dict[str, Any]]:
    """
    Pop-up: shows the ``saved`` scale line (if any); two clicks draw a new one, Enter asks length + unit.

    Returns ``{"p1", "p2", "px_length", "length", "unit", "um_per_px"}``, or None when the window is
    closed without Enter or there is no screen (the caller then keeps the current scale).
    """
    if not has_screen():
        print("No screen for the scale window: keeping the current scale (or set UM_PER_PX in Settings).")
        return None
    try:
        res = _run_worker(
            "scale",
            image,
            "Scale line: click 2 points on the scale bar",
            {
                "default_length": float(default_length),
                "default_unit": default_unit,
                "saved": saved,
                "window_title": "AMICS: scale",
            },
        )
    except WindowUnavailable as exc:
        print(exc)
        print("The scale window could not open: keeping the current scale (or set UM_PER_PX in Settings).")
        return None
    if res.get("confirmed") is not True:
        return None
    res.pop("confirmed", None)
    return res
