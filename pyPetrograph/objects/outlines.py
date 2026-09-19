"""
Stage 3 outlines: watershed baseline, N-channel U-Net (SEG weights), SAM2 + GrainPlot.

The Jupyter kernel never imports TensorFlow. Train / predict / GrainPlot run in
subprocesses of conda ``work``.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyPetrograph.common.constants import (
    DEFAULT_OUTLINE_EPOCHS,
    DEFAULT_OUTLINE_MAX_PATCHES,
    GRAIN_UNET_PREFIX,
    PathLike,
)
from pyPetrograph.common.paths import project_models_dir
from pyPetrograph.fuse.stack import ChannelStack
from pyPetrograph.objects.qc import (
    grain_qc_paths,
    launch_grain_qc,
    load_grain_mask,
    overlay_labels,
    seg_setup_hint,
    _n_labels,
    _seg_python,
    _tf_worker_env,
)
from pyPetrograph.objects.watershed import run_watershed


def grain_unet_path(channel_set_id: str) -> Path:
    """Notebook-level ``models/grain_unet_{channel_set}.keras``."""
    return project_models_dir() / f"{GRAIN_UNET_PREFIX}{channel_set_id}.keras"


def grain_unet_trained_marker(channel_set_id: str) -> Path:
    """Sidecar written after a finished train (init-only ``.keras`` is not trained)."""
    return Path(str(grain_unet_path(channel_set_id)) + ".trained.json")


def grain_unet_is_trained(channel_set_id: str) -> bool:
    """True if weights exist and a train pass wrote the sidecar."""
    return grain_unet_path(channel_set_id).exists() and grain_unet_trained_marker(
        channel_set_id
    ).exists()


def _unet_worker() -> Path:
    return Path(__file__).resolve().parent / "unet_worker.py"


def _run_worker(cmd: List[str], *, mpl: str = "Agg") -> subprocess.CompletedProcess:
    py = _seg_python()
    if py is None:
        raise RuntimeError("no Python for grain U-Net worker")
    env = _tf_worker_env(mpl_backend=mpl)
    print(" ".join(str(c) for c in cmd[1:6]), "…")
    return subprocess.run(cmd, env=env)


def _write_field(tmp: Path, cs: ChannelStack) -> Path:
    p = tmp / "field.npy"
    np.save(p, cs.stack)
    return p


def init_grain_unet(cs: ChannelStack, *, out: Optional[Path] = None) -> Dict[str, Any]:
    """
    Copy SEG RGB U-Net weights into an N-channel first layer.

    RGB kernel → ``cs.primary_rgb_indices()``; other slots start small-random.
    Writes ``models/grain_unet_{channel_set}.keras`` + ``.norm.json``.
    """
    py = _seg_python()
    if py is None:
        return {"ok": False, "error": "no Python for grain U-Net worker", "hint": seg_setup_hint()}
    seg = project_models_dir() / "seg_model_smooth_labels.keras"
    if not seg.exists():
        return {"ok": False, "error": f"missing {seg.name}", "hint": seg_setup_hint()}
    out_path = Path(out) if out is not None else grain_unet_path(cs.channel_set_id)
    rgb_idx = cs.primary_rgb_indices()
    cmd = [
        str(py),
        str(_unet_worker()),
        "--mode",
        "init",
        "--seg-unet",
        str(seg),
        "--n-in",
        str(cs.n_channels),
        "--rgb-indices",
        ",".join(str(i) for i in rgb_idx) if rgb_idx else "",
        "--kinds",
        json.dumps(list(cs.kinds)),
        "--names",
        json.dumps(list(cs.names)),
        "--out",
        str(out_path),
    ]
    print(f"grain U-Net init: {out_path.name}  n_in={cs.n_channels}  rgb←{rgb_idx}")
    proc = _run_worker(cmd)
    ok = proc.returncode == 0 and out_path.exists()
    note = None
    norm = Path(str(out_path) + ".norm.json")
    if norm.exists():
        try:
            note = json.loads(norm.read_text(encoding="utf-8")).get("init")
        except json.JSONDecodeError:
            note = None
    return {
        "ok": ok,
        "path": out_path,
        "n_in": cs.n_channels,
        "channel_set_id": cs.channel_set_id,
        "init": note,
        "error": None if ok else f"init exited {proc.returncode}",
        "hint": None if ok else seg_setup_hint(),
    }


def train_grain_unet(
    cs: ChannelStack,
    image_path: PathLike,
    *,
    epochs: int = DEFAULT_OUTLINE_EPOCHS,
    max_patches: int = DEFAULT_OUTLINE_MAX_PATCHES,
    out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train the N-channel U-Net from the QC mask beside ``image_path``."""
    py = _seg_python()
    if py is None:
        return {"ok": False, "error": "no Python for grain U-Net worker", "hint": seg_setup_hint()}
    mask = load_grain_mask(image_path)
    if mask is None:
        geo, mask_p = grain_qc_paths(image_path)
        return {
            "ok": False,
            "error": f"missing grain QC mask: {mask_p.name}",
            "hint": "Run launch_grain_qc first (watershed bootstrap is fine).",
        }
    out_path = Path(out) if out is not None else grain_unet_path(cs.channel_set_id)
    with tempfile.TemporaryDirectory(prefix="pypetrograph_grain_unet_") as tmp:
        tmp_p = Path(tmp)
        field_p = _write_field(tmp_p, cs)
        mask_p = tmp_p / "mask.png"
        from PIL import Image

        if int(np.asarray(mask).max()) <= 255:
            Image.fromarray(np.asarray(mask).astype(np.uint8), mode="L").save(mask_p)
        else:
            Image.fromarray(np.asarray(mask).astype(np.uint16), mode="I;16").save(mask_p)
        cmd = [
            str(py),
            str(_unet_worker()),
            "--mode",
            "train",
            "--field",
            str(field_p),
            "--mask",
            str(mask_p),
            "--out",
            str(out_path),
            "--epochs",
            str(int(epochs)),
            "--max-patches",
            str(int(max_patches)),
        ]
        print(f"grain U-Net train: {out_path.name}  epochs={epochs}")
        proc = _run_worker(cmd)
    ok = proc.returncode == 0 and out_path.exists()
    if ok:
        marker = grain_unet_trained_marker(cs.channel_set_id)
        marker.write_text(
            json.dumps(
                {
                    "ok": True,
                    "channel_set_id": cs.channel_set_id,
                    "epochs": int(epochs),
                    "path": str(out_path),
                },
                indent=2,
            )
            + "\n"
        )
    return {
        "ok": ok,
        "path": out_path,
        "n_in": cs.n_channels,
        "channel_set_id": cs.channel_set_id,
        "error": None if ok else f"train exited {proc.returncode}",
        "hint": None if ok else seg_setup_hint(),
    }


def predict_grain_unet(
    cs: ChannelStack,
    *,
    unet_path: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Predict 3-class probabilities + connected-component labels on ``cs``.

    Returns ``(probs HxWx3, labels HxW, info)``. Either array may be None on failure.
    """
    py = _seg_python()
    weights = Path(unet_path) if unet_path is not None else grain_unet_path(cs.channel_set_id)
    if py is None:
        return None, None, {
            "ok": False, "n": 0, "method": "grain_unet",
            "error": "no Python for grain U-Net worker", "hint": seg_setup_hint(),
        }
    if not weights.exists():
        return None, None, {
            "ok": False, "n": 0, "method": "grain_unet",
            "error": f"missing weights: {weights.name}",
            "hint": "Run init_grain_unet then train_grain_unet after GrainPlot QC.",
        }
    with tempfile.TemporaryDirectory(prefix="pypetrograph_grain_pred_") as tmp:
        tmp_p = Path(tmp)
        field_p = _write_field(tmp_p, cs)
        cmd = [
            str(py),
            str(_unet_worker()),
            "--mode",
            "predict",
            "--field",
            str(field_p),
            "--unet",
            str(weights),
            "--out-dir",
            str(tmp_p),
        ]
        print(f"grain U-Net predict: {weights.name}")
        proc = _run_worker(cmd)
        probs = np.load(tmp_p / "grain_unet_probs.npy") if (tmp_p / "grain_unet_probs.npy").exists() else None
        lab = np.load(tmp_p / "grain_unet_labels.npy").astype(np.int32) if (tmp_p / "grain_unet_labels.npy").exists() else None
        meta: Dict[str, Any] = {}
        if (tmp_p / "result.json").exists():
            try:
                meta = json.loads((tmp_p / "result.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
    ok = proc.returncode == 0 and lab is not None
    info = {
        "ok": ok,
        "n": int(meta.get("n") or (_n_labels(lab) if lab is not None else 0)),
        "method": str(meta.get("method") or "grain_unet"),
        "path": weights,
        "channel_set_id": cs.channel_set_id,
        "error": None if ok else (meta.get("error") or f"predict exited {proc.returncode}"),
        "hint": None if ok else "Train with train_grain_unet after GrainPlot QC.",
    }
    return probs, lab, info


def run_outlines(
    cs: ChannelStack,
    image_path: PathLike,
    *,
    method: str = "unet",
    train: bool = False,
    epochs: int = DEFAULT_OUTLINE_EPOCHS,
    qc: bool = True,
    reuse: bool = True,
    init: bool = False,
) -> Dict[str, Any]:
    """
    One-shot outline pass.

    ``method="watershed"``: no training; optional GrainPlot on those labels.
    ``method="unet"``: optional ``init``, optional ``train`` from saved QC, predict,
    then GrainPlot on the probability cube (SAM2 polish).
    """
    image_path = Path(image_path)
    out: Dict[str, Any] = {"method": method, "channel_set_id": cs.channel_set_id}
    ws_lab, ws_info = run_watershed(cs)
    out["watershed"] = {"labels": ws_lab, "info": ws_info}

    if method == "watershed":
        if qc:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                np.save(f.name, ws_lab)
                tmp_lab = Path(f.name)
            try:
                out["qc"] = launch_grain_qc(image_path, labels_path=tmp_lab, reuse=False)
            finally:
                tmp_lab.unlink(missing_ok=True)
        return out

    if init or not grain_unet_path(cs.channel_set_id).exists():
        out["init"] = init_grain_unet(cs)
        if not out["init"].get("ok"):
            return out
    if train:
        out["train"] = train_grain_unet(cs, image_path, epochs=epochs)
        if not out["train"].get("ok"):
            return out
    probs, unet_lab, unet_info = predict_grain_unet(cs)
    out["unet"] = {"probs": probs, "labels": unet_lab, "info": unet_info}
    if qc and probs is not None:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, probs)
            tmp_pred = Path(f.name)
        try:
            out["qc"] = launch_grain_qc(image_path, pred_path=tmp_pred, reuse=False)
        finally:
            tmp_pred.unlink(missing_ok=True)
    elif qc:
        out["qc"] = launch_grain_qc(image_path, reuse=reuse)
    return out


def compare_outlines(
    rgb: np.ndarray,
    *,
    watershed_lab: Optional[np.ndarray] = None,
    unet_lab: Optional[np.ndarray] = None,
    qc_lab: Optional[np.ndarray] = None,
    titles: Optional[Tuple[str, str, str]] = None,
    greyscale: bool = True,
) -> Any:
    """
    Overlay grain outlines on PPL. Only panels with labels are drawn.

    ``greyscale=True`` (default) uses a grey background. Outlines are darker
    red (watershed), green (U-Net), blue (GrainPlot QC).
    """
    import matplotlib.pyplot as plt

    titles = titles or ("watershed (no training)", "N-channel U-Net", "GrainPlot QC")
    colors = ((180, 15, 15), (0, 130, 45), (15, 45, 190))
    panels = [
        (lab, title, col)
        for lab, title, col in zip((watershed_lab, unet_lab, qc_lab), titles, colors)
        if lab is not None
    ]
    if not panels:
        raise ValueError("compare_outlines: no label maps to plot")

    from pyPetrograph.common.constants import clip_figure_dpi

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.4, 3.6), dpi=clip_figure_dpi())
    if n == 1:
        axes = np.asarray([axes])
    bg = np.asarray(rgb, dtype=np.uint8)
    if bg.ndim == 2:
        bg = np.stack([bg, bg, bg], axis=-1)
    if greyscale:
        grey = (
            0.299 * bg[..., 0] + 0.587 * bg[..., 1] + 0.114 * bg[..., 2]
        ).astype(np.uint8)
        bg = np.stack([grey, grey, grey], axis=-1)
    for ax, (lab, title, col) in zip(axes, panels):
        ax.imshow(overlay_labels(bg, lab, color=col))
        n_lab = int(np.sum(np.unique(lab) > 0))
        ax.set_title(f"{title}\nn={n_lab}")
        ax.axis("off")
    fig.tight_layout()
    return fig
