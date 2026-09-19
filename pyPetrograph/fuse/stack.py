"""
Stage 2: build the physics channel stack from an aligned ``Scene``.

Recipe (adapts to the layers present; a missing layer is skipped, never faked):

- primary photo (``ppl``, else ``bright``): Y-normalized RGB → ``ppl_r/g/b``
- XPL: ≥3 angles → sine fit → ``xpl_max``, ``xpl_amp``, ``xpl_phi_cos``,
  ``xpl_phi_sin`` + mean interference color ``xpl_mean_r/g/b``;
  a single XPL photo → raw ``xpl_r/g/b``
- ``bse`` gray when present
- extras (off by default): texture maps from ``build_feature_stack`` (``tex_*``),
  autoencoder field (``embed_*``)

``channel_set_id`` (e.g. ``ppl-xplfit-bse``) names the trained U-Net so a model is
only ever applied to a matching stack. Saved beside the image under
``channels/{stem}_channels.npz`` (stack + valid) and ``channels/{stem}_channels.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyPetrograph.align.scene import Scene
from pyPetrograph.common.constants import (
    CHANNELS_SUBDIR,
    DEFAULT_EMBED_FEATURE_TOGGLES,
    DEFAULT_Y_TARGET,
    PathLike,
)
from pyPetrograph.common.paths import stem_paths
from pyPetrograph.fuse.extinction import MIN_ANGLES, fit_extinction
from pyPetrograph.image_processing.brightness import normalize_brightness, relative_luminance

# Which channel groups to build. Extras are off by default.
DEFAULT_CHANNEL_TOGGLES: Dict[str, bool] = {
    "primary_rgb": True,
    "xpl": True,
    "bse": True,
    "extra_texture": False,
    "extra_embed": False,
}

# Per-channel value kind → how the U-Net worker scales it before z-scoring.
KIND_RGB255 = "rgb255"  # 0–255 color channel (SEG RGB weights can be copied here)
KIND_GRAY255 = "gray255"  # 0–255 single gray channel
KIND_POS255 = "pos255"  # 0–255-ish positive map (xpl_max, xpl_amp)
KIND_SIGNED = "signed"  # −1..1 (phi_cos, phi_sin)
KIND_FEATURE = "feature"  # arbitrary float map (texture / embedding)


@dataclass
class ChannelStack:
    """Working-grid channel stack with names, value kinds and a coverage mask."""

    stack: np.ndarray  # HxWxC float32
    valid: np.ndarray  # HxW bool — True where every used layer covers the pixel
    names: List[str]
    kinds: List[str]
    channel_set_id: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.stack.shape[-1])

    @property
    def shape_hw(self) -> Tuple[int, int]:
        return int(self.stack.shape[0]), int(self.stack.shape[1])

    def index(self, name: str) -> int:
        return self.names.index(name)

    def channel(self, name: str) -> np.ndarray:
        return self.stack[..., self.index(name)]

    def primary_rgb_indices(self) -> List[int]:
        """The three primary-photo RGB channels (SEG RGB weights are copied onto these)."""
        prim = str(self.meta.get("primary_kind") or "ppl")
        want = [f"{prim}_r", f"{prim}_g", f"{prim}_b"]
        return [self.names.index(n) for n in want if n in self.names]

    def summary(self) -> str:
        h, w = self.shape_hw
        lines = [
            f"channel set: {self.channel_set_id}   {self.n_channels} channels   {h}x{w} px   "
            f"valid {100.0 * float(self.valid.mean()):.1f}%",
        ]
        for i, (n, k) in enumerate(zip(self.names, self.kinds)):
            lines.append(f"  [{i:2d}] {n:<16} {k}")
        return "\n".join(lines)


def channel_set_id(groups: List[str]) -> str:
    """``ppl-xplfit-bse`` style id from the ordered group tags."""
    return "-".join(groups) if groups else "empty"


def _as_gray(rgb: np.ndarray) -> np.ndarray:
    return relative_luminance(np.asarray(rgb, dtype=np.float32)).astype(np.float32)


def build_channel_stack(
    scene: Scene,
    *,
    toggles: Optional[Dict[str, bool]] = None,
    y_target: float = DEFAULT_Y_TARGET,
    texture_toggles: Optional[Dict[str, bool]] = None,
    embed_field: Optional[np.ndarray] = None,
    min_xpl_angles: int = MIN_ANGLES,
) -> ChannelStack:
    """
    Build the channel stack on the scene's working grid.

    Parameters
    ----------
    scene : aligned Scene (transforms already saved / loaded)
    toggles : which groups to include (see ``DEFAULT_CHANNEL_TOGGLES``)
    y_target : brightness target for the primary photo (color ratios kept)
    texture_toggles : feature toggles for ``extra_texture`` (r/g/b/glcm forced off)
    embed_field : HxWxD array for ``extra_embed`` (e.g. from ``embed_scene``)
    min_xpl_angles : angles needed for the sine fit (default 3)
    """
    tg = dict(DEFAULT_CHANNEL_TOGGLES)
    tg.update(toggles or {})
    h, w = scene.working_hw()
    valid = np.ones((h, w), dtype=bool)
    chans: List[np.ndarray] = []
    names: List[str] = []
    kinds: List[str] = []
    groups: List[str] = []
    slots_used: List[str] = []
    notes: List[str] = []

    def _add(arr: np.ndarray, name: str, kind: str) -> None:
        chans.append(np.asarray(arr, dtype=np.float32))
        names.append(name)
        kinds.append(kind)

    # --- primary photo (color) ---
    prim_slot = scene.primary_slot()
    prim_layer = scene.layers[prim_slot]
    prim_kind = prim_layer.kind if prim_layer.kind in ("ppl", "bright") else "ppl"
    prim_rgb, prim_mask = scene.warp_layer(prim_slot)
    valid &= prim_mask
    if tg.get("primary_rgb", True):
        rgbn = normalize_brightness(prim_rgb, y_target=y_target, mask=prim_mask)
        for i, c in enumerate("rgb"):
            _add(rgbn[..., i], f"{prim_kind}_{c}", KIND_RGB255)
        groups.append(prim_kind)
        slots_used.append(prim_slot)

    # --- XPL: sine fit over angles, or one raw photo ---
    xpl_angles: List[float] = []
    if tg.get("xpl", True):
        xpls = scene.layers_of_kind("xpl")
        angled = [ly for ly in xpls if ly.angle_deg is not None]
        if len(angled) >= int(min_xpl_angles):
            grays: List[np.ndarray] = []
            rgb_sum = np.zeros((h, w, 3), dtype=np.float32)
            for ly in angled:
                rgb, m = scene.warp_layer(ly.slot)
                valid &= m
                grays.append(_as_gray(rgb))
                rgb_sum += rgb.astype(np.float32)
                slots_used.append(ly.slot)
                xpl_angles.append(float(ly.angle_deg))
            fit = fit_extinction(np.stack(grays, axis=0), xpl_angles, valid=valid)
            _add(fit["max"], "xpl_max", KIND_POS255)
            _add(fit["amp"], "xpl_amp", KIND_POS255)
            _add(fit["phi_cos"], "xpl_phi_cos", KIND_SIGNED)
            _add(fit["phi_sin"], "xpl_phi_sin", KIND_SIGNED)
            mean_rgb = rgb_sum / float(len(angled))
            for i, c in enumerate("rgb"):
                _add(mean_rgb[..., i], f"xpl_mean_{c}", KIND_RGB255)
            groups.append("xplfit")
            del grays, rgb_sum
        elif xpls:
            ly = xpls[0]
            rgb, m = scene.warp_layer(ly.slot)
            valid &= m
            for i, c in enumerate("rgb"):
                _add(rgb[..., i], f"xpl_{c}", KIND_RGB255)
            groups.append("xpl")
            slots_used.append(ly.slot)
            if len(xpls) > 1:
                notes.append(
                    f"{len(xpls)} XPL layers but fewer than {min_xpl_angles} with a known angle; "
                    f"used {ly.slot} only. Set angle_deg on each xpl_* layer for the sine fit."
                )

    # --- BSE gray ---
    if tg.get("bse", True):
        bses = scene.layers_of_kind("bse")
        if bses:
            rgb, m = scene.warp_layer(bses[0].slot)
            valid &= m
            _add(_as_gray(rgb), "bse", KIND_GRAY255)
            groups.append("bse")
            slots_used.append(bses[0].slot)

    # --- extras (off by default) ---
    if tg.get("extra_texture", False):
        from pyPetrograph.image_processing.features import build_feature_stack, resolve_feature_toggles

        tt = resolve_feature_toggles(
            texture_toggles if texture_toggles is not None else DEFAULT_EMBED_FEATURE_TOGGLES
        )
        for k in ("r", "g", "b", "y", "glcm"):
            tt[k] = False
        if any(tt.values()):
            feat, fnames = build_feature_stack(
                prim_rgb, toggles=tt, y_target=y_target, apply_norm=True, polygons=None
            )
            for i, n in enumerate(fnames):
                _add(feat[..., i], f"tex_{n}", KIND_FEATURE)
            groups.append("tex")
            del feat
        else:
            notes.append("extra_texture on but no texture toggle is True; skipped.")

    if tg.get("extra_embed", False):
        if embed_field is None:
            notes.append("extra_embed on but no embed_field given; skipped.")
        else:
            ef = np.asarray(embed_field, dtype=np.float32)
            if ef.shape[:2] != (h, w):
                raise ValueError(f"embed_field {ef.shape[:2]} does not match working grid {(h, w)}")
            for d in range(int(ef.shape[-1])):
                _add(ef[..., d], f"embed_{d:02d}", KIND_FEATURE)
            groups.append("embed")

    if not chans:
        raise RuntimeError("No channels built — every group is off or no layers are present.")

    stack = np.stack(chans, axis=-1).astype(np.float32)
    stack[~valid] = 0.0
    meta: Dict[str, object] = {
        "channel_set_id": channel_set_id(groups),
        "groups": groups,
        "primary_slot": prim_slot,
        "primary_kind": prim_kind,
        "slots": slots_used,
        "xpl_angles_deg": xpl_angles,
        "y_target": float(y_target),
        "shape_hw": [h, w],
        "toggles": tg,
        "notes": notes,
    }
    return ChannelStack(
        stack=stack,
        valid=valid,
        names=names,
        kinds=kinds,
        channel_set_id=channel_set_id(groups),
        meta=meta,
    )


# --- disk contract: channels/{stem}_channels.npz + channels/{stem}_channels.json ---


def channels_paths(image_path: PathLike) -> Tuple[Path, Path]:
    """``channels/{stem}_channels.npz`` and ``channels/{stem}_channels.json`` beside the image."""
    sp = stem_paths(image_path)
    return sp.channels, sp.channels_meta


def save_channel_stack(cs: ChannelStack, image_path: PathLike) -> Tuple[Path, Path]:
    npz, meta_p = channels_paths(image_path)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, stack=cs.stack.astype(np.float32), valid=cs.valid.astype(bool))
    meta = dict(cs.meta)
    meta.update(
        {
            "names": list(cs.names),
            "kinds": list(cs.kinds),
            "channel_set_id": cs.channel_set_id,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "source_image": str(image_path),
        }
    )
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz, meta_p


def load_channel_stack(image_path: PathLike) -> Optional[ChannelStack]:
    """Reload a saved stack, or ``None`` if the files are missing."""
    npz, meta_p = channels_paths(image_path)
    if not npz.exists() or not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    with np.load(npz) as blob:
        stack = np.asarray(blob["stack"], dtype=np.float32)
        valid = np.asarray(blob["valid"], dtype=bool)
    names = [str(n) for n in meta.get("names") or []]
    kinds = [str(k) for k in meta.get("kinds") or []]
    if len(names) != stack.shape[-1]:
        raise ValueError(f"{meta_p.name}: {len(names)} names for {stack.shape[-1]} channels")
    return ChannelStack(
        stack=stack,
        valid=valid,
        names=names,
        kinds=kinds,
        channel_set_id=str(meta.get("channel_set_id") or channel_set_id(list(meta.get("groups") or []))),
        meta=meta,
    )
