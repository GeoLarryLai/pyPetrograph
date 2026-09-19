"""Scene: named measurement layers on one working pixel grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.transform import SimilarityTransform, warp

from pyPetrograph.common.constants import DEFAULT_ALIGN_FADE, PathLike
from pyPetrograph.common.image_io import load_image


@dataclass
class Affine2D:
    """Map moving pixel (x, y) = (col, row) → working/base pixel."""

    scale: float = 1.0
    rotation_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def to_skimage(self) -> SimilarityTransform:
        return SimilarityTransform(
            scale=float(self.scale),
            rotation=np.deg2rad(float(self.rotation_deg)),
            translation=(float(self.tx), float(self.ty)),
        )

    def inverse(self) -> "Affine2D":
        inv = self.to_skimage().inverse
        rot = float(np.rad2deg(inv.rotation))
        sc = float(np.mean(inv.scale)) if np.ndim(inv.scale) else float(inv.scale)
        return Affine2D(scale=sc, rotation_deg=rot, tx=float(inv.translation[0]), ty=float(inv.translation[1]))

    def as_dict(self) -> Dict[str, float]:
        return {
            "scale": float(self.scale),
            "rotation_deg": float(self.rotation_deg),
            "tx": float(self.tx),
            "ty": float(self.ty),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Affine2D":
        return cls(
            scale=float(d.get("scale", 1.0)),
            rotation_deg=float(d.get("rotation_deg", 0.0)),
            tx=float(d.get("tx", 0.0)),
            ty=float(d.get("ty", 0.0)),
        )


def slot_kind(slot: str) -> str:
    """
    Layer kind from its slot name: ``"xpl_015"`` → ``"xpl"``, ``"ppl"`` → ``"ppl"``.

    Kinds: ``ppl`` (plane-polarized), ``xpl`` (cross-polarized; several angles allowed
    as ``xpl_000``, ``xpl_015``, …), ``bse`` (SEM), ``bright`` (bright field),
    ``cpol`` (circular polarized). Anything else is passed through as-is.
    """
    return str(slot).split("_")[0].lower()


def xpl_slot(angle_deg: float) -> str:
    """Slot name for one cross-polarized angle, e.g. ``xpl_015``."""
    return f"xpl_{int(round(float(angle_deg))):03d}"


@dataclass
class Layer:
    slot: str  # "ppl" | "xpl" | "xpl_015" | "bse" | "bright" | "cpol" | ...
    path: Path
    transform: Affine2D = field(default_factory=Affine2D)
    rgb: Optional[np.ndarray] = field(default=None, repr=False)
    angle_deg: Optional[float] = None  # polarizer angle for xpl_* series (None = unknown)

    @property
    def kind(self) -> str:
        return slot_kind(self.slot)

    def ensure_rgb(self) -> np.ndarray:
        if self.rgb is None:
            self.rgb = load_image(self.path)
        return self.rgb


@dataclass
class Scene:
    """One working grid (usually the train-on layer) plus other measurements."""

    working_slot: str = "ppl"
    layers: Dict[str, Layer] = field(default_factory=dict)
    point_tolerance_px: float = 8.0
    scale_range: Tuple[float, float] = (0.5, 2.0)
    click_pairs: List[Dict[str, List[float]]] = field(default_factory=list)
    json_path: Optional[Path] = None

    def add_layer(
        self,
        slot: str,
        path: PathLike,
        *,
        is_working: bool = False,
        angle_deg: Optional[float] = None,
    ) -> Layer:
        layer = Layer(
            slot=str(slot),
            path=Path(path),
            transform=Affine2D(),
            angle_deg=None if angle_deg is None else float(angle_deg),
        )
        self.layers[layer.slot] = layer
        if is_working or not self.layers:
            self.working_slot = layer.slot
            layer.transform = Affine2D()
        return layer

    def layers_of_kind(self, kind: str) -> List[Layer]:
        """Layers whose slot kind matches (e.g. all ``xpl_*``), sorted by angle then slot."""
        k = str(kind).lower()
        out = [ly for ly in self.layers.values() if ly.kind == k]
        return sorted(out, key=lambda ly: (ly.angle_deg if ly.angle_deg is not None else -1.0, ly.slot))

    def primary_slot(self) -> str:
        """Photo used for color: ``ppl`` if present, else ``bright``, else the working slot."""
        for k in ("ppl", "bright"):
            for slot, ly in self.layers.items():
                if ly.kind == k:
                    return slot
        return self.working_slot

    def working(self) -> Layer:
        if self.working_slot not in self.layers:
            raise KeyError(f"working slot {self.working_slot!r} is not in the scene")
        return self.layers[self.working_slot]

    def working_hw(self) -> Tuple[int, int]:
        rgb = self.working().ensure_rgb()
        return int(rgb.shape[0]), int(rgb.shape[1])

    def warp_layer(self, slot: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp ``slot`` onto the working grid.

        Returns (rgb_uint8 HxWx3, valid mask HxW bool). Pixels outside coverage
        are skipped (mask False) — never filled with fake data.
        """
        base = self.working().ensure_rgb()
        h, w = base.shape[:2]
        layer = self.layers[slot]
        src = layer.ensure_rgb()
        if slot == self.working_slot:
            valid = np.ones((h, w), dtype=bool)
            return base.copy(), valid
        tform = layer.transform.to_skimage()
        warped = warp(
            src.astype(np.float32) / 255.0,
            inverse_map=tform.inverse,
            output_shape=(h, w),
            order=1,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
        ones = np.ones(src.shape[:2], dtype=np.float32)
        coverage = warp(
            ones,
            inverse_map=tform.inverse,
            output_shape=(h, w),
            order=0,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
        rgb = np.clip(np.round(warped * 255.0), 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        valid = coverage >= 0.5
        rgb[~valid] = 0
        return rgb, valid.astype(bool)

    def composite(
        self,
        overlay_slot: str,
        fade: float = DEFAULT_ALIGN_FADE,
    ) -> np.ndarray:
        """Working RGB with ``overlay_slot`` faded on top where that layer is valid."""
        base = self.working().ensure_rgb().copy().astype(np.float32)
        if overlay_slot == self.working_slot:
            return np.clip(base, 0, 255).astype(np.uint8)
        over, valid = self.warp_layer(overlay_slot)
        a = float(np.clip(fade, 0.0, 1.0))
        out = base
        out[valid] = (1.0 - a) * base[valid] + a * over[valid].astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def coverage_mask(self, slot: str) -> np.ndarray:
        _, valid = self.warp_layer(slot)
        return valid
