"""Multi-layer lineup: register PPL / XPL / BSE onto one working grid."""

from pyPetrograph.align.embed import embed_scene, stack_feature_slots
from pyPetrograph.align.grains import (
    grain_qc_paths,
    launch_ppl_grain_qc,
    load_grain_mask,
    load_ppl_grain_qc,
    overlay_labels,
    seg_setup_hint,
)
from pyPetrograph.align.io import load_scene, save_scene
from pyPetrograph.align.register import auto_register, refine_register
from pyPetrograph.align.scene import Affine2D, Layer, Scene, slot_kind, xpl_slot
from pyPetrograph.align.structure import structure_map
from pyPetrograph.align.ui import launch_align

__all__ = [
    "Affine2D",
    "Layer",
    "Scene",
    "slot_kind",
    "xpl_slot",
    "structure_map",
    "auto_register",
    "refine_register",
    "save_scene",
    "load_scene",
    "launch_align",
    "embed_scene",
    "stack_feature_slots",
    "launch_ppl_grain_qc",
    "load_ppl_grain_qc",
    "grain_qc_paths",
    "load_grain_mask",
    "overlay_labels",
    "seg_setup_hint",
]
