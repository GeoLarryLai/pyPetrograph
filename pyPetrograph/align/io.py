"""Save / reload lineup JSON under {image_folder}/align/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pyPetrograph.align.scene import Affine2D, Layer, Scene
from pyPetrograph.common.constants import ALIGN_SUBDIR, PathLike
from pyPetrograph.common.paths import get_output_dir


def align_dir_for(image_path: PathLike) -> Path:
    p = Path(image_path).resolve()
    base = get_output_dir() or p.parent
    d = base / ALIGN_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_json_path(scene: Scene) -> Path:
    working = scene.working()
    return align_dir_for(working.path) / f"{working.path.stem}_align.json"


def scene_to_dict(scene: Scene) -> Dict[str, Any]:
    layers = []
    for slot, layer in scene.layers.items():
        item: Dict[str, Any] = {
            "slot": slot,
            "path": str(layer.path),
            "transform": layer.transform.as_dict(),
        }
        if layer.angle_deg is not None:
            item["angle_deg"] = float(layer.angle_deg)
        layers.append(item)
    return {
        "version": 1,
        "working_slot": scene.working_slot,
        "point_tolerance_px": float(scene.point_tolerance_px),
        "scale_range": [float(scene.scale_range[0]), float(scene.scale_range[1])],
        "click_pairs": list(scene.click_pairs),
        "layers": layers,
    }


def save_scene(scene: Scene, path: Optional[PathLike] = None) -> Path:
    out = Path(path) if path is not None else default_json_path(scene)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene_to_dict(scene), indent=2), encoding="utf-8")
    scene.json_path = out
    return out


def load_scene(path: PathLike) -> Scene:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    scene = Scene(
        working_slot=str(data.get("working_slot") or "ppl"),
        point_tolerance_px=float(data.get("point_tolerance_px", 8.0)),
        json_path=p,
    )
    sr = data.get("scale_range") or [0.5, 2.0]
    scene.scale_range = (float(sr[0]), float(sr[1]))
    scene.click_pairs = list(data.get("click_pairs") or [])
    for item in data.get("layers") or []:
        angle = item.get("angle_deg")
        layer = Layer(
            slot=str(item["slot"]),
            path=Path(item["path"]),
            transform=Affine2D.from_dict(item.get("transform") or {}),
            angle_deg=None if angle is None else float(angle),
        )
        scene.layers[layer.slot] = layer
    if scene.working_slot not in scene.layers and scene.layers:
        scene.working_slot = next(iter(scene.layers))
    return scene
