#!/usr/bin/env python3
"""
Experimental (incomplete): notebook 3 multi-image overlay / segmentation / classifier-training code; functions and results may change.

SegmentEveryGrain worker. Run in a subprocess, not inside the Jupyter kernel.

Safe to use conda env ``work``'s Python (TensorFlow 2.21 + numpy 2 imports).
The worker blocks pyarrow (native crash with TensorFlow). The notebook still
must not import this module in-process. The SEG code itself is shared with the
AMICS worker in ``objects/tf_env.py``; this file is the align CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401


_OBJECTS = Path(__file__).resolve().parent.parent / "objects"  # shared tf_env.py
if str(_OBJECTS) not in sys.path:
    sys.path.insert(0, str(_OBJECTS))

# shared SEG helpers; re-exported here for qc_worker / seg_qc_worker imports
from tf_env import (  # noqa: E402,F401
    _block_pyarrow,
    _device,
    _load_sam,
    _load_unet,
    _polygons_to_labels,
    _prepare_native_libs,
    _run_one,
)


def main(argv: Optional[List[str]] = None) -> int:
    _prepare_native_libs()
    p = argparse.ArgumentParser(description="SEG worker (subprocess; not the Jupyter kernel)")
    p.add_argument("--image", required=True)
    p.add_argument("--unet", required=True)
    p.add_argument("--sam", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--modes",
        default="unet",
        help="comma list: unet and/or sam",
    )
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = {m.strip() for m in str(args.modes).split(",") if m.strip()}
    result: Dict[str, Any] = {
        "unet": {"ok": False, "n": 0, "method": "seg_unet", "error": "not run"},
        "sam": {"ok": False, "n": 0, "method": "seg_unet_sam", "error": "not run"},
    }

    try:
        from PIL import Image

        im = Image.open(args.image)
        hw = (int(im.size[1]), int(im.size[0]))
        unet = _load_unet(args.unet)

        if "unet" in modes:
            one = _run_one(args.image, unet, None, use_sam=False, hw=hw)
            lab = one.pop("labels")
            result["unet"] = one
            import numpy as np

            np.save(out / "unet_labels.npy", lab)

        if "sam" in modes:
            if not args.sam:
                result["sam"]["error"] = "SAM2 weights path was empty"
            else:
                try:
                    sam_pred = _load_sam(args.sam)
                    one = _run_one(args.image, unet, sam_pred, use_sam=True, hw=hw)
                    lab = one.pop("labels")
                    result["sam"] = one
                    import numpy as np

                    np.save(out / "sam_labels.npy", lab)
                except Exception as sam_exc:
                    result["sam"]["error"] = f"{type(sam_exc).__name__}: {sam_exc}"
                    traceback.print_exc()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        for key in modes:
            if key in result and not result[key].get("ok"):
                result[key]["error"] = err
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(err, file=sys.stderr)
        return 1

    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    _prepare_native_libs()
    raise SystemExit(main())
