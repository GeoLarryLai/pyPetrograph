#!/usr/bin/env python3
"""
Multi-channel grain U-Net worker. Subprocess only (TensorFlow).

Modes:
  init    — copy SEG RGB U-Net weights into a new N-channel first layer
            (RGB kernel → primary photo channels; other slots start small-random)
  train   — 256-px patches from the channel stack + instance mask
  predict — tiled 3-class probabilities (bg / grain / boundary)

Import this file as a script, never from the Jupyter kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from tf_env import _prepare_native_libs  # noqa: E402

_prepare_native_libs()

# Kind → divide-by before any z-score. Matches fuse.stack.KIND_* (duplicated: no pyPetrograph).
_SCALE = {
    "rgb255": 255.0,
    "gray255": 255.0,
    "pos255": 255.0,
    "signed": 1.0,
    "feature": 1.0,
}


def _unet_n(n_in: int = 3, n_out: int = 3):
    """Same architecture as ``segmenteverygrain.Unet()`` except Input channels."""
    from keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        Input,
        MaxPooling2D,
        concatenate,
    )
    from keras.models import Model
    import tensorflow as tf

    tf.keras.backend.clear_session()
    inputs = Input((256, 256, int(n_in)), name="input")
    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)
    up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(int(n_out), (1, 1))(conv9)
    return Model(inputs=[inputs], outputs=[conv10])


def _weighted_layers(model):
    return [L for L in model.layers if L.weights]


def _copy_first_conv(src_w, dst_layer, rgb_indices: List[int], n_in: int) -> str:
    """Copy RGB kernel into dest channels listed in ``rgb_indices``; rest ~N(0, 0.1·σ)."""
    import numpy as np

    k, b = src_w[0], src_w[1]
    dk = np.zeros(dst_layer.get_weights()[0].shape, dtype=k.dtype)
    rng = np.random.RandomState(0)
    dk[:] = rng.normal(0.0, float(k.std()) * 0.1, size=dk.shape).astype(k.dtype)
    copied = 0
    for src_c, dst_c in enumerate(rgb_indices):
        if 0 <= int(dst_c) < n_in and src_c < k.shape[2]:
            dk[:, :, int(dst_c), :] = k[:, :, src_c, :]
            copied += 1
    dst_layer.set_weights([dk, b.astype(dk.dtype)])
    if copied == 0:
        return (
            "first conv: no rgb_indices landed in-range; left small-random "
            "(SEG RGB weights not transferred)"
        )
    return f"first conv: copied {copied} of 3 RGB kernels into dest channels {rgb_indices}"


def _scale_stack(field: np.ndarray, kinds: List[str]) -> np.ndarray:
    out = np.asarray(field, dtype=np.float32).copy()
    for c, kind in enumerate(kinds):
        s = float(_SCALE.get(kind, 1.0))
        if s != 1.0:
            out[..., c] = out[..., c] / s
    return out


def _parse_rgb_indices(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def _init(args: argparse.Namespace) -> int:
    from keras.saving import load_model
    from segmenteverygrain import weighted_crossentropy

    n_in = int(args.n_in)
    rgb_indices = _parse_rgb_indices(args.rgb_indices)
    src = load_model(args.seg_unet, custom_objects={"weighted_crossentropy": weighted_crossentropy})
    dst = _unet_n(n_in=n_in, n_out=3)
    src_w = _weighted_layers(src)
    dst_w = _weighted_layers(dst)
    if len(src_w) != len(dst_w):
        print(
            f"layer-count mismatch: SEG has {len(src_w)} weight layers, "
            f"N-channel U-Net has {len(dst_w)}. Falling back to random init."
        )
        note = "random init (architecture mismatch with SEG U-Net)"
        copied = 0
    else:
        copied = 0
        notes = []
        first = True
        for s, d in zip(src_w, dst_w):
            sw, dw = s.get_weights(), d.get_weights()
            if all(a.shape == b.shape for a, b in zip(sw, dw)):
                d.set_weights(sw)
                copied += 1
            elif first and s.__class__.__name__ == "Conv2D" and sw[0].ndim == 4:
                notes.append(_copy_first_conv(sw, d, rgb_indices, n_in))
                first = False
            else:
                notes.append(f"skip {s.name} { [a.shape for a in sw] } → {d.name} {[b.shape for b in dw]}")
        note = f"copied {copied}/{len(src_w)} matching layers. " + "; ".join(notes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dst.save(out)
    kinds = json.loads(args.kinds) if args.kinds else ["rgb255"] * n_in
    names = json.loads(args.names) if args.names else [f"c{i}" for i in range(n_in)]
    stats = {
        "n_in": n_in,
        "names": names,
        "kinds": kinds,
        "rgb_indices": rgb_indices,
        "mu": [0.0] * n_in,
        "sd": [1.0] * n_in,
        "init": note,
    }
    Path(str(out) + ".norm.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"saved {out}  n_in={n_in}  {note}")
    return 0


def _three_class_mask(instance):
    import numpy as np
    from skimage.segmentation import find_boundaries

    lab = np.asarray(instance)
    out = np.zeros(lab.shape[:2], dtype=np.uint8)
    out[lab > 0] = 1
    if int(lab.max()) > 0:
        out[find_boundaries(lab, mode="outer")] = 2
    return out


def _patches(field, mask, *, size: int = 256, stride: int = 128, max_n: int = 80) -> Tuple:
    import numpy as np

    h, w, c = field.shape
    xs, ys = [], []
    for y in range(0, max(1, h - size + 1), stride):
        for x in range(0, max(1, w - size + 1), stride):
            ye, xe = min(h, y + size), min(w, x + size)
            fp = np.zeros((size, size, c), dtype=np.float32)
            mp = np.zeros((size, size), dtype=np.uint8)
            fp[: ye - y, : xe - x] = field[y:ye, x:xe]
            mp[: ye - y, : xe - x] = mask[y:ye, x:xe]
            if int((mp > 0).sum()) < 32:
                continue
            xs.append(fp)
            ys.append(mp)
            if len(xs) >= int(max_n):
                return np.stack(xs), np.stack(ys)
    if not xs:
        fp = np.zeros((size, size, c), dtype=np.float32)
        mp = np.zeros((size, size), dtype=np.uint8)
        hh, ww = min(size, h), min(size, w)
        fp[:hh, :ww] = field[:hh, :ww]
        mp[:hh, :ww] = mask[:hh, :ww]
        xs.append(fp)
        ys.append(mp)
    return np.stack(xs), np.stack(ys)


def _onehot(y, n: int = 3):
    import numpy as np

    oh = np.zeros(y.shape + (n,), dtype=np.float32)
    for k in range(n):
        oh[..., k] = (y == k).astype(np.float32)
    return oh


def _flip_aug(X, Y):
    """Cheap 0/90/180/270 + mirrors so 80 patches become a few hundred."""
    import numpy as np

    xs, ys = [X], [Y]
    xs.append(np.flip(X, axis=1))
    ys.append(np.flip(Y, axis=1))
    xs.append(np.flip(X, axis=2))
    ys.append(np.flip(Y, axis=2))
    xs.append(np.rot90(X, k=1, axes=(1, 2)))
    ys.append(np.rot90(Y, k=1, axes=(1, 2)))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _load_norm(path: Path, n_in: int):
    import numpy as np

    mu = np.zeros((1, 1, 1, n_in), dtype=np.float32)
    sd = np.ones((1, 1, 1, n_in), dtype=np.float32)
    kinds = ["rgb255"] * n_in
    names = [f"c{i}" for i in range(n_in)]
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        kinds = [str(k) for k in blob.get("kinds") or kinds]
        names = [str(n) for n in blob.get("names") or names]
        if "mu" in blob:
            mu = np.asarray(blob["mu"], dtype=np.float32).reshape(1, 1, 1, -1)
        if "sd" in blob:
            sd = np.asarray(blob["sd"], dtype=np.float32).reshape(1, 1, 1, -1)
            sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd, kinds, names


def _train(args: argparse.Namespace) -> int:
    import numpy as np
    from keras.optimizers import Adam
    from keras.saving import load_model
    from PIL import Image
    from segmenteverygrain import weighted_crossentropy

    field = np.load(args.field).astype(np.float32)
    inst = np.asarray(Image.open(args.mask))
    if inst.ndim > 2:
        inst = inst[..., 0]
    if inst.shape[:2] != field.shape[:2]:
        raise ValueError(f"mask {inst.shape[:2]} vs field {field.shape[:2]}")
    mask = _three_class_mask(inst)
    n_in = int(field.shape[-1])
    out = Path(args.out)
    norm_path = Path(str(out) + ".norm.json")
    mu0, sd0, kinds, names = _load_norm(norm_path, n_in)
    del mu0, sd0
    Xraw, y = _patches(field, mask, max_n=int(args.max_patches))
    X = _scale_stack(Xraw, kinds)
    X, y = _flip_aug(X, y)
    Y = _onehot(y, 3)
    # z-score feature/pos channels over the (augmented) patch batch; RGB stays /255
    mu = X.mean(axis=(0, 1, 2), keepdims=True)
    sd = X.std(axis=(0, 1, 2), keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    for c, kind in enumerate(kinds):
        if kind in ("rgb255", "signed"):
            mu[..., c] = 0.0
            sd[..., c] = 1.0
    Xn = (X - mu) / sd
    if out.exists():
        model = load_model(str(out), custom_objects={"weighted_crossentropy": weighted_crossentropy})
        print(f"continue from {out.name}")
    else:
        model = _unet_n(n_in=n_in, n_out=3)
    model.compile(optimizer=Adam(), loss=weighted_crossentropy, metrics=["accuracy"])
    model.fit(Xn, Y, epochs=int(args.epochs), batch_size=4, verbose=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    stats = {
        "mu": mu.reshape(-1).tolist(),
        "sd": sd.reshape(-1).tolist(),
        "n_in": n_in,
        "names": names,
        "kinds": kinds,
    }
    if args.rgb_indices:
        stats["rgb_indices"] = _parse_rgb_indices(args.rgb_indices)
    elif norm_path.exists():
        old = json.loads(norm_path.read_text(encoding="utf-8"))
        if "rgb_indices" in old:
            stats["rgb_indices"] = old["rgb_indices"]
        if "init" in old:
            stats["init"] = old["init"]
    norm_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"saved {out}  patches={len(X)}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    import numpy as np
    from keras.saving import load_model
    from segmenteverygrain import weighted_crossentropy

    field = np.load(args.field).astype(np.float32)
    h, w, c = field.shape
    model = load_model(args.unet, custom_objects={"weighted_crossentropy": weighted_crossentropy})
    mu, sd, kinds, _names = _load_norm(Path(str(args.unet) + ".norm.json"), c)
    scaled = _scale_stack(field, kinds)
    size = 256
    logits = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, size // 2):
        for x in range(0, w, size // 2):
            ye, xe = min(h, y + size), min(w, x + size)
            tile = np.zeros((1, size, size, c), dtype=np.float32)
            tile[0, : ye - y, : xe - x] = scaled[y:ye, x:xe]
            tile = (tile - mu) / sd
            pred = model.predict(tile, verbose=0)[0]
            logits[y:ye, x:xe] += pred[: ye - y, : xe - x]
            weight[y:ye, x:xe] += 1.0
    weight = np.maximum(weight, 1e-6)
    logits /= weight[..., None]
    # softmax over class axis
    m = logits.max(axis=-1, keepdims=True)
    ex = np.exp(logits - m)
    probs = ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-6)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "grain_unet_probs.npy", probs.astype(np.float32))
    cls = np.argmax(probs, axis=-1).astype(np.int32)
    grain = cls == 1
    inst = np.zeros((h, w), dtype=np.int32)
    if grain.any():
        from skimage.measure import label as cc_label

        inst = cc_label(grain).astype(np.int32)
    np.save(out / "grain_unet_labels.npy", inst)
    n = int(np.sum(np.unique(inst) > 0))
    (out / "result.json").write_text(
        json.dumps({"ok": True, "n": n, "method": "grain_unet", "n_in": c}),
        encoding="utf-8",
    )
    print(f"grain U-Net n={n}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="N-channel grain U-Net worker")
    p.add_argument("--mode", choices=("init", "train", "predict"), required=True)
    p.add_argument("--field", default="")
    p.add_argument("--mask", default="")
    p.add_argument("--unet", default="")
    p.add_argument("--seg-unet", default="")
    p.add_argument("--out", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--n-in", type=int, default=3)
    p.add_argument("--rgb-indices", default="0,1,2")
    p.add_argument("--kinds", default="")  # JSON list
    p.add_argument("--names", default="")  # JSON list
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max-patches", type=int, default=80)
    args = p.parse_args(argv)
    try:
        if args.mode == "init":
            if not args.seg_unet or not args.out:
                raise ValueError("init needs --seg-unet and --out")
            return _init(args)
        if args.mode == "train":
            if not args.field or not args.mask or not args.out:
                raise ValueError("train needs --field, --mask, and --out")
            return _train(args)
        if not args.unet or not args.out_dir or not args.field:
            raise ValueError("predict needs --field, --unet, and --out-dir")
        return _predict(args)
    except Exception as exc:
        traceback.print_exc()
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
