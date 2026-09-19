"""Embedding field from an aligned Scene. Default C = small autoencoder; B = frozen net."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from pyPetrograph.align.scene import Scene
from pyPetrograph.common.constants import DEFAULT_EMBED_FEATURE_TOGGLES
from pyPetrograph.image_processing.features import build_feature_stack, resolve_feature_toggles

NAMED_SLOTS: Tuple[str, ...] = ("ppl", "xpl", "bse")


def _ordered_slots(scene: Scene) -> List[str]:
    slots = [s for s in NAMED_SLOTS if s in scene.layers]
    if scene.working_slot not in slots:
        slots = [scene.working_slot] + slots
    seen = set()
    ordered: List[str] = []
    for s in slots:
        if s not in seen and s in scene.layers:
            seen.add(s)
            ordered.append(s)
    return ordered


def stack_named_slots(scene: Scene) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Working-grid RGB stack and per-slot coverage (legacy RGB-only helper).

    Returns
    -------
    stack : float32 HxWx(3*n_slots), RGB in [0, 1]
    valid : bool HxWx n_slots — False where that measurement is missing (skip, not fake)
    names : slot names in order
    """
    h, w = scene.working_hw()
    ordered = _ordered_slots(scene)
    ch = np.zeros((h, w, 3 * len(ordered)), dtype=np.float32)
    valid = np.zeros((h, w, len(ordered)), dtype=bool)
    for i, slot in enumerate(ordered):
        rgb, mask = scene.warp_layer(slot)
        ch[:, :, i * 3 : (i + 1) * 3] = rgb.astype(np.float32) / 255.0
        valid[:, :, i] = mask
        ch[:, :, i * 3 : (i + 1) * 3][~mask] = 0.0
    return ch, valid, ordered


def stack_feature_slots(
    scene: Scene,
    *,
    feature_toggles: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Per-slot feature stacks on the working grid (shared with RGB notebook).

    For each aligned slot, call ``build_feature_stack`` (Y-norm on RGB, then
    toggled maps). Prefix channel names with ``ppl_``, ``xpl_``, ``bse_``.
    Missing / out-of-window pixels are zeroed and marked invalid.

    Returns
    -------
    stack : float32 HxWxC
    valid : bool HxW — True if any slot covers the pixel
    slot_names : slot order
    input_names : channel names (e.g. ``ppl_r``, ``xpl_local_grad``)
    """
    toggles = resolve_feature_toggles(
        feature_toggles if feature_toggles is not None else DEFAULT_EMBED_FEATURE_TOGGLES
    )
    # Embed never invents GLCM without polygons — force off.
    toggles["glcm"] = False
    ordered = _ordered_slots(scene)
    h, w = scene.working_hw()
    chunks: List[np.ndarray] = []
    input_names: List[str] = []
    any_valid = np.zeros((h, w), dtype=bool)
    for slot in ordered:
        rgb, mask = scene.warp_layer(slot)
        feat, names = build_feature_stack(
            rgb,
            toggles=toggles,
            apply_norm=True,
            polygons=None,
        )
        feat = np.asarray(feat, dtype=np.float32)
        feat[~mask] = 0.0
        chunks.append(feat)
        input_names.extend([f"{slot}_{n}" for n in names])
        any_valid |= mask
    if not chunks:
        raise RuntimeError("No layers in scene to embed.")
    stack = np.concatenate(chunks, axis=-1).astype(np.float32)
    return stack, any_valid, ordered, input_names


def _mlp_hidden(mlp: MLPRegressor, X: np.ndarray) -> np.ndarray:
    a = X
    # all hidden layers; last coefs_ map hidden → reconstruct
    for i in range(len(mlp.coefs_) - 1):
        a = np.maximum(0.0, a @ mlp.coefs_[i] + mlp.intercepts_[i])
    return a.astype(np.float32)


def embed_C(
    scene: Scene,
    *,
    hidden: int = 16,
    max_pixels: int = 80_000,
    random_state: int = 42,
    feature_toggles: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Small autoencoder on per-slot feature stacks (CPU MLP).

    Inputs come from ``build_feature_stack`` (shared with the RGB notebook).
    Default toggles: r/g/b + std 7/21/63 + grad + LBP r=1/3/5 + Gabor 0/45/90
    + Hessian ridge + entropy 7/21/63; no Y, no GLCM.

    Returns HxWxhidden float32 and a small info dict (includes ``input_names``).
    """
    stack, any_valid, slot_names, input_names = stack_feature_slots(
        scene, feature_toggles=feature_toggles
    )
    h, w, c = stack.shape
    X = stack.reshape(-1, c)
    idx = np.where(any_valid.ravel())[0]
    if len(idx) == 0:
        raise RuntimeError("No valid pixels to embed.")
    rng = np.random.RandomState(random_state)
    if len(idx) > max_pixels:
        idx = rng.choice(idx, size=max_pixels, replace=False)
    # Scale inputs to similar ranges so LBP codes and RGB share the MLP.
    mu = X[idx].mean(axis=0)
    sd = X[idx].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    mu = mu.astype(np.float32)

    def _z(block: np.ndarray) -> np.ndarray:
        return (block - mu) / sd

    mlp = MLPRegressor(
        hidden_layer_sizes=(int(hidden),),
        activation="relu",
        solver="adam",
        max_iter=80,
        random_state=random_state,
        verbose=False,
    )
    mlp.fit(_z(X[idx]), _z(X[idx]))
    emb = np.zeros((h * w, int(hidden)), dtype=np.float32)
    step = 200_000
    for s in range(0, h * w, step):
        e = min(h * w, s + step)
        emb[s:e] = _mlp_hidden(mlp, _z(X[s:e]))
    emb[~any_valid.ravel()] = 0.0
    field = emb.reshape(h, w, int(hidden))
    info: Dict[str, object] = {
        "method": "C",
        "slots": slot_names,
        "hidden": int(hidden),
        "input_names": input_names,
        "n_input": int(c),
        "feature_toggles": resolve_feature_toggles(
            feature_toggles if feature_toggles is not None else DEFAULT_EMBED_FEATURE_TOGGLES
        ),
    }
    return field, info


def embed_B(
    scene: Scene,
    *,
    model_name: str = "dinov2_vits14",
    feature_toggles: Optional[Dict[str, bool]] = None,  # unused; RGB-net path
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Frozen pretrained net per available layer; skip missing pixels (no fake BSE).

    Requires ``torch``. Each available RGB layer is encoded; votes are summed
    only where that layer is valid, then L2-normalized. Output length is fixed.
    ``feature_toggles`` is accepted for API symmetry with ``embed_C`` but ignored
    (DINOv2 expects RGB).
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "embed_B needs PyTorch. In conda env work: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc

    stack, valid, names = stack_named_slots(scene)
    h, w, _ = stack.shape
    dinov2 = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    dinov2.eval()
    votes = None
    weight = np.zeros((h, w), dtype=np.float32)
    for i, slot in enumerate(names):
        rgb = stack[:, :, i * 3 : (i + 1) * 3]
        mask = valid[:, :, i]
        if not mask.any():
            continue
        t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float()
        # DINOv2 expects ImageNet-sized patches; interpolate to multiple of 14
        ht = int(np.ceil(h / 14) * 14)
        wt = int(np.ceil(w / 14) * 14)
        t = F.interpolate(t, size=(ht, wt), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            out = dinov2.forward_features(t)
            feat = out["x_norm_patchtokens"]  # 1, N, D
            d = feat.shape[-1]
            gh, gw = ht // 14, wt // 14
            fmap = feat.reshape(1, gh, gw, d).permute(0, 3, 1, 2)
            fmap = F.interpolate(fmap, size=(h, w), mode="bilinear", align_corners=False)
            arr = fmap[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if votes is None:
            votes = np.zeros((h, w, d), dtype=np.float32)
        votes[mask] += arr[mask]
        weight[mask] += 1.0
    if votes is None:
        raise RuntimeError("No valid pixels to embed.")
    w3 = np.maximum(weight, 1e-6)[:, :, None]
    field = votes / w3
    nrm = np.linalg.norm(field, axis=-1, keepdims=True) + 1e-6
    field = field / nrm
    return field, {
        "method": "B",
        "slots": names,
        "model": model_name,
        "input_names": [f"{s}_rgb" for s in names],
    }


def embedding_to_rgb(field: np.ndarray, *, max_fit: int = 40_000) -> np.ndarray:
    """False-color view of an embedding field (first 3 PCA axes). Preview only."""
    h, w, d = field.shape
    x = np.asarray(field, dtype=np.float32).reshape(-1, d)
    n = int(x.shape[0])
    k = min(3, d)
    rng = np.random.RandomState(0)
    take = min(int(max_fit), n)
    idx = rng.choice(n, size=take, replace=False) if n > take else np.arange(n)
    pca = PCA(n_components=k, random_state=0)
    try:
        pca.set_params(svd_solver="covariance_eigh")
    except ValueError:
        pca.set_params(svd_solver="randomized")
    pca.fit(x[idx])
    vis = np.zeros((n, 3), dtype=np.float32)
    step = 200_000
    for s in range(0, n, step):
        e = min(n, s + step)
        vis[s:e, :k] = pca.transform(x[s:e]).astype(np.float32)
    for c in range(k):
        col = vis[:, c]
        lo, hi = np.percentile(col, (2, 98))
        if hi > lo:
            vis[:, c] = np.clip((col - lo) / (hi - lo), 0, 1)
    return np.clip(vis.reshape(h, w, 3) * 255.0, 0, 255).astype(np.uint8)


def embed_scene(scene: Scene, method: str = "C", **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run embedding. ``method`` is ``C`` (default autoencoder) or ``B`` (frozen net).

    Returns (field, rgb_view, info).
    """
    m = str(method).upper()
    if m == "B":
        field, info = embed_B(scene, **kwargs)
    else:
        field, info = embed_C(scene, **kwargs)
    return field, embedding_to_rgb(field), info
