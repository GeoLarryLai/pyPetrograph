# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
#
# Portions derived from segmenteverygrain (https://github.com/zsylvester/segmenteverygrain),
# Copyright Zoltan Sylvester and contributors, licensed under the Apache License 2.0.
# Modified here: batched U-Net tiles, bounding-box mask post-processing, edge-check fix.
"""
Faster drop-in replacements for three segmenteverygrain functions, used by ``seg_worker``.

* ``predict_image``: same tiles, weights and summation order as upstream, but the U-Net
  runs on ``BATCH`` tiles per call (identical output; about 3-4x faster on CPU).
* ``one_point_prompt``: post-processing (largest piece + contour) runs on the bounding box
  of the *full* SAM mask, padded by ``PAD`` px of zeros, so a grain is never cut by the crop.
  Upstream checked ``mask[0, -1]`` (one pixel) instead of the right column and shifted the
  contour by a guess; here every contour is traced on a zero-padded crop and shifted back
  exactly, so masks touching any patch edge are handled the same way.
* ``collect_polygon_from_mask``: label sizes with one ``np.bincount`` (same test as upstream).

``apply()`` patches the imported ``segmenteverygrain.segmenteverygrain`` module. Set the env
var ``PYPETROGRAPH_SEG_FAST=0`` to keep the original upstream code.
"""
from __future__ import annotations

import os

import numpy as np

BATCH = 32
PAD = 2


def enabled() -> bool:
    return os.environ.get("PYPETROGRAPH_SEG_FAST", "1").strip().lower() not in ("0", "false", "no", "off")


def _tile_jobs(r: int, c: int, I: int, I2: int, W, Wup, Wdown):
    """(row slice, col slice, weight) in upstream ``predict_image`` order."""
    jobs = []
    for i in range(c + 1):
        for j in range(1, 2 * r - 2):
            jobs.append((slice(j * I2, (j + 2) * I2), slice(i * I, (i + 1) * I), W))
    for i in range(c + 1):
        jobs.append((slice(0, 2 * I2), slice(i * I, (i + 1) * I), Wup))
    for i in range(c + 1):
        jobs.append((slice((2 * r - 2) * I2, 2 * r * I2), slice(i * I, (i + 1) * I), Wdown))
    for i in range(c):
        for j in range(1, 2 * r - 2):
            jobs.append((slice(j * I2, (j + 2) * I2), slice(i * I + I2, (i + 1) * I + I2), W))
    for i in range(c):
        jobs.append((slice(0, 2 * I2), slice(i * I + I2, (i + 1) * I + I2), Wup))
    for i in range(c):
        jobs.append((slice((2 * r - 2) * I2, 2 * r * I2), slice(i * I + I2, (i + 1) * I + I2), Wdown))
    return jobs


def tile_count(h: int, w: int, I: int = 256) -> int:
    """Tiles ``predict_image`` runs on an h x w image: len(_tile_jobs(...))."""
    r, c = h // I + 1, w // I + 1
    return (2 * c + 1) * (2 * r - 1)


def predict_image(image, model, I):
    """Batched version of segmenteverygrain ``predict_image`` (same result)."""
    from scipy.special import softmax

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D or 3D array.")
    if image.ndim == 3 and image.shape[2] not in [1, 3]:
        raise ValueError("3D input image must have 1 or 3 channels.")
    pad_rows = I - np.mod(image.shape[0], I)
    pad_cols = I - np.mod(image.shape[1], I)
    if image.ndim == 2:
        image = np.stack((image, image, image), axis=-1)
    image = np.vstack((image, np.zeros((pad_rows, image.shape[1], image.shape[2]))))
    image = np.hstack((image, np.zeros((image.shape[0], pad_cols, image.shape[2]))))
    r = int(np.floor(image.shape[0] / I))
    c = int(np.floor(image.shape[1] / I))
    I2 = int(I / 2)
    W = np.hanning(I) * np.hanning(I)[:, np.newaxis]
    Wup = W.copy()
    Wup[:I2, :] = np.tile(np.hanning(I), (I2, 1))
    Wdown = W.copy()
    Wdown[I2:, :] = np.tile(np.hanning(I), (I2, 1))
    image = np.hstack((np.zeros((r * I, I2, 3)), image, np.zeros((r * I, I2, 3))))
    image_pred = np.zeros((image.shape[0], image.shape[1], 3))
    jobs = _tile_jobs(r, c, I, I2, W, Wup, Wdown)
    for k in range(0, len(jobs), BATCH):
        chunk = jobs[k : k + BATCH]
        batch = np.stack([image[rs, cs, :] / 255.0 for rs, cs, _w in chunk])
        pred = np.asarray(model.predict(batch, batch_size=len(chunk), verbose=0))
        pred = softmax(pred, axis=-1)
        for (rs, cs, w), p in zip(chunk, pred):
            for layer in range(3):
                image_pred[rs, cs, layer] += p[:, :, layer] * w
    image_pred = image_pred[:, I2:-I2, :]
    image_pred = image_pred[:-pad_rows, :-pad_cols, :]
    return image_pred


def one_point_prompt(x, y, image, predictor, ax=False):
    """segmenteverygrain ``one_point_prompt`` with bounding-box post-processing and the edge fix."""
    from skimage import measure

    masks, scores, _logits = predictor.predict(
        point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True
    )
    h, w = image.shape[:2]
    if masks.ndim == 2:
        masks = masks[None]
    keep = [k for k in range(len(masks)) if np.count_nonzero(masks[k]) / (h * w) <= 0.5]
    if not keep:
        return [], [], np.zeros_like(image[:, :, 0])
    ind = max(keep, key=lambda k: scores[k])
    mask = np.asarray(masks[ind])
    if mask.dtype != bool:
        mask = mask > 0
    ys = np.flatnonzero(mask.any(axis=1))
    if len(ys) == 0:
        return [], [], mask
    xs = np.flatnonzero(mask.any(axis=0))
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    sub = mask[y0:y1, x0:x1]  # view into the full mask: edits below apply to it
    lab, n = measure.label(sub, return_num=True, connectivity=1)
    if n > 1:
        counts = np.bincount(lab.ravel())
        largest = np.argmax(counts[1:]) + 1
        sub[lab != largest] = 0
    contours = measure.find_contours(np.pad(sub, PAD, mode="constant"), 0.5)
    if not contours:
        return [], [], mask
    sx = contours[0][:, 1] + x0 - PAD
    sy = contours[0][:, 0] + y0 - PAD
    if ax:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        ax.fill(sx, sy, facecolor=color, edgecolor="k", alpha=0.5)
    return sx, sy, mask


def collect_polygon_from_mask(
    labels, mask, image_pred, all_grains, sx, sy, min_area=100, max_n_large_grains=10, max_bg_fraction=0.7
):
    """segmenteverygrain ``collect_polygon_from_mask`` with one ``np.bincount`` (same test)."""
    from shapely.geometry import Polygon

    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask > 0
    lm = labels[mask]
    if lm.size == 0:
        return all_grains
    n_large = int(np.count_nonzero(np.bincount(lm.ravel()) >= min_area))
    if n_large < max_n_large_grains and np.mean(image_pred[:, :, 0][mask]) < max_bg_fraction:
        poly = Polygon(np.vstack((sx, sy)).T)
        if not poly.is_valid:
            poly = poly.buffer(0)
        all_grains.append(poly)
    return all_grains


def apply(module=None) -> bool:
    """Patch segmenteverygrain in place unless ``PYPETROGRAPH_SEG_FAST=0``. Returns True if patched."""
    if not enabled():
        return False
    if module is None:
        from segmenteverygrain import segmenteverygrain as module
    module.predict_image = predict_image
    module.one_point_prompt = one_point_prompt
    module.collect_polygon_from_mask = collect_polygon_from_mask
    return True
