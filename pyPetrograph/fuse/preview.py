"""
Inline QC preview of a stage-2 ``ChannelStack``.

One panel per channel in a 3-column grid (colormap by channel kind, 2–98 %
color range over the valid pixels) with a composite row on top: primary true
color, extinction angle from ``xpl_phi_cos`` / ``xpl_phi_sin``, and ``xpl_max``.
Meant for ``%matplotlib inline`` notebooks; optionally saved beside the image as
``channels/{stem}_channels_preview.png``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyPetrograph.common.constants import (
    DEFAULT_FIGURE_DPI,
    DEFAULT_PREVIEW_DPI,
    PathLike,
    clip_figure_dpi,
)
from pyPetrograph.common.paths import relpath_display, stem_paths
from pyPetrograph.fuse.extinction import phi_deg_from_cos_sin
from pyPetrograph.fuse.stack import (
    KIND_FEATURE,
    KIND_GRAY255,
    KIND_POS255,
    KIND_SIGNED,
    ChannelStack,
)
from pyPetrograph.image_processing.features import downsample_to_approx_mp, resolve_view_target_mp

_NCOLS = 3
_COLORBAR_FRACTION = 0.046
_RANGE_PERCENTILES = (2.0, 98.0)
_AMP_MASK_PERCENTILE = 5.0
_RGB_SUFFIX_CMAPS: Dict[str, str] = {"_r": "Reds", "_g": "Greens", "_b": "Blues"}
_KIND_CMAPS: Dict[str, str] = {
    KIND_GRAY255: "gray",
    KIND_POS255: "viridis",
    KIND_SIGNED: "twilight",
    KIND_FEATURE: "magma",
}


def _channel_cmap(name: str, kind: str) -> str:
    """``*_r/_g/_b`` → Reds/Greens/Blues; otherwise by value kind."""
    for suffix, cmap in _RGB_SUFFIX_CMAPS.items():
        if name.endswith(suffix):
            return cmap
    return _KIND_CMAPS.get(kind, "viridis")


def _resize_float(ch: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of one float map to ``(h, w)``; always returns a new float32 copy."""
    from PIL import Image

    arr = np.array(ch, dtype=np.float32)  # copy: never touch cs.stack
    h, w = int(hw[0]), int(hw[1])
    if arr.shape[:2] == (h, w):
        return arr
    return np.asarray(
        Image.fromarray(np.ascontiguousarray(arr)).resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _valid_range(ch: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    """2nd–98th percentile of the valid pixels (min/max fallback when flat)."""
    vals = ch[valid]
    if vals.size == 0:
        vals = ch.ravel()
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, _RANGE_PERCENTILES)
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        lo, hi = float(vals.min()), float(vals.max())
        if not hi > lo:
            hi = lo + 1.0
    return lo, hi


def run_channel_preview(
    cs: ChannelStack,
    *,
    image_path: Optional[PathLike] = None,
    view_mp: Optional[float] = None,
    dpi: int = DEFAULT_PREVIEW_DPI,
    figure_dpi: int = DEFAULT_FIGURE_DPI,
) -> None:
    """
    Show the channel stack inline (``%matplotlib inline``): composite row + one panel per channel.

    Parameters
    ----------
    cs : ChannelStack from ``build_channel_stack`` / ``load_channel_stack``
    image_path : if given, the figure is also saved as
        ``channels/{stem}_channels_preview.png`` beside that image
    view_mp : display megapixels (default: sized from the screen)
    dpi : inline figure dpi
    figure_dpi : dpi of the saved PNG
    """
    import matplotlib.pyplot as plt

    print(cs.summary())
    mp = resolve_view_target_mp(view_mp)
    dpi = clip_figure_dpi(dpi)

    # Display size from the shared helper (same rule as the other previews). The
    # float channels are then resized to that size without a uint8 round trip.
    valid_small = downsample_to_approx_mp(cs.valid.astype(np.uint8) * 255, mp) > 127
    hw = (int(valid_small.shape[0]), int(valid_small.shape[1]))

    def _small(idx: int) -> np.ndarray:
        out = _resize_float(cs.stack[..., idx], hw)
        out[~valid_small] = 0.0
        return out

    # --- composite row: (title, image, imshow kwargs, colorbar?) ---
    panels: List[Tuple[str, np.ndarray, Dict[str, Any], bool]] = []
    rgb_idx = cs.primary_rgb_indices()
    if len(rgb_idx) == 3:
        rgb = np.clip(cs.stack[..., rgb_idx], 0, 255).astype(np.uint8)
        prim = str(cs.meta.get("primary_kind") or "ppl")
        panels.append((f"{prim} true color", downsample_to_approx_mp(rgb, mp), {}, False))
        del rgb
    if "xpl_phi_cos" in cs.names and "xpl_phi_sin" in cs.names:
        phi = phi_deg_from_cos_sin(_small(cs.index("xpl_phi_cos")), _small(cs.index("xpl_phi_sin")))
        hide = ~valid_small
        if "xpl_amp" in cs.names and valid_small.any():
            amp = _small(cs.index("xpl_amp"))
            thr = float(np.percentile(amp[valid_small], _AMP_MASK_PERCENTILE))
            hide |= amp < thr
        phi[hide] = np.nan
        panels.append(("extinction angle (°)", phi, {"cmap": "hsv", "vmin": 0.0, "vmax": 90.0}, True))
    if "xpl_max" in cs.names:
        mx = _small(cs.index("xpl_max"))
        vmin, vmax = _valid_range(mx, valid_small)
        panels.append(("xpl_max", mx, {"cmap": "gray", "vmin": vmin, "vmax": vmax}, True))

    # --- figure: optional composite row, then the channel grid ---
    n_ch = cs.n_channels
    n_grid_rows = max(1, math.ceil(n_ch / _NCOLS))
    top_rows = 1 if panels else 0
    nrows = n_grid_rows + top_rows
    fig, axes = plt.subplots(nrows, _NCOLS, figsize=(4.5 * _NCOLS, 3.4 * nrows), dpi=dpi)
    axes_arr = np.atleast_2d(axes)

    if panels:
        for c in range(_NCOLS):
            ax = axes_arr[0, c]
            if c >= len(panels):
                ax.axis("off")
                continue
            title, img, kw, cbar = panels[c]
            im = ax.imshow(img, **kw)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            if cbar:
                fig.colorbar(im, ax=ax, fraction=_COLORBAR_FRACTION, pad=0.04)

    for i, (name, kind) in enumerate(zip(cs.names, cs.kinds)):
        r, c = divmod(i, _NCOLS)
        ax = axes_arr[top_rows + r, c]
        small = _small(i)
        vmin, vmax = _valid_range(small, valid_small)
        im = ax.imshow(small, cmap=_channel_cmap(name, kind), vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=_COLORBAR_FRACTION, pad=0.04)
    for j in range(n_ch, n_grid_rows * _NCOLS):
        r, c = divmod(j, _NCOLS)
        axes_arr[top_rows + r, c].axis("off")

    fig.suptitle(f"channel set: {cs.channel_set_id}", fontsize=12)
    fig.tight_layout()

    if image_path is not None:
        sp = stem_paths(image_path)
        out_png = sp.channels_dir / f"{sp.stem}_channels_preview.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_png,
            dpi=clip_figure_dpi(figure_dpi),
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"  → {relpath_display(out_png)}")

    plt.show()
