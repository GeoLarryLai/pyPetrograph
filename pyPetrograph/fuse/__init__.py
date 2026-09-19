"""Stage 2: physics channel stack (PPL RGB + XPL extinction fit + BSE) from an aligned Scene."""

from pyPetrograph.fuse.extinction import MIN_ANGLES, fit_extinction, phi_deg_from_cos_sin
from pyPetrograph.fuse.preview import run_channel_preview
from pyPetrograph.fuse.stack import (
    DEFAULT_CHANNEL_TOGGLES,
    ChannelStack,
    build_channel_stack,
    channel_set_id,
    channels_paths,
    load_channel_stack,
    save_channel_stack,
)
from pyPetrograph.fuse.tiles import (
    add_axioscan_layers,
    apply_tiled,
    axioscan_layers,
    axioscan_scale_folder,
    iter_tiles,
)

__all__ = [
    "fit_extinction",
    "phi_deg_from_cos_sin",
    "MIN_ANGLES",
    "ChannelStack",
    "DEFAULT_CHANNEL_TOGGLES",
    "build_channel_stack",
    "channel_set_id",
    "channels_paths",
    "save_channel_stack",
    "load_channel_stack",
    "iter_tiles",
    "apply_tiled",
    "axioscan_layers",
    "axioscan_scale_folder",
    "add_axioscan_layers",
    "run_channel_preview",
]
