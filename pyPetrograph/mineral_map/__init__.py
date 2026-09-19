"""Flat-color mineral maps: legend snap → grains → per-class counts with uncertainty."""

from pyPetrograph.mineral_map.counts import (
    COUNT_COLS,
    count_grains,
    format_counts,
    pivot_counts,
    wilson_interval,
)
from pyPetrograph.mineral_map.grains import (
    absorb_class_specks,
    absorb_specks,
    class_rgb,
    grain_table,
    grains_to_geojson,
    label_grains,
)
from pyPetrograph.mineral_map.legend import (
    UNLISTED_NAME,
    UNKNOWN_NAME,
    check_legend,
    class_names,
    legend_from_image,
    read_legend_swatches,
    read_rgba,
    snap_to_legend,
    unknown_code,
)
from pyPetrograph.mineral_map.run import (
    MINERALMAP_SUBDIR,
    mineralmap_paths,
    process_folder,
    process_image,
)

__all__ = [
    "UNLISTED_NAME",
    "UNKNOWN_NAME",
    "unknown_code",
    "MINERALMAP_SUBDIR",
    "COUNT_COLS",
    "read_rgba",
    "read_legend_swatches",
    "legend_from_image",
    "check_legend",
    "class_names",
    "snap_to_legend",
    "absorb_specks",
    "absorb_class_specks",
    "label_grains",
    "grain_table",
    "grains_to_geojson",
    "class_rgb",
    "count_grains",
    "wilson_interval",
    "pivot_counts",
    "format_counts",
    "mineralmap_paths",
    "process_image",
    "process_folder",
]
