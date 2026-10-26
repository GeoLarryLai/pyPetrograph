"""
AMICS workflow (notebook ``pyPetrograph_4_AMICSworkflow.ipynb``): legend OCR,
BSE grain outlines (SEG in a subprocess), long / short axes, mineral vote per
grain, size curves, CSVs, knolling image.

Kernel-safe: nothing here imports TensorFlow, SEG, SAM2, torch, or Qt.
``seg_worker.py`` and ``window_worker.py`` are subprocess scripts (not imported).
"""
from pyPetrograph.amics.bse import (
    dark_mask,
    detect_footer,
    load_bse,
    plot_threshold_preview,
    read_footer_pixelsize,
    stretch,
    threshold_from_points,
    valid_mask,
)
from pyPetrograph.amics.export import (
    MINERAL_CSV_COLUMNS,
    read_grains_geojson,
    write_grains_geojson,
    write_mineral_csvs,
)
from pyPetrograph.amics.grain_filter import clean_polygons
from pyPetrograph.amics.knolling import grain_tile, make_knolling, save_knolling
from pyPetrograph.amics.legend_read import (
    legend_to_text,
    name_key,
    ocr_names,
    read_legend,
    read_legend_rows,
)
from pyPetrograph.amics.measure import (
    MEASURE_COLUMNS,
    add_size_metrics,
    ellipse_axes,
    measure_grains,
    polygon_moments,
)
from pyPetrograph.amics.mineral import (
    DEFAULT_SKIP_NAMES,
    VOTE_COLUMNS,
    count_table,
    load_mineral_map,
    match_map_colors,
    plot_composition,
    plot_missing_colors,
    plot_mixed,
    plot_overlay,
    plot_unmapped,
    recolor_from_matches,
    vote_grains,
)
from pyPetrograph.amics.naming import (
    PATH_SUFFIXES,
    amics_paths,
    mineral_csv,
    mineral_gsd_png,
    safe_name,
    sample_name,
)
from pyPetrograph.amics.plots import plot_gsd, plot_gsd_by_mineral, plot_legend, save_figure
from pyPetrograph.amics.seg_run import run_seg_bse
from pyPetrograph.amics.windows import UNIT_TO_UM, click_points, draw_scale, has_screen, pick_amics_files, pick_file

__all__ = [
    "DEFAULT_SKIP_NAMES",
    "MEASURE_COLUMNS",
    "MINERAL_CSV_COLUMNS",
    "PATH_SUFFIXES",
    "UNIT_TO_UM",
    "VOTE_COLUMNS",
    "add_size_metrics",
    "amics_paths",
    "clean_polygons",
    "click_points",
    "count_table",
    "dark_mask",
    "detect_footer",
    "draw_scale",
    "ellipse_axes",
    "grain_tile",
    "has_screen",
    "legend_to_text",
    "load_bse",
    "load_mineral_map",
    "make_knolling",
    "match_map_colors",
    "measure_grains",
    "mineral_csv",
    "mineral_gsd_png",
    "name_key",
    "ocr_names",
    "pick_amics_files",
    "pick_file",
    "plot_composition",
    "plot_gsd",
    "plot_gsd_by_mineral",
    "plot_legend",
    "plot_missing_colors",
    "plot_mixed",
    "plot_overlay",
    "plot_threshold_preview",
    "plot_unmapped",
    "polygon_moments",
    "read_footer_pixelsize",
    "recolor_from_matches",
    "read_grains_geojson",
    "read_legend",
    "read_legend_rows",
    "run_seg_bse",
    "safe_name",
    "sample_name",
    "save_figure",
    "save_knolling",
    "stretch",
    "threshold_from_points",
    "valid_mask",
    "vote_grains",
    "write_grains_geojson",
    "write_mineral_csvs",
]
