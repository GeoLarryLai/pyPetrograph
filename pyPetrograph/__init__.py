"""
RGB petrography modal-mineralogy toolkit.

Load thin-section images, draw training polygons, train RF / LightGBM / XGBoost,
predict class maps, and report area fractions.
"""

from __future__ import annotations

from pyPetrograph.common.constants import (
    CACHE_SUBDIR,
    DEFAULT_CLASSES,
    DEFAULT_FEATURE_TOGGLES,
    DEFAULT_FIGURE_DPI,
    DEFAULT_METHOD,
    DEFAULT_N_JOBS,
    DEFAULT_PREVIEW_DPI,
    DEFAULT_PREVIEW_TARGET_MP,
    DEFAULT_SURE_PROBA,
    DEFAULT_Y_TARGET,
    LABELS_SUBDIR,
    METHODS,
    MODELS_SUBDIR,
    PREDICTIONS_SUBDIR,
    REQUIRED_PACKAGES,
    SUPPORTED_EXTENSIONS,
    UNIVERSAL_MODEL_NAME,
)
from pyPetrograph.common.image_io import load_image
from pyPetrograph.common.notebook_runners import (
    print_label_summary,
    print_train_metrics,
    run_feature_preview,
    run_predict_cell,
    run_train_cell,
)
from pyPetrograph.common.paths import (
    check_and_install_packages,
    cleanup_legacy_outputs,
    relpath_display,
    stem_paths,
)
from pyPetrograph.common.session import Session
from pyPetrograph.common.ui import (
    build_ui,
    launch_app,
    select_images,
    # private subprocess entrypoints re-exported for child processes
    _run_label_gui_from_config,
    _run_qt_picker_from_config,
    launch_app_inplace,
)
from pyPetrograph.image_processing.brightness import normalize_brightness, relative_luminance
from pyPetrograph.image_processing.features import (
    build_feature_stack,
    downsample_to_approx_mp,
    resolve_view_target_mp,
    screen_view_target_mp,
)
from pyPetrograph.image_processing.fractions import compute_fractions, compute_fractions_uncertain
from pyPetrograph.labeling_ml.polygons import magic_wand_to_polygon
from pyPetrograph.labeling_ml.predict import predict_image, predict_image_proba

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_Y_TARGET",
    "DEFAULT_N_JOBS",
    "DEFAULT_METHOD",
    "METHODS",
    "DEFAULT_CLASSES",
    "REQUIRED_PACKAGES",
    "MODELS_SUBDIR",
    "PREDICTIONS_SUBDIR",
    "LABELS_SUBDIR",
    "CACHE_SUBDIR",
    "UNIVERSAL_MODEL_NAME",
    "stem_paths",
    "relpath_display",
    "cleanup_legacy_outputs",
    "load_image",
    "normalize_brightness",
    "relative_luminance",
    "DEFAULT_FEATURE_TOGGLES",
    "DEFAULT_PREVIEW_TARGET_MP",
    "DEFAULT_PREVIEW_DPI",
    "DEFAULT_FIGURE_DPI",
    "check_and_install_packages",
    "magic_wand_to_polygon",
    "build_feature_stack",
    "downsample_to_approx_mp",
    "screen_view_target_mp",
    "resolve_view_target_mp",
    "Session",
    "select_images",
    "build_ui",
    "launch_app",
    "print_label_summary",
    "run_feature_preview",
    "run_train_cell",
    "print_train_metrics",
    "run_predict_cell",
    "predict_image",
    "predict_image_proba",
    "compute_fractions",
    "compute_fractions_uncertain",
    "DEFAULT_SURE_PROBA",
]
