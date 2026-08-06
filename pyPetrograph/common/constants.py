"""Shared constants and type aliases for pyPetrograph."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".geotiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
}

DEFAULT_Y_TARGET = 128.0
DEFAULT_N_JOBS = 4
DEFAULT_MEDIAN_SIZE = 4
DEFAULT_RF_TREES = 50
DEFAULT_CHUNK_PIXELS = 1_000_000
DEFAULT_SURE_PROBA = 0.8  # "sure" fraction: winning class & max proba ≥ this
DEFAULT_POLY_FILL_ALPHA = 0.35  # label-summary / pred figure fills
LEGEND_FONTSIZE = 10
LEGEND_MAX_PER_ROW = 10
# HSV golden-angle step → bright, well-separated class colors (not turbo/rainbow)
_CLASS_COLOR_GOLDEN = 0.618033988749895
RANDOM_STATE = 42
COORD_DECIMALS = 1
DEFAULT_WAND_THRESHOLD = 70.0
DEFAULT_WAND_SIMPLIFY = 1.5
WAND_SLIDER_MIN = 10.0
WAND_SLIDER_MAX = 150.0
LOD_MAX_SIDE = 2048
DEFAULT_PREVIEW_TARGET_MP = 2.0  # used only if screen size cannot be read
DEFAULT_PREVIEW_DPI = 150
DEFAULT_FIGURE_DPI = 300
DEFAULT_LOAD_TARGET_MP: Optional[float] = None  # None = keep native size (after max_side)
DEFAULT_SCREEN_VIEW_FILL = 0.85  # fraction of screen pixels for UI/preview overview
DEFAULT_TEXTURE_WINDOW = 7
DEFAULT_LBP_P = 8
DEFAULT_LBP_R = 1.0
MODELS_SUBDIR = "models"
PREDICTIONS_SUBDIR = "predictions"
LABELS_SUBDIR = "labels"
CACHE_SUBDIR = "cache"  # legacy disk folder name; no longer written
UNIVERSAL_MODEL_NAME = "universal_model.joblib"

METHODS = ("lightgbm", "random_forest", "xgboost")
DEFAULT_METHOD = "lightgbm"
DEFAULT_CV_FOLDS = 5

# Feature toggles for train / predict / preview (GLCM = polygon-level only).
# Missing keys resolve to False — turn on only what you need in the notebook.
DEFAULT_FEATURE_TOGGLES: Dict[str, bool] = {
    "r": False,
    "g": False,
    "b": False,
    "y": False,
    "local_std": False,
    "local_grad": False,
    "lbp": False,
    "glcm": False,
}
GLCM_PROP_NAMES = ("contrast", "homogeneity")
FEATURE_PREVIEW_CMAPS: Dict[str, str] = {
    "r": "Reds",
    "g": "Greens",
    "b": "Blues",
    "y": "gray",
    "local_std": "viridis",
    "local_grad": "plasma",
    "lbp": "inferno",
    "glcm_contrast": "cividis",
    "glcm_homogeneity": "turbo",
}

# (import_name, pip_name) — checked/installed by check_and_install_packages
REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("geopandas", "geopandas"),
    ("shapely", "shapely"),
    ("rasterio", "rasterio"),
    ("affine", "affine"),
    ("joblib", "joblib"),
    ("pandas", "pandas"),
    ("lightgbm", "lightgbm"),
    ("skimage", "scikit-image"),
    ("PyQt5", "PyQt5"),  # label window (QtAgg); prefer conda-forge pyqt if pip fails
]

DEFAULT_CLASSES: Dict[int, str] = {
    1: "olivine",
    2: "clinopyroxene",
    3: "orthopyroxene",
    4: "chromium_spinel",
    5: "nickel_sulfide",
    6: "magnetite",
    7: "apatite",
    8: "holes",
}

PathLike = Union[str, Path]
