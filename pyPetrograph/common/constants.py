"""Shared constants and type aliases for pyPetrograph."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
DEFAULT_FIGURE_DPI = 150
MAX_FIGURE_DPI = 150


def clip_figure_dpi(dpi: Optional[int] = None) -> int:
    """Clamp a figure / preview dpi into [72, MAX_FIGURE_DPI] (default 150)."""
    d = int(dpi) if dpi not in (None, 0) else DEFAULT_FIGURE_DPI
    return max(72, min(d, MAX_FIGURE_DPI))
DEFAULT_LOAD_TARGET_MP: Optional[float] = None  # None = keep native size (after max_side)
DEFAULT_SCREEN_VIEW_FILL = 0.85  # fraction of screen pixels for UI/preview overview
DEFAULT_TEXTURE_WINDOW = 7
DEFAULT_LBP_P = 8
DEFAULT_LBP_R = 1.0
FEATURE_STD_WINDOWS = (7, 21, 63)
FEATURE_LBP_RADII = (1.0, 3.0, 5.0)
FEATURE_ENTROPY_WINDOWS = (7, 21, 63)
EMBED_UNET_NAME = "embed_unet.keras"  # retired in v0.1; optional extra only
GRAIN_UNET_PREFIX = "grain_unet_"  # models/grain_unet_{channel_set}.keras
DEFAULT_WATERSHED_MIN_DISTANCE = 24
DEFAULT_WATERSHED_MIN_AREA = 40
DEFAULT_OUTLINE_TILE = 256
DEFAULT_OUTLINE_EPOCHS = 15
DEFAULT_OUTLINE_MAX_PATCHES = 80
GRAINS_GEOJSON_SUFFIX = "_grains.geojson"
GRAINS_MASK_SUFFIX = "_grains_mask.png"
MODELS_SUBDIR = "models"
PREDICTIONS_SUBDIR = "predictions"
LABELS_SUBDIR = "labels"
ALIGN_SUBDIR = "align"
CHANNELS_SUBDIR = "channels"  # stage 2: physics channel stack beside the image
OBJECTS_SUBDIR = "objects"  # stages 4–6: object table, labels, classes, stats
CACHE_SUBDIR = "cache"  # legacy disk folder name; no longer written
DEFAULT_ALIGN_POINT_TOLERANCE_PX = 8.0
DEFAULT_ALIGN_SCALE_RANGE = (0.5, 2.0)
DEFAULT_ALIGN_FADE = 0.45
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
    "gabor": False,
    "hessian": False,
    "hessian_aniso": False,
    "entropy": False,
}
# Multilayer embed defaults (no Y, no GLCM).
DEFAULT_EMBED_FEATURE_TOGGLES: Dict[str, bool] = {
    "r": True,
    "g": True,
    "b": True,
    "y": False,
    "local_std": True,
    "local_grad": True,
    "lbp": True,
    "glcm": False,
    "gabor": True,
    "hessian": True,
    "hessian_aniso": False,
    "entropy": True,
}
GLCM_PROP_NAMES = ("contrast", "homogeneity")
DEFAULT_GABOR_FREQUENCY = 0.15
DEFAULT_GABOR_N_ORIENT = 3
DEFAULT_GABOR_ANGLES_DEG = (0, 45, 90)
DEFAULT_HESSIAN_SIGMA = 1.5
FEATURE_PREVIEW_CMAPS: Dict[str, str] = {
    "r": "Reds",
    "g": "Greens",
    "b": "Blues",
    "y": "gray",
    "local_std": "viridis",
    "local_std_7": "viridis",
    "local_std_21": "viridis",
    "local_std_63": "viridis",
    "local_grad": "plasma",
    "lbp": "inferno",
    "lbp_r1": "inferno",
    "lbp_r3": "inferno",
    "lbp_r5": "inferno",
    "glcm_contrast": "cividis",
    "glcm_homogeneity": "turbo",
    "gabor_0": "magma",
    "gabor_45": "magma",
    "gabor_90": "magma",
    "gabor_1": "magma",
    "gabor_2": "magma",
    "gabor_3": "magma",
    "hessian_ridge": "cividis",
    "hessian_aniso": "turbo",
    "entropy_7": "plasma",
    "entropy_21": "plasma",
    "entropy_63": "plasma",
}
FEATURE_DISPLAY_TITLES: Dict[str, str] = {
    "r": "R",
    "g": "G",
    "b": "B",
    "y": "Y",
    "local_std": "Y-std",
    "local_std_7": "Y-std 7",
    "local_std_21": "Y-std 21",
    "local_std_63": "Y-std 63",
    "local_grad": "Y-grad",
    "lbp": "microtexture (LBP)",
    "lbp_r1": "microtexture (LBP r=1)",
    "lbp_r3": "microtexture (LBP r=3)",
    "lbp_r5": "microtexture (LBP r=5)",
    "gabor_0": "orientation 0° (Gabor)",
    "gabor_45": "orientation 45° (Gabor)",
    "gabor_90": "orientation 90° (Gabor)",
    "hessian_ridge": "ridge (Hessian)",
    "hessian_aniso": "directionality (Hessian)",
    "entropy_7": "complexity (entropy 7)",
    "entropy_21": "complexity (entropy 21)",
    "entropy_63": "complexity (entropy 63)",
    "glcm_contrast": "GLCM contrast",
    "glcm_homogeneity": "GLCM homogeneity",
}
FEATURE_PREVIEW_ROWS: Tuple[Tuple[str, ...], ...] = (
    ("r", "g", "b"),
    ("local_std_7", "local_std_21", "local_std_63"),
    ("lbp_r1", "lbp_r3", "lbp_r5"),
    ("gabor_0", "gabor_45", "gabor_90"),
    ("entropy_7", "entropy_21", "entropy_63"),
)

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

# GrainPlot / SEG U-Net workers only. Never import these in the Jupyter kernel.
SEG_PACKAGES: List[Tuple[str, str]] = [
    ("tensorflow", "tensorflow>=2.16,<2.22"),
    ("keras", "keras>=3"),
    ("segmenteverygrain", "segmenteverygrain==0.5.0"),
]

# Optional SAM 2.1 refinement (also pulled by segmenteverygrain). Weights ~860 MB stay opt-in.
SAM2_PACKAGES: List[Tuple[str, str]] = [
    ("sam2", "sam2>=1.0"),
    ("torch", "torch"),
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
