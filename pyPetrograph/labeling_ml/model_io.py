"""Model bundle save/load (joblib)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib

from pyPetrograph.common.constants import PathLike

def save_model_bundle(path: PathLike, bundle: Dict[str, Any]) -> Path:
    """Save trained model + metadata with joblib."""
    path = Path(path)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: PathLike) -> Dict[str, Any]:
    """Load a joblib model bundle."""
    return joblib.load(Path(path))
