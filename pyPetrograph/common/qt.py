"""One Qt for pop-ups and the app: PySide6-Essentials (not a default dependency)."""
from __future__ import annotations

import importlib.util
from typing import Dict, MutableMapping

QT_INSTALL_HINT = "pip install PySide6-Essentials  (~350 MB)"


def has_qt_binding() -> bool:
    return any(importlib.util.find_spec(m) is not None for m in ("PySide6", "PyQt6", "PyQt5"))


def require_qt(what: str = "this pop-up window") -> None:
    """Raise a clear ImportError when no Qt binding is installed."""
    if not has_qt_binding():
        raise ImportError(f"{what} needs Qt. In your activated env:  {QT_INSTALL_HINT}")


def qt_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Point a QtAgg subprocess at PySide6 when it is installed (unless QT_API is set)."""
    if "QT_API" not in env and importlib.util.find_spec("PySide6") is not None:
        env["QT_API"] = "pyside6"
    return env
