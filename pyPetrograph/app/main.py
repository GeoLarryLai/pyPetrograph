# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Entry point: ``pypetrograph-app``."""
from __future__ import annotations

import os
import sys


# import name -> pip name; the app cannot start without these
REQUIRED = [("PySide6", "PySide6>=6.6"), ("numpy", "numpy"), ("pandas", "pandas"), ("shapely", "shapely"),
            ("matplotlib", "matplotlib"), ("PIL", "pillow"), ("skimage", "scikit-image")]
# optional: the app starts, with a reduced feature
OPTIONAL = [("rapidocr_onnxruntime", "rapidocr_onnxruntime>=1.3", "legend names are not read (swatch_1..N)")]


def check_dependencies(stream=None) -> bool:
    """Print what is missing in *this* Python env and how to install it; False if the app can't start."""
    import importlib.util

    stream = stream or sys.stderr
    missing = [(imp, pipn) for imp, pipn in REQUIRED if importlib.util.find_spec(imp) is None]
    for imp, pipn, effect in OPTIONAL:
        if importlib.util.find_spec(imp) is None:
            how = (f"  install: {sys.executable} -m pip install \"{pipn}\"" if sys.version_info < (3, 13)
                   else "  (no wheels for Python 3.13+ yet; use a Python 3.10-3.12 env for legend OCR)")
            print(f"pyPetrograph app: optional package {pipn!r} is missing: {effect}.\n{how}", file=stream)
    if not missing:
        return True
    names = ", ".join(imp for imp, _ in missing)
    print(
        f"pyPetrograph app: missing package(s) in this Python environment: {names}\n"
        f"  environment: {sys.prefix}\n"
        "Install them into this same environment, then launch again:\n"
        f"  {sys.executable} -m pip install \"pyPetrograph[app]\"\n"
        "  (from a repo checkout: pip install -e \".[app]\")",
        file=stream,
    )
    return False


def build(argv=None):
    os.environ.setdefault("MPLBACKEND", "Agg")  # figures are saved, never shown
    from pyPetrograph.app.amics_module import AmicsModule
    from pyPetrograph.app.shell import AppWindow, make_app

    app = make_app(argv)
    win = AppWindow()
    module = AmicsModule(win)
    return app, win, module


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        print("usage: pypetrograph-app [--help] [--version]\n\n"
              "pyPetrograph desktop app: the AMICS workflow (notebook 4) step by step.\n"
              "Needs the [app] extra: pip install \"pyPetrograph[app]\"")
        return 0
    if "--version" in args:
        from pyPetrograph import __version__

        print(f"pyPetrograph {__version__}")
        return 0
    if not check_dependencies():
        return 1
    app, win, _module = build(argv)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
