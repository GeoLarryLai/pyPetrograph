# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyPetrograph in the browser (``pypetrograph-web``): the AMICS workflow of the desktop app,
served by FastAPI on 127.0.0.1 and used in any browser. Same science calls as the desktop app
(``pyPetrograph.app.amics_pipeline``); no Qt needed. Install: ``pip install "pyPetrograph[web]"``."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pypetrograph-web", description="pyPetrograph AMICS workflow in the browser")
    ap.add_argument("--port", type=int, default=None, help="port (default: first free from 8766)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = ap.parse_args(argv)
    missing = [m for m in ("fastapi", "uvicorn") if __import__("importlib").util.find_spec(m) is None]
    if missing:
        print(f"pypetrograph-web needs {', '.join(missing)}: pip install \"pyPetrograph[web]\"", file=sys.stderr)
        return 1
    import matplotlib

    matplotlib.use("Agg")
    from pyPetrograph.webapp.routes import create_app
    from pyPetrograph.webapp.shell import serve

    serve(create_app(), "pyPetrograph", port=a.port, open_browser=not a.no_browser)
    return 0
