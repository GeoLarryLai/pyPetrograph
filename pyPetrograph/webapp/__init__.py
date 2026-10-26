# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyPetrograph app (``pypetrograph-app``, alias ``pypetrograph-web``): the AMICS workflow,
served by FastAPI on 127.0.0.1 and used in any browser. Same science calls as notebook 4
(``pyPetrograph.app.amics_pipeline``); no Qt needed. fastapi and uvicorn are base dependencies."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pypetrograph-app", description="pyPetrograph AMICS workflow in the browser")
    ap.add_argument("--port", type=int, default=None, help="port (default: first free from 8766)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = ap.parse_args(argv)
    missing = [m for m in ("fastapi", "uvicorn") if __import__("importlib").util.find_spec(m) is None]
    if missing:
        print(f"pypetrograph-app needs {', '.join(missing)}: pip install fastapi uvicorn", file=sys.stderr)
        return 1
    import matplotlib

    matplotlib.use("Agg")
    from pyPetrograph.webapp.routes import create_app
    from pyPetrograph.webapp.shell import serve

    serve(create_app(), "pyPetrograph", port=a.port, open_browser=not a.no_browser)
    return 0
