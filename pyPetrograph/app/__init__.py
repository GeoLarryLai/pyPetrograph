# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyPetrograph desktop app (PySide6). Launch with ``pypetrograph-app`` or ``python -m pyPetrograph.app``."""


def main(argv=None) -> int:
    from pyPetrograph.app.main import main as _main

    return _main(argv)
