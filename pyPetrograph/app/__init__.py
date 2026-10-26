# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Qt-free app backend (AMICS pipeline, SEG backend) used by the browser app ``pyPetrograph.webapp``.
``pypetrograph-app`` launches the browser app."""


def main(argv=None) -> int:
    from pyPetrograph.webapp import main as _main

    return _main(argv)
