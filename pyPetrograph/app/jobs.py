# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Background jobs: run a callable on a worker thread, deliver the result on the GUI thread."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.sig = _Signals()

    def run(self) -> None:
        try:
            res = self.fn()
        except Exception as exc:
            traceback.print_exc()
            self.sig.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.sig.done.emit(res)


class JobRunner(QObject):
    """One job at a time (the UI shows it as busy); models stay loaded in ``SegBackend``."""

    busyChanged = Signal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._tasks = set()
        self._queue = []  # jobs asked for while busy: (label, fn, on_done, on_error)
        self._label = ""
        self.busy = False

    def run(self, label: str, fn: Callable[[], Any], on_done: Optional[Callable[[Any], None]] = None,
            on_error: Optional[Callable[[str], None]] = None) -> bool:
        """Start ``fn`` now, or after the running job when busy (never dropped silently).

        A job with the same label as the running or a queued one is ignored (double clicks).
        Returns True when started or queued.
        """
        if self.busy:
            if label == self._label or any(q[0] == label for q in self._queue):
                return False
            self._queue.append((label, fn, on_done, on_error))
            return True
        self._label = label
        task = _Task(fn)
        self._tasks.add(task)
        task.setAutoDelete(False)

        def finish(ok: bool, payload: Any) -> None:
            self._tasks.discard(task)
            self.busy = False
            self._label = ""
            self.busyChanged.emit(False, "" if ok else str(payload))
            cb = on_done if ok else on_error
            try:
                if cb is not None:
                    cb(payload)
            finally:
                if self._queue and not self.busy:
                    self.run(*self._queue.pop(0))

        task.sig.done.connect(lambda r: finish(True, r))
        task.sig.failed.connect(lambda e: finish(False, e))
        self.busy = True
        self.busyChanged.emit(True, label)
        self.pool.start(task)
        return True

    def wait(self, msecs: int = -1) -> bool:
        return self.pool.waitForDone(msecs)

    @property
    def pending(self) -> int:
        return len(self._queue)
