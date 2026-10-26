# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
GUI-side client for the persistent SEG model server (``seg_server.py``).

This is the model-loading boundary of the app: the GUI only calls
``SegBackend.request(op, **kwargs)``. The server process keeps the U-Net / SAM
loaded between runs. Qt-free so the pipeline also runs headless.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_TAG = "@@PP@@ "
_SERVER = Path(__file__).resolve().parent / "seg_server.py"


class SegBackendError(RuntimeError):
    pass


class SegBackend:
    """One long-lived server subprocess; ``request`` is thread-safe and blocking."""

    def __init__(self, log: Optional[Callable[[str], None]] = None) -> None:
        self._log = log or (lambda s: None)
        self._proc: Optional[subprocess.Popen] = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._replies: Dict[int, Dict[str, Any]] = {}
        self._cv = threading.Condition()
        self._progress: Dict[int, Callable[[Dict[str, Any]], None]] = {}

    # -- process -------------------------------------------------------------
    def _start(self) -> None:
        from pyPetrograph.objects.qc import _seg_python, _tf_worker_env

        py = _seg_python()
        if py is None:
            raise SegBackendError(
                "No Python found for the SEG server. It runs in the environment the app was "
                f"launched from ({sys.prefix}, python {sys.executable}) unless "
                "PYPETROGRAPH_SEG_PYTHON points elsewhere; install the seg extra there: "
                f'{sys.executable} -m pip install "pyPetrograph[seg]"'
            )
        env = _tf_worker_env(mpl_backend="Agg")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [str(py), str(_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        threading.Thread(target=self._read, args=(self._proc.stdout, True), daemon=True).start()
        threading.Thread(target=self._read, args=(self._proc.stderr, False), daemon=True).start()

    def _read(self, stream, replies: bool) -> None:
        for line in stream:
            line = line.rstrip("\n")
            if replies and line.startswith(_TAG):
                msg = json.loads(line[len(_TAG):])
                if "progress" in msg:  # progress of a running request, not its reply
                    cb = self._progress.get(int(msg.get("id") or 0))
                    if cb is not None:
                        try:
                            cb(msg["progress"])
                        except Exception:
                            pass
                    continue
                with self._cv:
                    self._replies[int(msg.get("id") or 0)] = msg
                    self._cv.notify_all()
            elif line.strip():
                self._log(line)
        if replies:
            with self._cv:
                self._cv.notify_all()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def request(self, op: str, timeout: Optional[float] = None,
                progress: Optional[Callable[[Dict[str, Any]], None]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Send one op and wait for its reply; ``progress`` gets the server's progress dicts."""
        with self._lock:
            if not self.running:
                self._start()
            rid = next(self._ids)
            if progress is not None:
                self._progress[rid] = progress
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(json.dumps({"id": rid, "op": op, **kwargs}) + "\n")
            self._proc.stdin.flush()
            with self._cv:
                ok = self._cv.wait_for(lambda: rid in self._replies or not self.running, timeout)
                msg = self._replies.pop(rid, None)
            self._progress.pop(rid, None)
        if msg is None:
            raise SegBackendError("SEG server stopped" if ok else f"SEG server timed out on {op}")
        if not msg.get("ok"):
            raise SegBackendError(msg.get("error") or f"{op} failed")
        return msg

    def close(self) -> None:
        if self.running:
            try:
                self.request("quit", timeout=10)
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self._proc = None
