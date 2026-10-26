# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Headless GUI tests (pytest-qt). Run: QT_QPA_PLATFORM=offscreen pytest pyPetrograph/app -q

Covers Files → Legend → BSE & footer → Non-grain → Scale on the 10B example; segmentation is left to the
end-to-end run because it needs the models (minutes)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from pyPetrograph.app.shell import _own_plugins  # noqa: E402

_own_plugins()  # before pytest-qt creates the QApplication

EX = Path(__file__).resolve().parents[2] / "example_data"
need_data = pytest.mark.skipif(not (EX / "10B_BSE_Image.tiff").is_file(), reason="example_data not present")


@pytest.fixture
def ui(qtbot, tmp_path):
    from pyPetrograph.app.amics_module import AmicsModule
    from pyPetrograph.app.shell import AppWindow

    win = AppWindow()
    qtbot.addWidget(win)
    mod = AmicsModule(win)
    yield win, mod, tmp_path
    win.jobs.wait()


STEPS = ["Files", "Legend", "BSE & footer", "Non-grain regions", "Scale", "Grain segmentation",
         "Minerals", "Grain size", "Outputs"]


def _open_10b(win, mod, out, qtbot):
    o = mod.steps[0]
    o.edits["bse"].setText(str(EX / "10B_BSE_Image.tiff"))
    o.edits["mineral"].setText(str(EX / "10B_Mineral_Image.tiff"))
    o.edits["legend"].setText(str(EX / "10-Mineral_Legend.tiff"))
    o.out.setText(str(out))
    o.open()
    qtbot.waitUntil(lambda: bool(mod.session.legend) and not win.jobs.busy, timeout=60000)


def _to_nongrain(win, mod, out, qtbot):
    _open_10b(win, mod, out, qtbot)
    mod.steps[1].apply()
    qtbot.waitUntil(lambda: mod.session.bse_stretch is not None and not win.jobs.busy, timeout=60000)
    assert win.step_list.currentRow() == 2  # BSE & footer
    mod.steps[2].apply()
    assert win.step_list.currentRow() == 3  # Non-grain regions


def test_window_layout(ui):
    from PySide6.QtWidgets import QScrollArea

    win, mod, _ = ui
    win.show()
    assert [s.title for s in mod.steps] == STEPS
    assert win.findChild(type(win.canvas)) is win.canvas
    assert mod.steps[4].optional and "(optional)" in win.step_list.item(4).text()
    assert win.step_list.item(0).text().startswith("\u25B6")          # current
    assert win.step_list.item(1).text().startswith("\u25CB")          # to do
    names = {"Open", "Use this legend", "Use this footer", "Apply non-grain regions", "Apply scale",
             "Run segmentation", "Use this grain map", "Replot grain size curves",
             "Export CSV data sheets", "Export all figures"}
    got = {b.text() for st in mod.steps for b in st.primaries}
    assert names <= got and not any("Export copy" in t or "Vote" in t for t in got)
    for st in mod.steps:  # primary buttons + Controls are pinned: never inside the scrolling part
        for w in st.primaries + [st.next_btn, st.controls_box]:
            p = w.parent()
            while p is not None and p is not st.panel:
                assert not isinstance(p, QScrollArea), (st.title, w)
                p = p.parent()
            assert w.property("primary") or w is st.controls_box
    win.go(0)
    assert mod.steps[0].primaries[0].isVisible() and mod.steps[0].controls_box.isVisible()
    assert not mod.steps[0].next_btn.isEnabled()  # Next waits for Open
    txt = " ".join(st.panel.findChild(type(st.state_lbl)).text() for st in mod.steps)
    allt = " ".join(lb.text() for st in mod.steps for lb in st.panel.findChildren(type(st.state_lbl)))
    assert "dark" not in (txt + allt).lower()  # 'non-grain' everywhere in the UI


@need_data
def test_flow_nongrain_and_scale(ui, qtbot):
    from PySide6.QtCore import Qt

    win, mod, tmp = ui
    _to_nongrain(win, mod, tmp, qtbot)
    ng, sc = mod.steps[3], mod.steps[4]
    assert ng.controls_box.isVisibleTo(ng.panel) and "Right click" in ng.controls_box.findChildren(type(ng.state_lbl))[2 * 2].text()
    assert not ng.next_btn.isEnabled()  # required step: Next only after Apply
    assert mod.win.canvas._layers  # image + footer line
    for p in [(500.0, 400.0), (900.0, 700.0), (1200.0, 300.0)]:
        ng.on_click(*p, Qt.LeftButton)
    assert len(ng.points) == 3 and ng.state_lbl.property("state") == "unsaved"
    ng.undo()
    assert ng.points == [(500.0, 400.0), (900.0, 700.0)]
    ng.on_key("z")
    assert ng.points == [(500.0, 400.0)]
    ng.on_click(500.4, 400.3, Qt.LeftButton)      # click on a pick removes it
    assert ng.points == []
    ng.on_click(500.0, 400.0, Qt.LeftButton)
    ng.on_click(520.0, 410.0, Qt.LeftButton)
    ng.reset()
    assert ng.points == [] and ng.state_lbl.property("state") == "none"
    ng.thr.setValue(58.0)
    ng.on_key("enter")                            # Enter applies
    assert mod.session.threshold == 58.0 and "nongrain" in mod.done
    assert win.step_list.currentRow() == 4 and win.step_list.item(3).text().startswith("\u2713")
    assert ng.state_lbl.property("state") == "saved"
    # scale: optional, Next reads Skip until something is drawn
    assert sc.next_btn.isEnabled() and sc.next_btn.text().startswith("Skip")
    sc.on_click(100.0, 2520.0, Qt.LeftButton)
    assert "other end" in sc.info.text()
    sc.on_click(292.7, 2520.0, Qt.LeftButton)
    assert sc.drawn() and sc.state_lbl.property("state") == "unsaved"
    sc.on_key("z")
    assert len(sc.pts) == 1
    sc.on_click(292.7, 2520.0, Qt.LeftButton)
    sc.apply()
    S = mod.session.S
    assert abs(mod.session.um_per_px - 500.0 / 192.7) < 1e-9 and S["scale_json"].is_file()
    assert win.step_list.currentRow() == 5
    assert S["threshold_json"].is_file() and S["folder"].name == ".pypetrograph"
    # session files only: nothing in the export folder itself before an Export button
    assert [p.name for p in Path(mod.session.P["folder"]).iterdir() if not p.name.startswith(".")] == []


@need_data
def test_clear_saved_needs_confirm(ui, qtbot, monkeypatch):
    from PySide6.QtCore import Qt

    import pyPetrograph.app.amics_module as am

    win, mod, tmp = ui
    _to_nongrain(win, mod, tmp, qtbot)
    ng = mod.steps[3]
    ng.on_click(500.0, 400.0, Qt.LeftButton)
    ng.apply()
    f = mod.session.S["threshold_json"]
    assert f.is_file()
    win.go(3)
    monkeypatch.setattr(am, "confirm", lambda *a: False)
    ng.clear_saved()
    assert f.is_file() and ng.points == [(500.0, 400.0)]
    monkeypatch.setattr(am, "confirm", lambda *a: True)
    ng.clear_saved()
    assert not f.exists() and ng.points == [] and "nongrain" not in mod.done
    assert not ng.next_btn.isEnabled()


@need_data
def test_saved_settings_reload(ui, qtbot):
    from PySide6.QtCore import Qt

    win, mod, tmp = ui
    _to_nongrain(win, mod, tmp, qtbot)
    ng, sc = mod.steps[3], mod.steps[4]
    ng.on_click(500.0, 400.0, Qt.LeftButton)
    ng.on_click(900.0, 700.0, Qt.LeftButton)
    ng.apply()
    line = mod.session.scale_from_line((100.0, 2520.0), (292.7, 2520.0), 500.0, "µm")
    sc.pts = [tuple(line["p1"]), tuple(line["p2"])]
    sc.apply()

    # reopen the same output folder in a fresh window: saved picks and line come back
    from pyPetrograph.app.amics_module import AmicsModule
    from pyPetrograph.app.shell import AppWindow

    win2 = AppWindow()
    qtbot.addWidget(win2)
    m2 = AmicsModule(win2)
    _to_nongrain(win2, m2, tmp, qtbot)
    n2, s2 = m2.steps[3], m2.steps[4]
    assert n2.points == [(500.0, 400.0), (900.0, 700.0)]
    assert n2.state_lbl.property("state") == "saved" and "threshold.json" in n2.state_lbl.text()
    n2.apply()
    assert s2.saved_scale == line and "scale.json" in s2.state_lbl.text()
    assert "scale_saved" in win2.canvas.layer_names()        # orange dashed saved line
    s2.next()                                                 # Skip keeps the saved scale
    assert m2.session.scale == line and win2.step_list.currentRow() == 5
    win2.jobs.wait()


# -- gaps round: prompts, saved settings, extra legend, dependency check -------------
class _FakeBackend:
    """Stands in for the model server: records requests, answers a square grain."""

    def __init__(self):
        self.calls = []

    def request(self, op, **kw):
        self.calls.append((op, kw))
        if op == "prompt":
            x, y = (kw["points"][0] if kw["points"] else kw["box"][:2])
            return {"ok": True, "coords": [[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]]}
        return {"ok": True}

    def close(self):
        pass


def test_sam_prompts(ui, qtbot):
    from PySide6.QtCore import Qt
    from shapely.geometry import box

    from pyPetrograph.app.amics_module import SegmentStep

    win, mod, _ = ui
    seg = mod.step(SegmentStep)
    win.backend = _FakeBackend()
    seg.polys = [box(0, 0, 10, 10)]
    seg._sam_ready = True  # skip loading SAM
    seg.add.setChecked(True)
    keys = [w.text() for w in seg.controls_box.findChildren(type(seg.state_lbl))]
    assert "Right / \u21E7 click" in keys and "Drag" in keys          # add-mode controls
    seg.on_click(50, 60, Qt.LeftButton)        # positive point
    seg.on_click(70, 60, Qt.RightButton)       # background point
    seg.on_box(40, 50, 65, 75) if mod.current() is seg else seg.__setattr__("prompt_box", (40.0, 50.0, 65.0, 75.0)) or seg._run_prompt()
    qtbot.waitUntil(lambda: not win.jobs.busy and seg.preview is not None, timeout=5000)
    op, req = win.backend.calls[-1]
    assert op == "prompt"
    assert req["points"] == [[50.0, 60.0], [70.0, 60.0]] and req["labels"] == [True, False]
    assert req["box"] == [40.0, 50.0, 65.0, 75.0]
    seg.on_key("enter")
    assert len(seg.polys) == 2 and seg.edited and seg.preview is None and not seg.prompt_pts
    assert seg.state_lbl.property("state") == "unsaved"
    seg.on_key("z")
    assert len(seg.polys) == 1
    seg.add.setChecked(False)
    keys = [w.text() for w in seg.controls_box.findChildren(type(seg.state_lbl))]
    assert "M" in keys and "Esc" in keys                                 # edit-mode controls


def test_grain_select_deselect_reset(ui, monkeypatch):
    from PySide6.QtCore import Qt
    from shapely.geometry import box

    import pyPetrograph.app.amics_module as am

    win, mod, _ = ui
    seg = mod.step(am.SegmentStep)
    seg._got([box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)], "segmented")
    seg.on_click(5, 5, Qt.LeftButton)
    assert seg.sel == {0}
    seg.on_click(5, 5, Qt.LeftButton)          # click again: deselect
    assert seg.sel == set()
    seg.on_click(25, 5, Qt.LeftButton)
    seg.on_key("d")
    assert len(seg.polys) == 2 and seg.edited
    monkeypatch.setattr(am, "confirm", lambda *a: True)
    seg.reset()
    assert len(seg.polys) == 3 and not seg.edited and not seg.history


def test_box_drag_signal(ui, qtbot):
    import numpy as np
    from PySide6.QtCore import QPoint, Qt

    win, _, _ = ui
    c = win.canvas
    c.resize(400, 300)
    c.set_image(np.zeros((300, 400), np.uint8))
    c.set_box_mode(True)
    got = []
    c.boxDragged.connect(lambda *a: got.append(a))
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.LeftButton, pos=QPoint(50, 50))
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtWidgets import QApplication

    QApplication.sendEvent(vp, QMouseEvent(QEvent.MouseMove, QPointF(150, 120), QPointF(150, 120),
                                           Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    qtbot.mouseRelease(vp, Qt.LeftButton, pos=QPoint(150, 120))
    assert len(got) == 1
    x0, y0, x1, y1 = got[0]
    assert x1 > x0 and y1 > y0


def test_step_order(ui):
    win, mod, _ = ui
    from pyPetrograph.app.amics_module import ORDER, APP_DEFAULTS

    assert ORDER == ["open", "legend", "footer", "nongrain", "scale", "segment", "minerals", "gsd", "outputs"]
    assert [win.step_list.item(i).text().split("  ", 2)[2].replace("  (optional)", "") for i in range(9)] == STEPS
    seg = mod.steps[5]
    assert seg.min_px.value() == 10 and seg.drop.value() == 0.5 == APP_DEFAULTS["dark_drop_frac"]
    from pyPetrograph.app.amics_pipeline import AmicsSettings
    assert AmicsSettings().min_grain_px == 10  # same as notebook 4's MIN_GRAIN_PX


def test_delete_button_enabled(ui, qtbot):
    from PySide6.QtCore import Qt
    from shapely.geometry import box

    win, mod, _ = ui
    seg = mod.steps[5]
    win.show()
    win.go(5)
    seg._got([box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)], "segmented")
    assert not seg.del_btn.isEnabled() and not seg.merge_btn.isEnabled()
    seg.on_click(25, 5, Qt.LeftButton)
    assert seg.del_btn.isEnabled() and not seg.merge_btn.isEnabled()
    qtbot.mouseClick(seg.del_btn, Qt.LeftButton)          # the button, not the D key
    assert len(seg.polys) == 2 and seg.edited and not seg.del_btn.isEnabled()
    seg.on_click(5, 5, Qt.LeftButton)
    seg.on_click(45, 5, Qt.LeftButton)
    assert seg.merge_btn.isEnabled()


def _table_session():
    import numpy as np
    import pandas as pd

    from pyPetrograph.app.amics_pipeline import AmicsSession

    s = AmicsSession(log=lambda *_: None)
    rgb = np.full((4, 10, 3), 255, np.uint8)
    rgb[0, :4] = (255, 0, 0)        # A exact
    rgb[1, :3] = (0, 250, 0)        # B: near colour (within COLOR_TOL)
    rgb[2, :2] = (150, 100, 0)      # D: matched by AUTO_MATCH_COLORS
    rgb[3, :1] = (123, 45, 67)      # no name could take it
    s.footer = 4
    s.vote_legend = {"A": (255, 0, 0), "B": (0, 255, 0), "C": (0, 0, 255), "D": (200, 200, 0)}
    cm = pd.DataFrame({"map_red": [150, 123], "map_green": [100, 45], "map_blue": [0, 67], "pixels": [2, 1],
                       "assigned": ["D", ""], "distance": [20.0, 99.0], "status": ["free name", "unnamed"]})
    s.color_matches = cm
    s._raw_map = (rgb, np.ones((4, 10), bool), cm[["map_red", "map_green", "map_blue", "pixels"]])
    return s


def test_minerals_table_statuses(ui):
    from PySide6.QtGui import QColor

    from pyPetrograph.app.amics_module import REASSIGNED_BG, MineralsStep

    rows = {r["mineral"]: r for r in _table_session().mineral_table()}
    assert rows["A"]["status"] == "ok" and rows["A"]["map_rgbs"] == [(255, 0, 0)] and rows["A"]["pixels"] == 4
    assert rows["B"]["status"] == "reassigned" and rows["B"]["map_rgbs"] == [(0, 250, 0)]
    assert rows["D"]["status"] == "reassigned" and rows["D"]["map_rgbs"] == [(150, 100, 0)]
    assert rows["C"]["status"] == "missing" and rows["C"]["pixels"] == 0
    assert rows["(unlisted map colour)"]["status"] == "unlisted"
    assert abs(sum(r["pct"] for r in rows.values()) - 100.0) < 1e-9
    win, mod, _ = ui
    mn = mod.step(MineralsStep)
    mn.s.mineral_table = _table_session().mineral_table
    mn.fill()
    icons = [mn.table.item(i, 0).text() for i in range(mn.table.rowCount())]
    assert icons == ["\u2713", "\u25B2", "\u2717", "\u25B2", "\u2717"]
    assert mn.table.item(1, 1).background().color() == QColor(REASSIGNED_BG)   # reassigned row highlighted
    assert mn.table.item(0, 1).background().style() == __import__("PySide6.QtCore").QtCore.Qt.NoBrush                       # identical: plain row
    assert mn.table.item(1, 2).text() == "0,255,0" and mn.table.item(1, 3).text() == "0,250,0"
    assert [mn.view.itemText(i) for i in range(mn.view.count())] == ["Composition", "Overlay", "Mixed grains", "Unmapped grains"]
    assert mn.add_btn.text() == "+ Add map colour not in legend"
    assert mn.parse_rgb("(1, 2, 3)") == (1, 2, 3) and mn.parse_rgb("#ffd60a") == (255, 214, 10) and mn.parse_rgb("1 2 999") is None


def test_nothing_written_before_export(ui, tmp_path):
    import pandas as pd

    from pyPetrograph.app.amics_module import OutputsStep

    win, mod, _ = ui
    s = mod.session
    s.sample = "X"
    folder = tmp_path / "amics" / "X"
    keys = {"counts": "counts.csv", "minerals": "minerals.geojson", "knolling": "knolling.png", "gsd": "gsd.png",
            "legend_json": "legend.json", "threshold_json": "threshold.json", "scale_json": "scale.json", "grains": "grains.geojson"}
    cache = tmp_path / "cache"
    cache.mkdir()
    s.P = {"folder": folder, **{k: folder / f"X_{v}" for k, v in keys.items()}}
    s.S = {"folder": folder / ".pypetrograph", **{k: folder / ".pypetrograph" / f"X_{v}" for k, v in keys.items()}}
    s.C = {"folder": cache, **{k: cache / f"X_{v}" for k, v in keys.items()}}
    for k in ("counts", "minerals", "knolling", "gsd"):
        s.C[k].write_bytes(k.encode())
    (cache / "X_Rutile.csv").write_text("a\n")
    s.S["folder"].mkdir(parents=True)
    s.S["grains"].write_text("{}")
    out = mod.step(OutputsStep)
    assert out.export_folder() == folder and [p.name for p in folder.iterdir()] == [".pypetrograph"]
    assert out.open_btn.text() == "Open export folder" and not out.open_btn.property("primary")
    assert out.replot_btn.text() == "Replot grain knolling diagram"
    data = s.export_data()
    assert sorted(p.name for p in data) == ["X_Rutile.csv", "X_counts.csv", "X_grains.geojson", "X_minerals.geojson"]
    assert not (folder / "X_knolling.png").exists()
    figs = s.export_figures()
    assert sorted(p.name for p in figs) == ["X_gsd.png", "X_knolling.png"]
    assert (folder / "X_knolling.png").read_bytes() == b"knolling"


def test_progress_signal(ui, qtbot):
    import io
    import json
    import sys

    from pyPetrograph.app import backend as be
    from pyPetrograph.app.shell import progress_text

    got = []
    b = be.SegBackend()
    b._progress[7] = got.append
    d = {"stage": "U-Net", "done": 120, "total": 560, "unit": "tiles", "patch": 2, "patches": 4, "frac": 0.21}
    b._read(io.StringIO(be._TAG + json.dumps({"id": 7, "progress": d}) + "\n"), True)
    assert got == [d] and 7 not in b._replies              # progress is not taken as the reply
    t = progress_text(d, 42.0)
    assert t.startswith("U-Net: 120 / 560 tiles (21%)") and "0:42 elapsed" in t and "ETA 2:38" in t
    assert progress_text({"stage": "SAM", "done": 3, "total": 9, "unit": "prompts", "frac": 0.5, "patches": 1}, 10).startswith("SAM: 3 / 9 prompts (50%)")
    win, mod, _ = ui
    win.jobs.busy = True
    win.segProgress.emit(d)
    qtbot.waitUntil(lambda: "tiles (21%)" in win.seg_bar.format(), timeout=2000)
    assert win.seg_bar.value() == 210
    win.jobs.busy = False
    # the server's tile count equals the tiles predict_image really runs
    sys.path.insert(0, str(Path(be.__file__).parents[1] / "amics"))
    import fast_seg
    import numpy as np

    for h, w in ((2000, 2000), (800, 1313), (256, 512)):
        r, c = h // 256 + 1, w // 256 + 1
        W = np.ones((256, 256))
        assert fast_seg.tile_count(h, w) == len(fast_seg._tile_jobs(r, c, 256, 128, W, W, W))


def test_dependency_check(monkeypatch, capsys):
    import importlib.util

    import importlib

    m = importlib.import_module("pyPetrograph.app.main")

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("PySide6", "rapidocr_onnxruntime") else real(name, *a))
    assert m.check_dependencies() is False
    err = capsys.readouterr().err
    assert "PySide6" in err and 'pip install "pyPetrograph[app]"' in err and "rapidocr" in err
    assert m.main([]) == 1  # exits cleanly, no window, no traceback


def test_seg_env_message(monkeypatch):
    import sys as _sys

    from pyPetrograph.app import backend
    import pyPetrograph.objects.qc as qc

    monkeypatch.setattr(qc, "_seg_python", lambda: None)
    with pytest.raises(backend.SegBackendError) as e:
        backend.SegBackend()._start()
    msg = str(e.value)
    assert "'work'" not in msg and _sys.prefix in msg and "launched from" in msg


def _consistent(win, mod):
    """No step after a not-done required step may be current or ticked."""
    from pyPetrograph.app.amics_module import ORDER

    cur = win.step_list.currentRow()
    assert win.steps[cur].ready(), f"current step {cur} is not ready; done={sorted(mod.done)}"
    for i in range(cur):
        st = win.steps[i]
        mark = win.step_list.item(i).text()[0]
        assert st.optional or st.complete(), f"step {i} before current {cur} not done; done={sorted(mod.done)}"
        assert mark == ("\u2713" if st.complete() else "\u25CB")
    if cur >= ORDER.index("minerals"):
        assert "segment" in mod.done


@pytest.mark.parametrize("k", range(20))
def test_step_status_race(ui, qtbot, k):
    """Apply grains, move towards Outputs while the background compute runs, go back / forward."""
    import time

    from shapely.geometry import box

    from pyPetrograph.app.amics_module import ORDER

    win, mod, _ = ui
    s = mod.session
    for st in mod.steps[6:]:
        st.activate = lambda: None          # no real results in this test
    delay = 0.02 * (1 + k % 5)
    s.finalize_grains = lambda: None
    s.compute_downstream = lambda: time.sleep(delay)
    s.extra_legend = {"x": (1, 2, 3)}
    seg = mod.steps[5]
    mod.done = set(ORDER[:5])
    win.go(5)
    seg._got([box(0, 0, 10, 10)], "segmented")
    s.grain_polys = list(seg.polys)

    seg.apply()                              # first apply: later steps not ready yet
    assert win.jobs.busy
    for i in (8, 7, 6):
        win.go(i)                            # refused while nothing is computed
        assert win.step_list.currentRow() == 5
    qtbot.waitUntil(lambda: not win.jobs.busy and win.step_list.currentRow() == 6, timeout=3000)
    _consistent(win, mod)
    win.go(7); win.go(8)
    _consistent(win, mod)

    win.go(5)                                # go back and re-apply; user walks to Outputs meanwhile
    seg.apply()
    win.go(6); win.go(7); win.go(8)
    _consistent(win, mod)
    qtbot.waitUntil(lambda: not win.jobs.busy, timeout=3000)
    assert win.step_list.currentRow() == 8   # not pulled back to Minerals
    _consistent(win, mod)
    for i in (5, 6, 7, 8, 6, 8):
        win.go(i)
        _consistent(win, mod)
    assert {"segment"} <= mod.done


def test_jobs_queue_not_dropped(ui, qtbot):
    import time

    win, mod, _ = ui
    got = []
    assert win.jobs.run("a", lambda: time.sleep(0.1), lambda _: got.append("a"))
    assert win.jobs.run("b", lambda: 2, got.append)          # busy: queued, not dropped
    assert not win.jobs.run("b", lambda: 3, got.append)      # same label queued: ignored (double click)
    qtbot.waitUntil(lambda: got == ["a", 2] and not win.jobs.busy, timeout=3000)
    failed = []
    win.jobs.run("c", lambda: 1 / 0, None, failed.append)
    qtbot.waitUntil(lambda: bool(failed), timeout=3000)
    assert "ZeroDivisionError" in failed[0]


# -- editable legend (t120u) ------------------------------------------------------
@need_data
def test_legend_edit_add_delete_undo_reload(ui, qtbot, tmp_path):
    import json

    from PySide6.QtCore import Qt

    from pyPetrograph.app.amics_module import AmicsModule, LegendStep
    from pyPetrograph.app.shell import AppWindow

    win, mod, tmp = ui
    _open_10b(win, mod, tmp, qtbot)
    lg = mod.step(LegendStep)
    ocr = dict(mod.session.legend_auto)
    n0 = lg.table.rowCount()
    assert n0 == len(ocr) and [lg.table.horizontalHeaderItem(i).text() for i in range(3)] == ["Use", "RGB", "Mineral"]
    assert all(lg.table.item(i, 3).text() == "" for i in range(n0))
    # edit a name: marked + italic, revert brings OCR back, undo works
    old = lg.table.item(0, 2).text()
    lg.table.item(0, 2).setText("Renamed-X")
    assert lg.table.item(0, 3).text() == "\u270E" and lg.table.item(0, 2).font().italic()
    assert "Renamed-X" in lg.edited()[0] and old not in lg.edited()[0]
    lg.revert([0])
    assert lg.table.item(0, 2).text() == old and lg.table.item(0, 3).text() == ""
    lg.undo()
    assert lg.table.item(0, 2).text() == "Renamed-X"
    # add via the dialog (driven programmatically): validation first
    d = lg.add_dialog()
    d.name.setText("")
    assert not d.ok() and "name" in d.err.text()
    d.name.setText("New mineral")
    d.set_rgb(next(iter(ocr.values())))
    assert not d.ok() and "already used" in d.err.text()
    d.hex.setText("#00ff7f")
    d.hex.editingFinished.emit()
    assert d.rgb() == (0, 255, 127)
    d.r.setValue(1)
    assert d.hex.text() == "#01ff7f"
    d.drop_btn.setChecked(True)            # eyedropper samples the shown legend image
    img = lg._img
    lg.on_click(5.0, 5.0, Qt.LeftButton)
    assert d.rgb() == tuple(int(v) for v in img[5, 5, :3])
    d.drop_btn.setChecked(False)
    d.set_rgb((1, 255, 127))
    assert d.ok()
    assert lg.table.rowCount() == n0 + 1 and lg.table.item(n0, 3).text() == "+"
    assert lg.edited()[0]["New mineral"] == (1, 255, 127)
    # change colour of a row through the same dialog (double-click on the swatch)
    lg._dbl(n0, 0)
    lg._dialog.set_rgb((2, 250, 120))
    assert lg._dialog.ok() and lg.edited()[0]["New mineral"] == (2, 250, 120)
    # delete with undo
    lg.table.selectRow(1)
    gone = lg.table.item(1, 2).text()
    lg.on_key("delete")
    assert lg.table.rowCount() == n0 and gone not in lg.edited()[0]
    lg.undo()
    assert lg.table.rowCount() == n0 + 1 and gone in lg.edited()[0]
    lg.table.item(2, 0).setCheckState(Qt.Unchecked)
    skip_name = lg.table.item(2, 2).text()
    want, want_skip = lg.edited()
    assert skip_name in want_skip
    lg.apply()
    qtbot.waitUntil(lambda: "legend" in mod.done and not win.jobs.busy, timeout=60000)
    assert dict(mod.session.legend) == want
    side = mod.session.S["folder"] / f"{mod.session.sample}_legend_edit.json"
    assert json.loads(side.read_text())["skip"] == want_skip
    assert json.loads(mod.session.S["legend_json"].read_text())["New mineral"] == [2, 250, 120]
    # reopen in a new window: the edits come back (not the OCR result)
    win2 = AppWindow()
    qtbot.addWidget(win2)
    mod2 = AmicsModule(win2)
    _open_10b(win2, mod2, tmp, qtbot)
    lg2 = mod2.step(LegendStep)
    win2.go(1)
    qtbot.waitUntil(lambda: lg2.table.rowCount() == n0 + 1 and not win2.jobs.busy, timeout=60000)
    leg2, skip2 = lg2.edited()
    assert leg2 == want and skip2 == want_skip
    marks = {lg2.table.item(i, 2).text(): lg2.table.item(i, 3).text() for i in range(lg2.table.rowCount())}
    assert marks["New mineral"] == "+" and marks["Renamed-X"] == "\u270E" and marks[skip_name] == ""
    lg2.reset()
    assert lg2.edited()[0] == ocr
    win2.jobs.wait()
