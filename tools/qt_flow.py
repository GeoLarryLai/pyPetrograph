"""Run the Qt desktop app offscreen through the same 10B flow; export to $QT_OUT. Run with pytest."""
import os, shutil
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pyPetrograph.app.shell import _own_plugins
_own_plugins()
import sys; sys.path.insert(0, str(Path(__file__).parent))
from web_flow import EX, seed


def test_qt_flow(qtbot):
    from pyPetrograph.app.amics_module import AmicsModule, ScaleStep, SegmentStep, OutputsStep
    from pyPetrograph.app.shell import AppWindow
    out = Path(os.environ["QT_OUT"]); shutil.rmtree(out, ignore_errors=True); seed(out)
    win = AppWindow(); qtbot.addWidget(win); mod = AmicsModule(win)
    o = mod.steps[0]
    o.edits["bse"].setText(str(EX / "10B_BSE_Image.tiff")); o.edits["mineral"].setText(str(EX / "10B_Mineral_Image.tiff"))
    o.edits["legend"].setText(str(EX / "10-Mineral_Legend.tiff")); o.out.setText(str(out)); o.open()
    idle = lambda: not win.jobs.busy
    qtbot.waitUntil(lambda: bool(mod.session.legend) and idle(), timeout=120000)
    mod.steps[1].apply(); qtbot.waitUntil(lambda: mod.session.bse_stretch is not None and idle(), timeout=120000)
    mod.steps[2].apply(); mod.steps[3].apply()
    mod.step(ScaleStep).skip(go_next=True)
    seg = mod.step(SegmentStep); seg.activate(); qtbot.waitUntil(idle, timeout=60000)
    if not seg.polys:
        seg.load(); qtbot.waitUntil(lambda: bool(seg.polys) and idle(), timeout=120000)
    seg.apply(); qtbot.waitUntil(lambda: "segment" in mod.done and idle(), timeout=600000)
    ot = mod.step(OutputsStep); ot.export_data(); qtbot.waitUntil(idle, timeout=300000)
    ot.export_figures(); qtbot.waitUntil(idle, timeout=300000)
    win.jobs.wait()
