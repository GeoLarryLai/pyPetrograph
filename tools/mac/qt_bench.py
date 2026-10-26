"""Qt desktop app offscreen. usage: qt_bench.py SAMPLE OUT MODE(seg|load) RESULT.json"""
import os, sys, time
T0 = time.perf_counter()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from common import *
sample, out, mode, res = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
seed(out, sample, os.environ.get("SEED_GRAINS"))
tm = Timer(); info = {}; log = []
with Peak() as pk:
    with tm("startup (import+create app+GET /)"):
        from pyPetrograph.app.shell import _own_plugins; _own_plugins()
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from pyPetrograph.app.amics_module import AmicsModule, ScaleStep, SegmentStep, OutputsStep
        from pyPetrograph.app.shell import AppWindow
        win = AppWindow(); mod = AmicsModule(win); win.show(); app.processEvents()
    tm.t["startup incl. interpreter"] = round(time.perf_counter() - T0, 3)
    def wait(cond, to=1800):
        a = time.time()
        while not cond():
            app.processEvents(); time.sleep(0.01)
            if time.time() - a > to: raise TimeoutError
    idle = lambda: not win.jobs.busy
    try: win.segProgress.connect(lambda d: log.append(f"{d.get('done')} / {d.get('total')} {d.get('stage')}"))
    except Exception as e: print("no progress hook", e)
    b, m, l = files(sample); o = mod.steps[0]
    with tm("open + legend OCR"):
        o.edits["bse"].setText(b); o.edits["mineral"].setText(m); o.edits["legend"].setText(l); o.out.setText(str(out)); o.open()
        wait(lambda: bool(mod.session.legend) and idle())
    info["legend"] = [(k, list(map(int, v))) for k, v in mod.session.legend.items()] if isinstance(mod.session.legend, dict) else str(mod.session.legend)
    with tm("legend apply"):
        mod.steps[1].apply(); wait(lambda: mod.session.bse_stretch is not None and idle())
    with tm("footer apply"): mod.steps[2].apply(); wait(idle)
    with tm("non-grain apply"): mod.steps[3].apply(); wait(idle)
    with tm("scale skip"): mod.step(ScaleStep).skip(go_next=True); wait(idle)
    if mode == "noseg":
        info["peak_rss_MB"] = round(pk.peak / 2**20); info["times"] = tm.t; dump(res, info); print(tm.t); os._exit(0)
    seg = mod.step(SegmentStep); seg.activate(); wait(idle)
    if mode == "seg":
        seg.polys = []
        with tm("segmentation (SEG)"):
            seg.edited = False; seg.run(); wait(lambda: idle() and bool(seg.polys))
        info["seg_run1"] = polyhash(rings_of(seg.polys)); seg.polys = []
        if os.environ.get("SAMREFINE"):
            seg.sam.setChecked(True)
            with tm("segmentation SEG + SAM 2.1 refine"):
                seg.edited = False; seg.run(); wait(lambda: idle() and bool(seg.polys))
            info["seg_sam_refine"] = polyhash(rings_of(seg.polys)); seg.polys = []; seg.sam.setChecked(False)
        with tm("segmentation 2nd run (models loaded)"):
            seg.edited = False; seg.run(); wait(lambda: idle() and bool(seg.polys))
    else:
        with tm("load saved grains"):
            seg.polys = []; seg.load(); wait(lambda: bool(seg.polys) and idle())
    info["grains"] = len(seg.polys); info["grains_hash"] = polyhash(rings_of(seg.polys))
    if mode == "seg": info["seg_run2"] = info["grains_hash"]
    if os.environ.get("QTSAM"):
        import numpy as np
        n = len(seg.polys)
        with tm("SAM prepare"):
            seg.add.setChecked(True); wait(lambda: seg._sam_ready and idle())
        bx = np.asarray(rings_of(seg.polys[5:6])[0])
        with tm("SAM point prompt"):
            seg.add_point(float(bx[:, 0].mean()), float(bx[:, 1].mean())); wait(lambda: idle() and seg.preview is not None)
        seg.accept_prompt(); info["sam_point_poly"] = polyhash(rings_of(seg.polys[-1:])); assert len(seg.polys) == n + 1
        seg.undo(); assert len(seg.polys) == n
        lo, hi = bx.min(0) - 3, bx.max(0) + 3
        with tm("SAM box prompt"):
            seg.prompt_box = (*map(float, lo), *map(float, hi)); seg._run_prompt(); wait(lambda: idle() and seg.preview is not None)
        seg.accept_prompt(); info["sam_box_poly"] = polyhash(rings_of(seg.polys[-1:])); seg.undo(); assert len(seg.polys) == n
        seg.add.setChecked(False); app.processEvents()
    with tm("use this grain map (precompute)"):
        seg.apply(); wait(lambda: "segment" in mod.done and idle())
    ot = mod.step(OutputsStep)
    with tm("export CSV data"): ot.export_data(); wait(idle)
    with tm("export figures"): ot.export_figures(); wait(idle)
    win.jobs.wait()
info["peak_rss_MB"] = round(pk.peak / 2**20); info["times"] = tm.t; info["progress"] = log[-40:]
dump(res, info); print("peak MB", info["peak_rss_MB"], "grains", info["grains"]); os._exit(0)
