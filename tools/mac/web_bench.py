"""web app via HTTP API in-process. usage: web_bench.py SAMPLE OUT MODE(seg|load) RESULT.json [features]"""
import os, shutil, sys, time
T0 = time.perf_counter()
from common import *
from fastapi.testclient import TestClient
sample, out, mode, res = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
feat = "features" in sys.argv[5:]
seed(out, sample, os.environ.get("SEED_GRAINS"))
tm = Timer(); log = []
with Peak() as pk:
    with tm("startup (import+create app+GET /)"):
        from pyPetrograph.webapp.routes import create_app
        c = TestClient(create_app(testing=True)); assert c.get("/").status_code == 200
    tm.t["startup incl. interpreter"] = round(time.perf_counter() - T0, 3)
    def ok(r):
        assert r.status_code == 200, (str(r.request.url), r.status_code, r.text[:300]); return r.json()
    def job(url, body=None, show=False):
        j = ok(c.post(url, json=body or {}))["job"]; last = None; w = 0.04
        while j["state"] == "running":
            time.sleep(w); w = min(0.25, w * 2); j = ok(c.get(f"/shell/api/jobs/{j['id']}"))
            p = (j.get("n"), j.get("N"), j.get("message"))
            if show and p != last and j.get("N"):
                last = p; pct = 100 * (p[0] or 0) / max(1, p[1])
                s = f"{p[0]} / {p[1]} ({pct:.0f}%) {p[2]}"; log.append(s); print("  prog", s, flush=True)
        assert j["state"] == "done", j; return j.get("result")
    b, m, l = files(sample)
    with tm("open + legend OCR"):
        ok(c.post("/api/open", json={"bse": b, "mineral": m, "legend": l, "out": str(out), "sample": ""}))
        lg = ok(c.get("/api/legend"))
        if lg["need_read"]: job("/api/legend/read")
        lg = ok(c.get("/api/legend"))
    with tm("legend OCR (Read again)"): job("/api/legend/read"); lg = ok(c.get("/api/legend"))
    rows = lg["rows"]; info = {"legend": [(r["name"], r["rgb"]) for r in rows]}
    if feat:
        F = {}
        n0 = len(rows); nm = rows[0]["name"]
        ok(c.post("/api/legend/op", json={"op": "rename", "row": 0, "name": "TESTNAME"}))
        F["legend rename"] = ok(c.get("/api/legend"))["rows"][0]["name"] == "TESTNAME"
        r = c.post("/api/legend/op", json={"op": "entry", "rgb": [1, 2, 3], "name": "Zz"}); F["legend add"] = r.status_code == 200 and len(ok(c.get("/api/legend"))["rows"]) == n0 + 1
        r = c.post("/api/legend/op", json={"op": "entry", "rgb": [1, 2, 3], "name": "Zz2"}); F["legend add duplicate colour rejected"] = r.status_code != 200
        ok(c.post("/api/legend/op", json={"op": "delete", "rows": [n0]})); F["legend delete"] = len(ok(c.get("/api/legend"))["rows"]) == n0
        ok(c.post("/api/legend/op", json={"op": "undo"})); F["legend undo delete"] = len(ok(c.get("/api/legend"))["rows"]) == n0 + 1
        ok(c.post("/api/legend/op", json={"op": "reset"})); rr = ok(c.get("/api/legend"))["rows"]
        F["legend reset to OCR"] = len(rr) == n0 and rr[0]["name"] == nm
        F["eyedropper /api/pixel"] = c.get("/api/pixel", params={"src": "legend_img", "x": 5, "y": 5}).status_code == 200
    with tm("first display mineral map preview (1600 px)"): assert c.get("/api/img/mineral_img?max=1600").status_code == 200
    with tm("legend apply"): job("/api/legend/apply")
    with tm("preview stretched BSE 1600 (zoom-out tile)"): r = c.get("/api/img/bse_stretch?max=1600"); assert r.status_code == 200
    with tm("full-res BSE (zoom-in)"): r = c.get("/api/img/bse_stretch"); assert r.status_code == 200
    info["full_png_MB"] = round(len(r.content) / 1e6, 2)
    with tm("cached preview re-request (pan)"): c.get("/api/img/bse_stretch?max=1600")
    f = ok(c.get("/api/footer"))
    with tm("footer apply"): ok(c.post("/api/footer/apply", json={"row": f["footer"]}))
    ng = ok(c.get("/api/nongrain?activate=true"))
    if feat:
        w = ok(c.get("/api/imgsize/bse_stretch"))
        ok(c.post("/api/nongrain/op", json={"op": "click", "x": 10, "y": 10})); s1 = ok(c.get("/api/nongrain"))
        ok(c.post("/api/nongrain/op", json={"op": "undo"})); s2 = ok(c.get("/api/nongrain"))
        F["nongrain click/undo"] = s1 != s2
        ok(c.post("/api/nongrain/op", json={"op": "reset"})); ok(c.post("/api/nongrain/op", json={"op": "load_saved"}))
        F["nongrain reset+load saved"] = True
    with tm("non-grain apply"): ok(c.post("/api/nongrain/op", json={"op": "apply"}))
    ok(c.get("/api/scale?activate=true"))
    if feat:
        ok(c.post("/api/scale/op", json={"op": "click", "x": 100, "y": 100})); ok(c.post("/api/scale/op", json={"op": "click", "x": 300, "y": 100}))
        ok(c.post("/api/scale/op", json={"op": "set", "length": 100, "unit": "µm"})); sc = ok(c.get("/api/scale"))
        ok(c.post("/api/scale/op", json={"op": "undo"})); ok(c.post("/api/scale/op", json={"op": "reset"}))
        F["scale line/undo/reset"] = True; info["scale_view_after_line"] = {k: sc.get(k) for k in ("um", "info", "status") if k in sc}
        F["scale footer OCR"] = c.post("/api/scale/ocr", json={}).status_code == 200
    with tm("scale skip"): ok(c.post("/api/scale/op", json={"op": "skip"}))
    if mode == "noseg":
        info["peak_rss_MB"] = round(pk.peak / 2**20); info["times"] = tm.t; dump(res, info); print(tm.t); os._exit(0)
    sg = ok(c.get("/api/segment?activate=true")); info["avail"] = sg["avail"]
    if mode == "seg":
        with tm("segmentation (SEG)"): job("/api/segment/run", {}, show=True)
        info["seg_run1"] = polyhash(ok(c.get("/api/segment/polys"))["polys"])
        if os.environ.get("SAMREFINE"):
            with tm("segmentation SEG + SAM 2.1 refine"): job("/api/segment/run", {"use_sam2": True}, show=True)
            info["seg_sam_refine"] = polyhash(ok(c.get("/api/segment/polys"))["polys"])
            ok(c.post("/api/segment/settings", json={"use_sam2": False}))
        with tm("segmentation 2nd run (models loaded)"): job("/api/segment/run", {}, show=True)
        info["seg_run2"] = polyhash(ok(c.get("/api/segment/polys"))["polys"])
    else:
        with tm("load saved grains"): job("/api/segment/load")
    info["grains"] = ok(c.get("/api/segment"))["n"]
    info["grains_hash"] = polyhash(ok(c.get("/api/segment/polys"))["polys"])
    if feat:
        P = ok(c.get("/api/segment/polys"))["polys"]
        import numpy as np
        cen = [np.asarray(p).reshape(-1, 2).mean(0) for p in P[:2]]
        n = len(P)
        ok(c.post("/api/segment/op", json={"op": "click", "x": float(cen[0][0]), "y": float(cen[0][1])}))
        ok(c.post("/api/segment/op", json={"op": "delete"})); F["grain delete"] = ok(c.get("/api/segment"))["n"] == n - 1
        ok(c.post("/api/segment/op", json={"op": "undo"})); F["grain undo"] = ok(c.get("/api/segment"))["n"] == n
        if sg["avail"].get("sam2"):
            with tm("SAM prepare"): job("/api/segment/sam_prepare")
            bx = np.asarray(P[5]).reshape(-1, 2)
            ok(c.post("/api/segment/op", json={"op": "add_point", "x": float(bx[:, 0].mean()), "y": float(bx[:, 1].mean())}))
            with tm("SAM point prompt"): job("/api/segment/prompt")
            ok(c.post("/api/segment/op", json={"op": "accept"})); info["sam_point_poly"] = polyhash(ok(c.get("/api/segment/polys"))["polys"][-1:]); F["SAM point add+accept"] = ok(c.get("/api/segment"))["n"] == n + 1
            ok(c.post("/api/segment/op", json={"op": "undo"})); F["SAM add undo"] = ok(c.get("/api/segment"))["n"] == n
            lo, hi = bx.min(0) - 3, bx.max(0) + 3
            ok(c.post("/api/segment/op", json={"op": "box", "box": [*map(float, lo), *map(float, hi)]}))
            with tm("SAM box prompt"): job("/api/segment/prompt")
            ok(c.post("/api/segment/op", json={"op": "accept"})); info["sam_box_poly"] = polyhash(ok(c.get("/api/segment/polys"))["polys"][-1:]); F["SAM box add+accept"] = ok(c.get("/api/segment"))["n"] == n + 1
            ok(c.post("/api/segment/op", json={"op": "undo"})); F["SAM box undo"] = ok(c.get("/api/segment"))["n"] == n
    with tm("use this grain map (precompute)"): job("/api/segment/use")
    with tm("minerals view"): mi = ok(c.get("/api/minerals"))
    with tm("grain size view"): gs = ok(c.get("/api/gsd"))
    info["minerals"] = mi.get("summary"); info["gsd"] = gs.get("summary")
    if feat:
        sug = ok(c.post("/api/minerals/op", json={"op": "suggest"}))
        r = c.post("/api/minerals/op", json={"op": "add", "rgb": "1,2,3", "name": "TestMin"}); F["minerals add colour"] = r.status_code == 200
        if r.status_code == 200 and "job" in r.json():
            pass
        time.sleep(1)
        r = c.post("/api/minerals/op", json={"op": "remove", "mineral": "TestMin"}); F["minerals remove colour"] = r.status_code == 200
        time.sleep(3)
        F["gsd replot"] = c.post("/api/gsd/replot", json={}).status_code == 200; time.sleep(3)
    with tm("export CSV data"): job("/api/outputs/export_data")
    with tm("export figures"): job("/api/outputs/export_figures")
    if feat:
        F["outputs folder"] = c.post("/api/outputs/folder", json={}).status_code == 200
        F["outputs list"] = len(ok(c.get("/api/outputs")).get("files", [])) > 0
        info["features"] = F; print(F)
info["peak_rss_MB"] = round(pk.peak / 2**20); info["times"] = tm.t; info["progress"] = log[-40:]
dump(res, info); print("peak MB", info["peak_rss_MB"], "grains", info["grains"])
