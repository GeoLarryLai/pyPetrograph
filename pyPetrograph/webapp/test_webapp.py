# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Browser app tests (FastAPI TestClient, no browser). Run: pytest pyPetrograph/webapp -q"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from fastapi.testclient import TestClient  # noqa: E402

EX = Path(__file__).resolve().parents[2] / "example_data"
need_data = pytest.mark.skipif(not (EX / "10B_BSE_Image.tiff").is_file(), reason="example_data not present")


@pytest.fixture
def client():
    from pyPetrograph.webapp.routes import create_app

    return TestClient(create_app(testing=True))


def ok(r):
    assert r.status_code == 200, (r.request.url, r.status_code, r.text[:300])
    return r.json()


def job(c, url, body=None):
    j = ok(c.post(url, json=body or {}))["job"]
    while j["state"] == "running":
        time.sleep(0.1)
        j = ok(c.get(f"/shell/api/jobs/{j['id']}"))
    assert j["state"] == "done", j
    return j.get("result")


def open10b(c, out):
    ok(c.post("/api/open", json={"bse": str(EX / "10B_BSE_Image.tiff"), "mineral": str(EX / "10B_Mineral_Image.tiff"),
                                 "legend": str(EX / "10-Mineral_Legend.tiff"), "out": str(out)}))
    job(c, "/api/legend/read")


def files_under(p):
    """Export files only (session files in .pypetrograph/ and the remembered last_files.json are the
    app's own bookkeeping, written there too)."""
    return sorted(str(q.relative_to(p)) for q in Path(p).rglob("*") if q.is_file() and ".pypetrograph" not in q.parts
                  and q.name != "last_files.json")


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "mainBtns" in r.text and "cdn" not in r.text.lower()
    for f in ("/static/app.js", "/static/viewer.js", "/static/app.css", "/shell/static/shell.js"):
        assert client.get(f).status_code == 200, f
    st = ok(client.get("/api/state"))
    assert [s["title"] for s in st["steps"]][0] == "Files" and len(st["steps"]) == 9
    assert st["steps"][0]["ready"] and not st["steps"][1]["ready"]
    assert set(st["avail"]) >= {"seg", "sam2"}


def test_main_entry_help():
    from pyPetrograph.webapp import main

    with pytest.raises(SystemExit):
        main(["--help"])


@need_data
def test_legend_editing(client, tmp_path):
    open10b(client, tmp_path)
    lg = ok(client.get("/api/legend"))
    n0, old = len(lg["rows"]), lg["rows"][0]["name"]
    assert all(r["mark"] == "" for r in lg["rows"])
    lg = ok(client.post("/api/legend/op", json={"op": "rename", "row": 0, "name": "Renamed-X"}))
    assert lg["rows"][0]["mark"] == "\u270E" and lg["can_undo"]
    lg = ok(client.post("/api/legend/op", json={"op": "revert", "rows": [0]}))
    assert lg["rows"][0]["name"] == old
    lg = ok(client.post("/api/legend/op", json={"op": "undo"}))
    assert lg["rows"][0]["name"] == "Renamed-X"
    assert "name" in ok(client.post("/api/legend/check", json={"rgb": [1, 2, 3], "name": ""}))["problem"]
    used = lg["rows"][1]["rgb"]
    assert "already" in ok(client.post("/api/legend/check", json={"rgb": used, "name": "New"}))["problem"]
    assert client.post("/api/legend/op", json={"op": "entry", "rgb": used, "name": "New"}).status_code == 400
    lg = ok(client.post("/api/legend/op", json={"op": "entry", "rgb": [0, 255, 127], "name": "New mineral"}))
    assert len(lg["rows"]) == n0 + 1 and lg["rows"][-1]["mark"] == "+"
    lg = ok(client.post("/api/legend/op", json={"op": "delete", "rows": [n0]}))
    assert len(lg["rows"]) == n0
    lg = ok(client.post("/api/legend/op", json={"op": "reset"}))
    assert lg["rows"][0]["name"] == old and all(r["mark"] == "" for r in lg["rows"])
    px = ok(client.get("/api/pixel?src=legend_img&x=5&y=5"))
    assert len(px["rgb"]) == 3
    assert files_under(tmp_path) == []  # nothing written before export


@need_data
def test_flow_nongrain_scale_and_gating(client, tmp_path):
    open10b(client, tmp_path)
    job(client, "/api/legend/apply")
    f = ok(client.get("/api/footer"))
    ok(client.post("/api/footer/apply", json={"row": f["detected"]}))
    v = ok(client.get("/api/nongrain?activate=true"))
    assert v["points"] == []
    v = ok(client.post("/api/nongrain/op", json={"op": "click", "x": 100, "y": 100}))
    assert len(v["points"]) == 1 and "gray level" in v["info"]
    v = ok(client.post("/api/nongrain/op", json={"op": "undo"}))
    assert v["points"] == []
    v = ok(client.post("/api/nongrain/op", json={"op": "typed", "value": 62}))
    assert v["typed"] == 62
    assert client.get("/api/img/mask?max=300").status_code == 200
    ok(client.post("/api/nongrain/op", json={"op": "apply"}))
    s = ok(client.get("/api/scale?activate=true"))
    s = ok(client.post("/api/scale/op", json={"op": "click", "x": 10, "y": f["detected"] + 20}))
    s = ok(client.post("/api/scale/op", json={"op": "click", "x": 210, "y": f["detected"] + 20}))
    assert len(s["pts"]) == 2
    s = ok(client.post("/api/scale/op", json={"op": "set", "length": 100, "unit": "µm"}))
    assert "0.5" in s["info"]
    s = ok(client.post("/api/scale/op", json={"op": "reset"}))
    assert s["pts"] == []
    ok(client.post("/api/scale/op", json={"op": "skip"}))
    g = ok(client.get("/api/segment?activate=true"))
    if not g["avail"]["seg"]:
        assert client.post("/api/segment/run", json={}).status_code == 400
    if not g["avail"]["sam2"]:
        assert not g["settings"]["use_sam2"]
        assert client.post("/api/segment/sam_prepare").status_code == 400
    assert client.get("/api/minerals").status_code == 409   # not computed yet
    assert files_under(tmp_path) == []


@need_data
def test_full_flow_export(tmp_path):
    """Saved notebook grains -> Use this grain map -> exports; then the export buttons are the only writers."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    web_flow = pytest.importorskip("web_flow")
    if not web_flow.REF.is_dir():
        pytest.skip("no reference 10B grains")
    web_flow.seed(tmp_path)
    before = files_under(tmp_path)
    c = web_flow.run(tmp_path)
    after = files_under(tmp_path)
    assert len(after) > len(before) and any(f.endswith("10B_counts.csv") for f in after)
    st = ok(c.get("/api/state"))
    assert {s["name"] for s in st["steps"] if s["done"]} >= {"open", "legend", "footer", "nongrain", "scale", "segment", "outputs"}
    out = ok(c.get("/api/outputs"))
    assert out["files"] and "Exported" in out["state"][0]


def test_seg_env_message(monkeypatch):
    import sys as _sys

    import pytest

    from pyPetrograph.app import backend
    import pyPetrograph.objects.qc as qc

    monkeypatch.setattr(qc, "_seg_python", lambda: None)
    with pytest.raises(backend.SegBackendError) as e:
        backend.SegBackend()._start()
    msg = str(e.value)
    assert "'work'" not in msg and _sys.prefix in msg and "launched from" in msg


def test_seg_availability_probe():
    from pyPetrograph.app import backend

    av = backend.seg_availability(refresh=True)
    assert set(av) == {"seg", "sam2", "python"} and (av["seg"] or not av["sam2"])


# -- ported from the former Qt app tests (old test_app.py), at API level -------------------------


def _to_segment(c, out):
    """Open 10B, apply legend/footer/non-grain, skip scale, load the saved notebook grains."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    web_flow = pytest.importorskip("web_flow")
    if not web_flow.REF.is_dir():
        pytest.skip("no reference 10B grains")
    web_flow.seed(out)
    open10b(c, out)
    job(c, "/api/legend/apply")
    f = ok(c.get("/api/footer"))
    ok(c.post("/api/footer/apply", json={"row": f["footer"]}))
    ok(c.get("/api/nongrain?activate=true"))
    ok(c.post("/api/nongrain/op", json={"op": "apply"}))
    ok(c.get("/api/scale?activate=true"))
    ok(c.post("/api/scale/op", json={"op": "skip"}))
    g = ok(c.get("/api/segment?activate=true"))
    if not g["n"]:
        job(c, "/api/segment/load")
    return ok(c.get("/api/segment"))


def _centroids(c, k=40):
    from shapely.geometry import Polygon

    polys = ok(c.get("/api/segment/polys"))["polys"]
    out = []
    for i, r in enumerate(polys[:k]):
        p = Polygon(list(zip(r[0::2], r[1::2])))
        q = p.representative_point()
        out.append((i, q.x, q.y, p))
    return out


def test_step_order_gating(client):
    """Later steps refuse until earlier ones are done (Qt: test_step_order)."""
    st = ok(client.get("/api/state"))
    assert [s["ready"] for s in st["steps"]][1:] == [False] * 8
    for url in ("/api/segment", "/api/minerals", "/api/gsd", "/api/outputs"):
        assert client.get(url).status_code in (400, 409), url
    for url in ("/api/outputs/export_data", "/api/outputs/export_figures", "/api/segment/use"):
        assert client.post(url, json={}).status_code in (400, 409), url


@need_data
def test_legend_tick_and_entry_validation(client, tmp_path):
    open10b(client, tmp_path)
    lg = ok(client.post("/api/legend/op", json={"op": "tick", "row": 0, "use": False}))
    assert lg["rows"][0]["use"] is False
    assert client.post("/api/legend/op", json={"op": "entry", "rgb": [9, 9, 9], "name": ""}).status_code == 400
    assert files_under(tmp_path) == []


@need_data
def test_grain_edits_select_delete_merge_undo(client, tmp_path):
    v = _to_segment(client, tmp_path)
    n0 = v["n"]
    assert n0 > 10 and v["edits"] == 0 and not v["edited"]
    cs = _centroids(client)
    i, x, y, _ = cs[0]
    v = ok(client.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    assert v["sel"] == [i]
    v = ok(client.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    assert v["sel"] == []                                      # click again deselects
    ok(client.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    v = ok(client.post("/api/segment/op", json={"op": "deselect"}))
    assert v["sel"] == []
    ok(client.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    v = ok(client.post("/api/segment/op", json={"op": "delete"}))
    assert v["n"] == n0 - 1 and v["edits"] == 1 and v["edited"]
    v = ok(client.post("/api/segment/op", json={"op": "undo"}))
    assert v["n"] == n0 and v["edits"] == 0 and not v["edited"]
    # merge two neighbouring grains (SEG leaves a few-px gap between them; Merge must bridge it)
    from pyPetrograph.webapp.state import MERGE_GAP_PX
    a, b = min(((a, b) for a in cs for b in cs if a[0] < b[0]), key=lambda ab: ab[0][3].distance(ab[1][3]))
    assert a[3].distance(b[3]) < 2 * MERGE_GAP_PX
    for g in (a, b):
        ok(client.post("/api/segment/op", json={"op": "click", "x": g[1], "y": g[2]}))
    v = ok(client.post("/api/segment/op", json={"op": "merge"}))
    assert v["n"] == n0 - 1 and v["sel"] == [] and v["edits"] == 1 and v["status"].startswith("merged 2")
    ok(client.post("/api/segment/op", json={"op": "undo"}))
    ok(client.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    ok(client.post("/api/segment/op", json={"op": "delete"}))
    v = ok(client.post("/api/segment/op", json={"op": "reset"}))
    assert v["n"] == n0 and v["edits"] == 0
    assert all(".pypetrograph" in f or "grains" in f or "threshold" in f for f in files_under(tmp_path))


@need_data
def test_seg_sam_missing_disables(tmp_path):
    """Without SEG: Run segmentation 400, edited grains cannot be used; without SAM: refine/add off."""
    from pyPetrograph.webapp.routes import create_app

    app = create_app(testing=True)
    c = TestClient(app)
    app.state.st.avail = {"seg": False, "sam2": False, "python": ""}
    v = _to_segment(c, tmp_path)
    assert v["avail"] == {"seg": False, "sam2": False, "python": ""}
    assert c.post("/api/segment/run", json={}).status_code == 400
    assert c.post("/api/segment/sam_prepare").status_code == 400
    assert not v["settings"]["use_sam2"] and v["use_ok"]          # unedited saved grains still usable
    i, x, y, _ = _centroids(c, 1)[0]
    ok(c.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    v = ok(c.post("/api/segment/op", json={"op": "delete"}))
    assert v["edited"] and not v["use_ok"]                         # edits need SEG to be saved
    p = ok(c.post("/api/segment/prompt"))
    assert p["job"] is None                                        # SAM not ready: no prompt job


@need_data
def test_minerals_add_colour_and_export_gating(tmp_path):
    from pyPetrograph.webapp.routes import create_app

    c = TestClient(create_app(testing=True))
    _to_segment(c, tmp_path)
    before = files_under(tmp_path)
    job(c, "/api/segment/use")
    m = ok(c.get("/api/minerals"))
    assert m["rows"] and "Added colours: 0" in m["summary"]
    assert c.post("/api/minerals/op", json={"op": "add", "rgb": "", "name": "X"}).status_code == 400
    assert c.post("/api/minerals/op", json={"op": "add", "rgb": "1,2,3", "name": ""}).status_code == 400
    sug = ok(c.post("/api/minerals/op", json={"op": "suggest", "row": 0}))["rgb"]
    assert len(sug) == 3
    job(c, "/api/minerals/op", {"op": "add", "rgb": "#123456", "name": "Testite"})
    assert "Added colours: 1" in ok(c.get("/api/minerals"))["summary"]
    job(c, "/api/minerals/op", {"op": "remove", "mineral": "Testite"})
    assert "Added colours: 0" in ok(c.get("/api/minerals"))["summary"]
    g = ok(c.get("/api/gsd"))
    assert g["figures"] and "grains measured" in g["summary"]
    out = ok(c.get("/api/outputs"))
    assert out["state"] == ["Nothing exported yet", "none"] and out["files"] == []
    assert not [f for f in files_under(tmp_path) if f.endswith(".csv") and f not in before]   # export-only writes
    job(c, "/api/outputs/export_data")
    out = ok(c.get("/api/outputs"))
    assert out["n_csv"] > 0 and "Exported" in out["state"][0]


def test_grain_merge_bridges_seg_gap_and_reports():
    """Regression: SEG leaves a few-px gap between neighbouring grains, so an exact union never
    merged them (Merge silently did nothing). Merge now bridges small gaps and always reports."""
    from shapely.geometry import box
    from pyPetrograph.webapp.state import WebState, MERGE_GAP_PX

    st = WebState()
    st.got_grains([box(0, 0, 10, 10), box(15, 0, 25, 10), box(100, 0, 110, 10)], "segmented")
    st.grain_click(5, 5); st.grain_click(20, 5)                # plain clicks add to the selection
    assert st.sel == {0, 1}
    st.grain_op("merge")
    assert len(st.polys) == 2 and st.sel == set() and st.edited and "merged 2" in st.status
    m = st.polys[-1]
    assert m.geom_type == "Polygon" and abs(m.area - (100 + 100 + 50)) < 5     # gap filled, shape kept
    assert m.bounds == (0.0, 0.0, 25.0, 10.0)
    st.status = ""
    st.grain_click(105, 5); st.grain_click(5, 5)               # far apart: refused, with a message
    st.grain_op("merge")
    assert len(st.polys) == 2 and st.sel == set() and "px apart" in st.status and 2 * MERGE_GAP_PX < 75
    st.grain_click(105, 5); st.grain_op("delete")
    assert len(st.polys) == 1 and "deleted 1 grain" in st.status
    st.grain_op("undo"); st.grain_op("undo")
    assert len(st.polys) == 3 and not st.edited


@need_data
def test_seg_view_passes_status(client, tmp_path):
    v = _to_segment(client, tmp_path)
    x, y = _centroids(client)[0][1:3]
    ok(client.post("/api/segment/op", json={"op": "click", "x": x, "y": y}))
    v = ok(client.post("/api/segment/op", json={"op": "delete"}))
    assert v["status"].startswith("deleted 1 grain")
    assert ok(client.get("/api/segment"))["status"] == ""      # shown once
