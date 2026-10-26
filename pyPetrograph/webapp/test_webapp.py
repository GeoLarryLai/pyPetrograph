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
    desktop app's own bookkeeping, written there too)."""
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
