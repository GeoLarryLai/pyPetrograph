# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Start page, optical job (notebook 1), Advanced settings and the colour-match report."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from fastapi.testclient import TestClient  # noqa: E402

from pyPetrograph.webapp.test_webapp import EX, _to_segment, job, ok  # noqa: E402

PPL = [EX / n for n in ("AV03_ppl_10x_pic03t3.tif", "AV11_ppl_10x_pic04t4.tif", "AV12_ppl_10x_pic03t3.tif")]
need_ppl = pytest.mark.skipif(not all(p.is_file() for p in PPL), reason="example PPL photos not present")
LARRY = Path(__file__).resolve().parent / "testdata" / "larry_v005"     # Larry's notebook-1 labels (v0.0.5)
need_larry = pytest.mark.skipif(not LARRY.is_dir(), reason="label fixture not present")
# Verified with notebook 1's functions on these labels (pores %, AV03 / AV11 / AV12)
SHARED = {"AV03": 15.72, "AV11": 2.39, "AV12": 0.07}
AV03_ON_ALL = {"AV03": 15.68, "AV11": 2.70, "AV12": 0.13}
TOL = 0.05


def seed_labels(out):
    """Larry's saved labels: AV03 103 outlines, AV11 56, AV12 none (as in notebook 1)."""
    import shutil

    (Path(out) / "labels").mkdir(parents=True, exist_ok=True)
    for f in LARRY.glob("*_polygons.geojson"):
        shutil.copy(f, Path(out) / "labels" / f.name)


def pores(rows):
    return {r["image"][:4]: r["porosity_pct"] for r in rows}


def close(got, want):
    assert set(got) == set(want) and all(abs(got[k] - want[k]) <= TOL for k in want), (got, want)


@pytest.fixture
def client():
    from pyPetrograph.webapp.routes import create_app

    return TestClient(create_app(testing=True))


def test_start_page_has_two_jobs(client):
    html = client.get("/").text
    assert 'id="goOptical"' in html and 'id="goAmics"' in html and 'id="homeBtn"' in html and "<svg" in html
    assert client.get("/static/optical.js").status_code == 200


def test_porosity_summary(tmp_path):
    from pyPetrograph import porosity_summary, set_output_dir
    from pyPetrograph.common.paths import get_output_dir, stem_paths

    prev = get_output_dir()
    try:
        set_output_dir(tmp_path)
        imgs = [tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"]
        for p, pct in zip(imgs[:2], (10.0, 20.0)):
            csv = stem_paths(p).fractions_csv
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_text("class_id,name,count,area_pct,area_pct_plus,area_pct_minus\n"
                           f"1,Pores,5,{pct},1.5,0.5\n2,grains,5,{100 - pct},0,1\n")
        tab, st = porosity_summary(imgs, verbose=False)
    finally:
        set_output_dir(prev)
    assert list(tab["image"]) == ["a", "b"] and list(tab["porosity_pct"]) == [10.0, 20.0]
    assert st["n"] == 2 and st["mean"] == 15.0 and abs(st["sd"] - 7.0710678) < 1e-6 and (st["min"], st["max"]) == (10.0, 20.0)


@need_ppl
@need_larry
def test_optical_flow(client, tmp_path):
    assert client.get("/api/opt/label").status_code == 409                    # gated before photos
    seed_labels(tmp_path)
    job(client, "/api/opt/open", {"paths": [str(p) for p in PPL], "out": str(tmp_path)})
    st = ok(client.get("/api/opt/state"))
    assert [s["ready"] for s in st["steps"]] == [True, True, True, True, True, False, False]
    v = ok(client.post("/api/opt/classes", json={"names": ["pores", "grains"]}))
    assert [c["name"] for c in v["classes"]] == ["pores", "grains"]
    assert client.post("/api/opt/classes", json={"names": ["a", "A"]}).status_code == 400
    v = ok(client.get("/api/opt/label"))
    assert v["n_labelled"] == 2 and v["counts"] == {"1": 44, "2": 59}          # Larry's AV03 outlines
    # marking tools on AV12 (no labels, as in notebook 1): add, then undo back to none
    ok(client.post("/api/opt/label/op", json={"op": "open", "i": 2}))
    ok(client.post("/api/opt/label/op", json={"op": "class", "cid": 1}))
    v = ok(client.post("/api/opt/label/op", json={"op": "wand", "x": 800, "y": 500}))
    v = ok(client.post("/api/opt/label/op", json={"op": "poly", "pts": [[100, 100], [140, 100], [140, 130]]}))
    assert sum(v["counts"].values()) >= 2 and v["n_labelled"] == 3
    while sum(v["counts"].values()):
        v = ok(client.post("/api/opt/label/op", json={"op": "undo"}))
    assert v["n_labelled"] == 2
    assert (tmp_path / "labels" / "AV11_ppl_10x_pic04t4_polygons.geojson").is_file()     # shared label files
    job(client, "/api/opt/features", {"toggles": {k: True for k in ("r", "g", "b", "local_std", "local_grad")}})
    assert all(m["ready"] for m in ok(client.get("/api/opt/features"))["images"])
    job(client, "/api/opt/train")
    t = ok(client.get("/api/opt/train"))
    assert t["accuracy"] > 0.9 and "stratified" in t["text"]
    assert (tmp_path / "models" / "universal_model.joblib").is_file()
    job(client, "/api/opt/predict")
    assert len(ok(client.get("/api/opt/predict"))["images"]) == 3
    for k in ("photo", "feat:0", "pred:2", "conf:1"):
        assert client.get(f"/api/opt/img/{k}?max=400").status_code == 200, k
    por = ok(client.get("/api/opt/porosity"))
    assert por["stats"]["n"] == 3
    close(pores(por["rows"]), SHARED)
    r = job(client, "/api/opt/porosity/export")
    assert Path(r["path"]).name == "porosity_summary.csv" and Path(r["path"]).is_file()


def test_settings_and_colour_matches(tmp_path):
    from pyPetrograph.webapp.routes import create_app

    c = TestClient(create_app(testing=True))
    s = ok(c.get("/api/settings"))
    keys = {f["key"] for f in s["fields"]}
    assert {"color_tol", "white_tol", "match_max_dist", "knoll_width_px"} <= keys and not s["computed"]
    assert c.post("/api/settings", json={"color_tol": "abc"}).status_code == 400
    try:
        _to_segment(c, tmp_path)
    except Exception as e:  # pragma: no cover - needs example data
        pytest.skip(f"no 10B data: {e}")
    job(c, "/api/segment/use")
    m = ok(c.get("/api/minerals"))
    assert "matches" in m and "auto_match" in m
    r = job(c, "/api/settings", {"overlay_alpha": 0.42})
    assert r["changed"] == ["overlay_alpha"]
    r = job(c, "/api/settings", {"overlay_alpha": 0.42})
    assert r["changed"] == []


@need_ppl
@need_larry
def test_optical_label_summary_per_photo_models_and_load(client, tmp_path):
    """Notebook 1 extras: label summary figures, Train scope per_image, and Train mode load."""
    from pyPetrograph.webapp.routes import create_app

    two = [str(p) for p in PPL[:2]]
    seed_labels(tmp_path)
    job(client, "/api/opt/open", {"paths": two, "out": str(tmp_path)})
    r = job(client, "/api/opt/labelsummary")
    pngs = [Path(m["png"]) for m in r["images"]]
    assert [p.name for p in pngs] == [Path(p).stem + "_labels.png" for p in two]
    assert all(p.is_file() and p.parent == tmp_path / "labels" for p in pngs)       # notebook file + place
    s = ok(client.get("/api/opt/labelsummary"))
    assert "Label summary" in s["text"] and s["text"].count("polygons:") == 2
    assert client.get("/api/opt/img/labels:1?max=300").status_code == 200

    job(client, "/api/opt/train", {"mode": "retrain", "scope": "per_image"})
    per = [tmp_path / "models" / f"{Path(p).stem}_model.joblib" for p in two]
    assert all(p.is_file() for p in per) and not (tmp_path / "models" / "universal_model.joblib").exists()
    t = ok(client.get("/api/opt/train"))
    assert t["scope"] == "per_image" and t["accuracy"] > 0.9 and all(Path(p).stem in t["text"] for p in two)
    assert sorted(Path(m).name for m in t["models"]) == sorted(p.name for p in per)
    job(client, "/api/opt/predict")
    st = client.app.state.opt
    assert [Path(e["model_used"]) for e in st.pred] == per        # each photo used its own model

    # a fresh app, a new output folder, no examples: load the saved AV03 model and skip training
    c2 = TestClient(create_app(testing=True))
    out2 = tmp_path / "second"
    job(c2, "/api/opt/open", {"paths": [str(p) for p in PPL], "out": str(out2)})
    assert ok(c2.get("/api/opt/train"))["n_labelled"] == 0
    nolab = ok(c2.post("/api/opt/train", json={"mode": "retrain"}))["job"]
    while nolab["state"] == "running":
        nolab = ok(c2.get(f"/shell/api/jobs/{nolab['id']}"))
    assert nolab["state"] == "error" and "No labeled images" in nolab["error"]
    bad = ok(c2.post("/api/opt/train", json={"mode": "load", "model_path": str(tmp_path / "nope.joblib")}))["job"]
    while bad["state"] == "running":
        bad = ok(c2.get(f"/shell/api/jobs/{bad['id']}"))
    assert bad["state"] == "error" and "not found" in bad["error"]
    job(c2, "/api/opt/train", {"mode": "load", "model_path": str(per[0])})
    t2 = ok(c2.get("/api/opt/train"))
    assert t2["mode"] == "load" and t2["loaded"] == str(per[0].resolve()) and "training skipped" in t2["text"]
    job(c2, "/api/opt/predict")
    st2 = c2.app.state.opt
    assert [Path(e["model_used"]) for e in st2.pred] == [per[0].resolve()] * 3
    close(pores(ok(c2.get("/api/opt/porosity"))["rows"]), AV03_ON_ALL)            # AV12 ≈ 0.13 %
    assert not (out2 / "models" / "universal_model.joblib").exists()               # nothing copied
    assert all(Path(e["fractions_csv"]).is_file() for e in st2.pred)
    assert "universal_model_path" not in st2.S.__dict__                             # lookup restored
