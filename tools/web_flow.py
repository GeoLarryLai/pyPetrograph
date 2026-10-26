"""Drive the browser app's API (no browser) through the whole 10B flow; export to OUT."""
import shutil, sys, time
from pathlib import Path
from fastapi.testclient import TestClient
from pyPetrograph.webapp.routes import create_app

EX = Path(__file__).resolve().parents[1] / "example_data"
REF = Path("/workspace/ppq/outputs/amics/10B")


def seed(out):
    d = Path(out) / "amics" / "10B"; d.mkdir(parents=True, exist_ok=True)
    for f in ("10B_grains.geojson", "10B_threshold.json"):
        shutil.copy(REF / f, d / f)


def run(out):
    c = TestClient(create_app(testing=True))
    def ok(r):
        assert r.status_code == 200, (r.request.url, r.status_code, r.text[:400]); return r.json()
    def job(url, body=None):
        j = ok(c.post(url, json=body or {}))["job"]
        while j["state"] == "running":
            time.sleep(0.2); j = ok(c.get(f"/shell/api/jobs/{j['id']}"))
        assert j["state"] == "done", j; return j.get("result")
    ok(c.post("/api/open", json={"bse": str(EX/"10B_BSE_Image.tiff"), "mineral": str(EX/"10B_Mineral_Image.tiff"),
                                 "legend": str(EX/"10-Mineral_Legend.tiff"), "out": str(out), "sample": ""}))
    lg = ok(c.get("/api/legend"))
    if lg["need_read"]: job("/api/legend/read")
    print("legend rows", len(ok(c.get("/api/legend"))["rows"]))
    job("/api/legend/apply")
    f = ok(c.get("/api/footer")); ok(c.post("/api/footer/apply", json={"row": f["footer"]}))
    ng = ok(c.get("/api/nongrain?activate=true")); print("nongrain", ng["info"], ng["state"])
    ok(c.post("/api/nongrain/op", json={"op": "apply"}))
    ok(c.get("/api/scale?activate=true")); ok(c.post("/api/scale/op", json={"op": "skip"}))
    sg = ok(c.get("/api/segment?activate=true")); print("segment", sg["n"], sg["state"], sg["avail"])
    if not sg["n"]:
        job("/api/segment/load")
    print("grains", ok(c.get("/api/segment"))["n"])
    job("/api/segment/use")
    print(ok(c.get("/api/minerals"))["summary"]); print(ok(c.get("/api/gsd"))["summary"])
    job("/api/outputs/export_data"); job("/api/outputs/export_figures")
    for k in ("bse_stretch", "mask", "legend_plot", "knoll", "fig:Composition"):
        assert c.get(f"/api/img/{k}?max=400").status_code == 200, k
    return c


if __name__ == "__main__":
    out = Path(sys.argv[1]); shutil.rmtree(out, ignore_errors=True); seed(out); run(out)
    print(sorted(p.name for p in (out/"amics"/"10B").iterdir()))
