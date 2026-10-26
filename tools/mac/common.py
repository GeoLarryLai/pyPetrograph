import json, os, threading, time
from pathlib import Path
import psutil
ROOT = Path(__file__).resolve().parents[2]; EX = ROOT / "example_data"
SAMPLES = {"10B": ("10B_BSE_Image.tiff", "10B_Mineral_Image.tiff", "10-Mineral_Legend.tiff"),
           "Con-1": ("Con-1_BSE_Image.tiff", "Con-1_Mineral_Image.tiff", "Con-1_Mineral_Legend.tiff")}
def files(s):
    if s.startswith("big"):  # 2x nearest-upscaled 10B made by make_big.py
        d = Path.home() / "pp_webapp_test_out" / "big"; return [str(d / f) for f in ("big_BSE_Image.tiff", "big_Mineral_Image.tiff", "10-Mineral_Legend.tiff")]
    return [str(EX / f) for f in SAMPLES[s]]
class Peak:
    """peak RSS of this process + children (seg server), sampled every 0.1 s"""
    def __init__(self): self.peak = 0; self._s = False
    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self
    def _run(self):
        me = psutil.Process()
        while not self._s:
            try:
                r = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                self.peak = max(self.peak, r)
            except Exception: pass
            time.sleep(0.1)
    def __exit__(self, *a): self._s = True
class Timer:
    def __init__(self): self.t = {}
    def __call__(self, k):
        T = self
        class C:
            def __enter__(s): s.a = time.perf_counter()
            def __exit__(s, *a): T.t[k] = round(time.perf_counter() - s.a, 3); print(f"[t] {k}: {T.t[k]} s", flush=True)
        return C()
def dump(path, d): Path(path).write_text(json.dumps(d, indent=1))
def seed(out, s, grains=None):
    d = Path(out) / "amics" / s; d.mkdir(parents=True, exist_ok=True)
    (d / f"{s}_threshold.json").write_text('{\n "threshold": 62.0,\n "points": []\n}')
    if grains:
        import shutil; shutil.copy(grains, d / f"{s}_grains.geojson")
def polyhash(rings):
    """rings: list of flat or Nx2 coordinate lists -> (n, md5 of 1-decimal coords, total area)"""
    import hashlib, numpy as np
    h = hashlib.md5(); area = 0.0
    for r in rings:
        a = np.round(np.asarray(r, float).reshape(-1, 2), 1); h.update(a.tobytes())
        x, y = a[:, 0], a[:, 1]; area += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return [len(rings), h.hexdigest()[:12], round(area, 1)]
def rings_of(polys):
    out = []
    for p in polys:
        g = p if p.geom_type == "Polygon" else max(getattr(p, "geoms", [p]), key=lambda q: q.area)
        out.append(list(g.exterior.coords))
    return out
