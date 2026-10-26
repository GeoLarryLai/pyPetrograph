# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Browser app state: the step logic of the desktop app (``pyPetrograph.app.amics_module``)
without Qt. Every science call goes to ``AmicsSession`` (``pyPetrograph.app.amics_pipeline``)
with the same arguments as the desktop app, so both write the same files.

This module only keeps what the widgets kept in the desktop app: legend table rows, picks,
the scale line, grain edits, undo stacks, the extra (hand-named) map colours.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyPetrograph.app.amics_pipeline import AmicsSession, AmicsSettings

ORDER = ["open", "legend", "footer", "nongrain", "scale", "segment", "minerals", "gsd", "outputs"]
TITLES = ["Files", "Legend", "BSE & footer", "Non-grain regions", "Scale", "Grain segmentation",
          "Minerals", "Grain size", "Outputs"]
APP_DEFAULTS = dict(min_grain_px=10, dark_drop_frac=0.5)   # same as the desktop app / notebook 4
MINERAL_FIGURES = [("Composition", "composition"), ("Overlay", "overlay"),
                   ("Mixed grains", "mixed"), ("Unmapped grains", "unmapped")]


class UserError(ValueError):
    """Shown to the user as a message, not a traceback."""


def short_count(n: int) -> str:
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.3g}k"
    return f"{n / 1_000_000:.3g}M"


def parse_rgb(text: str):
    """'R, G, B' or '#rrggbb' -> (r, g, b) or None (same rule as the desktop Minerals step)."""
    t = str(text).strip()
    if t.startswith("#") and len(t) == 7:
        try:
            return tuple(int(t[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            return None
    parts = [p for p in t.replace(",", " ").replace("(", " ").replace(")", " ").split() if p]
    if len(parts) == 3 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        return tuple(int(p) for p in parts)
    return None


class WebState:
    """One sample at a time, one browser tab."""

    def __init__(self, settings: Optional[AmicsSettings] = None, log=None) -> None:
        import matplotlib

        matplotlib.use("Agg")
        self.logs: List[str] = []
        self.session = AmicsSession(settings or AmicsSettings(**APP_DEFAULTS), log=log or self.log)
        self.done: set = set()
        self.version = 0
        self.status = ""
        # legend
        self.rows: List[Dict[str, Any]] = []       # {rgb, name, use, origin: [name, rgb] | None}
        self.legend_hist: List[List[Dict[str, Any]]] = []
        self.legend_read_for = None
        # non-grain
        self.points: List[Tuple[float, float]] = []
        self.thr_typed: Optional[float] = None
        self.ng_filled_for = None
        # scale
        self.pts: List[Tuple[float, float]] = []
        self.length = float(self.session.s.scale_length)
        self.unit = self.session.s.scale_unit
        self.um_text = ""
        self.saved_scale: Optional[dict] = None
        # grains
        self.polys: List[Any] = []
        self.base: List[Any] = []
        self.source = ""
        self.sel: set = set()
        self.ghist: List[List[Any]] = []
        self.edited = False
        self._tree = None
        self.poly_version = 0
        self.sam_ready = False
        self.prompt_pts: List[Tuple[float, float, bool]] = []
        self.prompt_box = None
        self.preview = None
        self.avail: Optional[Dict[str, Any]] = None
        self.exported = {"data": [], "figures": []}

    # -- misc ---------------------------------------------------------------------------
    def log(self, text: str) -> None:
        self.logs.append(str(text))
        self.logs = self.logs[-200:]

    def finish(self, name: str) -> None:
        i = ORDER.index(name)
        self.done -= set(ORDER[i + 1:])
        self.done.add(name)

    def undone(self, name: str) -> None:
        self.done -= set(ORDER[ORDER.index(name):])

    def ready(self, i: int) -> bool:
        needs = {1: "open", 2: "legend", 3: "footer", 4: "nongrain", 5: "nongrain"}.get(i)
        if i >= 6:
            needs = "segment"
        return not needs or needs in self.done

    # -- 1 files ----------------------------------------------------------------------------
    @staticmethod
    def last_files() -> Dict[str, str]:
        from pyPetrograph.amics.windows import load_last_files
        from pyPetrograph.common.paths import get_output_dir, project_root

        last = {k: str(v) for k, v in (load_last_files() or {}).items()}
        last["out"] = str(get_output_dir() or (project_root() / "outputs"))
        return last

    @staticmethod
    def complete_files(key: str, cur: Dict[str, Optional[str]]) -> Dict[str, str]:
        """Same companion-file guess as the desktop '…' buttons."""
        from pyPetrograph.amics.file_match import complete

        args = {k: (v if k == key else (v if v and Path(v).is_file() else None))
                for k, v in cur.items() if k in ("legend", "bse", "mineral")}
        paths, auto = complete(**args)
        return {k: str(paths[k]) for k in auto if paths.get(k)}

    def open(self, legend: str, bse: str, mineral: str, out: str, sample: str = "") -> str:
        from pyPetrograph import set_output_dir

        set_output_dir(out.strip() or None)
        try:
            self.session.set_files(legend, bse, mineral, sample.strip() or None)
        except Exception as exc:
            raise UserError(str(exc))
        self.log(f"sample {self.session.sample}")
        self.finish("open")
        self.legend_read_for = None
        self.ng_filled_for = None
        self.polys, self.base, self.sel, self.ghist, self.edited = [], [], set(), [], False
        self.poly_version += 1
        self.exported = {"data": [], "figures": []}
        return self.session.sample

    # -- 2 legend ---------------------------------------------------------------------------
    def ocr_rows(self):
        return list((self.session.legend_auto or {}).items())

    def read_legend(self) -> None:
        s = self.session
        self.legend_read_for = s.legend_file
        legend = s.read_legend()
        self.legend_hist = []
        sv = s.saved()
        if sv.get("legend") and dict(sv["legend"]) != dict(legend):
            self._fill(sv["legend"], skip=(sv.get("legend_edit") or {}).get("skip"))
            self.status = "showing your saved legend edits (Reset all to OCR drops them)"
        else:
            self._fill(legend, skip=(sv.get("legend_edit") or {}).get("skip"))

    def _fill(self, legend, skip=None) -> None:
        skip = set(self.session.s.skip_names if skip is None else skip)
        ocr = self.ocr_rows()
        by_name = {n: (n, tuple(c)) for n, c in ocr}
        by_rgb = {tuple(c): (n, tuple(c)) for n, c in ocr}
        used, rows = set(), []
        for name, rgb in dict(legend).items():
            rgb = tuple(int(c) for c in rgb)
            org = by_name.get(name) or by_rgb.get(rgb)
            if org is not None and org in used:
                org = None
            if org is not None:
                used.add(org)
            rows.append({"rgb": list(rgb), "name": name, "use": name not in skip,
                         "origin": None if org is None else [org[0], list(org[1])]})
        self.rows = rows

    def _push_legend(self) -> None:
        import copy

        self.legend_hist.append(copy.deepcopy(self.rows))
        self.legend_hist = self.legend_hist[-50:]

    def legend_rows(self) -> List[Dict[str, Any]]:
        out = []
        for r in self.rows:
            o, name, rgb = r["origin"], r["name"].strip(), tuple(r["rgb"])
            if o is None:
                mark, tip = "+", "added entry (not read by OCR)"
            elif o[0] != name or tuple(o[1]) != rgb:
                what = []
                if o[0] != name:
                    what.append(f"OCR name '{o[0]}'")
                if tuple(o[1]) != rgb:
                    what.append(f"OCR colour {tuple(o[1])}")
                mark, tip = "\u270E", "edited; " + ", ".join(what)
            else:
                mark, tip = "", "as read by OCR"
            out.append({**r, "mark": mark, "tip": tip})
        return out

    def entry_problem(self, rgb, name: str, row: Optional[int]) -> str:
        name, rgb = name.strip(), tuple(int(c) for c in rgb)
        if not name:
            return "Type a mineral name."
        if any(not (0 <= c <= 255) for c in rgb) or len(rgb) != 3:
            return "R, G, B must be 0-255."
        for i, r in enumerate(self.rows):
            if i == row:
                continue
            if tuple(r["rgb"]) == rgb:
                return f"RGB {rgb} is already used by '{r['name'].strip()}'."
            if r["name"].strip() == name:
                return f"'{name}' is already in the legend."
        return ""

    def legend_op(self, op: str, **kw) -> None:
        """rename / tick / entry (add or change colour+name) / delete / revert / undo / reset / load_saved."""
        rows = self.rows
        if op == "rename":
            i = int(kw["row"])
            if rows[i]["name"] != kw["name"]:
                self._push_legend()
                rows[i]["name"] = str(kw["name"])
        elif op == "tick":
            rows[int(kw["row"])]["use"] = bool(kw["use"])   # not an undo step (as in the desktop app)
        elif op == "entry":
            row = kw.get("row")
            row = None if row is None else int(row)
            p = self.entry_problem(kw["rgb"], kw["name"], row)
            if p:
                raise UserError(p)
            self._push_legend()
            rgb = [int(c) for c in kw["rgb"]]
            if row is None:
                rows.append({"rgb": rgb, "name": kw["name"].strip(), "use": True, "origin": None})
            else:
                rows[row].update(rgb=rgb, name=kw["name"].strip())
        elif op == "delete":
            idx = {int(i) for i in kw.get("rows", [])}
            if idx:
                self._push_legend()
                self.rows = [r for i, r in enumerate(rows) if i not in idx]
                n = len(idx)
                self.status = f"deleted {n} legend entr{'y' if n == 1 else 'ies'} (Undo brings {'it' if n == 1 else 'them'} back)"
        elif op == "revert":
            idx = [int(i) for i in kw.get("rows", []) if 0 <= int(i) < len(rows) and rows[int(i)]["origin"] is not None]
            if idx:
                self._push_legend()
                for i in idx:
                    o = rows[i]["origin"]
                    rows[i].update(name=o[0], rgb=list(o[1]))
        elif op == "undo":
            if self.legend_hist:
                self.rows = self.legend_hist.pop()
        elif op == "reset":
            if self.session.legend_auto:
                self._push_legend()
                self._fill(self.session.legend_auto)
        elif op == "load_saved":
            sv = self.session.saved()
            if "legend" not in sv:
                raise UserError("no saved legend for this sample")
            self._push_legend()
            self._fill(sv["legend"], skip=(sv.get("legend_edit") or {}).get("skip"))
            self.status = f"loaded {len(sv['legend'])} legend rows from {Path(sv['folder']).name}/"
        elif op == "clear_saved":
            self.session.clear_saved("legend")
        else:
            raise UserError(f"unknown legend op {op}")

    def legend_edited(self):
        legend, skip = {}, []
        for r in self.rows:
            name = r["name"].strip()
            if not name or name in legend:
                continue
            legend[name] = tuple(int(c) for c in r["rgb"])
            if not r["use"]:
                skip.append(name)
        return legend, skip

    def legend_state(self) -> Tuple[str, str]:
        s = self.session
        sv = s.saved()
        legend, _ = self.legend_edited()
        n_ed = sum(1 for r in self.legend_rows() if r["mark"])
        n_del = max(0, len(self.ocr_rows()) - sum(1 for r in self.rows if r["origin"] is not None))
        ed = f" ({n_ed} edited / added, {n_del} deleted)" if (n_ed or n_del) else ""
        if "legend" in sv and dict(sv["legend"]) == legend:
            return f"Saved: {len(legend)} names{ed}", "saved"
        if "legend" in sv:
            return f"Saved legend differs from the table ({len(legend)} names{ed}): not saved yet", "unsaved"
        if legend:
            return f"{len(legend)} names{ed}: not saved yet", "unsaved"
        return "", "none"

    def apply_legend(self, job=None) -> None:
        s = self.session
        legend, skip = self.legend_edited()
        if not legend:
            raise UserError("no legend rows")
        rgbs = list(legend.values())
        if len(set(rgbs)) != len(rgbs):
            raise UserError("two legend entries have the same colour: change one first")
        s.set_legend(None if legend == dict(s.legend_auto) else legend)
        default = AmicsSettings().skip_names
        if set(skip) != {n for n in default if n in legend}:
            s.s.skip_names = [n for n in default if n not in legend] + skip
        s.save_legend_edit(legend, skip, self.ocr_rows())
        if job:
            job.progress(1, 2, "Loading BSE")
        s.prepare_bse()
        self.finish("legend")

    def legend_plot(self):
        from pyPetrograph.amics import plot_legend

        legend, skip = self.legend_edited()
        if not legend:
            return None
        return plot_legend(legend, [n for n in self.session.s.skip_names if n in legend] + skip)

    # -- 3 footer ---------------------------------------------------------------------------
    def apply_footer(self, row: int) -> None:
        s = self.session
        F = int(row)
        if F != s.footer:
            s.set_footer(None if F == s.detected_footer() else F)
        self.finish("footer")

    # -- 4 non-grain --------------------------------------------------------------------------
    def ng_load_saved(self, force: bool = False) -> bool:
        s = self.session
        key = (str(s.bse_file), str(s.P.get("folder")))
        if not force and (self.ng_filled_for == key or self.points or self.thr_typed is not None):
            return False
        self.ng_filled_for = key
        sv = s.saved()
        if "threshold" not in sv:
            if force:
                self.status = "nothing saved for this sample yet"
            return False
        self.points = [tuple(map(float, p)) for p in sv.get("points") or []]
        self.thr_typed = None if self.points else float(sv["threshold"])
        return True

    def ng_current_threshold(self) -> Optional[float]:
        if self.thr_typed is not None:
            return float(self.thr_typed)
        if self.points:
            return self.session.threshold_from(self.points)
        return None

    def ng_click(self, x: float, y: float, button: int, scale: float = 1.0) -> None:
        if button == 0:
            i = self._near(self.points, x, y, scale)
            if i is None:
                self.points.append((float(x), float(y)))
            else:
                self.points.pop(i)
        elif button == 2 and self.points:
            d = [(px - x) ** 2 + (py - y) ** 2 for px, py in self.points]
            self.points.pop(int(np.argmin(d)))

    @staticmethod
    def _near(pts, x, y, scale: float, px: float = 9.0) -> Optional[int]:
        if not pts:
            return None
        d = [math.hypot(a - x, b - y) for a, b, *_ in pts]
        i = int(np.argmin(d))
        return i if d[i] * (scale or 1.0) <= px else None

    def ng_is_saved(self) -> bool:
        sv = self.session.saved()
        if "threshold" not in sv:
            return False
        if self.thr_typed is not None:
            return not sv.get("points") and abs(sv["threshold"] - self.thr_typed) < 1e-9
        return [tuple(map(float, p)) for p in sv.get("points") or []] == [tuple(map(float, p)) for p in self.points]

    def ng_state(self) -> Tuple[str, str]:
        sv = self.session.saved()
        n = len(self.points)
        if "threshold" in sv and self.ng_is_saved():
            n_saved = len(sv.get("points") or [])
            how = f"from {n_saved} picks" if n_saved else "typed"
            return (f"Saved: gray level {sv['threshold']:.1f} {how} (threshold.json)"
                    + ("" if "nongrain" in self.done else ". Press Apply to use it."), "saved")
        if n or self.thr_typed is not None:
            return f"{n} picks: not applied / not saved yet", "unsaved"
        return "No picks yet: click on epoxy or holes", "none"

    def ng_mask(self):
        """(mask, info text) for the blue tint."""
        s = self.session
        try:
            t = self.ng_current_threshold()
        except ValueError:
            t = None
        if t is None:
            return ~s.bse_valid, "no picks yet"
        m = s.preview_dark(t)
        return m, f"gray level {t:.1f}; non-grain (blue) {100 * float(m.mean()):.1f}% of the image"

    def ng_apply(self) -> None:
        try:
            self.session.set_threshold(self.thr_typed, self.points)
        except ValueError as exc:
            raise UserError(str(exc))
        self.finish("nongrain")

    def ng_clear_saved(self) -> None:
        self.session.clear_saved("threshold")
        self.points, self.thr_typed = [], None
        self.undone("nongrain")

    # -- 5 scale ------------------------------------------------------------------------------
    def sc_activate(self) -> None:
        sc = self.session.saved().get("scale")
        self.saved_scale = dict(sc) if sc else None

    def sc_typed(self) -> Optional[float]:
        try:
            v = float(str(self.um_text).strip())
        except ValueError:
            return None
        return v if v > 0 else None

    def sc_drawn(self):
        if len(self.pts) != 2 or self.length <= 0:
            return None
        try:
            return self.session.scale_from_line(self.pts[0], self.pts[1], float(self.length), self.unit)
        except ValueError:
            return None

    def sc_click(self, x, y, button: int, scale: float = 1.0) -> None:
        if button == 0:
            i = self._near(self.pts, x, y, scale)
            if i is not None:
                self.pts.pop(i)
            else:
                if len(self.pts) >= 2:
                    self.pts = []
                self.pts.append((float(x), float(y)))
        elif button == 2 and self.pts:
            self.pts.pop()

    def sc_info(self) -> str:
        t, d = self.sc_typed(), self.sc_drawn()
        if t:
            return f"{t:.4f} µm/px (typed, not saved)"
        if d:
            return f"{d['length']:g} {d['unit']} = {d['px_length']:.1f} px → {d['um_per_px']:.4f} µm/px"
        if len(self.pts) == 1:
            return "click the other end of the scale bar"
        return "click both ends of the scale bar"

    def sc_state(self) -> Tuple[str, str]:
        s = self.session
        ss, t, d = self.saved_scale, self.sc_typed(), self.sc_drawn()
        if t or d:
            v = t or d["um_per_px"]
            return f"New scale {v:.4f} µm/px: not applied" + ("" if t else " / not saved"), "unsaved"
        if ss:
            used = "scale" in self.done and s.scale == ss
            what = (f"{ss['length']:g} {ss.get('unit', '')} line = " if ss.get("length") else "")
            src = "scale.json" if s.saved_path("scale_json") else "grains file"
            return (f"Saved: {what}{float(ss['um_per_px']):.4f} µm/px ({src})"
                    + ("" if used else ". Apply or Skip keeps it."), "saved")
        if "scale" in self.done and s.scale:
            return f"Using {s.um_per_px:.4f} µm/px", "saved"
        return "No scale: sizes will be in pixels", "none"

    def sc_apply(self) -> None:
        t, d = self.sc_typed(), self.sc_drawn()
        if t is None and d is None and len(self.pts) == 1:
            raise UserError("click the other end of the scale bar first")
        if t is None and d is None:
            self.sc_skip()
            return
        self.session.set_scale(um_per_px=t, drawn=d if t is None else None)
        if t is None:
            self.saved_scale, self.pts = dict(d), []
        self.finish("scale")

    def sc_skip(self) -> None:
        s = self.session
        ss = self.saved_scale if self.saved_scale is not None else s.saved().get("scale")
        if ss and ss.get("p1") and ss.get("p2"):
            s.set_scale(drawn=dict(ss))
        elif ss:
            s.scale = dict(ss)
        else:
            s.set_scale()
        self.finish("scale")

    def sc_clear_saved(self) -> None:
        self.session.clear_saved("scale")
        self.saved_scale = None
        self.session.set_scale()
        self.undone("scale")

    # -- 6 grains -----------------------------------------------------------------------------
    def availability(self) -> Dict[str, Any]:
        if self.avail is None:
            from pyPetrograph.app import backend as _backend

            try:
                self.avail = _backend.seg_availability()
            except Exception:
                self.avail = {"seg": False, "sam2": False, "python": ""}
        return self.avail

    def sync_settings(self, use_sam2=None, min_px=None, drop=None) -> None:
        st = self.session.s
        av = self.availability()
        if use_sam2 is not None:
            st.use_sam2 = bool(use_sam2) and bool(av.get("sam2"))
        elif not av.get("sam2"):
            st.use_sam2 = False
        if min_px is not None:
            st.min_grain_px = float(int(min_px))
        if drop is not None:
            st.dark_drop_frac = float(drop)

    def seg_activate(self) -> None:
        if "scale" not in self.done:   # scale step not visited: keep the saved scale, as the notebook does
            self.sc_activate()
            self.sc_skip()

    def got_grains(self, polys, source: str = "") -> None:
        self.polys, self.sel, self.ghist, self.edited = list(polys), set(), [], False
        self.base = list(polys)
        self.source = source or self.source
        self._tree = None
        self.poly_version += 1
        self.log(f"{len(self.polys)} grains")

    def tree(self):
        if self._tree is None:
            from shapely.strtree import STRtree

            self._tree = STRtree(self.polys) if self.polys else None
        return self._tree

    def _gpush(self) -> None:
        self.ghist.append(list(self.polys))
        self.edited = True
        self._tree = None

    def grain_click(self, x, y) -> None:
        from shapely.geometry import Point

        t = self.tree()
        if t is None:
            return
        hits = [int(i) for i in t.query(Point(x, y), predicate="intersects")]
        if not hits:
            return
        i = min(hits, key=lambda k: self.polys[k].area)
        self.sel.symmetric_difference_update({i})

    def grain_op(self, op: str) -> None:
        import shapely
        from shapely.geometry import Polygon

        if op == "deselect":
            self.sel.clear()
        elif op == "delete" and self.sel:
            self._gpush()
            self.polys = [p for i, p in enumerate(self.polys) if i not in self.sel]
            self.sel.clear()
            self.poly_version += 1
        elif op == "merge" and len(self.sel) >= 2:
            u = shapely.unary_union([self.polys[i] for i in self.sel])
            if u.geom_type != "Polygon":
                self.status = "selected grains do not touch: not merged"
                self.sel.clear()
                return
            self._gpush()
            self.polys = [p for i, p in enumerate(self.polys) if i not in self.sel] + [Polygon(u.exterior.coords)]
            self.sel.clear()
            self.poly_version += 1
        elif op == "undo" and self.ghist:
            self.polys = self.ghist.pop()
            self.sel.clear()
            self._tree = None
            self.edited = bool(self.ghist)
            self.poly_version += 1
        elif op == "reset" and self.ghist:
            self.got_grains(self.base)

    def accept_prompt(self) -> None:
        if self.preview is None:
            return
        self._gpush()
        self.polys.append(self.preview)
        self.poly_version += 1
        self.clear_prompts()

    def clear_prompts(self) -> None:
        self.prompt_pts, self.prompt_box, self.preview = [], None, None

    def prompt_request(self) -> dict:
        return {"points": [[x, y] for x, y, _ in self.prompt_pts],
                "labels": [p for _, _, p in self.prompt_pts],
                "box": list(self.prompt_box) if self.prompt_box else None}

    def seg_state(self) -> Tuple[str, str]:
        s = self.session
        p = s.saved_path("grains") if s.P else None
        on_disk = p is not None
        n = len(self.polys)
        seg_ok = self.availability().get("seg")
        if not n:
            if not seg_ok:
                return ("No grains yet: Load saved grains" if on_disk else
                        "No grains yet: no saved grains and SegmentEveryGrain is not installed"), "none"
            return "No grains yet: Run segmentation" + (" or Load saved grains" if on_disk else ""), "none"
        if self.edited and not seg_ok:
            return f"{n} grains, {len(self.ghist)} edits: saving edits needs SegmentEveryGrain", "unsaved"
        if self.edited:
            return f"{n} grains, {len(self.ghist)} edits not saved: 'Use this grain map' saves them", "unsaved"
        if on_disk and self.source in ("saved", "segmented", "measured"):
            return (f"Saved: {n} grains ({Path(p).name})"
                    + ("" if "segment" in self.done else ". 'Use this grain map' measures them."), "saved")
        return f"{n} grains: not saved yet", "unsaved"

    def use_grain_map(self, backend, job=None) -> None:
        """Desktop 'Use this grain map': save edits, measure, then sections 3 + 4 into the cache."""
        s = self.session
        if not self.polys:
            raise UserError("no grains")
        if self.edited and not self.availability().get("seg"):
            raise UserError("saving edited grains needs SegmentEveryGrain")
        if not s.extra_legend:
            s.set_extra_legend(s.saved().get("extra_legend") or {})
        N = 5
        step = (lambda n, m: job.progress(n, N, m)) if job else (lambda n, m: None)
        step(0, "saving grains" if self.edited else "measuring grains")
        if self.edited:
            s.save_edits(backend, self.polys)
        else:
            s.grain_polys = list(self.polys)
        s.finalize_grains()
        self._downstream(step, 1)
        self.got_grains(s.grain_polys, "measured")
        self.finish("segment")

    def _downstream(self, step, k0: int) -> None:
        """``AmicsSession.compute_downstream`` in its three parts, with progress."""
        s = self.session
        step(k0, "colours and overlay")
        s.minerals()
        step(k0 + 1, "mineral vote, composition")
        s.vote()
        step(k0 + 2, "tables and knolling")
        s.outputs()
        s.log(f"{len(s.mapped)} mapped grains; figures and tables ready (not exported yet)")
        self.version += 1
        self.exported = {"data": [], "figures": []}

    def seg_clear_saved(self) -> None:
        s = self.session
        if (s.saved_path("grains") if s.P else None) is None:
            raise UserError("no saved grains for this sample")
        s.clear_saved("grains")
        self.undone("segment")

    # -- 7 minerals ---------------------------------------------------------------------------
    def recount(self, extra, job=None) -> None:
        s = self.session
        s.set_extra_legend(extra)
        if "segment" not in self.done:
            return
        N = 3
        self._downstream((lambda n, m: job.progress(n, N, m)) if job else (lambda n, m: None), 0)
        self.finish("segment")

    def suggest_rgb(self, row: Optional[int]):
        rows = self.session.mineral_table()
        if row is not None and 0 <= row < len(rows) and rows[row]["map_rgbs"]:
            return rows[row]["map_rgbs"][-1]
        unl = [r for r in rows if r["status"] == "unlisted"]
        return unl[0]["map_rgbs"][0] if unl else (0, 0, 0)
