// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// viewer.js - canvas image viewer: zoom / pan, a downsampled preview when zoomed out and the
// full-size image (loaded once, on demand) when zoomed in, a mask layer, vector overlays in
// image coordinates, click / drag-box events in image coordinates. No dependencies.
"use strict";
const Viewer = (() => {
  const PREVIEW = 1600;
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  let W = 0, H = 0, k = 1, tx = 0, ty = 0, dpr = 1;
  let base = null, mask = null, maskOn = true, overlay = null, boxMode = false;
  let onClick = null, onBox = null, onKey = null, dragging = null, boxNow = null;
  function layer(url, w, h, nearest) {
    const L = {url, w, h, nearest, prev: null, full: null, loadingFull: false};
    const im = new Image(); im.onload = () => { L.prev = im; draw(); if (Math.max(w, h) > PREVIEW) setTimeout(() => pick(L, true), 300); }; im.src = url + (url.includes("?") ? "&" : "?") + "max=" + PREVIEW;
    return L;
  }
  function pick(L, preload) {   // preview when it is sharp enough, else the full image (pre-loaded in the background)
    const need = k * dpr * Math.max(L.w, L.h) > PREVIEW * 1.05 && Math.max(L.w, L.h) > PREVIEW;
    if ((need || preload) && !L.full && !L.loadingFull) {
      L.loadingFull = true; const im = new Image(); im.onload = () => { L.full = im; draw(); }; im.src = L.url;
    }
    return (need && L.full) ? L.full : (L.prev || L.full);
  }
  function resize() {
    dpr = window.devicePixelRatio || 1; const r = cv.getBoundingClientRect();
    W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); draw();
  }
  function fit() {
    if (!base) return; k = Math.min(W / base.w, H / base.h) * 0.98;
    tx = (W - base.w * k) / 2; ty = (H - base.h * k) / 2; draw();
  }
  function zoomTo(x0, y0, x1, y1) {
    k = Math.min(W / (x1 - x0), H / (y1 - y0)); tx = -x0 * k + (W - (x1 - x0) * k) / 2; ty = -y0 * k + (H - (y1 - y0) * k) / 2; draw();
  }
  function setImage(url, w, h, opts = {}) {
    const same = base && base.w === w && base.h === h;
    base = layer(url, w, h, false); document.getElementById("empty").hidden = true;
    if (!same || opts.fit) fit(); else draw();
  }
  function clear() { base = null; mask = null; overlay = null; draw(); }
  function setMask(url) { mask = url && base ? layer(url, base.w, base.h, true) : null; draw(); }
  function showMask(on) { maskOn = on; draw(); }
  function setOverlay(fn) { overlay = fn; draw(); }
  const T = {x: x => tx + x * k, y: y => ty + y * k, get k() { return k; }, get tx() { return tx; }, get ty() { return ty; }, get dpr() { return dpr; }};
  // Overlay cache: the overlay (e.g. 2,644 grain outlines) is drawn once into an offscreen canvas.
  // While the view moves (wheel / drag / buttons) that bitmap is scaled and shifted instead of
  // being redrawn; ~140 ms after the last move the overlay is drawn sharp again.
  const oc = {cv: document.createElement("canvas"), k: 1, tx: 0, ty: 0, ok: false};
  let raf = 0, moving = false, settle = 0, lastPaintMs = 0;
  function draw() { oc.ok = false; if (!raf) raf = requestAnimationFrame(paint); }
  function moved() { moving = true; clearTimeout(settle); settle = setTimeout(() => { moving = false; draw(); }, 140);
    if (!raf) raf = requestAnimationFrame(paint); }
  function paint() {
    const t0 = performance.now();
    raf = 0; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    if (!base) return;
    for (const L of [base, maskOn ? mask : null]) {
      if (!L) continue; const im = pick(L); if (!im) continue;
      ctx.imageSmoothingEnabled = !L.nearest && k < 2;
      ctx.drawImage(im, tx, ty, L.w * k, L.h * k);
    }
    if (overlay) {
      if (moving && oc.ok) { const s = k / oc.k; ctx.imageSmoothingEnabled = true;
        ctx.drawImage(oc.cv, 0, 0, oc.cv.width, oc.cv.height, tx - oc.tx * s, ty - oc.ty * s, W * s, H * s); }
      else {
        if (oc.cv.width !== cv.width || oc.cv.height !== cv.height) { oc.cv.width = cv.width; oc.cv.height = cv.height; }
        const o = oc.cv.getContext("2d"); o.setTransform(1, 0, 0, 1, 0, 0); o.clearRect(0, 0, oc.cv.width, oc.cv.height);
        o.setTransform(dpr, 0, 0, dpr, 0, 0); o.save(); overlay(o, T); o.restore();
        oc.k = k; oc.tx = tx; oc.ty = ty; oc.ok = true;
        ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(oc.cv, 0, 0); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }
    if (boxNow) { ctx.strokeStyle = "#64d2ff"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.strokeRect(T.x(Math.min(boxNow[0], boxNow[2])), T.y(Math.min(boxNow[1], boxNow[3])), Math.abs(boxNow[2] - boxNow[0]) * k, Math.abs(boxNow[3] - boxNow[1]) * k); ctx.setLineDash([]); }
    lastPaintMs = performance.now() - t0;
  }
  const img = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left - tx) / k, (e.clientY - r.top - ty) / k]; };
  cv.addEventListener("wheel", e => {
    e.preventDefault(); const r = cv.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
    const f = Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)), nk = Math.min(64, Math.max(0.01, k * f));
    tx = sx - (sx - tx) * nk / k; ty = sy - (sy - ty) * nk / k; k = nk; moved();
  }, {passive: false});
  cv.addEventListener("contextmenu", e => e.preventDefault());
  cv.addEventListener("mousedown", e => {
    cv.focus(); const box = boxMode && e.button === 0 && !e.altKey;
    dragging = {x: e.clientX, y: e.clientY, tx, ty, b: e.button, moved: false, box, p0: img(e), shift: e.shiftKey};
  });
  window.addEventListener("mousemove", e => {
    const c = document.getElementById("coord");
    if (base && e.target === cv) { const [x, y] = img(e); c.textContent = (x >= 0 && y >= 0 && x < base.w && y < base.h) ? `x ${x | 0}  y ${y | 0}` : ""; }
    if (!dragging) return;
    const dx = e.clientX - dragging.x, dy = e.clientY - dragging.y;
    if (Math.hypot(dx, dy) > 4) dragging.moved = true;
    if (!dragging.moved) return;
    if (dragging.box) { const p = img(e); boxNow = [dragging.p0[0], dragging.p0[1], p[0], p[1]]; moved(); return; }
    cv.classList.add("pan"); tx = dragging.tx + dx; ty = dragging.ty + dy; moved();
  });
  window.addEventListener("mouseup", e => {
    if (!dragging) return; const d = dragging; dragging = null; cv.classList.remove("pan");
    if (d.box && d.moved && boxNow) { const b = boxNow; boxNow = null; draw();
      if (onBox) onBox(Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])); return; }
    if (!d.moved && onClick && base) { const [x, y] = d.p0; if (x >= 0 && y >= 0 && x < base.w && y < base.h) onClick(x, y, d.b, d.shift || e.shiftKey, k); }
  });
  cv.addEventListener("dblclick", fit);
  document.getElementById("fitBtn").onclick = fit;
  const zc = f => { tx = W / 2 - (W / 2 - tx) * f; ty = H / 2 - (H / 2 - ty) * f; k *= f; moved(); };
  document.getElementById("zin").onclick = () => zc(1.4); document.getElementById("zout").onclick = () => zc(1 / 1.4);
  window.addEventListener("resize", resize); setTimeout(resize, 0);
  if (window.ResizeObserver) new ResizeObserver(() => { const r = cv.getBoundingClientRect(); if (r.width !== W || r.height !== H) { const f = !!base; resize(); if (f) fit(); } }).observe(cv);
  return {setImage, setMask, showMask, setOverlay, fit, zoomTo, clear, draw,
    set onClick(f) { onClick = f; }, set onBox(f) { onBox = f; }, set onKey(f) { onKey = f; },
    set boxMode(b) { boxMode = b; boxNow = null; }, get paintMs() { return lastPaintMs; }, screen: (x, y) => { const r = cv.getBoundingClientRect(); return [r.left + tx + x * k, r.top + ty + y * k]; }, get size() { return [W, H]; }, get k() { return k; }, focus: () => cv.focus()};
})();
// overlay helpers (image coordinates)
const Draw = {
  crosses(c, T, pts, color, size = 9) { c.strokeStyle = color; c.lineWidth = 2; for (const [x, y] of pts) { const X = T.x(x), Y = T.y(y), s = size / 2;
    c.beginPath(); c.moveTo(X - s, Y - s); c.lineTo(X + s, Y + s); c.moveTo(X + s, Y - s); c.lineTo(X - s, Y + s); c.stroke(); } },
  line(c, T, p1, p2, color, width = 1.5, dashed = false, dots = 0) { c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dashed ? [6, 4] : []);
    c.beginPath(); c.moveTo(T.x(p1[0]), T.y(p1[1])); c.lineTo(T.x(p2[0]), T.y(p2[1])); c.stroke(); c.setLineDash([]);
    if (dots) { c.fillStyle = color; for (const p of [p1, p2]) { c.beginPath(); c.arc(T.x(p[0]), T.y(p[1]), dots, 0, 7); c.fill(); } } },
  label(c, T, x, y, text, color) { c.font = "12px -apple-system,system-ui,sans-serif"; c.fillStyle = "rgba(0,0,0,.55)";
    const w = c.measureText(text).width; c.fillRect(T.x(x) - 2, T.y(y), w + 6, 16); c.fillStyle = color; c.fillText(text, T.x(x) + 1, T.y(y) + 12); },
  ring(c, T, flat) { c.moveTo(T.x(flat[0]), T.y(flat[1])); for (let i = 2; i < flat.length; i += 2) c.lineTo(T.x(flat[i]), T.y(flat[i + 1])); c.closePath(); },
  // one Path2D per outline list (cached by list identity), stroked in image coordinates
  polys(c, T, list, color, width = 1, fill = null) { let p = Draw._paths.get(list);
    if (!p) { p = new Path2D(); for (const f of list) { if (!f || f.length < 4) continue; p.moveTo(f[0], f[1]); for (let i = 2; i < f.length; i += 2) p.lineTo(f[i], f[i + 1]); p.closePath(); }
      Draw._paths.set(list, p); }
    c.save(); c.setTransform(T.dpr * T.k, 0, 0, T.dpr * T.k, T.dpr * T.tx, T.dpr * T.ty);
    if (fill) { c.fillStyle = fill; c.fill(p); } c.strokeStyle = color; c.lineWidth = width / T.k; c.stroke(p); c.restore(); },
  _paths: new WeakMap(),
  points(c, T, pts, color, r = 5) { c.fillStyle = color; c.strokeStyle = "#fff"; c.lineWidth = 1; for (const [x, y] of pts) { c.beginPath(); c.arc(T.x(x), T.y(y), r, 0, 7); c.fill(); c.stroke(); } },
};
