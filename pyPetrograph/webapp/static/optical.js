// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// optical.js - the optical photo job (notebook 1): photos -> classes -> mark examples -> features
// -> train -> predict -> porosity %. The server (webapp/optical.py) runs the package functions.
"use strict";
const OPT_STEPS = [];
JOBS.optical = {steps: OPT_STEPS, api: "/api/opt", name: "Thin-section photos",
  empty: "Thin-section photos (PPL / CPL, notebook 1), step by step.<br>Start with <b>1 · Photos</b> on the right.",
  doc: G => G.n ? `Photos · ${G.n} open` + (G.photo ? ` · showing ${G.photo}` : "") + (G.folder ? ` · output folder ${G.folder}` : "") : "Thin-section photos · none open"};
const OV = {tool: "wand", pts: []};                       // optical view state (tool, outline in progress)
const rgba = (h, a) => { const c = fromHex(h) || [255, 0, 0]; return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
function drawLabels(v) {
  const by = {}; for (const p of v.polys) (by[p.cid] = by[p.cid] || []).push(p.flat);
  Viewer.setOverlay((c, T) => {
    for (const cl of v.classes) if (by[cl.id]) Draw.polys(c, T, by[cl.id], cl.color, 2, rgba(cl.color, 0.28));
    if (OV.pts.length) { const col = v.classes.find(x => x.id === v.current)?.color || "#ff3b30";
      for (let i = 1; i < OV.pts.length; i++) Draw.line(c, T, OV.pts[i - 1], OV.pts[i], col, 2);
      Draw.points(c, T, OV.pts, col, 4); }
  });
}
// 1 Photos
OPT_STEPS.push({title: "Photos", lead: "Pick the microscope photos (PPL, or a CPL colour photo) and where results go.",
  help: "One photo per line. Labels you mark are kept in &lt;output&gt;/labels/, the same files notebook 1 uses.",
  controls: [["Photos", "one image file per line (.tif, .png, .jpg)"], ["Add photo…", "browse for a photo and add it to the list"],
    ["Output folder", "labels/, models/ and predictions/ go here (notebook 1 uses outputs/)"], ["Open photos", "load the list and show the first photo"]],
  async enter() {
    const f = await api("GET", "/api/opt/files");
    body(`<div class="hintt">Photos (one per line)</div><textarea id="opPaths" rows="6" style="width:100%;font-size:11.5px">${esc(f.paths.join("\n"))}</textarea>
      <div class="btns"><button id="opAdd" title="Browse for a photo">Add photo…</button></div>
      <div class="f"><label>Output folder</label><input type="text" id="opOut" value="${esc(f.out)}"><button id="opOutB" title="Browse">…</button></div>
      <p class="hintt">Wheel = zoom, drag = pan, F or double-click = fit.</p>`);
    on("opAdd", "onclick", async () => { const lines = $("opPaths").value.split("\n").filter(Boolean); const p = await browse(lines[lines.length - 1] || "", false, "Pick a photo");
      if (p) $("opPaths").value = [...lines, p].join("\n"); });
    on("opOutB", "onclick", async () => { const p = await browse($("opOut").value, true, "Pick the output folder"); if (p) $("opOut").value = p; });
    if (G.n) showImg("photo", {fit: true});
    mainBtns([{label: "Open photos", tip: "Load the photos (and any labels saved before)", fn: async () => {
      await runJob("/api/opt/open", {paths: $("opPaths").value.split("\n").map(s => s.trim()).filter(Boolean), out: $("opOut").value}, "Opening photos");
      await finish(1); }}]);
  }});
// 2 Classes
OPT_STEPS.push({title: "Classes", lead: "Name what you want to measure. Porosity needs a class called “pores”.",
  help: "Keep it simple: pores and grains. Add a class only if you also want its area %.",
  controls: [["Name boxes", "rename a class"], ["＋ Add class", "add another class (e.g. cement)"], ["✕", "remove a class (its outlines on this photo go too)"], ["Use these classes", "save and go on to marking"]],
  async enter() {
    const v = await api("GET", "/api/opt/classes"); showImg("photo", {fit: true});
    const rows = names => names.map((n, i) => `<div class="clsrow"><i style="background:${v.classes[i]?.color || "#999"}"></i><input type="text" class="cn" value="${esc(n)}" style="flex:1"><button class="ghost cx" data-i="${i}" title="Remove">✕</button></div>`).join("");
    let names = v.classes.map(c => c.name);
    const render = () => { body(`<div id="cls">${rows(names)}</div><div class="btns"><button id="cAdd">＋ Add class</button></div>`);
      const grab = () => { names = [...document.querySelectorAll(".cn")].map(e => e.value); };
      on("cAdd", "onclick", () => { grab(); names.push(""); render(); });
      document.querySelectorAll(".cx").forEach(b => b.onclick = () => { grab(); names.splice(+b.dataset.i, 1); render(); });
      document.querySelectorAll(".cn").forEach(e => e.oninput = grab); };
    render();
    mainBtns([{label: "Use these classes", fn: async () => { names = [...document.querySelectorAll(".cn")].map(e => e.value);
      await api("POST", "/api/opt/classes", {names}); await finish(2); }}]);
  }});
// 3 Mark examples
OPT_STEPS.push({title: "Mark examples", lead: "Show the computer a few examples of each class on one or more photos.",
  help: "Pick a class, then click inside a pore or grain: the wand outlines the similar colour around it. A few clean examples per class are enough.",
  controls: [["Class buttons", "what the next click marks"], ["Wand", "click: outline the similar colour around the click"], ["Outline", "click points around a region, Enter (or click the first point) to close, Backspace removes the last point, Esc cancels"],
    ["Tolerance", "how different a colour may be and still join the wand (notebook 1: 70)"], ["Undo (Ctrl+Z)", "remove the last change"], ["Clear class", "remove this class's outlines on this photo"], ["Photos", "switch photo; labels save as you go"],
    ["Label summary", "notebook 1's summary: each photo in grey with its outlines, saved as labels/&lt;photo&gt;_labels.png"], ["Summary 1 2 3", "show that photo's summary figure; Back to marking returns to the photo"]],
  async enter() { OV.pts = []; await this.render(true); },
  async render(fit) {
    const v = this.v = await api("GET", "/api/opt/label"); const self = this;
    body(`<div class="hintt">Photo</div><div class="clsbtn">${v.images.map((n, i) => `<button class="${i === v.index ? "on" : ""}" data-im="${i}" title="${esc(n)}">${i + 1}</button>`).join("")}</div>
      <div class="hintt">${esc(v.images[v.index] || "")}</div>
      <div class="hintt">Mark</div><div class="clsbtn">${v.classes.map(c => `<button class="${c.id === v.current ? "on" : ""}" data-c="${c.id}"><i class="sw" style="width:12px;height:12px;border-radius:3px;background:${c.color}"></i>${esc(c.name)} <span class="hintt">${v.counts[c.id] || 0}</span></button>`).join("")}</div>
      <div class="hintt">Tool</div><div class="btns"><button id="tW" class="${OV.tool === "wand" ? "on" : ""}">Wand</button><button id="tO" class="${OV.tool === "outline" ? "on" : ""}">Outline</button></div>
      <div class="f"><label>Tolerance</label><input type="range" id="tTol" min="5" max="200" step="5" value="${v.threshold}"><span id="tTolV">${v.threshold}</span></div>
      <div class="btns"><button id="tU">Undo</button><button id="tC" class="danger">Clear class</button></div>
      <p class="hintt">${v.n_labelled} of ${v.images.length} photo(s) have examples. Saved in ${esc(v.labels_dir)}</p>
      <div class="hintt">Label summary</div><div class="btns"><button id="tS" title="One figure per photo, saved as labels/&lt;photo&gt;_labels.png">Label summary</button></div>
      <div class="clsbtn" id="sumB"></div><pre class="rep" id="sumT" style="display:none"></pre>`);
    on("tS", "onclick", async () => { await runJob("/api/opt/labelsummary", {}, "Label summary figures"); await this.summary(0); });
    this.summary();
    document.querySelectorAll("[data-im]").forEach(b => b.onclick = () => this.op({op: "open", i: +b.dataset.im}, true));
    document.querySelectorAll("[data-c]").forEach(b => b.onclick = () => this.op({op: "class", cid: +b.dataset.c}));
    on("tW", "onclick", () => { OV.tool = "wand"; OV.pts = []; this.render(); }); on("tO", "onclick", () => { OV.tool = "outline"; this.render(); });
    on("tTol", "oninput", () => { $("tTolV").textContent = $("tTol").value; }); on("tTol", "onchange", () => this.op({op: "threshold", t: +$("tTol").value}));
    on("tU", "onclick", () => this.op({op: "undo"})); on("tC", "onclick", () => this.op({op: "clear"}));
    if (fit !== undefined) await showImg("photo", {fit: !!fit, v: "o" + v.index + "_" + G.version});
    drawLabels(v); Viewer.onClick = this.click = (x, y) => self.clickAt(x, y);
    mainBtns([{label: "Next: features", tip: v.n_labelled ? "Labels are already saved" : "No examples yet: fine only if you will load a saved model in Train", fn: () => finish(3)}]);
    pill(v.polys.length ? `${v.polys.length} outline(s) on this photo · saved` : "No examples on this photo yet", v.polys.length ? "saved" : "");
  },
  async summary(showI) {
    const s = await api("GET", "/api/opt/labelsummary"); const box = $("sumB"); if (!box) return;
    if (!s.text) { box.innerHTML = ""; return; }
    box.innerHTML = s.images.map((m, i) => `<button data-si="${i}" ${m.ready ? "" : "disabled"} title="${esc(m.png)}">${i + 1}</button>`).join("") +
      `<button id="sumBack" title="Show the photo again to keep marking">Back to marking</button>`;
    $("sumT").textContent = s.text; $("sumT").style.display = "";
    const view = i => { OV.summary = true; Viewer.setOverlay(null); Viewer.onClick = null; showImg("labels:" + i, {fit: true, v: "s" + G.version});
      pill("Summary figure: " + s.images[i].png, "saved"); };
    box.querySelectorAll("[data-si]").forEach(b => b.onclick = () => view(+b.dataset.si));
    on("sumBack", "onclick", () => { OV.summary = false; this.render(true); });
    if (showI !== undefined && s.images[showI]?.ready) { view(showI); toast("Saved " + s.images.filter(m => m.ready).length + " summary figure(s) in " + s.labels_dir); }
  },
  async op(b, fit) { const r = await api("POST", "/api/opt/label/op", b); if (r.msg) toast(r.msg); await refreshGlobal(); await this.render(fit ? true : undefined); },
  async clickAt(x, y) {
    if (OV.tool === "wand") return this.op({op: "wand", x, y});
    const p0 = OV.pts[0];
    if (p0 && OV.pts.length >= 3 && Math.hypot(x - p0[0], y - p0[1]) * Viewer.k < 9) return this.close();
    OV.pts.push([x, y]); drawLabels(this.v);
  },
  async close() { if (OV.pts.length < 3) { toast("An outline needs at least 3 points."); return; } const pts = OV.pts; OV.pts = []; await this.op({op: "poly", pts}); },
  key(k) {
    if (k === "z") { this.op({op: "undo"}); return true; }
    if (OV.tool !== "outline") return false;
    if (k === "enter") { this.close(); return true; }
    if (k === "esc") { OV.pts = []; drawLabels(this.v); return true; }
    if (k === "delete") { OV.pts.pop(); drawLabels(this.v); return true; }
    return false;
  }});
// 4 Features
OPT_STEPS.push({title: "Features", lead: "Choose what the computer looks at in each pixel, and preview it.",
  help: "Notebook 1 uses red, green, blue, local texture and edges. Preview shows each feature as a grey image.",
  controls: [["Ticks", "features used for training and prediction"], ["Preview features", "draw every ticked feature for each photo"], ["Photo buttons", "show that photo's preview"]],
  async enter() {
    const v = await api("GET", "/api/opt/features");
    const show = i => showImg("feat:" + i, {fit: true, v: "f" + G.version});
    body(`${v.features.map(f => `<label class="clsrow"><input type="checkbox" class="ft" data-k="${f.key}" ${f.on ? "checked" : ""}> ${esc(f.label)}</label>`).join("")}
      <div class="hintt">Preview</div><div class="clsbtn">${v.images.map((m, i) => `<button data-fi="${i}" ${m.ready ? "" : "disabled"} title="${esc(m.name)}">${i + 1}</button>`).join("")}</div>`);
    document.querySelectorAll("[data-fi]").forEach(b => b.onclick = () => show(+b.dataset.fi));
    if (v.images[0]?.ready) show(0); else showImg("photo", {fit: true});
    const tog = () => Object.fromEntries([...document.querySelectorAll(".ft")].map(e => [e.dataset.k, e.checked]));
    mainBtns([{label: "Preview features", fn: async () => { await runJob("/api/opt/features", {toggles: tog()}, "Feature preview"); await finish(3); }},
      {label: "Next: train", disabled: !G.steps[4]?.ready, fn: async () => { await runJob("/api/opt/features", {toggles: tog()}, "Feature preview"); await finish(4); }}]);
  }});
// 5 Train
OPT_STEPS.push({title: "Train", lead: "Teach the model from your examples, or load one saved before (notebook 1's Train cell).",
  help: "Accuracy is checked by cross-validation: the model is tested on example pixels it did not learn from. Notebook 1 uses one LightGBM model for all photos.",
  controls: [["Train a new model", "learn from the marked examples (mode “retrain”)"], ["Load a saved model", "use a .joblib from models/ and skip training (mode “load”)"],
    ["One model for all photos", "pool every photo's examples: models/universal_model.joblib (notebook 1)"], ["One model per photo", "a model for each marked photo: models/&lt;photo&gt;_model.joblib; Predict uses each photo's own model"],
    ["Report", "precision / recall per class, as in the notebook's Accuracy cell"]],
  async enter() {
    const v = await api("GET", "/api/opt/train"); showImg("photo", {fit: true});
    const mode = V.optMode || v.mode, scope = V.optScope || v.scope;
    const opts = v.models.map(m => `<option value="${esc(m)}" ${m === v.loaded ? "selected" : ""}>${esc(m.split("/").pop())}</option>`).join("");
    body(`<div class="hintt">Model</div>
      <label class="clsrow"><input type="radio" name="tm" value="retrain" ${mode === "retrain" ? "checked" : ""}> Train a new model</label>
      <div id="trNew" style="margin-left:22px">
        <label class="clsrow"><input type="radio" name="ts" value="universal" ${scope === "universal" ? "checked" : ""}> One model for all photos</label>
        <label class="clsrow"><input type="radio" name="ts" value="per_image" ${scope === "per_image" ? "checked" : ""}> One model per photo</label>
        <p class="hintt">${v.n_labelled} of ${v.n} photo(s) have examples${scope === "per_image" ? " (photos without examples get no model)" : ""}. Features: ${esc(v.features.join(", "))}.</p></div>
      <label class="clsrow"><input type="radio" name="tm" value="load" ${mode === "load" ? "checked" : ""}> Load a saved model (skip training)</label>
      <div id="trLoad" style="margin-left:22px">
        <div class="f"><label>Saved model</label><select id="tmSel"><option value="">${v.models.length ? "pick one…" : "none in models/ yet"}</option>${opts}</select><span></span></div>
        <div class="f"><label>or file</label><input type="text" id="tmPath" placeholder="path to a .joblib" value="${esc(v.loaded && !v.models.includes(v.loaded) ? v.loaded : "")}"><button id="tmB" title="Browse">…</button></div>
        <p class="hintt">Saved models are in ${esc(v.models_dir)}. The model must use the same features as when it was trained.</p></div>
      ${v.accuracy != null ? `<div class="hintt">Accuracy (cross-validation${v.scope === "per_image" && v.mode === "retrain" ? ", mean of the photos" : ""})</div><div class="big-num">${(v.accuracy * 100).toFixed(2)}%</div>` : ""}
      ${v.text ? `<pre class="rep">${esc(v.text)}</pre>` : ""}`);
    const sync = () => { const m = document.querySelector("[name=tm]:checked").value; V.optMode = m;
      V.optScope = document.querySelector("[name=ts]:checked").value;
      $("trNew").style.opacity = m === "retrain" ? 1 : 0.45; $("trLoad").style.opacity = m === "load" ? 1 : 0.45;
      const b = $("mainBtns").querySelector("button"); if (b) b.textContent = m === "load" ? "Load model" : (v.text && v.mode === "retrain" ? "Train again" : "Train"); };
    document.querySelectorAll("[name=tm],[name=ts]").forEach(e => e.onchange = sync);
    on("tmSel", "onchange", () => { if ($("tmSel").value) { $("tmPath").value = ""; document.querySelector("[name=tm][value=load]").checked = true; sync(); } });
    on("tmB", "onclick", async () => { const p = await browse($("tmPath").value || v.models_dir, false, "Pick a saved model (.joblib)", ".joblib"); if (p) { $("tmPath").value = p; $("tmSel").value = ""; document.querySelector("[name=tm][value=load]").checked = true; sync(); } });
    mainBtns([{label: "Train", fn: async () => { const m = V.optMode || "retrain";
        if (m === "retrain" && !v.n_labelled) return toast("Mark examples first (step 3), or load a saved model.", true);
        const model_path = $("tmPath").value.trim() || $("tmSel").value;
        if (m === "load" && !model_path) return toast("Pick a saved model to load.", true);
        await runJob("/api/opt/train", {mode: m, scope: V.optScope || "universal", model_path}, m === "load" ? "Loading the saved model" : "Training the pixel classifier"); await finish(4); }},
      ...(v.text ? [{label: "Next: predict", fn: () => go(5)}] : [])]);
    sync();
  }});
// 6 Predict
OPT_STEPS.push({title: "Predict", lead: "Colour every pixel of every photo with its most likely class.",
  help: "The confidence map is bright where the model is sure and dark where it hesitates.",
  controls: [["Predict all photos", "run the model on every photo; maps go to predictions/"], ["Photo / Class map / Confidence", "what the left side shows"]],
  async enter() {
    const v = await api("GET", "/api/opt/predict"); V.optLay = V.optLay || "pred"; V.optI = Math.min(V.optI || 0, Math.max(0, v.images.length - 1));
    const show = () => { if (v.images.length) showImg(`${V.optLay}:${V.optI}`, {v: "p" + G.version}); else showImg("photo", {fit: true}); };
    const tbl = e => `<table class="t"><tr><th>class</th><th>area %</th><th>range</th></tr>${Object.values(e.fractions).map(f => `<tr><td>${esc(f.name)}</td><td>${f.area_pct.toFixed(2)}</td><td class="hintt">+${f.area_pct_plus.toFixed(2)} / −${f.area_pct_minus.toFixed(2)}</td></tr>`).join("")}</table>`;
    const render = () => body(v.images.length ? `<div class="hintt">Photo</div><div class="clsbtn">${v.images.map((m, i) => `<button data-pi="${i}" class="${i === V.optI ? "on" : ""}" title="${esc(m.name)}">${i + 1}</button>`).join("")}</div>
      <div class="hintt">${esc(v.images[V.optI].name)}</div>
      <div class="btns">${[["photo", "Photo"], ["pred", "Class map"], ["conf", "Confidence"]].map(([k, l]) => `<button data-l="${k}" class="${V.optLay === k ? "on" : ""}">${l}</button>`).join("")}</div>${tbl(v.images[V.optI])}`
      : `<p class="hintt">Not run yet.</p>`);
    const wire = () => { render(); document.querySelectorAll("[data-pi]").forEach(b => b.onclick = () => { V.optI = +b.dataset.pi; wire(); show(); });
      document.querySelectorAll("[data-l]").forEach(b => b.onclick = () => { V.optLay = b.dataset.l; wire(); show(); }); };
    wire(); show();
    mainBtns([{label: v.images.length ? "Predict again" : "Predict all photos", fn: async () => { await runJob("/api/opt/predict", {}, "Predicting every photo"); await finish(5); }},
      ...(v.images.length ? [{label: "Next: porosity %", fn: () => go(6)}] : [])]);
  }});
// 7 Porosity
OPT_STEPS.push({title: "Porosity %", lead: "Porosity is the share of pixels called pores, per photo and on average.",
  help: "plus / minus is a confidence range (unsure pixels dropped or added), not a lab error bar. sd = how much the photos differ.",
  controls: [["Pore class", "which class counts as pore space"], ["Save porosity table", "write predictions/porosity_summary.csv"]],
  async enter() {
    const pore = V.pore || "pores"; const v = await api("GET", "/api/opt/porosity?pore=" + encodeURIComponent(pore)); const s = v.stats;
    if (G.n) showImg(`pred:${V.optI || 0}`, {v: "p" + G.version});
    body(`<div class="f"><label>Pore class</label><select id="pPore">${v.classes.map(c => `<option ${c === pore ? "selected" : ""}>${esc(c)}</option>`).join("")}</select><span></span></div>
      ${s.n ? `<div class="hintt">Mean ± sd over ${s.n} photo(s)</div><div class="big-num">${s.mean.toFixed(2)}% ± ${s.sd.toFixed(2)}%</div><div class="hintt">range ${s.min.toFixed(2)} – ${s.max.toFixed(2)}%</div>
      <table class="t"><tr><th>photo</th><th>porosity %</th><th>plus</th><th>minus</th></tr>${v.rows.map(r => `<tr><td title="${esc(r.image)}">${esc(r.image)}</td><td>${r.porosity_pct.toFixed(2)}</td><td>+${r.plus.toFixed(2)}</td><td>−${r.minus.toFixed(2)}</td></tr>`).join("")}</table>`
      : `<p class="warnbox">No “${esc(pore)}” rows: run Predict, or pick the pore class.</p>`}`);
    on("pPore", "onchange", () => { V.pore = $("pPore").value; go(6); });
    mainBtns([{label: "Save porosity table (CSV)", disabled: !s.n, fn: async () => { const r = await runJob("/api/opt/porosity/export", {pore}, "Saving porosity table"); toast("Saved " + r.path); await finish(6); }}]);
  }});
