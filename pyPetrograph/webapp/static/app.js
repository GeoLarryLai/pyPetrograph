// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// app.js - the nine AMICS steps (notebook 4 order). The server holds the state; every step
// renders from GET /api/<step> and changes it with POST /api/<step>/op.
"use strict";
const {api, runJob, toast, $} = Shell;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const hex = rgb => "#" + rgb.map(c => (+c).toString(16).padStart(2, "0")).join("");
const fromHex = h => { const m = /^#?([0-9a-f]{6})$/i.exec(h.trim()); return m ? [0, 2, 4].map(i => parseInt(m[1].substr(i, 2), 16)) : null; };
let G = {steps: []}, cur = 0, V = {};          // global state, current step, per-step view
let tick = 0;                                    // cache-buster for images that change
async function showImg(key, opts = {}) {
  try { const s = await api("GET", "/api/imgsize/" + encodeURIComponent(key)); Viewer.setImage(`/api/img/${encodeURIComponent(key)}?v=${opts.v ?? G.version}`, s.w, s.h, opts); }
  catch (e) { Viewer.clear(); }
}
function pill(msg, kind) { const p = $("pill"); p.textContent = msg || ""; p.className = "pill " + (kind || ""); }
function ctrls(list) { $("ctrls").innerHTML = "<h4>Controls</h4>" + list.map(([k, d]) => `<div><b>${k}</b>${d}</div>`).join(""); }
function mainBtns(list) {
  $("mainBtns").innerHTML = ""; list.forEach((b, i) => { const e = document.createElement("button");
    e.textContent = b.label; e.title = b.tip || ""; e.className = i === 0 ? "main" : ""; e.id = b.id || ""; e.disabled = !!b.disabled;
    e.onclick = async () => { if (e.dataset.busy) return; try { await b.fn(); } catch (x) { console.warn(x); } }; $("mainBtns").appendChild(e); });
}
const body = html => { $("stepBody").innerHTML = html; };
const on = (id, ev, fn) => { const e = $(id); if (e) e[ev] = fn; };
async function refreshGlobal() {
  G = await api("GET", "/api/state"); if (G.status) toast(G.status);
  $("doc").textContent = G.sample ? `AMICS · sample ${G.sample}` + (G.folder ? ` · export folder ${G.folder}` : "") : "AMICS workflow · no sample open";
  $("logp").textContent = G.log.join("\n"); $("logp").scrollTop = 1e9;
  $("rail").innerHTML = "<ol>" + G.steps.map((s, i) => `<li data-i="${i}" title="${s.ready ? "" : "finish the earlier steps first"}" class="${i === cur ? "cur" : ""} ${s.done ? "done" : ""} ${s.ready ? "" : "locked"}">${esc(s.title)}</li>`).join("") + "</ol>";
  $("rail").querySelectorAll("li").forEach(li => li.onclick = () => { const i = +li.dataset.i; if (G.steps[i].ready) go(i); });
}
async function go(i) {
  cur = i; Viewer.setOverlay(null); Viewer.setMask(null); Viewer.boxMode = false; Viewer.onClick = null; Viewer.onBox = null;
  closeDlg(); await refreshGlobal(); const s = STEPS[i];
  $("stepTitle").textContent = G.steps[i]?.title || s.title; $("stepLead").textContent = s.lead; pill(""); ctrls(s.controls || []);
  Shell.help(s.help || ""); try { await s.enter(); } catch (e) { body(`<p class="warnbox">${esc(e.message)}</p>`); mainBtns([]); }
}
async function finish(next) { await refreshGlobal(); if (next !== undefined) go(next); }
const confirmClear = what => confirm(`Delete the saved ${what} for this sample? This cannot be undone.`);
// ---- generic dialog -------------------------------------------------------------------------
const dlg = $("dlg");
function openDlg(html, modal = true) { dlg.innerHTML = html; dlg.className = modal ? "" : "nonmodal"; if (dlg.open) dlg.close(); modal ? dlg.showModal() : dlg.show(); }
function closeDlg() { if (dlg.open) dlg.close(); Viewer.onClick = STEPS[cur]?.click || null; }
// file / folder browser (server side, the files never leave the computer)
function browse(start, dirsOnly, title) {
  return new Promise(res => {
    let path = start || "";
    async function load(p) {
      const r = await api("GET", `/api/ls?path=${encodeURIComponent(p)}&dirs_only=${dirsOnly}`); path = r.path;
      openDlg(`<h3>${esc(title)}</h3><div class="hintt">${esc(r.path)}</div><div class="list files">
        <div data-p="${esc(r.parent)}" data-d="1">⬆ ..</div>${r.items.map(it => `<div data-p="${esc(r.path + "/" + it.name)}" data-d="${it.dir ? 1 : ""}">${it.dir ? "📁 " : "🖼 "}${esc(it.name)}</div>`).join("")}</div>
        <div class="btns"><button id="bwCancel">Cancel</button>${dirsOnly ? '<button class="main" id="bwHere">Use this folder</button>' : ""}</div>`);
      dlg.querySelectorAll(".list div").forEach(d => d.onclick = () => d.dataset.d ? load(d.dataset.p) : (closeDlg(), res(d.dataset.p)));
      on("bwCancel", "onclick", () => { closeDlg(); res(null); }); on("bwHere", "onclick", () => { closeDlg(); res(path); });
    }
    load(path);
  });
}
// legend entry dialog (add / change): RGB, hex, palette, eyedropper, name, live errors
const PALETTE = ["#ff0000", "#ff8000", "#ffff00", "#80ff00", "#00ff00", "#00ff80", "#00ffff", "#0080ff", "#0000ff", "#8000ff", "#ff00ff", "#ff0080",
  "#800000", "#804000", "#808000", "#408000", "#008000", "#008040", "#008080", "#004080", "#000080", "#400080", "#800080", "#800040",
  "#000000", "#404040", "#808080", "#c0c0c0", "#ffffff", "#ffc0c0", "#ffe0c0", "#ffffc0", "#c0ffc0", "#c0ffff", "#c0c0ff", "#ffc0ff"];
function entryDialog({title, rgb, name, row, check, submit, dropSrc}) {
  openDlg(`<h3>${esc(title)}</h3>
    <div class="hintt">Set the colour exactly as it appears in the mineral map: type R, G, B or #hex, pick a swatch / the colour picker, or use the eyedropper and click the image.</div>
    <div class="row">R <input type="number" id="eR" min="0" max="255"> G <input type="number" id="eG" min="0" max="255"> B <input type="number" id="eB" min="0" max="255"></div>
    <div class="row">Hex <input type="text" id="eHex" size="8"> <input type="color" id="ePick" title="Colour palette"> <button id="eDrop" title="Eyedropper: click a pixel of the image">Eyedropper</button></div>
    <div class="pal">${PALETTE.map(c => `<i style="background:${c}" data-c="${c}"></i>`).join("")}</div>
    <div class="prev" id="ePrev">sample</div>
    <div class="row">Name <input type="text" id="eName" style="flex:1" placeholder="mineral name"></div>
    <div class="err" id="eErr"></div>
    <div class="btns"><button id="eCancel">Cancel</button><button class="main" id="eOk">OK</button></div>`, false);
  let c = rgb.slice();
  const upd = (from) => {
    if (from !== "rgb") { $("eR").value = c[0]; $("eG").value = c[1]; $("eB").value = c[2]; }
    if (from !== "hex") $("eHex").value = hex(c); $("ePick").value = hex(c.map(v => Math.max(0, Math.min(255, v | 0))));
    $("ePrev").style.background = hex(c.map(v => Math.max(0, Math.min(255, v | 0)))); $("ePrev").style.color = (c[0] + c[1] + c[2]) > 380 ? "#000" : "#fff";
    $("ePrev").textContent = `${$("eName").value || "sample"} · (${c.join(", ")})`; validate();
  };
  let vt = 0; async function validate() { clearTimeout(vt); vt = setTimeout(async () => { const p = (await check(c, $("eName").value, row)); $("eErr").textContent = p; $("eErr").className = "err " + (p ? "e" : ""); $("eOk").disabled = !!p; }, 120); }
  ["eR", "eG", "eB"].forEach((id, i) => $(id).oninput = () => { c[i] = parseInt($(id).value || "0"); upd("rgb"); });
  $("eHex").oninput = () => { const v = fromHex($("eHex").value); if (v) { c = v; upd("hex"); } else { $("eErr").textContent = "Hex looks like #rrggbb"; } };
  $("ePick").oninput = () => { c = fromHex($("ePick").value); upd(); };
  dlg.querySelectorAll(".pal i").forEach(i => i.onclick = () => { c = fromHex(i.dataset.c); upd(); });
  $("eName").value = name || ""; $("eName").oninput = () => upd("name");
  let dropping = false;
  $("eDrop").onclick = () => { dropping = !dropping; $("eDrop").classList.toggle("on", dropping);
    Viewer.onClick = dropping ? async (x, y) => { const r = await api("GET", `/api/pixel?src=${dropSrc()}&x=${x}&y=${y}`); c = r.rgb; upd(); } : STEPS[cur].click || null;
    if (dropping) toast("Eyedropper on: click the image (zoom in for a single pixel). Click Eyedropper again to stop."); };
  $("eCancel").onclick = closeDlg;
  $("eOk").onclick = async () => { try { await submit(c, $("eName").value); closeDlg(); } catch (e) { $("eErr").textContent = e.message; } };
  upd(); $("eName").focus();
}
// ---- steps ----------------------------------------------------------------------------------
const STEPS = [];
// 1 Files
STEPS.push({title: "Files", lead: "Pick the three AMICS images of one sample and where exports go. Picking one file fills in its companions.",
  help: "Use <b>…</b> to browse. Nothing is written until you press an Export button in step 9.",
  controls: [["BSE image", "the back-scattered electron image (.tif)"], ["Mineral map", "the AMICS mineral colour map"], ["Legend", "the legend image the colours are read from (OCR)"],
    ["Output folder", "exports go to &lt;folder&gt;/amics/&lt;sample&gt;/"], ["Sample", "name used in file names (blank = from the BSE file name)"], ["Open", "load the sample and read its legend (step 2)"]],
  async enter() {
    const last = V.files || await api("GET", "/api/files"); V.files = last;
    const f = (k, label, tip) => `<div class="f"><label title="${tip}">${label}</label><input type="text" id="f_${k}" value="${esc(last[k] || "")}" title="${tip}"><button id="b_${k}" title="Browse">…</button></div>`;
    body(f("bse", "BSE image", "BSE image file") + f("mineral", "Mineral map", "Mineral colour map") + f("legend", "Legend", "Legend image") +
      f("out", "Output folder", "Exports go here, in amics/<sample>/") +
      `<div class="f"><label>Sample</label><input type="text" id="f_sample" value="${esc(last.sample || "")}" placeholder="from the BSE file name"></div>
      <p class="hintt">The BSE image is shown on the left once it is picked. Wheel = zoom, drag = pan, F or double-click = fit.</p>`);
    const cur_ = () => ({bse: $("f_bse").value, mineral: $("f_mineral").value, legend: $("f_legend").value});
    const after = async k => { const fill = await api("POST", "/api/files/complete", {key: k, cur: cur_()});
      for (const [kk, v] of Object.entries(fill)) $("f_" + kk).value = v; V.files = {...V.files, ...cur_()}; preview(); };
    const preview = async () => { const r = await api("POST", "/api/files/preview", {bse: $("f_bse").value}); if (r.ok) showImg("bse_gray", {v: "p" + (++tick), fit: true}); };
    for (const k of ["bse", "mineral", "legend"]) {
      on("b_" + k, "onclick", async () => { const p = await browse($("f_" + k).value || $("f_bse").value, false, "Pick the " + k + " image"); if (p) { $("f_" + k).value = p; after(k); } });
      on("f_" + k, "onchange", () => after(k));
    }
    on("b_out", "onclick", async () => { const p = await browse($("f_out").value, true, "Pick the output folder"); if (p) $("f_out").value = p; });
    if (last.bse) preview();
    mainBtns([{label: "Open", tip: "Load the sample and read its legend", fn: async () => {
      const b = {...cur_(), out: $("f_out").value, sample: $("f_sample").value}; V.files = {...V.files, ...b};
      await api("POST", "/api/open", b); await finish(1); }}]);
  }});
// 2 Legend
STEPS.push({title: "Legend", lead: "Check the mineral names read from the legend (OCR). Untick what to leave out, fix names and colours.",
  help: "Double-click a name to edit it. Click a colour square to change the colour. Changes are only saved by <b>Use this legend</b>.",
  controls: [["Show", "legend image / legend as used / mineral map on the left"], ["Use ☑", "untick to leave a mineral out (grains count as Unmapped)"],
    ["Name", "click to rename (✎ = edited, + = added)"], ["Colour square", "change colour (RGB / hex / palette / eyedropper)"], ["Row click", "select (Shift / Ctrl for more)"],
    ["+ Add legend entry", "a colour the OCR missed"], ["Delete entry (Del)", "remove the selected rows"], ["Undo (Z)", "undo the last edit"],
    ["Revert to OCR", "selected rows back to what OCR read"], ["Reset all to OCR", "every row back to the OCR result"], ["Load saved", "the legend saved for this sample"],
    ["Read again (OCR)", "run OCR again"], ["Clear saved…", "delete the saved legend"], ["Use this legend", "save it and load the BSE image (step 3)"]],
  sel: new Set(), view: "legend_img",
  async enter() {
    const st = await api("GET", "/api/legend");
    if (st.need_read) { await runJob("/api/legend/read", {}, "Reading legend (OCR)"); }
    this.sel = new Set(); body(`<div class="f"><label>Show</label><select id="lgView"><option value="legend_img">legend image</option><option value="legend_plot">legend as used</option><option value="mineral_img">mineral map</option></select><span></span></div>
      <div class="tbl"><table class="t" id="lgT"></table></div>
      <div class="btns"><button id="lgAdd" title="Add a colour the OCR missed">+ Add legend entry</button><button id="lgDel" title="Delete the selected rows (Del)">Delete entry</button></div>
      <div class="btns"><button id="lgUndo" title="Undo the last edit (Z)">Undo</button><button id="lgRev" title="Selected rows back to OCR">Revert to OCR</button><button id="lgReset" title="All rows back to OCR">Reset all to OCR</button></div>
      <div class="btns"><button id="lgLoad">Load saved</button><button id="lgOcr" title="Run OCR on the legend again">Read again (OCR)</button><button id="lgClear" class="danger">Clear saved…</button></div>
      <pre class="hintt" id="lgRep" style="white-space:pre-wrap"></pre>`);
    $("lgView").value = this.view; $("lgView").onchange = () => { this.view = $("lgView").value; this.show(); };
    const op = async (o, extra = {}) => { this.draw(await api("POST", "/api/legend/op", {op: o, ...extra})); };
    this.op = op;
    on("lgAdd", "onclick", () => this.entry(null));
    on("lgDel", "onclick", () => this.sel.size ? op("delete", {rows: [...this.sel]}).then(() => this.sel.clear()) : toast("Select rows first"));
    on("lgUndo", "onclick", () => op("undo")); on("lgRev", "onclick", () => this.sel.size ? op("revert", {rows: [...this.sel]}) : toast("Select rows first"));
    on("lgReset", "onclick", () => op("reset")); on("lgLoad", "onclick", () => op("load_saved"));
    on("lgOcr", "onclick", async () => { await runJob("/api/legend/read", {}, "Reading legend (OCR)"); this.draw(await api("GET", "/api/legend")); });
    on("lgClear", "onclick", () => confirmClear("legend") && op("clear_saved"));
    mainBtns([{label: "Use this legend", tip: "Save the legend and load the BSE image", fn: async () => { await runJob("/api/legend/apply", {}, "Loading BSE"); await finish(2); }}]);
    this.draw(await api("GET", "/api/legend")); this.show();
  },
  show() { showImg(this.view, {v: this.view === "legend_plot" ? "lp" + (++tick) : G.version, fit: true}); },
  draw(st) {
    this.st = st; pill(st.state[0], st.state[1]); $("lgRep").textContent = st.report; $("lgUndo").disabled = !st.can_undo; $("lgLoad").disabled = !st.saved; $("lgClear").disabled = !st.saved;
    $("lgT").innerHTML = "<tr><th>Use</th><th>RGB</th><th>Mineral</th><th></th></tr>" + st.rows.map((r, i) =>
      `<tr data-i="${i}" class="${this.sel.has(i) ? "sel" : ""}"><td><input type="checkbox" ${r.use ? "checked" : ""} title="use this mineral"></td>
       <td class="sw" title="click to change the colour">${r.rgb.join(",")}<i style="background:${hex(r.rgb)}"></i></td>
       <td class="nm ${r.origin ? "" : "it"}"><input type="text" value="${esc(r.name)}" title="${esc(r.tip)}"></td><td class="mk" title="${esc(r.tip)}">${r.mark}</td></tr>`).join("");
    $("lgT").querySelectorAll("tr[data-i]").forEach(tr => { const i = +tr.dataset.i;
      tr.onclick = e => { if (e.target.tagName === "INPUT") return; if (!(e.shiftKey || e.ctrlKey || e.metaKey)) this.sel.clear(); this.sel.has(i) ? this.sel.delete(i) : this.sel.add(i); this.draw(this.st); };
      tr.querySelector("input[type=checkbox]").onchange = e => this.op("tick", {row: i, use: e.target.checked});
      const nm = tr.querySelector("td.nm input"); nm.onchange = () => this.op("rename", {row: i, name: nm.value});
      nm.onkeydown = e => { if (e.key === "Enter") nm.blur(); if (e.key === "Escape") { nm.value = st.rows[i].name; nm.blur(); } };
      tr.querySelector("td.sw").onclick = e => { e.stopPropagation(); this.entry(i); };
      tr.oncontextmenu = e => { e.preventDefault(); this.sel = new Set([i]); this.draw(this.st);
        openDlg(`<h3>${esc(st.rows[i].name)}</h3><div class="btns" style="flex-direction:column"><button id="cmCol">Change colour / name…</button><button id="cmRev" ${st.rows[i].origin ? "" : "disabled"}>Revert to OCR</button><button id="cmDel" class="danger">Delete entry</button><button id="cmX">Close</button></div>`);
        on("cmCol", "onclick", () => this.entry(i)); on("cmRev", "onclick", () => { closeDlg(); this.op("revert", {rows: [i]}); });
        on("cmDel", "onclick", () => { closeDlg(); this.op("delete", {rows: [i]}); this.sel.clear(); }); on("cmX", "onclick", closeDlg); };
    });
    if (this.view === "legend_plot") this.show();
  },
  entry(row) {
    const r = row == null ? null : this.st.rows[row];
    entryDialog({title: r ? "Change legend entry" : "Add legend entry", rgb: r ? r.rgb : [128, 128, 128], name: r ? r.name : "", row,
      dropSrc: () => this.view === "mineral_img" ? "mineral_img" : "legend_img",
      check: async (c, n, rw) => (await api("POST", "/api/legend/check", {rgb: c, name: n, row: rw})).problem,
      submit: async (c, n) => this.op("entry", {row, rgb: c, name: n})});
    if (this.view === "legend_plot") { this.view = "legend_img"; $("lgView").value = "legend_img"; this.show(); }
  },
  key(k, e) { if (k === "delete" && this.sel.size) { this.op("delete", {rows: [...this.sel]}); this.sel.clear(); return true; }
    if (k === "z") { this.op("undo"); return true; } }});
// 3 BSE & footer
STEPS.push({title: "BSE & footer", lead: "The red dashed line marks where the image ends and the footer (scale bar, text) starts.",
  help: "Change the footer row if the line is off, or press <b>Use detected</b>. Everything below the line is ignored.",
  controls: [["Footer row", "first row of the footer strip"], ["Use detected", "go back to the automatically detected row"], ["Use this footer", "keep it and go on (step 4)"]],
  async enter() {
    const f = await api("GET", "/api/footer"); this.f = f;
    body(`<div class="f"><label>Footer row</label><input type="number" id="ftRow" min="1" max="${f.h}" value="${f.footer}"><button id="ftDet" title="row ${f.detected}">Use detected</button></div>
      <p class="hintt">Detected footer at row ${f.detected} of ${f.h}. Image ${f.w} × ${f.h} px.</p>`);
    const ov = () => Viewer.setOverlay((c, T) => { const r = +$("ftRow").value; Draw.line(c, T, [0, r], [f.w, r], "#ff3b30", 2, true); Draw.label(c, T, 10, r + 4, `footer starts at row ${r}`, "#ff9f9a"); });
    $("ftRow").oninput = ov; on("ftDet", "onclick", () => { $("ftRow").value = f.detected; ov(); });
    await showImg("bse_stretch", {fit: true}); ov(); pill(f.done ? `Footer at row ${f.footer} (used)` : "Check the footer line", f.done ? "saved" : "unsaved");
    mainBtns([{label: "Use this footer", fn: async () => { await api("POST", "/api/footer/apply", {row: +$("ftRow").value}); await finish(3); }}]);
  }});
// 4 Non-grain
STEPS.push({title: "Non-grain", lead: "Click a few dark non-grain spots (epoxy, holes). Everything as dark as the brightest pick turns blue.",
  help: "Click = pick, right-click a pick = remove it. Or type a gray level. Blue = non-grain.",
  controls: [["Click", "pick a dark non-grain spot"], ["Right-click", "remove the nearest pick"], ["Z", "undo last pick"], ["C", "reset picks"], ["T", "show / hide the blue tint"],
    ["Gray level threshold", "type one instead of picking (blank = from the picks)"], ["Load saved", "the threshold saved for this sample"], ["Clear saved…", "delete it"],
    ["Enter / Apply non-grain regions", "keep it and go on (step 5)"]],
  async enter() {
    const v = await api("GET", "/api/nongrain?activate=true");
    body(`<div class="btns"><button id="ngTint" class="on" title="T">Show blue tint (T)</button></div>
      <div class="f"><label>Gray level threshold</label><input type="number" id="ngThr" min="0" max="255" step="1" placeholder="from picks"><span></span></div>
      <p class="hintt" id="ngInfo"></p>
      <div class="btns"><button id="ngUndo" title="Z">Undo last pick</button><button id="ngReset" title="C">Reset picks</button></div>
      <div class="btns"><button id="ngLoad">Load saved</button><button id="ngClear" class="danger">Clear saved…</button></div>`);
    this.tint = true; const op = async (o, x = {}) => this.draw(await api("POST", "/api/nongrain/op", {op: o, ...x})); this.op = op;
    on("ngTint", "onclick", () => this.toggle()); on("ngUndo", "onclick", () => op("undo")); on("ngReset", "onclick", () => op("reset"));
    on("ngLoad", "onclick", () => op("load_saved")); on("ngClear", "onclick", () => confirmClear("non-grain threshold") && op("clear_saved"));
    $("ngThr").onchange = () => op("typed", {value: $("ngThr").value});
    this.click = (x, y, b, sh, k) => op("click", {x, y, button: b === 2 ? 2 : 0, scale: k}); Viewer.onClick = this.click;
    await showImg("bse_stretch", {fit: true}); this.draw(v);
    mainBtns([{label: "Apply non-grain regions", fn: () => this.apply()}]);
  },
  async apply() { await api("POST", "/api/nongrain/op", {op: "apply"}); await finish(4); },
  toggle() { this.tint = !this.tint; $("ngTint").classList.toggle("on", this.tint); Viewer.showMask(this.tint); },
  draw(v) {
    this.v = v; $("ngInfo").textContent = v.info; pill(v.state[0], v.state[1]); if (document.activeElement !== $("ngThr")) $("ngThr").value = v.typed ?? "";
    $("ngLoad").disabled = $("ngClear").disabled = !v.saved; $("ngUndo").disabled = !v.points.length;
    Viewer.setMask("/api/img/mask?v=" + (++tick)); Viewer.showMask(this.tint);
    Viewer.setOverlay((c, T) => { Draw.crosses(c, T, v.points.map(p => [p[0], p[1]]), "#ff3b30"); Draw.line(c, T, [0, v.footer], [v.w, v.footer], "#ff3b30", 1, true); });
  },
  key(k) { if (k === "z") this.op("undo"); else if (k === "c") this.op("reset"); else if (k === "t") this.toggle(); else if (k === "enter") this.apply(); else return false; return true; }});
// 5 Scale
STEPS.push({title: "Scale", lead: "Optional: click both ends of the scale bar and type its length, or type µm per pixel. Without a scale, sizes stay in pixels.",
  help: "Click = line end (2 clicks), right-click = remove the nearest end. Orange dashed = the saved scale.",
  controls: [["Click ×2", "the two ends of the scale bar"], ["Right-click", "remove the nearest end"], ["B", "whole image / footer strip"], ["Bar length + Unit", "the number on the bar"],
    ["µm/px", "type it directly instead (wins over the line)"], ["Undo last point (Z)", ""], ["Reset line (C)", ""], ["Check footer text (OCR)", "compare with the µm/px printed in the footer"],
    ["Clear saved…", "delete the saved scale"], ["Apply scale", "keep it and go on (step 6)"], ["Skip", "go on without a new scale (keeps the saved one, if any)"]],
  async enter() {
    const v = await api("GET", "/api/scale?activate=true");
    body(`<div class="btns"><button id="scZoom" title="B">Show footer strip (B)</button></div>
      <div class="f"><label>Bar length</label><input type="number" id="scLen" min="0" step="any"><select id="scUnit">${v.units.map(u => `<option>${u}</option>`).join("")}</select></div>
      <div class="f"><label>µm/px</label><input type="text" id="scUm" placeholder="from the line"><span></span></div>
      <p class="hintt" id="scInfo"></p>
      <div class="btns"><button id="scUndo" title="Z">Undo last point</button><button id="scReset" title="C">Reset line</button></div>
      <div class="btns"><button id="scOcr">Check footer text (OCR)</button><button id="scClear" class="danger">Clear saved…</button></div><p class="hintt" id="scOcrT"></p>`);
    const op = async (o, x = {}) => this.draw(await api("POST", "/api/scale/op", {op: o, ...x})); this.op = op; this.strip = false;
    $("scLen").onchange = () => op("set", {length: $("scLen").value}); $("scUnit").onchange = () => op("set", {unit: $("scUnit").value}); $("scUm").onchange = () => op("set", {um: $("scUm").value});
    on("scUndo", "onclick", () => op("undo")); on("scReset", "onclick", () => op("reset")); on("scZoom", "onclick", () => this.zoom());
    on("scClear", "onclick", () => confirmClear("scale") && op("clear_saved"));
    on("scOcr", "onclick", async () => { const r = await runJob("/api/scale/ocr", {}, "Reading footer"); $("scOcrT").textContent = r.text; });
    this.click = (x, y, b, sh, k) => op("click", {x, y, button: b === 2 ? 2 : 0, scale: k}); Viewer.onClick = this.click;
    await showImg("bse_gray", {fit: true}); this.draw(v);
    mainBtns([{label: "Apply scale", fn: () => this.apply()}, {label: "Skip (no new scale)", tip: "keep the saved scale, if any", fn: async () => { await api("POST", "/api/scale/op", {op: "skip"}); await finish(5); }}]);
  },
  async apply() { await api("POST", "/api/scale/op", {op: "apply"}); await finish(5); },
  zoom() { this.strip = !this.strip; const v = this.v; $("scZoom").textContent = this.strip ? "Show whole image (B)" : "Show footer strip (B)";
    this.strip ? Viewer.zoomTo(0, v.footer - 20, v.w, v.h) : Viewer.fit(); },
  draw(v) {
    this.v = v; pill(v.state[0], v.state[1]); $("scInfo").textContent = v.info; $("scClear").disabled = !v.saved;
    for (const [id, val] of [["scLen", v.length || ""], ["scUnit", v.unit], ["scUm", v.um]]) if (document.activeElement !== $(id)) $(id).value = val;
    Viewer.setOverlay((c, T) => {
      Draw.line(c, T, [0, v.footer], [v.w, v.footer], "#ff3b30", 1, true);
      const s = v.saved; if (s && s.p1 && s.p2) Draw.line(c, T, s.p1, s.p2, "#ff9f0a", 2, true, 4);
      if (v.pts.length === 2) Draw.line(c, T, v.pts[0], v.pts[1], "#ff3b30", 2, false, 4); else Draw.points(c, T, v.pts, "#ff3b30", 4);
    });
  },
  key(k) { if (k === "z") this.op("undo"); else if (k === "c") this.op("reset"); else if (k === "b") this.zoom(); else if (k === "enter") this.apply(); else return false; return true; }});
// 6 Segmentation
STEPS.push({title: "Segmentation", lead: "Find the grains, check them, fix mistakes, then use the grain map.",
  help: "Click a grain to select it (Shift = add). D delete, M merge, Z undo. With Add grain (A): click points or drag a box, Enter accepts.",
  controls: [["Run segmentation", "SegmentEveryGrain (U-Net + SAM); progress shows n / N"], ["SAM 2.1 refine", "optional extra pass over SEG's grains"],
    ["Min grain (px)", "smaller grains are dropped"], ["Non-grain drop fraction", "drop grains mostly inside the blue non-grain area"], ["Load saved grains", "the grains saved for this sample"],
    ["Click / Shift-click", "select grains"], ["Add grain (A)", "SAM: click = point (right-click = negative point), drag = box"], ["Accept grain (Enter)", "keep the SAM preview"], ["Clear prompts (Esc)", ""],
    ["Delete (D)", "remove selected grains"], ["Merge (M)", "join selected grains"], ["Undo last edit (Z)", ""], ["Deselect all (Esc)", ""], ["Reset edits", "back to the segmentation result"],
    ["Clear saved grains…", "delete the saved grains"], ["Use this grain map", "save the edits, measure grains, vote minerals, draw figures (steps 7–9)"]],
  polys: [], pv: -1, add: false,
  async enter() {
    let v = await api("GET", "/api/segment?activate=true"); const a = v.avail;
    body(`<p class="hintt"><b>SEG finds the grains; SAM 2.1 refine is an optional extra pass, not an alternative.</b></p>
      ${a.seg ? "" : `<div class="warnbox">SegmentEveryGrain is not installed, so <b>Run segmentation</b> is off. Install it from <a href="${esc(G.seg_url)}" target="_blank" rel="noopener">${esc(G.seg_url)}</a>, then restart. Saved grains can still be loaded.</div>`}
      ${a.sam2 ? "" : `<div class="warnbox">SAM 2.1 (sam2 + torch) is not installed: the refine pass and Add grain (A) are off.</div>`}
      <label title="${a.sam2 ? "" : "needs sam2 + torch"}"><input type="checkbox" id="sgSam" ${a.sam2 ? "" : "disabled"}> SAM 2.1 refine (optional extra pass)</label>
      <div class="f"><label>Min grain (px)</label><input type="number" id="sgMin" min="0" step="1"><span></span></div>
      <div class="f"><label>Non-grain drop fraction</label><input type="number" id="sgDrop" min="0" max="1" step="0.05"><span></span></div>
      <div class="btns"><button id="sgLoad">Load saved grains</button><button id="sgClear" class="danger">Clear saved grains…</button></div>
      <h4>Edit grains</h4><p class="hintt" id="sgInfo"></p>
      <div class="btns"><button id="sgAdd" title="A" ${a.sam2 ? "" : "disabled"}>Add grain (A)</button><button id="sgUndo" title="Z">Undo last edit (Z)</button></div>
      <div class="btns" id="sgPromptB" hidden><button id="sgAcc" title="Enter">Accept grain (Enter)</button><button id="sgClr" title="Esc">Clear prompts (Esc)</button></div>
      <div class="btns"><button id="sgDel" title="D">Delete (D)</button><button id="sgMerge" title="M">Merge (M)</button><button id="sgDes" title="Esc">Deselect all (Esc)</button></div>
      <div class="btns"><button id="sgReset" class="danger">Reset edits</button></div>`);
    const set = () => api("POST", "/api/segment/settings", {use_sam2: $("sgSam").checked, min_px: +$("sgMin").value, drop: +$("sgDrop").value}).then(v => this.draw(v));
    $("sgSam").checked = v.settings.use_sam2; $("sgMin").value = v.settings.min_px; $("sgDrop").value = v.settings.drop;
    ["sgSam", "sgMin", "sgDrop"].forEach(id => $(id).onchange = set);
    const op = async (o, x = {}) => this.draw(await api("POST", "/api/segment/op", {op: o, ...x})); this.op = op;
    on("sgLoad", "onclick", async () => { await runJob("/api/segment/load", {}, "Loading grains"); this.draw(await api("GET", "/api/segment")); await refreshGlobal(); });
    on("sgClear", "onclick", () => confirmClear("grains") && op("clear_saved"));
    on("sgUndo", "onclick", () => op("undo")); on("sgDel", "onclick", () => op("delete")); on("sgMerge", "onclick", () => op("merge"));
    on("sgDes", "onclick", () => op("deselect")); on("sgReset", "onclick", () => confirm("Throw away all grain edits?") && op("reset"));
    on("sgAdd", "onclick", () => this.addMode(!this.add)); on("sgAcc", "onclick", () => op("accept")); on("sgClr", "onclick", () => op("clear_prompts"));
    this.click = (x, y, b, sh) => this.add ? this.prompt({op: "add_point", x, y, positive: b !== 2}) : op("click", {x, y, shift: sh});
    Viewer.onClick = this.click; Viewer.onBox = (x0, y0, x1, y1) => this.prompt({op: "box", box: [x0, y0, x1, y1]});
    this.add = false; await showImg("bse_stretch", {fit: true}); await this.draw(v);
    mainBtns([{label: "Run segmentation", id: "sgRun", disabled: !a.seg, tip: a.seg ? "" : "SegmentEveryGrain is not installed", fn: async () => {
        await runJob("/api/segment/run", {use_sam2: $("sgSam").checked, min_px: +$("sgMin").value, drop: +$("sgDrop").value}, "Segmenting");
        this.draw(await api("GET", "/api/segment")); await refreshGlobal(); }},
      {label: "Use this grain map", id: "sgUse", fn: async () => {
        await runJob("/api/segment/use", {use_sam2: $("sgSam").checked, min_px: +$("sgMin").value, drop: +$("sgDrop").value}, "Measuring grains, voting minerals, drawing figures");
        await finish(6); }}]);
    if (v.n && !a.seg) { $("sgUse").className = "main"; $("sgRun").className = ""; }
    this.v = v; this.draw(v);
  },
  async addMode(b) {
    if (b && !this.v.sam_ready) { try { await runJob("/api/segment/sam_prepare", {}, "Preparing SAM 2.1"); } catch (e) { return; } }
    this.add = b; Viewer.boxMode = b; $("sgAdd").classList.toggle("on", b); $("sgPromptB").hidden = !b; if (!b) this.op("clear_prompts"); else this.draw(await api("GET", "/api/segment"));
  },
  async prompt(x) { await api("POST", "/api/segment/op", x); const r = await api("POST", "/api/segment/prompt");
    if (r.job) { try { await runJob("/api/segment/prompt", undefined, "SAM"); } catch (e) {} } this.draw(await api("GET", "/api/segment")); },
  async draw(v) {
    this.v = v; pill(v.state[0], v.state[1]);
    $("sgInfo").textContent = `${v.n} grains · ${v.sel.length} selected · ${v.edits} edits` + (v.edited ? " (edited)" : "");
    $("sgLoad").disabled = $("sgClear").disabled = !v.has_saved; $("sgUndo").disabled = !v.edits; $("sgDel").disabled = !v.sel.length; $("sgMerge").disabled = v.sel.length < 2;
    const u = $("sgUse"); if (u) u.disabled = !v.use_ok; const r = $("sgRun"); if (r && !v.avail.seg) r.disabled = true;
    if (v.n && u && !u.disabled) { u.className = "main"; if (r) r.className = ""; }
    if (v.poly_version !== this.pv) { const p = await api("GET", "/api/segment/polys?v=" + v.poly_version); this.polys = p.polys; this.pv = p.v; }
    const sel = new Set(v.sel);
    Viewer.setOverlay((c, T) => {
      Draw.polys(c, T, this.polys.filter((_, i) => !sel.has(i)), "#ffd60a", 1);
      if (sel.size) Draw.polys(c, T, v.sel.map(i => this.polys[i]).filter(Boolean), "#ff2d55", 2, "rgba(255,45,85,.3)");
      if (v.preview) Draw.polys(c, T, [v.preview.flat()], "#30d158", 2, "rgba(48,209,88,.3)");
      if (v.box) { c.strokeStyle = "#64d2ff"; c.lineWidth = 1.5; c.strokeRect(T.x(v.box[0]), T.y(v.box[1]), (v.box[2] - v.box[0]) * T.k, (v.box[3] - v.box[1]) * T.k); }
      Draw.points(c, T, v.prompts.filter(p => p[2]), "#30d158"); Draw.points(c, T, v.prompts.filter(p => !p[2]), "#ff3b30");
    });
  },
  key(k) { const o = this.op;
    if (k === "a" && !$("sgAdd").disabled) this.addMode(!this.add); else if (k === "z") o("undo"); else if (k === "d" || k === "delete") o("delete"); else if (k === "m") o("merge");
    else if (k === "enter" && this.add) o("accept"); else if (k === "esc") this.add && (this.v.prompts.length || this.v.box) ? o("clear_prompts") : o("deselect"); else return false; return true; }});
// 7 Minerals
const ST = {ok: ["✓", "#30a14e", "colour matches the legend"], reassigned: ["▲", "#c98a00", "near colours were given to this mineral"],
  missing: ["✗", "#d70015", "in the legend but not in the map"], unlisted: ["✗", "#d70015", "map colour not in the legend: counts as Unknown"]};
STEPS.push({title: "Minerals", lead: "How the map colours were matched to legend minerals. Add a map colour the legend misses.",
  help: "✓ exact · ▲ near colours reassigned · ✗ missing / not in legend. Pick a figure on the left.",
  controls: [["Figure", "composition / overlay / mixed grains / unmapped grains"], ["+ Add map colour not in legend", "name a ✗ map colour so its grains count"],
    ["Remove added colour", "remove the selected added colour"], ["Clear saved…", "delete the saved added colours"], ["Next", "to the grain size curves (step 8)"]],
  async enter() {
    const v = await api("GET", "/api/minerals"); this.v = v; this.selRow = null; pill(v.state[0], v.state[1]);
    body(`<div class="f"><label>Figure</label><select id="mnFig">${v.figures.map(f => `<option>${f}</option>`).join("")}</select><span></span></div>
      <div class="tbl"><table class="t" id="mnT"></table></div><p class="hintt">${esc(v.summary)}</p>
      <div class="btns"><button id="mnAdd">+ Add map colour not in legend</button></div>
      <div class="btns"><button id="mnRem">Remove added colour</button><button id="mnClear" class="danger">Clear saved…</button></div>`);
    $("mnT").innerHTML = "<tr><th></th><th>Mineral</th><th>Legend</th><th>Map colours</th><th>Pixels</th></tr>" + v.rows.map((r, i) =>
      `<tr data-i="${i}" title="${ST[r.status][2]}"><td class="st" style="color:${ST[r.status][1]}">${ST[r.status][0]}</td><td>${esc(r.mineral || "(not in legend)")}${r.extra ? " +" : ""}${r.skipped ? " (left out)" : ""}</td>
       <td class="sw">${r.legend_rgb ? `<i style="background:${hex(r.legend_rgb)}"></i>` : ""}</td><td class="sw">${(r.map_rgbs || []).slice(0, 6).map(c => `<i style="background:${hex(c)}" title="${c}"></i>`).join("")}</td><td>${r.pixels_text}</td></tr>`).join("");
    $("mnT").querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => { $("mnT").querySelectorAll("tr").forEach(t => t.classList.remove("sel")); tr.classList.add("sel"); this.selRow = +tr.dataset.i; });
    const job = async (b, label) => { await runJob("/api/minerals/op", b, label); await finish(7); };
    on("mnAdd", "onclick", async () => { const s = await api("POST", "/api/minerals/op", {op: "suggest", row: this.selRow});
      entryDialog({title: "Add map colour not in legend", rgb: s.rgb, name: "", row: null, dropSrc: () => "mineral_img",
        check: async (c, n) => !n.trim() ? "Type a mineral name." : (c.some(x => !(x >= 0 && x <= 255)) ? "R, G, B must be 0-255." : ""),
        submit: (c, n) => job({op: "add", rgb: c.join(","), name: n}, "Counting minerals again")}); });
    on("mnRem", "onclick", () => { const r = v.rows[this.selRow]; if (!r || !r.extra) return toast("Select an added colour (+) first"); job({op: "remove", mineral: r.mineral}, "Counting minerals again"); });
    on("mnClear", "onclick", () => confirmClear("added colours") && job({op: "clear_saved"}, "Counting minerals again"));
    $("mnFig").onchange = () => showImg("fig:" + $("mnFig").value, {fit: true}); showImg("fig:" + v.figures[0], {fit: true});
    mainBtns([{label: "Next: grain size", fn: () => go(7)}]);
  }});
// 8 Grain size
STEPS.push({title: "Grain size", lead: "Grain size distribution curves: all grains and one per mineral.",
  help: "Pick a curve on the left. Replot after changing settings.", controls: [["Figure", "all grains or one mineral"], ["Replot grain size curves", "draw the curves again"], ["Next", "to the outputs (step 9)"]],
  async enter() {
    const v = await api("GET", "/api/gsd"); pill(v.state[0], v.state[1]);
    body(`<div class="f"><label>Figure</label><select id="gsFig">${v.figures.map(f => `<option>${esc(f)}</option>`).join("")}</select><span></span></div>
      <p class="hintt">${esc(v.summary)}</p><div class="btns"><button id="gsRe">Replot grain size curves</button></div>`);
    const sh = () => v.figures.length && showImg("fig:" + $("gsFig").value, {fit: true}); $("gsFig").onchange = sh; sh();
    on("gsRe", "onclick", async () => { await runJob("/api/gsd/replot", {}, "Replotting"); await finish(7); });
    mainBtns([{label: "Next: outputs", fn: () => go(8)}]);
  }});
// 9 Outputs
STEPS.push({title: "Outputs", lead: "Nothing has been written yet. Export the CSV data sheets and the figures here.",
  help: "Exports go to the folder shown below. Click a file name to view an exported figure.",
  controls: [["Export CSV data sheets", "grains, counts, composition, geojson…"], ["Export all figures", "PNG figures"], ["Replot grain knolling diagram", "draw it again"],
    ["Open export folder", "show the folder in your file manager"]],
  async enter() {
    const v = await api("GET", "/api/outputs"); pill(v.state[0], v.state[1]);
    body(`<p class="hintt">Export folder: <b>${esc(v.folder)}</b><br>Ready: ${v.n_csv} data sheets, ${v.n_png} figures.</p>
      <div class="btns"><button id="otFig">Export all figures</button></div>
      <div class="btns"><button id="otKn">Replot grain knolling diagram</button><button id="otOpen">Open export folder</button></div>
      <h4>Exported files</h4><div class="files" id="otFiles">${v.files.map(f => `<div data-f="${esc(f)}">${esc(f)}</div>`).join("") || '<span class="hintt">none yet</span>'}</div>`);
    $("otFiles").querySelectorAll("div").forEach(d => d.onclick = () => /\.(png|jpg)$/i.test(d.dataset.f) ? showImg("export:" + d.dataset.f, {fit: true, v: "e" + (++tick)}) : toast(d.dataset.f + " is a data file"));
    on("otFig", "onclick", async () => { await runJob("/api/outputs/export_figures", {}, "Exporting figures"); await finish(8); });
    on("otKn", "onclick", async () => { await runJob("/api/outputs/replot", {}, "Replotting"); await finish(8); });
    on("otOpen", "onclick", async () => { const r = await api("POST", "/api/outputs/folder"); toast("Opened " + r.opened); });
    showImg("knoll", {fit: true});
    mainBtns([{label: "Export CSV data sheets", fn: async () => { await runJob("/api/outputs/export_data", {}, "Exporting CSV data sheets"); await finish(8); }}]);
  }});
// ---- keys, log, start -----------------------------------------------------------------------
document.addEventListener("keydown", e => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || dlg.open && dlg.className !== "nonmodal") return;
  const k = {Enter: "enter", Escape: "esc", Delete: "delete", Backspace: "delete"}[e.key] || e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z") { STEPS[cur].key?.("z", e); e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === "f") { Viewer.fit(); return; }
  if (STEPS[cur].key && STEPS[cur].key(k, e)) e.preventDefault();
});
$("logBtn").onclick = () => { $("logp").hidden = !$("logp").hidden; };
Shell.init({steps: [], onStep() {}});
setInterval(() => { if (!$("logp").hidden) refreshGlobal(); }, 3000);
go(0);
