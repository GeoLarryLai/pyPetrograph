---
doc_type: AI_EDITING_LOG
schema_version: 1
canonical_filename: AI_EDITING_LOG.md
purpose: LLM bootstrap for any new Cursor chat. Read THIS before exploring code.
project: ML petrography / Modal RGB petrography ML
primary_notebook: Modal_rgb_petrographyML.ipynb
module_path: pyPetrograph/
status: v2_multi_image
last_updated: 2026-08-12
conda_env: work
---

# AI_EDITING_LOG

## A. AGENT BOOTSTRAP (do first, every chat)
1. Read this entire file.
2. If task touches API/details: skim `pyPetrograph/README.md`.
3. If editing UI flow: skim notebook markdown cells in `Modal_rgb_petrographyML.ipynb`.
4. Only then open the relevant module under `pyPetrograph/` (see LAYOUT / WHERE TO EDIT).
5. Do NOT re-derive history from `old codes/` or agent transcripts unless user asks.
6. After any meaningful edit: APPEND a row to §G EDIT LOG (date, what, why). Do not rewrite history.

## B. USER — COMMUNICATION (MUST)
- Short, simple plain language. Jargon OK; define briefly on first use.
- Pointed answers. No multi-topic essay dumps. Bold sparingly. No long restatements.
- Plan mode: ONE subject/problem per turn. State understanding → options. Interactive Q&A. Do not decide for user. Do not start next subject without confirmation.
- Function explain: (1) one-sentence summary (2) params (3) outputs.
- User-requested markdown: do not prefix every line with `#` or `%`; `#` only for real headings.
- Mirror rule file: `.cursor/rules/user-working-prefs.mdc` (alwaysApply).

## C. USER — WORK / CODE (MUST / NEVER)
MUST:
- Follow user instructions carefully; minimal diffs only for the goal.
- Error report → fix that issue; preserve unrelated behavior.
- Minimize tests; ask before running. Python → conda env `work`.
- Commit/push only when user explicitly asks (user git/PR rules).
- Keep output file-type compatibility (§E) unless user changes that.
- Large redesign → Plan mode first, one question at a time.

NEVER:
- Resurrect old 37-cell spaghetti from `old codes/`.
- ~~Expand to multi-file package without ask (v1 = one `__init__.py`).~~ **Superseded 2026-08-06:** user approved split into `common/`, `image_processing/`, `labeling_ml/`; keep top-level `from pyPetrograph import …`.
- Merge other notebooks unless asked.
- Break stem naming / companion file contracts without ask.
- Silent broad refactors, drive-by cleanup, unsolicited docs.

## D. PROJECT SNAPSHOT
GOAL: RGB thin-section → multi-image queue → polygons/wand → train (per-image or universal) → predict maps → modal (area) fractions.

LAYOUT:
```
AI_EDITING_LOG.md                 # THIS file (LLM track record)
Modal_rgb_petrographyML.ipynb     # thin UI + pipeline cells
pyPetrograph/
  __init__.py                     # re-exports public API (notebooks import here)
  common/                         # constants, paths, image_io, session, ui, notebook_runners
  image_processing/               # brightness, features, fractions, viz
  labeling_ml/                    # polygons, train, predict, model_io
  README.md                       # API docs
old codes/                        # archived pre-refactor notebooks (reference only)
```

ENTRY:
- `%matplotlib inline` for notebook figures; **Label** via `launch_app` → separate desktop process (Matplotlib **QtAgg** / PyQt5)
- deps → imports → one Settings cell → select → `launch_app`
- then summary → `run_feature_preview` → `run_train_cell` → `print_train_metrics` → `run_predict_cell`
- Settings: set attrs on `session` (classes, `image_dir`, load size, `method`, `n_jobs`, `preview_dpi`, `figure_dpi`); call `remember_settings_classes()` after class_names; advanced as comments. Train/predict modes live in those cells.
- Core state object: `Session` (multi-image queue)

STACK: numpy, pillow, matplotlib (Qt popups + PolygonSelector + LOD), sklearn, scipy, geopandas/shapely, rasterio/affine, joblib, pandas, scikit-image (LBP/GLCM); optional lightgbm/xgboost.

## E. FILE CONTRACTS (compat — do not break)
Original images stay where they are. Beside each image: only `labels/` and `predictions/`. Models live under notebook-level `models/` (next to the `.ipynb`). No `cache/` folder; no image-folder `models/`.

Under `{image_folder}/labels/` (Save writes polygons+session only; train/reload from polygons):
- `{stem}_polygons.geojson` — FeatureCollection; properties.class + geometry
- `{stem}_session.json` — `class_names` (id→name, incl. Label UI adds) + `current_class` only; Settings knobs never read/written here
- `{stem}_labels.png` — label summary **figure** export only (was `*_summary.png`; NOT uint8 class ids)

Under notebook-level `models/` (beside the `.ipynb`):
- `{stem}_model.joblib` — per-image train bundle
- `universal_model.joblib` — pooled train bundle

Under `{image_folder}/predictions/`:
- `{stem}_pred.png` — HSV RGB class map
- `{stem}_pred_rgb.png` — overlay
- `{stem}_confidence.npy` — float32 HxW, P(assigned label) after smooth (0–1)
- `{stem}_fractions.csv` — class_id, name, count, area_pct, area_pct_low, area_pct_high, area_pct_plus, area_pct_minus, sure_threshold, model_used
- `{stem}_fig_pred.png`, `{stem}_fig_confidence.png` — predict cell figures
- `{stem}_features.png` — feature preview cell figure (saved at `figure_dpi`)

RGB working image: RAM only until kernel restart (`ensure_rgb_cache` / `_RGB_RAM_CACHE`). No permanent `*_rgb_cache.npy` on disk.

No legacy flat companions beside the image (`*_labels.png` next to the TIFF, `labels_image.png`, `polygons.geojson`, `*_image_small.png`, beside-image `*_model.joblib`). `select_images` deletes those flat leftovers once per folder.

Prints use notebook-relative abbreviated paths + `subfolder/filename` lists (not absolute paths). Predict print: `name: 41.20%  +2.80%/−4.40%  (n=…)` — no `[low–high]` in print; CSV still has low/high columns.

DEFAULTS: method=`lightgbm`; y_target=`128`; n_jobs=`4`; preview_dpi=`150`; figure_dpi=`300`; median_size=`4`; scale=`1.0`; load_target_mp=`None` (native); preview_target_mp=`None` (screen-auto); draw_mode=`wand`; wand_threshold=`70`; wand_simplify=`1.5`; wand slider 10–150; random_state=`42`
CLASSES default ids 1..8: olivine, clinopyroxene, orthopyroxene, chromium_spinel, nickel_sulfide, magnetite, apatite, holes (editable in UI; files store ints)

## F. LOCKED DESIGN DECISIONS (v2)
- Package is a multi-module tree under `pyPetrograph/` (`common/`, `image_processing/`, `labeling_ml/`); public import stays `from pyPetrograph import …` via `__init__.py` re-exports. (Was one-file until 2026-08-06 user-approved split.)
- Scope: this notebook only (not merging mineralogy clones yet).
- Multi-format load → RGB uint8 HxWx3 (`load_image`: PIL first, rasterio fallback).
- Multi-image ordered queue; dropdown switch; **Save** only writes labels to disk (overwrite).
- Auto-load companions on open (labels/polygons/session; model if present).
- Label UI: **subprocess** desktop window with Matplotlib **QtAgg** (PyQt5), same backend idea as DMG-07 `%matplotlib qt`. Left controls + right canvas. Never use tkinter for screen size / GUI (macOS crash). `select_images()` / Add-more pickers are Qt **subprocess** (kernel stays Qt-free; all OS); cleans flat legacy files; RGB via RAM cache (`ensure_rgb_cache` / `_RGB_RAM_CACHE`).
- Enter finishes hand polygon; Esc stops editing window (not clear-poly).
- Draw modes: default **wand** (threshold 70, slider 10–150, simplify 1.5); or polygon; flood-fill + simplify; threshold in UI.
- View: downsample fit-all ≈ `preview_target_mp` (`None` = screen-auto; else fixed MP); high-res crop when zoomed; **minimap in dedicated lower-left axes** (not inset); wheel zoom; Ctrl/Cmd+drag pan. Outline-only turbo polygons. No napari/VisPy/GPU.
- Features (toggles): R,G,B,Y, local_std, local_grad, LBP; GLCM polygon-only (optional). Train at full working res; preview downsampled inline (all queue, 1 column, `preview_dpi`).
- Train: load `.joblib` | retrain; retrain scope per_image | universal; skip unlabeled; CV metrics via `print_train_metrics`.
- Predict: all queued images; universal | per_image model; missing model → error + stop (plain fix hint); `predict_proba` → hard/sure/soft area % with +/− bounds (`DEFAULT_SURE_PROBA=0.8`); print `name: 41.20%  +2.80%/−4.40%  (n=…)` (no `[low–high]` in print; CSV keeps low/high); single confidence map = P(assigned) after smooth; one fig/image (pred + confidence) at `figure_dpi`.
- Models under notebook-level `models/` (beside `.ipynb`: `universal_model.joblib`, `{stem}_model.joblib`); beside each image only `labels/` + `predictions/`; no `cache/`, no image-folder `models/`. RGB cache = RAM only until kernel restart (B1: `ensure_rgb_cache` / `_RGB_RAM_CACHE`; no permanent `*_rgb_cache.npy`).
- `session.image_dir`: browse start folder only (`Path(".")` → notebook folder when Jupyter started there).
- Class dropdown by name; editable name + Add label; **Save** → disk (all on-screen labels, overwrite); **Reload** → load saved labels from disk (undoable); **Restart** → clear current-class labels in memory only (not class names, not disk); no autosave on Prev/Next or window close; `session.json` = class names + `current_class` only (Settings cell is source of truth for `y_target`, `preview_dpi`, `figure_dpi`, `draw_mode`, `wand_threshold`, method, feature toggles, etc.).
- Undo = general step stack (includes Restart and Reload).
- One train API family; methods: lightgbm (default) | random_forest | xgboost.
- Train evaluates with stratified k-fold CV (default 5 folds; `n_jobs` parallelizes folds), then fits the saved model on **all** labeled pixels (no 80/20 holdout).
- Always Y-brightness normalize (keep color ratios); default Y=128 editable.
- uint8 on disk; float32 features; chunked predict; compact polygon coords.

OUT OF SCOPE (ask before doing): histogram/cross-slide matching beyond Y-norm; heavy polygon vertex editor; napari/VisPy viewer; Wavelets/Gabor/fractal (see §J).

## G. EDIT LOG (append-only)
| date | change | why / notes |
|------|--------|-------------|
| pre-2026-07 | Legacy RF cell notebooks | In-notebook logic; hard paths; no model save; geojson/all_polygons bug risk; class-name mismatch; duplicated notebooks → later moved to `old codes/` |
| 2026-07-17 | Plan redesign | package+thin notebook; stem I/O; dual UI; LGBM default; Y-norm; float32/chunk/n_jobs=4; keep PNG/GeoJSON compat |
| 2026-07-17 | Implement v1 | `petrography_ml/` + rewritten notebook + README; train/predict/save/autoload checked in conda `work` |
| 2026-07-17 | Add/rename AI_EDITING_LOG.md | LLM bootstrap + prefs + decisions + log; Cursor alwaysApply rule points here |
| 2026-07-17 | Add user-working-prefs.mdc | Refined global-style prefs as alwaysApply project rule; synced §B–§C |
| 2026-07-17 | Soften AI_EDITING_LOG rule | Load log only if present; if missing do not ask; use working prefs |
| 2026-07-19 | v2 multi-image pipeline | Queue+autosave; Enter/Esc; wand; LOD view; models/ + predictions/; universal/per-image train; predict stop-on-missing; notebook 8-section rewrite; README+contracts |
| 2026-07-19 | Rewrite notebook to 14-cell pipeline | title→deps→imports→params→label→summary→train→predict; thin UI over petrography_ml; empty outputs |
| 2026-07-19 | Docs catch-up for v2 | Rewrite `petrography_ml/README.md` for queue/train/predict helpers + file layout; sync §D goal/ENTRY; confirm §E/§F/§J/§L |
| 2026-07-19 | Color+texture features | RGB/Y/std/grad/LBP (+ optional polygon GLCM); feature toggles + inline preview MP/DPI; IMAGE_DIR browse; notebook categorized short md; Wavelets/Gabor/fractal deferred to §J |
| 2026-07-19 | Notebook settings collapse | One Settings cell; drop MODELS/PREDICTIONS notebook knobs; IMAGE_DIR default notebook folder (`Path(".")`); accepted types in README; keep n_jobs + preview_dpi; advanced as comments |
| 2026-07-19 | Path/knob polish | `ensure_output_dirs` always beside image folder (ignore sticky/session custom dirs); Settings tune tips; Features defaults table; `{image_folder}` wording in contracts |
| 2026-07-19 | Settings direct on session | Drop intermediate METHOD/N_JOBS/… vars; assign `session.*` in place; train/predict modes set at call sites with option comments |
| 2026-07-19 | Qt-only Label UI | Replace ipywidgets inline panel with Qt control popup; Qt file dialogs (no tkinter); fixes Select-images kernel death in Cursor |
| 2026-07-19 | Old-style label popup | Match `Modal_mineralogy_pp`: one `plt` figure + PolygonSelector; macOS Finder pick; drop control-panel/QApplication crash path |
| 2026-07-19 | Label GUI subprocess | `launch_app` spawns separate process (`macosx`); Cursor Jupyter kernel no longer hosts GUI (fixes hard crash) |
| 2026-07-19 | Left-panel LabelWindow | DMG-07-style popup: left buttons/fields, right image canvas; Finder select; still subprocess-launched |
| 2026-07-19 | select_images before label | Finder-only multi-folder pick cell; remove Select from LabelWindow (was crashing); launch_app requires queue |
| 2026-07-20 | Fix Tk crash; QtAgg like DMG-07 | `get_screen_size_px` no longer uses tkinter; label child uses QtAgg/PyQt5 (DMG-07 style) |
| 2026-07-20 | Feature preview defaults/cmaps | Undefined toggles → False; preview cmaps Reds/Greens/Blues, y gray, std/grad/lbp viridis/plasma/inferno |
| 2026-07-20 | Stratified k-fold CV then fit-all | Score via stratified k-fold (default 5); print classification report in train cell; saved model fit on all labels; replaces 80/20 holdout |
| 2026-07-20 | RGB cache + Label UI + figures | `*_rgb_cache.npy` after select; drop `_image_small`; minimap LL; short status + editable name/Add label; Restart→Settings classes; turbo outlines; feature 1-col all queue; label summary gray+polys; Accuracy cell; predict 1 fig/image; `figure_dpi=300` |
| 2026-07-20 | Docs: Label UI / cache / figures | Notebook+README+§E/§F: rgb_cache.npy; figure_dpi 300; minimap LL; Accuracy cell; drop image_small contract |
| 2026-07-20 | Review fixes: Restart defaults + UI | Stop re-snapshotting class defaults at launch_app; return poly class_ids; clamp status above minimap |
| 2026-07-20 | Notebook trim + wand defaults | Drop DEFAULT_* imports; Session figure_dpi default; Label default draw_mode=wand, wand_threshold=50 |
| 2026-07-20 | Outputs in subfolders + short prints | labels/ cache/ models/ predictions/; drop legacy flat companions; prints use relative paths + subfolder/file names |
| 2026-07-20 | Restore `_pick_files_native` / `_ask_add_more_images` | Accidental drop during layout rewrite caused `select_images` NameError; Finder/osascript first, Qt fallback |
| 2026-07-20 | Restore `_downsample_for_view`; check PyQt5 in deps | LabelWindow NameError; PyQt5 in REQUIRED_PACKAGES + conda hint |
| 2026-07-20 | Label summary turbo outlines like UI | Drop muted fill; raster fallback uses discrete `class_color_rgba` |
| 2026-07-20 | Predict uncertainty + confidence map | `predict_proba`; fractions main/low/high +/−; `{stem}_confidence.npy`; CSV extra cols; sure thr 0.8; §J item done |
| 2026-07-20 | Fix mixed-folder label wipe | `save_all` skips empty overwrite of disk; summary reloads disk; per-image `paths_for`; autosave on label window close |
| 2026-07-20 | HSV class colors + legends / predict figs | Distinct HSV (not turbo); UI outline-only; summary fill α=0.35+legend; pred+confidence as 2 figs; inferno h-colorbar |
| 2026-07-20 | Per-folder predict/train outputs | Writes beside each image; manifests per folder; universal model copied to each queue parent; clear path prints |
| 2026-07-20 | CV metrics as % + class names in report | Accuracy printed as `99.736%` (3 dp); report rows `1 (pores)` not bare ids |
| 2026-07-20 | Locked layout: models + RAM cache | Models → notebook-level `models/`; beside image only labels/+predictions/; RGB RAM-only (`ensure_rgb_cache`); wand 70 / simplify 1.5; predict print without `[low–high]` |
| 2026-07-20 | Simplify package check print + pip progress | Env assumed selected; no per-pkg dump / no dict echo; stream install; YAML TF/SAM stack not required |
| 2026-07-20 | Drop run_manifest.json | Not read by pipeline; stop writing; delete existing |
| 2026-07-20 | Rename package petrography_ml → pyPetrograph | Folder + imports + docs; joblib filenames unchanged |
| 2026-07-20 | Package check: list once, then install progress | Restore required-pkg status table; pip only after; no dict echo |
| 2026-07-20 | Label Save-only; Restart clears current class | No autosave Prev/Next/close; Reload undoable; wand/mode from Settings not session.json |
| 2026-07-20 | Fix pred/labels PNG visibility + savefig cell figures | pred=HSV RGB; pred_rgb=overlay; pred_ids; labels_rgb; summary/fig exports |
| 2026-07-20 | Trim label/pred PNG clutter | summary→labels.png; drop labels_rgb/pred_ids; features.png at figure_dpi; train from polygons |
| 2026-07-20 | session.json = class_names only | Settings knobs stay in notebook; figure_dpi no longer overwritten on load |
| 2026-07-29 | OS-agnostic file pickers | Qt creates QApplication if needed; Add-more via Qt on Win/Linux; notebook/README wording off Finder-only |
| 2026-07-29 | Qt picker subprocess (all OS) | why: in-kernel QApplication crashed Windows Jupyter (3221226505); match launch_app pattern |
| 2026-08-06 | Split pyPetrograph into purpose folders | User-approved; cut monolith `__init__.py` → `common/`, `image_processing/`, `labeling_ml/`; `__init__.py` re-exports; notebook import name unchanged |
| 2026-08-06 | Apache-2.0 then AGPL-3.0+ | User first chose Apache-2.0, then preferred AGPL-3.0-or-later for pyPetrograph (and pyRockImageLab); LICENSE + README updated |
| 2026-08-11 | Restore `@dataclass` on `StemPaths` | Missing decorator → `StemPaths() takes no arguments` on `select_images` |
| 2026-08-12 | Plain-language features markdown | Notebook: define toggles; clearer r/g/b/y/grad wording |
| 2026-08-12 | Features md: what std/grad/LBP/GLCM measure | Spell out acronyms + how each is computed |
| 2026-08-12 | Fix predict `Image` NameError | `run_predict_cell` used `PIL.Image.open` without import |

## H. WHERE TO EDIT
| Intent | Touch |
|--------|-------|
| Constants / paths / image I/O / Session / Label UI / notebook runners | `pyPetrograph/common/` |
| Brightness / features / fractions / viz | `pyPetrograph/image_processing/` |
| Polygons / train / predict / model_io | `pyPetrograph/labeling_ml/` |
| Public re-exports only | `pyPetrograph/__init__.py` |
| User-facing steps / expected outcomes text | notebook markdown cells |
| API documentation | `pyPetrograph/README.md` |
| Catch-up / prefs / decisions / history | THIS file (§G append) |
| Ignore unless migrating | `old codes/*` |

## I. ANTI-PATTERNS (seen before — avoid)
- Putting training loops back into many notebook cells.
- Saving geojson from “current batch only” instead of full polygon list.
- Full-image pandas DataFrame predict.
- Hard-coded absolute image paths.
- Changing mineral_names list without syncing class_names / label ids.
- Dumping long plan options in one message (user rejects this).

## J. FUTURE IDEAS (not committed)
- Click-select polygon delete/reassign
- Image↔pred overlay slider
- ~~predict_proba confidence map~~ (done 2026-07-20)
- Class-set presets
- napari/VisPy viewer if LOD stays laggy on huge slides
- Wavelets (LL/LH/HL/HH), Gabor filters, fractal texture — considered for classification features; deferred (complexity/cost); revisit if RGB+Y+std+grad+LBP is not enough

## K. CROSS-PROJECT CONVENTION
- Filename always: `AI_EDITING_LOG.md` at project root (or beside primary artifact).
- Cursor rule: if the file exists → read/follow/append; if missing → do not ask; fall back to user working preferences only.
- Keep §B–§C (user prefs) stable across projects; replace §D–§J with project-specific content.

## L. END_RITUAL
After meaningful edits:
1. README matches §E file contracts and Label/Train/Predict behavior.
2. Notebook cells match pipeline (deps → imports → params → label → summary → train → predict).
3. §G EDIT_LOG has a new row; §E/§F updated if contracts/decisions changed.
4. Do not git commit/push unless the user asked.
