# pyPetrograph

RGB thin-section **modal mineralogy** toolkit: label one or many images, train a pixel classifier (LightGBM / Random Forest / XGBoost), predict class maps, and report mineral area fractions.

```python
from pyPetrograph import Session, launch_app, check_and_install_packages
```

---

## Quick start (notebook cells)

Typical cell order in `Modal_rgb_petrographyML.ipynb`:

1. **deps** — `check_and_install_packages()`
2. **imports** — `%matplotlib qt`
3. **Settings** — assign on `session` (classes, `image_dir`, load size, `method`, `n_jobs`, `preview_dpi`, `figure_dpi`; `remember_settings_classes()`; advanced as commented overrides). Train/predict modes are set in those cells.
4. **select_images** / **launch_app** — OS-agnostic Qt file picker (**subprocess**, not in-kernel) → Label window (subprocess)
5. **print_label_summary**
6. **run_feature_preview** — toggle RGB/Y/std/grad/LBP/GLCM; inline downsampled maps
7. **run_train_cell** — `load` | `retrain` (`per_image` | `universal`)
8. **print_train_metrics** — CV accuracy after train
9. **run_predict_cell** — all queued images

```python
%matplotlib qt
from pyPetrograph import (
    launch_app,
    print_label_summary,
    print_train_metrics,
    run_feature_preview,
    run_train_cell,
    run_predict_cell,
)

session = launch_app()
print_label_summary(session)
run_feature_preview(session)
train_results = run_train_cell(session, mode="retrain", scope="universal")
print_train_metrics(train_results, session=session)
run_predict_cell(session, model_source="universal")
```

`session.image_dir = Path(".")` → file picker **starts** in the folder that contains this notebook (when Jupyter was started there).  
Example: `session.image_dir = Path(r"/path/to/images")`.

Beside each image: only `labels/` and `predictions/`. Models live under notebook-level `models/` (next to the `.ipynb`). RGB working image is RAM-only until kernel restart (no permanent `*_rgb_cache.npy`).

Accepted image types: tif/tiff/geotiff, jpg/jpeg, png, bmp, webp, gif.

---

## Package structure

Logic is split by purpose under `pyPetrograph/`. Notebooks still import the package top level:

```python
from pyPetrograph import Session, launch_app, check_and_install_packages
```

```
pyPetrograph/
  __init__.py                 # re-exports public API
  common/
    constants.py              # defaults, extensions, feature toggles
    paths.py                  # stem paths, legacy cleanup, package check
    image_io.py               # load/save + RGB RAM cache
    session.py                # Session state
    ui.py                     # LabelWindow, pickers, launch_app
    notebook_runners.py       # summary / preview / train / predict cells
  image_processing/
    brightness.py             # Y-norm / luminance
    features.py               # feature stack + view helpers
    fractions.py              # modal area fractions
    viz.py                    # class colors, overlays, label PNGs
  labeling_ml/
    polygons.py               # GeoJSON / wand / rasterize
    train.py                  # classifiers + CV
    predict.py                # predict / smooth / confidence
    model_io.py               # joblib bundles
  README.md
```

---

## File layout

Original images stay put. Beside each image: `labels/` + `predictions/` only. Models at notebook-level `models/`.

```
{notebook_dir}/
  models/{stem}_model.joblib
  models/universal_model.joblib

{image_folder}/
  labels/{stem}_polygons.geojson     # Save; train/reload from polygons
  labels/{stem}_session.json         # Save: class_names + current_class only
  labels/{stem}_labels.png           # label summary figure export only (NOT uint8 class ids)
  predictions/{stem}_pred.png        # HSV RGB class map
  predictions/{stem}_pred_rgb.png    # overlay
  predictions/{stem}_fig_pred.png    # predict cell: class map figure
  predictions/{stem}_fig_confidence.png
  predictions/{stem}_features.png    # feature preview cell figure (figure_dpi)
  predictions/{stem}_confidence.npy  # float32 HxW, P(assigned label) after smooth (0–1)
  predictions/{stem}_fractions.csv   # class_id,name,count,area_pct,area_pct_low,area_pct_high,area_pct_plus,area_pct_minus,sure_threshold,model_used
```

No `cache/` folder; no image-folder `models/`. RGB cache = RAM only (`ensure_rgb_cache` / `_RGB_RAM_CACHE`) until kernel restart.

Flat legacy companions beside the image are **not** used; `select_images` deletes them. Notebook prints use paths relative to the notebook cwd plus `subfolder/filename`.

Polygon GeoJSON coords are rounded (~0.1 px) for compact files.

---

## Label UI

**Important (Cursor / VS Code Jupyter):** opening a Matplotlib GUI *inside* the kernel often **kills the kernel**.  
`launch_app(session)` therefore starts a **separate Python process** with a desktop window (Matplotlib **QtAgg** / PyQt5 — cross-platform).

Layout (DMG-07-style popup):
- **Before Label:** `select_images(session)` — OS-agnostic Qt file picker via **subprocess** (not in-kernel; same pattern as Label window); multi-select; **Add more** for other subfolders (also Qt subprocess; no Matplotlib). RGB via RAM cache (`ensure_rgb_cache`).
- **Left:** Prev/Next, Save/Reload/Undo/Restart, Cycle class, Mode (default **wand**, threshold **70**, slider 10–150, simplify 1.5), Fit, editable class name + **Add label**, short status (incl. current-class polygon count)
- **Right:** image canvas + outline-only polygons; **minimap** in dedicated axes (lower-left). Enter finishes polygon; Esc starts a new one.
- **Save** — only control that writes all on-screen labels to disk (overwrite). **Reload** — load saved labels from disk into memory (undoable). **Restart** — clear only the **current class** labels in the scene (polygons + pixels); does not reset class names or write disk. **Prev/Next**, window close, and **Cycle class** do **not** autosave. **Undo** rewinds in-memory steps (including Restart and Reload). Notebook **Settings** cell is the source of truth for `y_target`, `preview_dpi`, `figure_dpi`, `draw_mode`, `wand_threshold`, method, feature toggles, etc. — never read/written from `session.json`.

After **Save**, run `print_label_summary(session)` in the notebook (one figure per image: gray background + colored polygons; DPI = `figure_dpi`, default 300). Also writes `labels/{stem}_labels.png` per image (figure export only). **Save** writes `{stem}_polygons.geojson` + `{stem}_session.json` (class_names + current_class only); train/reload rasterizes from polygons.

For plain IPython/script (not Jupyter): `launch_app_inplace(session)`.

---

## Features (color + texture)

Per-pixel train/predict features (toggles in the notebook). **Training uses full working resolution**; preview maps are downsampled to screen size (or `preview_target_mp` if set).

| Toggle | Meaning |
|--------|---------|
| `r`,`g`,`b` | RGB after optional Y-norm |
| `y` | luminance (weighted gray) |
| `local_std` | local std of Y (window, default 7) |
| `local_grad` | Sobel magnitude of Y |
| `lbp` | uniform LBP on Y (`scikit-image`) |
| `glcm` | polygon-level Haralick contrast + homogeneity (broadcast onto polygon pixels; **not** full-image dense GLCM) |

Default: all **off** unless you set a toggle `True` in the notebook (missing keys → False). If every toggle is off, train/predict falls back to RGB. Model bundle stores `feature_names` + toggles. Use RGB (not CMYK) for thin sections.

Preview colormaps: `r`/`g`/`b` → Reds/Greens/Blues; `y` → gray; `local_std`/`local_grad`/`lbp` → viridis/plasma/inferno.

`run_feature_preview(session)` plots enabled maps **inline** for **all queued images**, **one column** (DPI = `preview_dpi`, default 150). Also saves `predictions/{stem}_features.png` per image at `figure_dpi` (default 300). Label summary / predict figures use `figure_dpi` too. Qt popups are for the Label UI only.

---

## Train

Via `run_train_cell` / `Session.train_queue`:

| Option | Behavior |
|--------|----------|
| `mode="load"` | Load a `.joblib` (file picker or `model_path=…`) |
| `mode="retrain"` | Retrain from labeled pixels in the queue |
| `scope="per_image"` | One model per labeled image → notebook-level `models/{stem}_model.joblib` |
| `scope="universal"` | Pool all labeled pixels → notebook-level `models/universal_model.joblib` |

Unlabeled queue images are **skipped** (listed in the result).

Retrain scores with stratified **k-fold CV** (default 5 folds), then fits the saved model on **all** labeled pixels. Use `print_train_metrics(train_results, session=session)` in the notebook Accuracy cell.

---

## Predict

Via `run_predict_cell` / `Session.predict_queue`:

- Runs on **all queued** images.
- `model_source="universal"` or `"per_image"`.
- **Missing model → error and stop** with a plain message saying what to train/load.
- Uses `predict_image_proba` + `compute_fractions_uncertain` (`DEFAULT_SURE_PROBA = 0.8`):
  - **main** = hard argmax area %
  - **low** = sure % (assigned class & P ≥ 0.8)
  - **high** = soft mean P(class)
  - print like `name: 41.20%  +2.80%/−4.40%  (n=…)` (+/− from high−main / main−low, floored at 0; no `[low–high]` in print; CSV still has low/high columns)
- Confidence map (option A): single map = P(assigned label) after smooth → `predictions/{stem}_confidence.npy` (float32 0–1). Notebook shows pred + confidence side-by-side (viridis, vmin=0 vmax=1, colorbar).

Writes under `{image_folder}/predictions/`:
- `{stem}_pred.png` — HSV RGB class map
- `{stem}_pred_rgb.png` — overlay
- `{stem}_fig_pred.png`, `{stem}_fig_confidence.png` — notebook figure exports
- `{stem}_confidence.npy`, `{stem}_fractions.csv`

Models load from notebook-level `models/`. Shows **one figure per image** at `figure_dpi` (also saved as `*_fig_*.png`).

---

## Main API

### Helpers (notebook cells)

| Function | Role |
|----------|------|
| `check_and_install_packages(…)` | Assumes kernel env already selected; checks required pkgs and pip-installs gaps (esp. PyQt5); optional `xgboost` not auto-checked |
| `print_label_summary(session)` | Stats + one gray+colored-poly figure per queued image |
| `print_train_metrics(train_results, session=…)` | Print stratified CV metrics after train |
| `run_train_cell(session, mode=…, scope=…)` | Load or retrain (see Train above) |
| `run_predict_cell(session, model_source=…)` | Predict queue + print fractions + one fig/image |
| `magic_wand_to_polygon(image, row, col, …)` | Color flood → simplified polygon vertices |

### `Session` — queue + ML state

| Method | What it does |
|--------|----------------|
| `set_image_queue(paths)` | Set ordered multi-image queue |
| `switch_image(idx\|path, autosave=False)` | Open another queue image (no autosave by default) |
| `open_image(path, …)` | Load image, init labels, auto-load companions |
| `add_polygon` / `add_wand_at` | Undo-aware labeling |
| `save_all` / `load_labels_only` | Disk I/O beside the image |
| `train_queue(scope=…)` | Retrain per_image or universal |
| `predict_queue(model_source=…)` | Predict all queued images |
| `train` / `predict` / `fractions` | Single-image ML (current image) |
| `add_class` / `set_class_name` / `remove_class` | Class registry |
| `undo` / `restart_labels` | Edit history |

### Image / labels / ML (library)

| Function | Role |
|----------|------|
| `load_image` | Multi-format → RGB uint8 |
| `stem_paths` | Companion path builder (notebook `models/` + image `labels/`/`predictions/`) |
| `normalize_brightness` | Y-norm keeping color ratios |
| `rasterize_polygons` / GeoJSON I/O | Label burn + compact polygon save/load |
| `train_classifier` / `predict_image` / `predict_image_proba` | Core ML |
| `smooth_prediction` / `colorize_prediction` / `compute_fractions` / `compute_fractions_uncertain` | Post-process |
| `save_model_bundle` / `load_model_bundle` | joblib persistence |
| `build_ui` / `launch_app` | ipywidgets panel + Qt label window |

---

## Dependencies

Required: numpy, pillow, matplotlib, scikit-learn, scipy, geopandas, shapely, rasterio, affine, joblib, pandas, lightgbm, scikit-image, **PyQt5** (label window / QtAgg; `conda install -c conda-forge pyqt` if missing). Optional: `xgboost` (not auto-checked by `check_and_install_packages`).

Assumes the notebook kernel already uses your conda/venv; `check_and_install_packages()` fills any missing required packages into that env.

```bash
conda activate work
pip install xgboost   # optional
```

Or from a notebook: `check_and_install_packages()`.

---

## Design notes

- **Memory:** uint8 on disk for labels/preds; RGB working image RAM-only until kernel restart; float32 features; chunked predict.
- **Parallelism:** default `n_jobs=4`.
- **Multi-image:** queue labeling, universal or per-image train, batch predict — all in scope.
- **Layout (locked):** notebook-level `models/`; beside image only `labels/` + `predictions/`; no disk RGB cache.

---

## License

**pyPetrograph** is licensed under the [GNU Affero General Public License 3.0 or later](LICENSE).
