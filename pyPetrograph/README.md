# pyPetrograph

See the [root README](../README.md) for what the package does, the under-development warning, and the SegmentEveryGrain / SAM 2.1 note.

v0.0.6: hand-label RGB features → LightGBM area %; mineral-map grain counts; multi-image overlay grain segmentation (PPL + XPL, SegmentEveryGrain U-Net).

```python
from pyPetrograph import Session, launch_app, check_and_install_packages, build_channel_stack
```

---

## Notebooks

| Notebook | What it does |
|----------|----------------|
| `pyPetrograph_rgb_porosity.ipynb` | Wand/polygon pore + grain labels → LightGBM → porosity % with uncertainty |
| `pyPetrograph_mineralmap_count.ipynb` | Flat-color mineral maps (legend TIFF) → snap colors → absorb specks → split touching grains → per-class counts with uncertainty |
| `pyPetrograph_multimodal_grains.ipynb` | PPL + XPL → physics channels → SEG **U-Net** outlines (SAM off by default) → object table → names → stats |

Each multimodal stage starts with `LOAD_SAVED_* = True`. Kernel **never imports TensorFlow**. GrainPlot, the N-channel U-Net, and the montage run in subprocesses of conda `work`.

v0.0.3 (multilayer embed notebook + fake-RGB / 16-ch student) lives in `old codes/pyPetrograph_v0.0.3/`.

---

## Quick start (RGB)

Typical cell order in `pyPetrograph_rgb_porosity.ipynb`:

1. **deps** — `check_and_install_packages()`
2. **imports** — `%matplotlib qt`
3. **Settings** — classes, `image_dir`, method, `n_jobs`
4. **select_images** / **launch_app** — Qt file picker then Label window (both subprocesses)
5. **print_label_summary**
6. **run_feature_preview**
7. **run_train_cell** — `load` | `retrain` (`per_image` | `universal`)
8. **print_train_metrics**
9. **run_predict_cell**

`session.image_dir = Path(".")` → file picker starts in the notebook folder.

---

## Quick start (multimodal)

Edit only the Settings cell in `pyPetrograph_multimodal_grains.ipynb` (default: `Test images` AV03 PPL + XPL).

1. Align (auto + tweak window; Axioscan 2% stack is already on one grid — skip auto)
2. Physics channels (`build_channel_stack` → `channels/{stem}_channels.npz`)
3. Outlines: watershed (no training) or SEG **U-Net** (~26 MB). Leave `USE_SAM2 = False`. SAM refine (~860 MB) is only for finer grain shapes.
4. Object table (grains + leftover pores/cement)
5. k-means + SEG montage labeler
6. LightGBM + overlay + stats

`INIT_UNET` / `TRAIN_UNET` stay False until you have a GrainPlot mask you trust.

---

## Quick start (mineral map)

Edit only the Settings cell in `pyPetrograph_mineralmap_count.ipynb` (`IMAGE_DIR`, `LEGEND` name → RGB in legend order, `COLOR_TOL`, `SPECK_PX`, `UNKNOWN_SPECK_PX`, `MIN_GRAIN_PX`, `SPLIT`, `BRIDGE_PX`, `H_MIN`, `NOTCH_SOLIDITY`, `NOTCH_HALVES`, `UNKNOWN_FRAC`, `DISPLAY_RGB`, `N_JOBS`). Call `set_output_dir("outputs")` so files land under `outputs/mineralmap/`.

Workflow figure (repo only, not in the PyPI package): [`project images/workflow.png`](../project%20images/workflow.png). Far-from-legend colors group into Unknown. A grain is named by its largest known color; Unknown only if Unknown pixels ≥ `unknown_frac` (default 0.95).

1. Legend check (`read_legend_swatches` + `check_legend` against the legend TIFF)
2. Run all images (`process_folder`, one loky worker per image, `LOAD_SAVED` reuses `mineralmap/{stem}_counts.csv`)
3. Preview one overlay + colors grouped into Unknown
4. Stacked share chart

Counts table cells read `n ± √n [n_lo–n_hi]`: Poisson counting error, then the method range (no-split count, min grain size halved / doubled).

---

## Package structure

```
pyPetrograph/
  __init__.py
  common/          # constants, paths, Session, Label UI, notebook runners
  image_processing/ # Y-norm, feature stack, fractions, colors
  labeling_ml/     # polygons, LightGBM train/predict, joblib
  align/           # Scene, register, lineup window; embed.py is an optional extra
  fuse/            # extinction sine fit, channel stack, tiles, Axioscan helpers
  objects/         # watershed, N-ch U-Net worker, GrainPlot, table, cluster, classify, stats
  mineral_map/     # legend snap, speck absorb, colour regions + bridge / h-maxima / notch split, grain table, counts with uncertainty, parallel driver
```

---

## File layout

Original images stay put. Default: beside each image `labels/` + `predictions/` + `align/` + `channels/` + `objects/` + `mineralmap/`; models at the project `models/`. `set_output_dir("outputs")` moves all of those under `outputs/` (demo notebooks).

```
{notebook_dir}/
  models/{stem}_model.joblib
  models/universal_model.joblib
  models/seg_model_smooth_labels.keras   # shipped SEG RGB U-Net
  models/sam2.1_hiera_large.pt
  models/grain_unet_{channel_set}.keras
  models/grain_unet_{channel_set}.keras.norm.json
  models/object_model_{channel_set}.joblib

{image_folder}/
  labels/{stem}_polygons.geojson
  labels/{stem}_session.json
  labels/{stem}_labels.png              # figure export only
  labels/{stem}_grains.geojson         # GrainPlot (do not overwrite mineral polygons)
  labels/{stem}_grains_mask.png
  align/{stem}_align.json
  channels/{stem}_channels.npz
  channels/{stem}_channels.json
  channels/{stem}_channels_preview.png
  objects/{stem}_objects.csv
  objects/{stem}_objects_mask.png
  objects/{stem}_object_labels.json
  objects/{stem}_classes.csv
  objects/{stem}_classes_rgb.png
  objects/{stem}_stats.csv
  predictions/{stem}_pred.png
  predictions/{stem}_pred_rgb.png
  predictions/{stem}_confidence.npy
  predictions/{stem}_fractions.csv
  mineralmap/{stem}_grains.csv          # one row per grain: class, area, shape, frac_{class}
  mineralmap/{stem}_grains.geojson      # polygons, property "class" = legend name
  mineralmap/{stem}_class_rgb.png       # overlay preview
  mineralmap/{stem}_counts.csv          # per-class counts with uncertainty
  mineralmap/{stem}_unlisted.csv        # colors grouped into Unknown (audit)
  mineralmap/all_counts.csv             # all images stacked
```

No `cache/` folder. RGB cache = RAM only until kernel restart.

---

## Channel recipe (stage 2)

Adapts to the layers present; a missing layer is skipped, never faked. `channel_set_id` (e.g. `ppl-xplfit-bse`) names the U-Net so a model is only applied to a matching stack.

- primary photo (`ppl`, else `bright`): Y-normalized RGB
- XPL: ≥3 angles → `xpl_max`, `xpl_amp`, `xpl_phi_cos`, `xpl_phi_sin` + mean RGB; one photo → raw `xpl_r/g/b`
- `bse` gray when present
- extras off by default: texture maps, autoencoder field (`align/embed.py`)

---

## Outlines (stage 3)

- **Watershed** — no training; edge map = z-scored Sobel per channel.
- **N-channel U-Net** — copy SEG RGB weights; first conv gets primary RGB kernels; other channels start small-random. Train on GrainPlot 3-class masks. Subprocess; pyarrow blocked (TF 2.21 SIGSEGV).
- **GrainPlot** — click add / D delete / M merge. Same files as the PPL tutorial: `labels/{stem}_grains.geojson` + `_grains_mask.png`.

---

## Objects (stages 4–6)

One table row per GrainPlot grain, plus leftover blobs (`kind=leftover`). k-means → montage (click a crop; Shift-click names the group) → LightGBM on named rows → every object named. Stats: area % per class, porosity / cement / matrix, IGV, QFL when those names exist, Folk & Ward if `PX_PER_UM` is set.

---

## Label UI (RGB notebook)

**Important (Cursor / VS Code Jupyter):** a Matplotlib GUI *inside* the kernel often **kills the kernel**. `launch_app(session)` starts a **separate Python process** (QtAgg / PyQt5).

Save writes `{stem}_polygons.geojson` + `{stem}_session.json`. Train/reload rasterizes from polygons. Settings cell is the source of truth for `y_target`, DPI, method, feature toggles — never read/written from `session.json`.

---

## Features (RGB train / predict)

Per-pixel toggles. Training uses full working resolution. RGB train toggles stay **default-off** in constants. If every toggle is off, train/predict falls back to RGB. Lighting = Y-norm, not a channel. Skip whole-image GLCM heatmap and 2D-CWT.

Retrain: stratified k-fold CV (default 5), then fit on all labeled pixels. Predict uses `DEFAULT_SURE_PROBA = 0.8`.

---

## Main API (multimodal)

| Function | Role |
|----------|------|
| `Scene` / `launch_align` / `save_scene` / `load_scene` | Lineup |
| `add_axioscan_layers` / `axioscan_scale_folder` | Zeiss 02-scale / 25-scale folders |
| `build_channel_stack` / `save_channel_stack` / `load_channel_stack` | Physics channels |
| `run_channel_preview` | Inline channel grid |
| `run_watershed` / `init_grain_unet` / `train_grain_unet` / `predict_grain_unet` | Outlines |
| `launch_grain_qc` / `load_grain_qc` / `load_grain_mask` | GrainPlot (`interactive=False` writes grains without a window) |
| `ensure_seg_weights` | Download SEG U-Net (~26 MB). Pass `sam2=True` only if you want the ~860 MB SAM 2.1 weights |
| `compare_outlines` | Watershed \| U-Net \| QC |
| `build_object_table_from_stack` | One row per object |
| `cluster_objects` / `launch_montage_labeler` | Group and name |
| `train_object_classifier` / `predict_object_classes` / `save_classes` | LightGBM names |
| `compute_object_stats` | Area % / QFL / Folk & Ward |
| `launch_ppl_grain_qc` | Alias for the PPL tutorial |
| `embed_scene` | Optional extra channel source (not the spine) |

---

## Main API (mineral map)

| Function | Role |
|----------|------|
| `read_legend_swatches` / `legend_from_image` / `check_legend` | Swatches from image; build a legend dict (names given in order, or `swatch_1..N`); compare with a typed dict |
| `snap_to_legend` | Pixel colors → class codes (0 background, 1..K legend; far colors → Unknown) + remap audit table |
| `absorb_specks` / `absorb_class_specks` / `label_grains` | Specks under 30 px; Unknown/black under 50 px; grain = one connected blob of one legend colour (colour change = boundary; black is a gap, enclosed black filled, standalone black = Unknown grain); same-colour touching grains split at thin bridges (`bridge_px`), distance-map necks (`h_min`), and deep notches when both halves are compact (`notch_solidity`, `notch_halves`) |
| `grain_table` / `grains_to_geojson` / `class_rgb` | Class = largest known color; Unknown only if Unknown pixels ≥ `unknown_frac` (default 0.95); polygons; overlay (`display_rgb` optional overrides, default none) |
| `count_grains` / `format_counts` / `pivot_counts` | Counts with Poisson / Wilson / method range; class × image tables |
| `process_image` / `process_folder` | One image; many images in parallel (`load_saved`); forwards `bridge_px`, `h_min`, `notch_solidity`, `notch_halves`, `unknown_frac`, `display_rgb`, `output_dir` |
| `mineralmap_paths` | Output paths for the files above (`output_dir` optional) |
| `set_output_dir` / `get_output_dir` | Optional folder for labels / predictions / models / align / channels / objects / mineralmap. Default is beside each image (unchanged). |

---

## Align

Auto-align uses structure maps (`local_grad` + `local_std` + LBP), not color. Save/reload `{image_folder}/align/{stem}_align.json`. Missing or smaller-coverage BSE is **skipped** (mask), never filled. JSON v1 still loads; optional `angle_deg` for XPL series.

---

## Dependencies

Declared in the repo-root [`environment.yml`](../environment.yml) and [`pyproject.toml`](../pyproject.toml).

| Extra | Packages | Used by |
|-------|----------|---------|
| *(core)* | numpy, pillow, matplotlib, scikit-learn, scipy, geopandas, shapely, rasterio, affine, joblib, pandas, lightgbm, scikit-image, **PyQt5** | RGB + mineral-map |
| `[seg]` | TensorFlow, Keras, `segmenteverygrain` | **optional** multimodal U-Net / GrainPlot / montage (finer grain boundaries) |
| `[sam2]` | torch, torchvision, `sam2` | optional SAM shape refine — **do not** download the ~860 MB weights unless `USE_SAM2 = True` |
| `[xgb]` | xgboost | optional RGB classifier |
| `[all]` | seg + sam2 + xgb | only if you need SAM refine and xgboost |

From PyPI: `pip install pypetrograph`. Alternatively, see the [root README](../README.md#install) conda path. After `conda activate pypetrograph`, optional SEG is `pip install segmenteverygrain` (into this env), not SEG’s separate conda env.

---

## Design notes

- **Memory:** uint8 on disk for labels/preds; RGB working image RAM-only until kernel restart; float32 channels; tiled predict; mineral map: uint32-packed color snap, float32 distance transform, one image per worker, workers return only the counts table.
- **Layout (locked):** notebook-level `models/`; beside image `labels/` + `predictions/` + `align/` + `channels/` + `objects/`.
- **Do not re-derive live code from `old codes/`** except as a backup snapshot.

---

## License

**pyPetrograph** is licensed under the [GNU Affero General Public License 3.0 or later](LICENSE).
