# Function documentation

Public functions and classes in `pyPetrograph`, grouped by module. Each entry is the name, the file, and the first line of the docstring. If that line is missing, the entry says "No docstring."

Names that start with `_` are omitted, except `_run_qt_picker_from_config` and `_run_label_gui_from_config`, which are exported from `pyPetrograph/__init__.py`.

## align

### `pyPetrograph.align.embed`

- `stack_named_slots` (`pyPetrograph/align/embed.py`): Working-grid RGB stack and per-slot coverage (legacy RGB-only helper).
- `stack_feature_slots` (`pyPetrograph/align/embed.py`): Per-slot feature stacks on the working grid (shared with RGB notebook).
- `embed_C` (`pyPetrograph/align/embed.py`): Small autoencoder on per-slot feature stacks (CPU MLP).
- `embed_B` (`pyPetrograph/align/embed.py`): Frozen pretrained net per available layer; skip missing pixels (no fake BSE).
- `embedding_to_rgb` (`pyPetrograph/align/embed.py`): False-color view of an embedding field (first 3 PCA axes). Preview only.
- `embed_scene` (`pyPetrograph/align/embed.py`): Run embedding. ``method`` is ``C`` (default autoencoder) or ``B`` (frozen net).

### `pyPetrograph.align.io`

- `align_dir_for` (`pyPetrograph/align/io.py`): No docstring.
- `default_json_path` (`pyPetrograph/align/io.py`): No docstring.
- `scene_to_dict` (`pyPetrograph/align/io.py`): No docstring.
- `save_scene` (`pyPetrograph/align/io.py`): No docstring.
- `load_scene` (`pyPetrograph/align/io.py`): No docstring.

### `pyPetrograph.align.register`

- `auto_register` (`pyPetrograph/align/register.py`): Always-run auto lineup on structure maps.
- `refine_register` (`pyPetrograph/align/register.py`): Re-run auto around the current transform of ``slot``.
- `fit_click_pairs` (`pyPetrograph/align/register.py`): Click pairs are loose hints: each dict has ``base`` [x, y] and ``moving`` [x, y].

### `pyPetrograph.align.scene`

- `Affine2D` (`pyPetrograph/align/scene.py`): Map moving pixel (x, y) = (col, row) → working/base pixel.
- `Affine2D.to_skimage` (`pyPetrograph/align/scene.py`): No docstring.
- `Affine2D.inverse` (`pyPetrograph/align/scene.py`): No docstring.
- `Affine2D.as_dict` (`pyPetrograph/align/scene.py`): No docstring.
- `Affine2D.from_dict` (`pyPetrograph/align/scene.py`): No docstring.
- `slot_kind` (`pyPetrograph/align/scene.py`): Layer kind from its slot name: ``"xpl_015"`` → ``"xpl"``, ``"ppl"`` → ``"ppl"``.
- `xpl_slot` (`pyPetrograph/align/scene.py`): Slot name for one cross-polarized angle, e.g. ``xpl_015``.
- `Layer` (`pyPetrograph/align/scene.py`): No docstring.
- `Layer.kind` (`pyPetrograph/align/scene.py`): No docstring.
- `Layer.ensure_rgb` (`pyPetrograph/align/scene.py`): No docstring.
- `Scene` (`pyPetrograph/align/scene.py`): One working grid (usually the train-on layer) plus other measurements.
- `Scene.add_layer` (`pyPetrograph/align/scene.py`): No docstring.
- `Scene.layers_of_kind` (`pyPetrograph/align/scene.py`): Layers whose slot kind matches (e.g. all ``xpl_*``), sorted by angle then slot.
- `Scene.primary_slot` (`pyPetrograph/align/scene.py`): Photo used for color: ``ppl`` if present, else ``bright``, else the working slot.
- `Scene.working` (`pyPetrograph/align/scene.py`): No docstring.
- `Scene.working_hw` (`pyPetrograph/align/scene.py`): No docstring.
- `Scene.warp_layer` (`pyPetrograph/align/scene.py`): Warp ``slot`` onto the working grid.
- `Scene.composite` (`pyPetrograph/align/scene.py`): Working RGB with ``overlay_slot`` faded on top where that layer is valid.
- `Scene.coverage_mask` (`pyPetrograph/align/scene.py`): No docstring.

### `pyPetrograph.align.seg_qc_worker`

- `main` (`pyPetrograph/align/seg_qc_worker.py`): No docstring.

### `pyPetrograph.align.seg_worker`

- `main` (`pyPetrograph/align/seg_worker.py`): No docstring.

### `pyPetrograph.align.structure`

- `structure_map` (`pyPetrograph/align/structure.py`): HxW float32 map for matching: local_grad + local_std + LBP.
- `structure_map_preview` (`pyPetrograph/align/structure.py`): Downsampled structure map for fast auto-align / UI.

### `pyPetrograph.align.ui`

- `AlignWindow` (`pyPetrograph/align/ui.py`): Overlay working image + one moving layer; auto first, then drag/scale/re-auto.
- `AlignWindow.do_auto` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.do_reauto` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.do_save` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.do_load` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.do_cycle` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.do_toggle_click` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.do_fit` (`pyPetrograph/align/ui.py`): No docstring.
- `AlignWindow.run` (`pyPetrograph/align/ui.py`): No docstring.
- `launch_align` (`pyPetrograph/align/ui.py`): Open lineup UI in a subprocess (kernel stays Qt-free). Always call auto in the window.

## common

### `pyPetrograph.common.constants`

- `clip_figure_dpi` (`pyPetrograph/common/constants.py`): Clamp a figure / preview dpi into [72, MAX_FIGURE_DPI] (default 150).

### `pyPetrograph.common.image_io`

- `load_image` (`pyPetrograph/common/image_io.py`): Load a common image format and return RGB uint8 (H, W, 3).
- `save_rgb_png` (`pyPetrograph/common/image_io.py`): Save an RGB or single-channel array as PNG.
- `save_rgb_cache` (`pyPetrograph/common/image_io.py`): Store working RGB in RAM only (no disk). Returns a synthetic path for callers.
- `load_rgb_cache` (`pyPetrograph/common/image_io.py`): Return RAM-cached RGB if present and load settings match.
- `ensure_rgb_cache` (`pyPetrograph/common/image_io.py`): Return working RGB from RAM cache or load_image (store in RAM; no disk).
- `clear_rgb_ram_cache` (`pyPetrograph/common/image_io.py`): Drop all RAM RGB caches (normally cleared on kernel restart).

### `pyPetrograph.common.notebook_runners`

- `print_label_summary` (`pyPetrograph/common/notebook_runners.py`): Print label stats; one gray+polygon figure per queued image (figure_dpi).
- `feature_display_title` (`pyPetrograph/common/notebook_runners.py`): No docstring.
- `run_feature_preview` (`pyPetrograph/common/notebook_runners.py`): Inline preview of toggled feature maps for every queued image (3-column grid).
- `run_scene_feature_preview` (`pyPetrograph/common/notebook_runners.py`): Same 3-column layout as ``run_feature_preview``, for a lined-up ``Scene``.
- `run_train_cell` (`pyPetrograph/common/notebook_runners.py`): Notebook Train cell helper.
- `print_train_metrics` (`pyPetrograph/common/notebook_runners.py`): Print stratified CV accuracy (%) + classification report (Accuracy notebook cell).
- `run_predict_cell` (`pyPetrograph/common/notebook_runners.py`): Notebook Predict cell: predict queue; pred + confidence figures at figure_dpi.

### `pyPetrograph.common.paths`

- `set_output_dir` (`pyPetrograph/common/paths.py`): Optional folder for labels / predictions / models / align / channels /
- `get_output_dir` (`pyPetrograph/common/paths.py`): Current ``set_output_dir`` folder, or ``None`` (default beside-image layout).
- `project_root` (`pyPetrograph/common/paths.py`): Folder that contains the notebook + ``models/`` (parent of ``pyPetrograph/``).
- `project_models_dir` (`pyPetrograph/common/paths.py`): ``models/`` under ``get_output_dir()`` if set, else the project root.
- `StemPaths` (`pyPetrograph/common/paths.py`): Companion paths under labels/ cache/ models/ predictions/ channels/ objects/
- `StemPaths.labels_dir` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.cache_dir` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.labels` (`pyPetrograph/common/paths.py`): Label-summary figure export (`print_label_summary`).
- `StemPaths.polygons` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.grains` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.grains_mask` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.session` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.model` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.rgb_cache` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.rgb_cache_meta` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.pred` (`pyPetrograph/common/paths.py`): RGB class map (HSV colors) — human-viewable.
- `StemPaths.pred_rgb` (`pyPetrograph/common/paths.py`): Original image with class-color overlay.
- `StemPaths.fractions_csv` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.confidence` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.channels_dir` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.channels` (`pyPetrograph/common/paths.py`): Channel stack (``stack`` + ``valid``) as npz.
- `StemPaths.channels_meta` (`pyPetrograph/common/paths.py`): Channel names / kinds / channel_set_id as json.
- `StemPaths.objects_dir` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.objects_csv` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.objects_mask` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.object_labels` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.classes_csv` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.classes_rgb` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.stats_csv` (`pyPetrograph/common/paths.py`): No docstring.
- `StemPaths.ensure_dirs` (`pyPetrograph/common/paths.py`): Create only labels/ and predictions/ beside the image (not cache/ or models/).
- `stem_paths` (`pyPetrograph/common/paths.py`): Companion paths under subfolders of ``image_path``'s parent, or under
- `relpath_display` (`pyPetrograph/common/paths.py`): Short path relative to ``base`` (default: cwd / notebook folder).
- `cleanup_legacy_outputs` (`pyPetrograph/common/paths.py`): Delete obsolete beside-image companions in ``image_dir`` (not under subfolders).
- `delete_legacy_image_small` (`pyPetrograph/common/paths.py`): No-op kept for call sites; legacy flat companions cleaned by cleanup_legacy_outputs.
- `check_and_install_packages` (`pyPetrograph/common/paths.py`): Check packages needed by this toolbox; optionally pip-install missing ones

### `pyPetrograph.common.session`

- `Session` (`pyPetrograph/common/session.py`): Project state for a queue of images: labels, polygons, models, params.
- `Session.remember_settings_classes` (`pyPetrograph/common/session.py`): Snapshot current class_names as Settings defaults (kept for launch config).
- `Session.on_change` (`pyPetrograph/common/session.py`): No docstring.
- `Session.set_status` (`pyPetrograph/common/session.py`): No docstring.
- `Session.paths` (`pyPetrograph/common/session.py`): Companion paths for the current image (beside that image's folder).
- `Session.work_directory` (`pyPetrograph/common/session.py`): Folder for shared prints when the queue has a single parent.
- `Session.queue_parent_dirs` (`pyPetrograph/common/session.py`): Unique parent folders of queued images (sorted).
- `Session.ensure_output_dirs` (`pyPetrograph/common/session.py`): Create labels/ + predictions/; models/ via ``project_models_dir``.
- `Session.paths_for` (`pyPetrograph/common/session.py`): Labels/predictions beside the image; models under notebook ``models/``.
- `Session.model_path_for_stem` (`pyPetrograph/common/session.py`): Per-image model under notebook-level ``models/{stem}_model.joblib``.
- `Session.universal_model_path` (`pyPetrograph/common/session.py`): ``models/universal_model.joblib`` next to the notebook.
- `Session.save_universal_model` (`pyPetrograph/common/session.py`): Write universal model once under notebook-level ``models/``.
- `Session.cleanup_unused_image_subfolders` (`pyPetrograph/common/session.py`): Remove leftover image-folder ``cache/`` and ``models/`` (keep labels/).
- `Session.push_undo` (`pyPetrograph/common/session.py`): No docstring.
- `Session.undo` (`pyPetrograph/common/session.py`): No docstring.
- `Session.set_image_queue` (`pyPetrograph/common/session.py`): Set ordered image queue from multi-select; optionally open the first.
- `Session.cache_queue_images` (`pyPetrograph/common/session.py`): Warm RAM RGB cache for every queued image (no disk writes).
- `Session.switch_image` (`pyPetrograph/common/session.py`): Open another queue image. Disk write only if autosave=True (default off).
- `Session.view_target_mp` (`pyPetrograph/common/session.py`): MP for label overview / feature preview (screen-based if unset).
- `Session.open_image` (`pyPetrograph/common/session.py`): Load image (prefer rgb_cache), init blank labels, optionally auto-load companions.
- `Session.save_all` (`pyPetrograph/common/session.py`): Save polygons + session under labels/ (no class-id PNG).
- `Session.reload_current_from_disk` (`pyPetrograph/common/session.py`): Re-load labels/polygons/session for the current image from disk.
- `Session.load_labels_only` (`pyPetrograph/common/session.py`): Reload polygons from disk into memory (pushes undo; no write).
- `Session.rebuild_labels_from_polygons` (`pyPetrograph/common/session.py`): No docstring.
- `Session.add_polygon` (`pyPetrograph/common/session.py`): Add one training polygon for the current (or given) class.
- `Session.add_wand_at` (`pyPetrograph/common/session.py`): Magic-wand click: grow similar color → polygon for current class.
- `Session.restart_labels` (`pyPetrograph/common/session.py`): Clear current class labels in memory only (no disk write).
- `Session.set_class_name` (`pyPetrograph/common/session.py`): No docstring.
- `Session.add_class` (`pyPetrograph/common/session.py`): No docstring.
- `Session.remove_class` (`pyPetrograph/common/session.py`): No docstring.
- `Session.has_training_labels` (`pyPetrograph/common/session.py`): No docstring.
- `Session.label_stats` (`pyPetrograph/common/session.py`): No docstring.
- `Session.save_session_file` (`pyPetrograph/common/session.py`): Persist only class id→name map (Settings knobs stay in the notebook cell).
- `Session.load_session_file` (`pyPetrograph/common/session.py`): Restore class names only; never overwrite Settings-cell parameters.
- `Session.train` (`pyPetrograph/common/session.py`): No docstring.
- `Session.predict` (`pyPetrograph/common/session.py`): No docstring.
- `Session.fractions` (`pyPetrograph/common/session.py`): No docstring.
- `Session.load_model_from_path` (`pyPetrograph/common/session.py`): No docstring.
- `Session.train_queue` (`pyPetrograph/common/session.py`): Retrain models for the queue.
- `Session.predict_queue` (`pyPetrograph/common/session.py`): Predict every queued image. Writes under that image's ``predictions/``.

### `pyPetrograph.common.ui`

- `UndoState` (`pyPetrograph/common/ui.py`): No docstring.
- `UndoStack` (`pyPetrograph/common/ui.py`): General undo stack for labeling / class-list steps.
- `UndoStack.push` (`pyPetrograph/common/ui.py`): No docstring.
- `UndoStack.pop` (`pyPetrograph/common/ui.py`): No docstring.
- `UndoStack.clear` (`pyPetrograph/common/ui.py`): No docstring.
- `_run_qt_picker_from_config` (`pyPetrograph/common/ui.py`): Child-process entry: Qt file / yes-no dialogs (never run inside Jupyter kernel).
- `select_images` (`pyPetrograph/common/ui.py`): Desktop file-picker popup to choose images **before** ``launch_app``.
- `LabelWindow` (`pyPetrograph/common/ui.py`): DMG-07-style one-figure label UI: left control panel + right image axes.
- `LabelWindow.refresh_overlays` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_prev` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_next` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_save` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_load` (`pyPetrograph/common/ui.py`): Reload saved labels from disk into the scene (undoable).
- `LabelWindow.do_undo` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_restart` (`pyPetrograph/common/ui.py`): Clear current class in the scene only (memory; no disk write).
- `LabelWindow.do_cycle_class` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_toggle_mode` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_fit` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_add_class` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.do_rename_class` (`pyPetrograph/common/ui.py`): No docstring.
- `LabelWindow.run` (`pyPetrograph/common/ui.py`): Block until the window is closed (DMG-07 QuadPicker style).
- `build_ui` (`pyPetrograph/common/ui.py`): Create LabelWindow; returns (session, win). DMG-07 style (no plt.ion).
- `launch_app` (`pyPetrograph/common/ui.py`): Open the label window in a **separate Python process** (DMG-07 Qt popup style).
- `_run_label_gui_from_config` (`pyPetrograph/common/ui.py`): Child-process entry: QtAgg popup like DMG-07, block until closed.
- `launch_app_inplace` (`pyPetrograph/common/ui.py`): Open the label UI **inside** this process (for plain Python / IPython, not Cursor Jupyter).

## fuse

### `pyPetrograph.fuse.extinction`

- `design_matrix` (`pyPetrograph/fuse/extinction.py`): Rows [1, cos4θ, sin4θ] for each angle (float64, n×3).
- `check_angles` (`pyPetrograph/fuse/extinction.py`): Raise a plain-language ValueError if the angles cannot support the 3-term fit.
- `fit_extinction` (`pyPetrograph/fuse/extinction.py`): Per-pixel sine fit over a stack of gray XPL photos.
- `phi_deg_from_cos_sin` (`pyPetrograph/fuse/extinction.py`): Recover the max-brightness angle in [0, 90) from the two circle channels.

### `pyPetrograph.fuse.preview`

- `run_channel_preview` (`pyPetrograph/fuse/preview.py`): Show the channel stack inline (``%matplotlib inline``): composite row + one panel per channel.

### `pyPetrograph.fuse.stack`

- `ChannelStack` (`pyPetrograph/fuse/stack.py`): Working-grid channel stack with names, value kinds and a coverage mask.
- `ChannelStack.n_channels` (`pyPetrograph/fuse/stack.py`): No docstring.
- `ChannelStack.shape_hw` (`pyPetrograph/fuse/stack.py`): No docstring.
- `ChannelStack.index` (`pyPetrograph/fuse/stack.py`): No docstring.
- `ChannelStack.channel` (`pyPetrograph/fuse/stack.py`): No docstring.
- `ChannelStack.primary_rgb_indices` (`pyPetrograph/fuse/stack.py`): The three primary-photo RGB channels (SEG RGB weights are copied onto these).
- `ChannelStack.summary` (`pyPetrograph/fuse/stack.py`): No docstring.
- `channel_set_id` (`pyPetrograph/fuse/stack.py`): ``ppl-xplfit-bse`` style id from the ordered group tags.
- `build_channel_stack` (`pyPetrograph/fuse/stack.py`): Build the channel stack on the scene's working grid.
- `channels_paths` (`pyPetrograph/fuse/stack.py`): ``channels/{stem}_channels.npz`` and ``channels/{stem}_channels.json`` beside the image.
- `save_channel_stack` (`pyPetrograph/fuse/stack.py`): No docstring.
- `load_channel_stack` (`pyPetrograph/fuse/stack.py`): Reload a saved stack, or ``None`` if the files are missing.

### `pyPetrograph.fuse.tiles`

- `iter_tiles` (`pyPetrograph/fuse/tiles.py`): Yield ``(r0, r1, c0, c1)`` windows covering an ``h×w`` grid with overlap.
- `apply_tiled` (`pyPetrograph/fuse/tiles.py`): Run ``fn(tile_HxWxC) -> HxWxK`` over overlapping tiles and blend the overlaps.
- `axioscan_layers` (`pyPetrograph/fuse/tiles.py`): Map Zeiss Axioscan exports in one ``NN scale`` folder to slots.
- `axioscan_scale_folder` (`pyPetrograph/fuse/tiles.py`): Find the ``02 scale`` / ``25scale`` / ``50 scale`` folder for ``scale`` (0.02, 0.25, 0.5).
- `add_axioscan_layers` (`pyPetrograph/fuse/tiles.py`): Add every matching Axioscan export in ``folder`` to ``scene``.

## image_processing

### `pyPetrograph.image_processing.brightness`

- `relative_luminance` (`pyPetrograph/image_processing/brightness.py`): Relative luminance Y from RGB (0–255 or float).
- `normalize_brightness` (`pyPetrograph/image_processing/brightness.py`): Scale RGB so mean luminance matches ``y_target``, keeping color ratios.
- `to_display_uint8` (`pyPetrograph/image_processing/brightness.py`): Clip float/int image to uint8 for display or PNG preview.

### `pyPetrograph.image_processing.features`

- `default_feature_toggles` (`pyPetrograph/image_processing/features.py`): No docstring.
- `resolve_feature_toggles` (`pyPetrograph/image_processing/features.py`): Known features default to False unless explicitly set True.
- `downsample_to_approx_mp` (`pyPetrograph/image_processing/features.py`): Shrink image so H×W ≈ target_mp megapixels (approximate).
- `get_screen_size_px` (`pyPetrograph/image_processing/features.py`): Primary screen width × height in pixels (fallback 1920×1080).
- `screen_view_target_mp` (`pyPetrograph/image_processing/features.py`): Megapixels to roughly fill the primary screen (for label overview / feature preview).
- `resolve_view_target_mp` (`pyPetrograph/image_processing/features.py`): Use explicit MP if set; otherwise size from the screen.
- `max_side_for_target_mp` (`pyPetrograph/image_processing/features.py`): Square-side length whose area ≈ target_mp (for LOD overview).
- `compute_local_std` (`pyPetrograph/image_processing/features.py`): Local standard deviation of a 2D float image (odd window).
- `compute_sobel_mag` (`pyPetrograph/image_processing/features.py`): Sobel gradient magnitude of a 2D image.
- `compute_lbp_map` (`pyPetrograph/image_processing/features.py`): Uniform LBP codes on gray (float32).
- `compute_gabor_maps` (`pyPetrograph/image_processing/features.py`): Gabor filter magnitudes — stripes at given angles (cleavage / twins).
- `compute_hessian_maps` (`pyPetrograph/image_processing/features.py`): Hessian eigenvalue maps — “ridge vs blob?” (crack vs grain).
- `compute_local_entropy` (`pyPetrograph/image_processing/features.py`): Local entropy of Y in an odd window (how mixed the grays are).
- `compute_fft_power_spectrum` (`pyPetrograph/image_processing/features.py`): Centered log power spectrum of a 2D image (whole-field fabric fingerprint).
- `glcm_feature_maps` (`pyPetrograph/image_processing/features.py`): Polygon-level GLCM props broadcast onto pixels inside each polygon.
- `build_feature_stack` (`pyPetrograph/image_processing/features.py`): Build HxWxC float32 feature stack and ordered feature names.
- `extract_training_table` (`pyPetrograph/image_processing/features.py`): Build float32 feature matrix X (N, C) and y (N,) from labeled pixels.

### `pyPetrograph.image_processing.fractions`

- `compute_fractions` (`pyPetrograph/image_processing/fractions.py`): Area fractions per class, excluding label 0 (background / white margins).
- `compute_fractions_uncertain` (`pyPetrograph/image_processing/fractions.py`): Hard area fractions plus asymmetric bounds from class probabilities.

### `pyPetrograph.image_processing.viz`

- `class_color_rgba` (`pyPetrograph/image_processing/viz.py`): Bright, well-separated color for a class id (HSV wheel + golden-angle step).
- `colorize_labels_hsv` (`pyPetrograph/image_processing/viz.py`): Map label ids → RGB uint8 using ``class_color_rgba`` (0 → black).
- `overlay_class_map` (`pyPetrograph/image_processing/viz.py`): Blend HSV class colors onto the original RGB where labels > 0.
- `add_class_legend` (`pyPetrograph/image_processing/viz.py`): Legend below / outside the axes: up to ``max_per_row`` entries per row.
- `save_labels_png` (`pyPetrograph/image_processing/viz.py`): Save a label raster as uint8 PNG (compatible format).
- `load_labels_png` (`pyPetrograph/image_processing/viz.py`): Load a label PNG to uint8 2D array (handles float PNG read quirks).
- `try_load_legacy_label_ids` (`pyPetrograph/image_processing/viz.py`): Load old uint8 class-id ``*_labels.png`` only if mode is ``L``.
- `colorize_prediction` (`pyPetrograph/image_processing/viz.py`): Map label ids to RGB using the training mean-color palette.

## labeling_ml

### `pyPetrograph.labeling_ml.model_io`

- `save_model_bundle` (`pyPetrograph/labeling_ml/model_io.py`): Save trained model + metadata with joblib.
- `load_model_bundle` (`pyPetrograph/labeling_ml/model_io.py`): Load a joblib model bundle.

### `pyPetrograph.labeling_ml.polygons`

- `remove_duplicate_polys` (`pyPetrograph/labeling_ml/polygons.py`): Deduplicate shapely polygons by WKB hex.
- `rasterize_polygons` (`pyPetrograph/labeling_ml/polygons.py`): Rasterize polygons into a label map (uint8). Later polygons overwrite earlier.
- `round_polygon_coords` (`pyPetrograph/labeling_ml/polygons.py`): Return a copy of ``poly`` with exterior coords rounded for compact GeoJSON.
- `save_polygons_geojson` (`pyPetrograph/labeling_ml/polygons.py`): Save polygons as GeoJSON with a ``class`` column (compact rounded coords).
- `magic_wand_to_polygon` (`pyPetrograph/labeling_ml/polygons.py`): Grow a similar-color region from (row, col) and return a simplified polygon.
- `load_polygons_geojson` (`pyPetrograph/labeling_ml/polygons.py`): Load GeoJSON → (polygons, class_ids).

### `pyPetrograph.labeling_ml.predict`

- `predict_image` (`pyPetrograph/labeling_ml/predict.py`): Predict labels for every pixel using the bundle's feature stack.
- `predict_image_proba` (`pyPetrograph/labeling_ml/predict.py`): Predict labels + class probabilities for every pixel.
- `smooth_prediction` (`pyPetrograph/labeling_ml/predict.py`): Median-filter a label map.
- `confidence_for_labels` (`pyPetrograph/labeling_ml/predict.py`): Per-pixel probability of the assigned label (0 where pred==0).

### `pyPetrograph.labeling_ml.train`

- `format_classification_report_pct` (`pyPetrograph/labeling_ml/train.py`): Classification report with precision / recall / f1 / accuracy as percent (3 dp).
- `train_classifier` (`pyPetrograph/labeling_ml/train.py`): Train RF / LightGBM / XGBoost on labeled pixels with selected features.
- `train_classifier_from_xy` (`pyPetrograph/labeling_ml/train.py`): Train from pre-built float32 X (N,C) and int y (N,). CV then fit-all (see train_classifier).

## mineral_map

### `pyPetrograph.mineral_map.counts`

- `wilson_interval` (`pyPetrograph/mineral_map/counts.py`): Wilson score interval for ``k`` successes in ``n`` trials (fractions).
- `count_grains` (`pyPetrograph/mineral_map/counts.py`): Count grains per class for one image.
- `pivot_counts` (`pyPetrograph/mineral_map/counts.py`): Class × image table of one column (e.g. ``n`` or ``share_pct``).
- `format_counts` (`pyPetrograph/mineral_map/counts.py`): Class × image table of ``n ± poisson_sd [n_lo–n_hi]`` strings.

### `pyPetrograph.mineral_map.grains`

- `absorb_specks` (`pyPetrograph/mineral_map/grains.py`): Replace same-color blobs smaller than ``speck_px`` with the class of the
- `absorb_class_specks` (`pyPetrograph/mineral_map/grains.py`): Absorb blobs of one class (e.g. Unknown/black) smaller than ``speck_px``.
- `label_grains` (`pyPetrograph/mineral_map/grains.py`): Grains = connected same-colour pixels (a colour change is a boundary;
- `grain_table` (`pyPetrograph/mineral_map/grains.py`): One row per grain: class, area, shape, and ``frac_{name}`` columns.
- `grains_to_geojson` (`pyPetrograph/mineral_map/grains.py`): Write one polygon per grain (pixel coords, x = col, y = row) with properties
- `class_rgb` (`pyPetrograph/mineral_map/grains.py`): Paint each grain with its class color (white background, dark grain edges).

### `pyPetrograph.mineral_map.legend`

- `class_names` (`pyPetrograph/mineral_map/legend.py`): Names indexed by class code: ``[background, *legend]``.
- `unknown_code` (`pyPetrograph/mineral_map/legend.py`): Class code of Unknown (name match, else the black swatch).
- `unlisted_code` (`pyPetrograph/mineral_map/legend.py`): Deprecated alias: far colors now use ``unknown_code``.
- `read_rgba` (`pyPetrograph/mineral_map/legend.py`): Read an image as ``(rgb uint8 HxWx3, alpha uint8 HxW or None)``.
- `read_legend_swatches` (`pyPetrograph/mineral_map/legend.py`): Swatch colors from a legend image, top to bottom.
- `legend_from_image` (`pyPetrograph/mineral_map/legend.py`): Build a legend dict from a swatch image (colors top to bottom).
- `check_legend` (`pyPetrograph/mineral_map/legend.py`): Compare typed ``legend`` (name → rgb) to swatches read from the legend image.
- `snap_to_legend` (`pyPetrograph/mineral_map/legend.py`): Map every pixel to a legend class by nearest RGB color.
- `format_unlisted` (`pyPetrograph/mineral_map/legend.py`): Rename the Unknown-color audit table to the public column names.

### `pyPetrograph.mineral_map.run`

- `mineralmap_paths` (`pyPetrograph/mineral_map/run.py`): Output paths for one image (folder ``mineralmap/`` beside the image, or under ``output_dir``).
- `process_image` (`pyPetrograph/mineral_map/run.py`): Full pass for one image: snap → specks → grains → table → files.
- `process_folder` (`pyPetrograph/mineral_map/run.py`): Run ``process_image`` on many images in parallel (one image per worker).

## objects

### `pyPetrograph.objects.classify`

- `object_model_path` (`pyPetrograph/objects/classify.py`): Notebook-level ``models/object_model_{channel_set_id}.joblib``.
- `train_object_classifier` (`pyPetrograph/objects/classify.py`): Train on rows with ``class_id > 0`` (from ``apply_object_labels``).
- `predict_object_classes` (`pyPetrograph/objects/classify.py`): Predict ``class_id`` / ``proba`` / ``class_name`` for every finite row.
- `save_classes` (`pyPetrograph/objects/classify.py`): Write ``objects/{stem}_classes.csv`` and ``objects/{stem}_classes_rgb.png``.
- `load_classes` (`pyPetrograph/objects/classify.py`): Reload ``objects/{stem}_classes.csv``, or None.

### `pyPetrograph.objects.cluster`

- `numeric_feature_matrix` (`pyPetrograph/objects/cluster.py`): Numeric columns for clustering / LightGBM.
- `cluster_objects` (`pyPetrograph/objects/cluster.py`): Standardise → optional PCA → k-means. Adds a ``cluster_id`` column (copy).
- `load_object_labels` (`pyPetrograph/objects/cluster.py`): Reload ``objects/{stem}_object_labels.json``, or None.
- `apply_object_labels` (`pyPetrograph/objects/cluster.py`): Add ``class_id`` / ``class_name`` from a montage JSON blob.
- `launch_montage_labeler` (`pyPetrograph/objects/cluster.py`): Open SEG ClusterMontageLabeler (subprocess). Close the window to save.

### `pyPetrograph.objects.montage_worker`

- `main` (`pyPetrograph/objects/montage_worker.py`): No docstring.

### `pyPetrograph.objects.outlines`

- `grain_unet_path` (`pyPetrograph/objects/outlines.py`): Notebook-level ``models/grain_unet_{channel_set}.keras``.
- `grain_unet_trained_marker` (`pyPetrograph/objects/outlines.py`): Sidecar written after a finished train (init-only ``.keras`` is not trained).
- `grain_unet_is_trained` (`pyPetrograph/objects/outlines.py`): True if weights exist and a train pass wrote the sidecar.
- `init_grain_unet` (`pyPetrograph/objects/outlines.py`): Copy SEG RGB U-Net weights into an N-channel first layer.
- `train_grain_unet` (`pyPetrograph/objects/outlines.py`): Train the N-channel U-Net from the QC mask beside ``image_path``.
- `predict_grain_unet` (`pyPetrograph/objects/outlines.py`): Predict 3-class probabilities + connected-component labels on ``cs``.
- `run_outlines` (`pyPetrograph/objects/outlines.py`): One-shot outline pass.
- `compare_outlines` (`pyPetrograph/objects/outlines.py`): Overlay grain outlines on PPL. Only panels with labels are drawn.

### `pyPetrograph.objects.qc`

- `ensure_seg_weights` (`pyPetrograph/objects/qc.py`): Download SegmentEveryGrain U-Net (26 MB) into ``models/`` if missing.
- `grain_qc_paths` (`pyPetrograph/objects/qc.py`): ``labels/{stem}_grains.geojson`` and ``labels/{stem}_grains_mask.png``.
- `overlay_labels` (`pyPetrograph/objects/qc.py`): Draw darker grain outlines (slightly thick) on an RGB copy.
- `seg_setup_hint` (`pyPetrograph/objects/qc.py`): No docstring.
- `load_grain_mask` (`pyPetrograph/objects/qc.py`): Load saved grain QC mask, or None.
- `load_grain_qc` (`pyPetrograph/objects/qc.py`): Load grains already saved by GrainPlot (this notebook or SegmentEveryGrain_PPL).
- `launch_ppl_grain_qc` (`pyPetrograph/objects/qc.py`): Alias for ``launch_grain_qc`` (PPL teacher / ``SegmentEveryGrain_PPL``).
- `load_ppl_grain_qc` (`pyPetrograph/objects/qc.py`): Alias for ``load_grain_qc``.
- `launch_grain_qc` (`pyPetrograph/objects/qc.py`): Open GrainPlot on ``image_path`` (subprocess; kernel never imports TensorFlow).

### `pyPetrograph.objects.qc_worker`

- `main` (`pyPetrograph/objects/qc_worker.py`): No docstring.

### `pyPetrograph.objects.stats`

- `folk_ward_percentiles` (`pyPetrograph/objects/stats.py`): Area-weighted φ percentiles. ``φp`` is where cumulative area reaches p% of grain area.
- `compute_object_stats` (`pyPetrograph/objects/stats.py`): Per-class area rows, then summary rows (porosity, cement, matrix, IGV, QFL, Folk & Ward).
- `save_object_stats` (`pyPetrograph/objects/stats.py`): Write ``objects/{stem}_stats.csv``.
- `load_object_stats` (`pyPetrograph/objects/stats.py`): Reload ``objects/{stem}_stats.csv``, or ``None`` if missing.

### `pyPetrograph.objects.table`

- `build_object_table` (`pyPetrograph/objects/table.py`): Instance grains + leftover blobs → table and a combined id map.
- `build_object_table_from_stack` (`pyPetrograph/objects/table.py`): ``build_object_table`` using a ``ChannelStack``.
- `save_object_table` (`pyPetrograph/objects/table.py`): Write ``objects/{stem}_objects.csv`` and ``objects/{stem}_objects_mask.png``.
- `load_object_table` (`pyPetrograph/objects/table.py`): Reload CSV + mask, or ``None`` if either file is missing.

### `pyPetrograph.objects.unet_worker`

- `main` (`pyPetrograph/objects/unet_worker.py`): No docstring.

### `pyPetrograph.objects.watershed`

- `gradient_magnitude` (`pyPetrograph/objects/watershed.py`): HxW edge strength: per-channel Sobel magnitude, z-scored, then weighted sum.
- `watershed_labels` (`pyPetrograph/objects/watershed.py`): Instance labels from an edge-elevation map.
- `run_watershed` (`pyPetrograph/objects/watershed.py`): Watershed on the channel stack.
- `instance_to_three_class` (`pyPetrograph/objects/watershed.py`): Instance mask → HxW uint8 {0 background, 1 grain, 2 outer boundary} (SEG layout).
- `three_class_to_probs` (`pyPetrograph/objects/watershed.py`): HxW {0,1,2} → HxWx3 float32 one-hot (bg, grain, boundary) for ``label_grains``.

## Files with no public function or class

These files define no public function or class:

- `pyPetrograph/__init__.py`
- `pyPetrograph/align/__init__.py`
- `pyPetrograph/align/grains.py`
- `pyPetrograph/common/__init__.py`
- `pyPetrograph/fuse/__init__.py`
- `pyPetrograph/image_processing/__init__.py`
- `pyPetrograph/labeling_ml/__init__.py`
- `pyPetrograph/mineral_map/__init__.py`
- `pyPetrograph/objects/__init__.py`
- `pyPetrograph/objects/tf_env.py`
