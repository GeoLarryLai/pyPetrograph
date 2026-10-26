# Changelog

## Unreleased

### Changed
- rapidocr_onnxruntime (legend OCR; Python <3.13, it has no 3.13+ wheels) is now a base dependency (~220 MB with onnxruntime and OpenCV).
- Removed the `app` and `amics` extras; `all` is now `[xgb]`. The desktop app needs only `pip install PySide6-Essentials` (~350 MB); `pypetrograph-app` says so when it is missing.
- PyQt5 removed from the base dependencies (~200 MB smaller default install). Notebook pop-up windows now use whichever Qt binding is installed via Matplotlib (PySide6-Essentials recommended, the same one the app uses; PyQt5/PyQt6 still work). Without one, pop-ups say `pip install PySide6-Essentials`.

## 0.1.1 (2026-10-26)

### Changed
- Removed the `seg` and `sam2` extras (and from `all`). SegmentEveryGrain, which brings TensorFlow, PyTorch and SAM 2, is installed with its own instructions (https://github.com/zsylvester/segmenteverygrain); notebooks and the app point there and never install it.
- Desktop app: checks SegmentEveryGrain / SAM 2.1 in the SEG server's Python in the background. Without SegmentEveryGrain, *Run segmentation* (and saving edited grains) is disabled with an inline hint and link; *Load saved grains*, editing and *Use this grain map* on unedited grains still work. Without SAM 2.1, *SAM 2.1 refine* and *Add grain* are disabled. One-line note on SEG vs the optional SAM 2.1 refine in the Grain segmentation step.
- README / PyPI readme: one install command per line and a separate *SegmentEveryGrain (optional)* section.

## 0.1.0 (2026-10-26)

### Added
- **Desktop app** (`pip install "pyPetrograph[app]"`, then run `pypetrograph-app`): the notebook 4 AMICS workflow in one window, nine steps (Files → Legend → BSE & footer → Non-grain regions → Scale → Grain segmentation → Minerals → Grain size → Outputs). Live segmentation progress, background jobs, per-step saved / not-saved state, session files in a hidden `.pypetrograph/` folder; nothing is written to the export folder until you export. Exports are byte-identical to notebook 4.
- **Editable legend** in the app: rename OCR names, add entries (R/G/B, hex, colour palette, eyedropper), delete with undo, recolour, revert to OCR; edits are saved with the session and used by every export. "+ Add map colour not in legend" (`EXTRA_LEGEND`) stays separate in the Minerals step.
- Notebook 4 `pyPetrograph_4_AMICSworkflow.ipynb` and the `pyPetrograph.amics` package (legend OCR, BSE threshold and scale, SEG on BSE, mineral vote per grain, CSV / GeoJSON, composition, size curves, knolling).
- New `app` extra: PySide6-Essentials (Qt core/gui/widgets only, ~350 MB instead of ~1.2 GB for full PySide6; full PySide6 also works) and rapidocr_onnxruntime. New `amics` extra: rapidocr_onnxruntime (legend OCR, Python <3.13). Neither is in the base install; `pypetrograph-app` prints how to install them if missing. New `test` extra; headless pytest-qt suite for the app.

### Changed
- Faster segmentation and downstream steps in notebook 4 and the app (same outputs).
- Default minimum grain size `MIN_GRAIN_PX = 10` (was 50) in notebook 4 and the app.
- README and PyPI readme rewritten around overview, purpose and example figures.
- Package metadata: SPDX licence `AGPL-3.0-or-later`, full AGPL-3.0 text in `LICENSE`, `CITATION.cff`.

## 0.0.6
- PyPI release with a short PyPI readme and demo figures.
