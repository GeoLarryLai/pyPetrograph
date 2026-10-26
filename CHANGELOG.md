# Changelog

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
