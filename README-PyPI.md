# pyPetrograph

v0.1.1

[![PyPI version](https://img.shields.io/pypi/v/pypetrograph.svg)](https://pypi.org/project/pypetrograph/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23659316.svg)](https://doi.org/10.5281/zenodo.23659316)

pyPetrograph is a machine-learning toolkit for thin-section petrography.

**Warning:** This tool is under development. Use with caution. APIs and numbers can still change.

## What it does

**Hand-label features and map their area %.** Paint a few examples of each class you care about (pores, grains, cement, …) from color, edge sharpness, and microtexture. A LightGBM pixel classifier (random forest or XGBoost optional) maps the rest of the photo and reports area % of each class.

**Count grains on a painted mineral map.** Match pixels to a color legend, split grains that touch, and count how many grains sit in each class.

**Segment grains from a multi-image overlay.** Line up several image types of the same field (plane-polarized light, crossed polars, or others) and refine grain outlines with a U-Net, so boundaries are not drawn from one lighting alone.

**AMICS workflow (notebook 4 and the desktop app).** From a BSE image, a mineral map and its legend: read the legend (OCR, editable), outline every grain on the BSE, name each grain from the map, and export counts, per-mineral grain sizes and figures.

| Grain segmentation | Mineral composition | Knolling |
|---|---|---|
| ![segmentation](https://raw.githubusercontent.com/GeoLarryLai/pyPetrograph/main/docs/images/amics_10B_segmentation.jpg) | ![composition](https://raw.githubusercontent.com/GeoLarryLai/pyPetrograph/main/docs/images/amics_10B_composition.jpg) | ![knolling](https://raw.githubusercontent.com/GeoLarryLai/pyPetrograph/main/docs/images/amics_10B_knolling.jpg) |

The desktop app (`pypetrograph-app`) runs the same workflow step by step; nothing is written until you export.

![Legend step of the desktop app](https://raw.githubusercontent.com/GeoLarryLai/pyPetrograph/main/docs/images/app_legend.jpg)

## Install

```bash
pip install pyPetrograph
```

Or from the repository:

```bash
git clone https://github.com/GeoLarryLai/pyPetrograph.git
cd pyPetrograph
pip install .
```

Python 3.10–3.12 recommended (legend OCR has no 3.13+ wheels yet). On Linux, the desktop app's Qt needs some system libraries; see the [GitHub README](https://github.com/GeoLarryLai/pyPetrograph#linux-system-libraries). The default install includes legend OCR (rapidocr_onnxruntime with onnxruntime and OpenCV, **~220 MB**). Optional extras: `xgb`, `test`, `all`.

## SegmentEveryGrain (optional, for grain segmentation)

Grain segmentation (notebooks 3 and 4, and *Run segmentation* in the app) uses [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain). Install it into the same environment as pyPetrograph (it brings TensorFlow, PyTorch and SAM 2.1):

```bash
pip install segmenteverygrain
```

pyPetrograph downloads the SegmentEveryGrain U-Net and, only if *SAM 2.1 refine* is on, the SAM 2.1 checkpoint (~860 MB) on first use. See the [SegmentEveryGrain repository](https://github.com/zsylvester/segmenteverygrain#installation) for full details and troubleshooting.

Without it, everything else works: notebooks 1 and 2, the non-segmentation steps of notebooks 3 and 4, and every app step, including *Load saved grains* and editing them. In the app, *Run segmentation* is greyed out with a link here.

## Desktop app

To run the app, install [PySide6-Essentials](https://pypi.org/project/PySide6-Essentials/) (⚠️ **~350 MB**; not part of the base install; full PySide6 also works):

```bash
pip install PySide6-Essentials
pypetrograph-app
# or
python -m pyPetrograph.app
```

## Demos

Example notebooks, test images, figures, and optional extras are in the [GitHub repository](https://github.com/GeoLarryLai/pyPetrograph).

## Credits

pyPetrograph is developed by **Larry Syu-Heng Lai** (larrysyuhenglai@gmail.com). Part of the concept and base design (in Notebook 1) originated from early source code written by **Zoltan Sylvester** at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. This project was initiated as part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups and the [e-MAGE lab](https://www.beg.utexas.edu/research/labs/sem), and Larry intends to continue it after his departure from the Bureau.

## License

```text
Program: pyPetrograph
A toolkit for semi-automated, ML-assisted thin-section petrography:
porosity estimation, mineral counting, grain segmentation and micro-
texture analysis
Copyright (C) 2026  Larry Syu-Heng Lai
```

GNU Affero General Public License 3.0 or later (`LICENSE`).
