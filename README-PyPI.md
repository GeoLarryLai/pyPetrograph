# pyPetrograph

v0.1.0

[![PyPI version](https://img.shields.io/pypi/v/pypetrograph.svg)](https://pypi.org/project/pypetrograph/)

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

Or from the repository: `git clone https://github.com/GeoLarryLai/pyPetrograph.git`, `cd pyPetrograph`, `pip install .` (or `pip install -e .`).

Python 3.10–3.12 recommended (legend OCR has no 3.13+ wheels yet). On Linux, PySide6 needs some system libraries; see the [GitHub README](https://github.com/GeoLarryLai/pyPetrograph#linux-system-libraries). Optional extras: `seg`, `sam2`, `xgb`, `test`, `all`.

## Desktop app

Included in the install. Launch with:

```bash
pypetrograph-app
# or
python -m pyPetrograph.app
```

## Demos

Example notebooks, test images, figures, and optional extras are in the [GitHub repository](https://github.com/GeoLarryLai/pyPetrograph).

## Credits

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. This development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups and the [e-MAGE lab](https://www.beg.utexas.edu/research/labs/sem) in the Bureau of Economic Geology.

## License

```text
Program: pyPetrograph
A toolkit for semi-automated, ML-assisted thin-section petrography:
porosity estimation, mineral counting, grain segmentation and micro-
texture analysis
Copyright (C) 2026  Larry Syu-Heng Lai
```

GNU Affero General Public License 3.0 or later (`LICENSE`).
