# pyPetrograph

v0.0.6

[![PyPI version](https://img.shields.io/pypi/v/pypetrograph.svg)](https://pypi.org/project/pypetrograph/)

pyPetrograph is a machine-learning toolkit for thin-section petrography.

**Warning:** This tool is under development. Use with caution. APIs and numbers can still change.

## What it does

**Hand-label features and map their area %.** Paint a few examples of each class you care about (pores, grains, cement, …) from color, edge sharpness, and microtexture. A LightGBM pixel classifier (random forest or XGBoost optional) maps the rest of the photo and reports area % of each class.

**Count grains on a painted mineral map.** Match pixels to a color legend, split grains that touch, and count how many grains sit in each class.

**Segment grains from a multi-image overlay.** Line up several image types of the same field (plane-polarized light, crossed polars, or others) and refine grain outlines with a U-Net, so boundaries are not drawn from one lighting alone.

For finer grain-boundary mapping there is optional support for [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain) (Zoltan Sylvester). [SAM 2.1](https://github.com/facebookresearch/sam2) weights (~860 MB) are a further opt-in on their page, not part of the default install.

## Install

```bash
pip install pypetrograph
```

```bash
pip install segmenteverygrain
```

The second line is optional (SegmentEveryGrain’s existing-environment path).

## Demos

Example notebooks, test images, and figures are in the [GitHub repository](https://github.com/GeoLarryLai/pyPetrograph).

## Credits

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. This development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups and the [e-MAGE lab](https://www.beg.utexas.edu/research/labs/sem) in the Bureau of Economic Geology.

## License

GNU Affero General Public License 3.0 or later (`LICENSE`).
