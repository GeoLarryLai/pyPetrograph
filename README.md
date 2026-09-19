# pyPetrograph

![PyPI version](https://img.shields.io/pypi/v/pypetrograph.svg) ![PyPI downloads](https://img.shields.io/pypi/dm/pypetrograph.svg)

v0.0.6

pyPetrograph is a machine-learning toolkit for thin-section petrography. You can hand-label features on an RGB photo and train a LightGBM pixel model to map area % of each class (porosity in the demo). You can count grains on a painted mineral map by matching a color legend. Or you can stack several image types of the same field (PPL, XPL, or others) and segment grains from that overlay.

> [!WARNING]
> This tool is under development. Use with caution. Notebooks, APIs, and numbers can still change.

> [!NOTE]
> For more careful grain-boundary detection and mapping, there is optional support for using [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain) developed by **Zoltan Sylvester**. [SAM 2.1](https://github.com/facebookresearch/sam2) weights (~860 MB) are a further opt-in on their page, not part of the default install.

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. This development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups and the [e-MAGE lab](https://www.beg.utexas.edu/research/labs/sem) in the Bureau of Economic Geology.

Example thin-section images are in `Test images/`. Analyzed results go to `outputs/`.

---
## Install

From [PyPI](https://pypi.org/project/pypetrograph/):

```bash
pip install pypetrograph
```

Alternatively, from this repo:

```bash
conda env create -f environment.yml
conda activate pypetrograph
pip install -e .
```

```bash
# Optional: finer grain-boundary mapping (SegmentEveryGrain)
# https://github.com/zsylvester/segmenteverygrain
pip install segmenteverygrain
```

This is SegmentEveryGrain’s “existing environment” path (install into the activated `pypetrograph` env). SAM 2.1 weights (~860 MB) are a further opt-in on their page, not part of this pip install.

---
## Demo notebooks

### 1. Hand-label features and map their area %

[`pyPetrograph_rgb_porosity.ipynb`](pyPetrograph_rgb_porosity.ipynb)

You paint a few examples of each feature you care about (pores, grains, cement, …) using what you see — color, edge sharpness, and microtexture. A LightGBM pixel classifier (random forest or XGBoost optional) learns those examples, maps the rest of the photo, and reports area % of each class with a confidence range. The demo uses pores vs grains and treats pore area as porosity.

![Wand/polygon label window on a PPL thin section](project%20images/Manual_label_UI.png)

---

### 2. Grain counts on a painted mineral map

[`pyPetrograph_mineralmap_count.ipynb`](pyPetrograph_mineralmap_count.ipynb)

You already have a color mineral map (one flat color per mineral) and a legend. The notebook matches pixels to the legend, splits grains that touch, and counts how many grains sit in each class.

![Mineral-map grain counts, table, and class share bar](project%20images/Mineralmap-demo.png)

---

### 3. Segment grains from a multi-image overlay

[`pyPetrograph_multimodal_grains.ipynb`](pyPetrograph_multimodal_grains.ipynb)

A boundary drawn from PPL alone can miss contacts that only show under crossed polars (XPL), or the other way around. This notebook lines up PPL and XPL of the same field, stacks them as one overlay, and refines grain outlines with a [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain) U-Net (Zoltan Sylvester), then optional GrainPlot QC. Left: watershed — a no-training first guess from brightness. Right: GrainPlot QC — the trained refine on that overlay, so touching grains split more cleanly. [SAM 2.1](https://github.com/facebookresearch/sam2) (~860 MB, off by default) is only if you need still finer shapes.

![Watershed vs GrainPlot QC on an aligned PPL + XPL overlay](project%20images/Grain-mapping-mutiimage-overlay.png)

---
## License

GNU Affero General Public License 3.0 or later (`LICENSE`).