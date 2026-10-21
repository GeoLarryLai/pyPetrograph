# pyPetrograph

![PyPI version](https://img.shields.io/pypi/v/pypetrograph.svg) ![PyPI downloads](https://img.shields.io/pypi/dm/pypetrograph.svg) [![pepy downloads](https://static.pepy.tech/badge/pypetrograph)](https://pepy.tech/project/pypetrograph)

v0.0.7

pyPetrograph is a machine-learning toolkit for thin-section petrography. You can hand-label features on an RGB photo and train a LightGBM pixel model to map area % of each class (porosity in the demo). You can count grains on a painted mineral map by matching a color legend. Or you can stack several image types of the same field (PPL, XPL, or others) and segment grains from that overlay.

> [!WARNING]
> This tool is under development. Use with caution. Notebooks, APIs, and numbers can still change.

> [!NOTE]
> For more careful grain-boundary detection and mapping, there is optional support for using [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain) developed by **Zoltan Sylvester**. [SAM 2.1](https://github.com/facebookresearch/sam2) weights (~860 MB) are a further opt-in on their page, not part of the default install.

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. This development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups and the [e-MAGE lab](https://www.beg.utexas.edu/research/labs/sem) in the Bureau of Economic Geology.

Example thin-section images are in `example_data/`. Analyzed results go to `outputs/`.

## Installation

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

## Citation

Lai, L.S.-H. (2026) pyPetrograph.

Contact: larrysyuhenglai@gmail.com

## Key Features

- Hand-label features and map their area %: paint a few examples of each class (pores, grains, cement, …); a LightGBM pixel classifier maps the photo and reports area % of each class. The demo treats pore area as porosity.
- Grain counts on a painted mineral map: match pixels to a color legend, split grains that touch, and count how many grains sit in each class.
- Segment grains from a multi-image overlay: line up PPL and XPL of the same field, stack them, and outline grains (watershed, optional SegmentEveryGrain, and the shipped N-channel U-Net).

## Example notebooks

### 1. Hand-label features and map their area %

[`pyPetrograph_1_rgb_porosity.ipynb`](pyPetrograph_1_rgb_porosity.ipynb)

You paint a few examples of each feature you care about (pores, grains, cement, …) using what you see — color, edge sharpness, and microtexture. A LightGBM pixel classifier (random forest or XGBoost optional) learns those examples, maps the rest of the photo, and reports area % of each class with a confidence range. The demo uses pores vs grains and treats pore area as porosity.

![Wand/polygon label window on a PPL thin section](image/Manual_label_UI.png)

---

### 2. Grain counts on a painted mineral map

[`pyPetrograph_2_mineralmap_count.ipynb`](pyPetrograph_2_mineralmap_count.ipynb)

You already have a color mineral map (one flat color per mineral) and a legend. The notebook matches pixels to the legend, splits grains that touch, and counts how many grains sit in each class.

![Mineral-map grain counts, table, and class share bar](image/Mineralmap-demo.png)

---

### 3. Segment grains from a multi-image overlay

[`pyPetrograph_3_multimodal_grains.ipynb`](pyPetrograph_3_multimodal_grains.ipynb)

A boundary drawn from PPL alone can miss contacts that only show under crossed polars (XPL), or the other way around. This notebook lines up PPL and XPL of the same field and stacks them as one overlay. Grain outlines go left to right in the same order as the work:

1. **Watershed** — first guess from brightness, no training.
2. **SEG mask** — optional [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain) U-Net (Zoltan Sylvester). This saved mask is the **train target**.
3. **N-channel U-Net** (~25 MB, already in [`outputs/models/grain_unet_ppl-xpl.keras`](outputs/models/grain_unet_ppl-xpl.keras)) — trained on that PPL + XPL stack to copy the SEG mask. The notebook **loads** it and does not train again.

GrainPlot QC is a further optional window. [SAM 2.1](https://github.com/facebookresearch/sam2) (~860 MB, off by default) is only if you need still finer shapes.

![Watershed, SEG train target, and N-channel U-Net on an aligned PPL + XPL overlay](image/Grain-mapping-mutiimage-overlay.png)

Notebook steps and the package layout are in GUIDE.md.

## Package structure

```
README.md
README-PyPI.md
GUIDE.md
FUNCTION_DOCUMENTATION.md
example_data/
image/
pyPetrograph/
    align/
    common/
    fuse/
    image_processing/
    labeling_ml/
    mineral_map/
    objects/
```

## Dependencies

Requires Python >=3.10.

Installed with the package: numpy (>=1.24), pillow (>=10), matplotlib (>=3.7), scikit-learn (>=1.3), scipy (>=1.10), geopandas (>=0.14), shapely (>=2), rasterio (>=1.3), affine (>=2.4), joblib (>=1.3), pandas (>=2), lightgbm (>=4), scikit-image (>=0.22), PyQt5 (>=5.15).

Optional extras (not installed by default):

- segmenteverygrain / tensorflow (`seg`: tensorflow >=2.16,<2.22, keras >=3, segmenteverygrain ==0.5.0)
- sam2 / torch (`sam2`: torch >=2.2, torchvision, sam2 >=1.0)
- xgboost (`xgb`: xgboost >=2)

## License

pyPetrograph is licensed under the GNU Affero General Public License 3.0 or later. See [LICENSE](LICENSE).
