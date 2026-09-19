# pyPetrograph

v0.0.5

pyPetrograph measures pores, grains, and minerals on thin-section photos and painted mineral maps. You can label pixels on a plane-polarized image and get porosity; count grains on a flat-color mineral map; or line up PPL and XPL photos, outline grains, and build one table row per object.

> [!WARNING]
> This tool is under development. Use with caution. Notebooks, APIs, and numbers can still change.

> [!NOTE]
> Grain outlines adapt [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain). Optional [SAM 2.1](https://github.com/facebookresearch/sam2) can refine those polygons along grain boundaries. Turning that on downloads a large weight file (~860 MB).

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. Development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups in the Bureau of Economic Geology.

## Install

**Conda (all three demos, including SegmentEveryGrain):**

```bash
conda env create -f environment.yml
conda activate pypetrograph
pip install -e .
```

This repo’s working kernel is conda `work`. To refresh that env: `conda env update -n work -f environment.yml`.

**PyPI (when published):**

```bash
pip install pyPetrograph              # RGB + mineral-map (no TensorFlow)
pip install "pyPetrograph[seg]"       # multimodal U-Net (TensorFlow / Keras / SegmentEveryGrain)
pip install "pyPetrograph[all]"       # only if you also want SAM refine + xgboost
```

From a notebook (kernel-safe: TensorFlow is never imported here):

```python
from pyPetrograph import check_and_install_packages
check_and_install_packages()                    # core
check_and_install_packages(include_seg=True)    # also check U-Net worker deps
```

U-Net weights (~26 MB) download on the first multimodal run. SAM weights do **not**. TensorFlow is only for the U-Net **subprocess**, never in the Jupyter kernel.

---

# Demo notebooks

| Notebook | What it does |
|----------|----------------|
| [`pyPetrograph_rgb_porosity.ipynb`](pyPetrograph_rgb_porosity.ipynb) | Wand/polygon labels on one PPL set → LightGBM → **porosity %** with uncertainty |
| [`pyPetrograph_mineralmap_count.ipynb`](pyPetrograph_mineralmap_count.ipynb) | Flat-color mineral maps + legend TIFF → grains per class with Poisson / Wilson / method-range uncertainty |
| [`pyPetrograph_multimodal_grains.ipynb`](pyPetrograph_multimodal_grains.ipynb) | PPL + XPL → align → physics channels → **U-Net** outlines → object table → names / stats. Optional SAM 2.1 refine (`USE_SAM2 = True`) |

Package API and file layout: [`pyPetrograph/README.md`](pyPetrograph/README.md).

---

## Test images

| Files | Demo |
|-------|------|
| `AV03` / `AV11` / `AV12` `*_ppl_*.tif` | RGB porosity |
| matching `*_xpl_*.tif` | multimodal (default pair is AV03) |
| `10A_` / `10C_` / `10E_Mineral_Image.tiff` + `Mineral_Legend.tiff` | mineral-map counts |

Test-image sources: PPL/XPL frames already shipped with this repo; mineral maps and legend from the Rich Kyle–BEG aggregate mapping set (three maps only).

---

## Outputs

Every demo notebook calls `set_output_dir("outputs")`. Results land under [`outputs/`](outputs/README.md) (`labels/`, `predictions/`, `mineralmap/`, `align/`, `channels/`, `objects/`, `models/`). If you omit `set_output_dir`, companions stay beside each image and models at the repo root — the package default, unchanged.

Example labels for the porosity demo are already in `outputs/labels/` (two classes: pores, grains). Example mineral-map counts for 10A / 10C / 10E are in `outputs/mineralmap/`.

---

## License

GNU Affero General Public License 3.0 or later (`LICENSE`).
