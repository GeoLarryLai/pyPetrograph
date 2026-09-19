# pyPetrograph

v0.0.5 — thin-section petrography in Python.

**This tool is under development.** Treat v0.0.5 as a research preview. APIs, notebooks, and numbers can still change.

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. Development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups in the Bureau of Economic Geology.

**Warning — U-Net vs SAM.** RGB and mineral-map demos need neither. Multimodal outlines use the SegmentEveryGrain **U-Net** (~26 MB). That is enough for grain counts, the object table, and area %. **SAM 2.1** (~860 MB) only refines polygon *shape*. Leave `USE_SAM2 = False`. Do not download SAM weights unless you need that extra detail.

Three public demos, all on files in `Test images/`:

| Notebook | What it does |
|----------|----------------|
| [`pyPetrograph_rgb_porosity.ipynb`](pyPetrograph_rgb_porosity.ipynb) | Wand/polygon labels on one PPL set → LightGBM → **porosity %** with uncertainty |
| [`pyPetrograph_mineralmap_count.ipynb`](pyPetrograph_mineralmap_count.ipynb) | Flat-color mineral maps + legend TIFF → grains per class with Poisson / Wilson / method-range uncertainty |
| [`pyPetrograph_multimodal_grains.ipynb`](pyPetrograph_multimodal_grains.ipynb) | PPL + XPL → align → physics channels → **U-Net** outlines → object table → names / stats |

Package API and file layout: [`pyPetrograph/README.md`](pyPetrograph/README.md).

Mineral-map workflow:

![Mineral-map workflow](project%20images/workflow.png)

---

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
