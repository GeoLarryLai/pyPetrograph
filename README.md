# pyPetrograph

v0.0.5

pyPetrograph is a machine-learning toolkit for thin-section petrography. You can draw pores and grains by hand in an interactive label window (wand or polygon), train a LightGBM pixel model, and get porosity with a confidence range. You can count grains on a painted mineral map by matching legend colors and splitting grains that touch. Or you can line up plane-polarized (PPL) and cross-polarized (XPL) photos, let a U-Net outline the grains, and build one table row per object for grouping and naming.

> [!WARNING]
> This tool is under development. Use with caution. Notebooks, APIs, and numbers can still change.

> [!NOTE]
> For more careful grain-boundary detection and mapping, there is optional support for using [SegmentEveryGrain](https://github.com/zsylvester/segmenteverygrain) developed by **Zoltan Sylvester**. That extra step requires a large [SAM 2.1](https://github.com/facebookresearch/sam2) model download (~860 MB).

The current version is primarily developed by **Larry Syu-Heng Lai**, based on early source code scripted by **Zoltan Sylvester**, at the Quantitative Clastic Laboratory, Bureau of Economic Geology, The University of Texas at Austin. This development is also part of a collaboration with **Priyanka Periwal**, **Lucy Tingwei Ko**, **Kelly Hattori**, and **Amanda Calle** across multiple research groups and the [e-MAGE lab](https://www.beg.utexas.edu/research/labs/sem) in the Bureau of Economic Geology.

## Install

```bash
conda env create -f environment.yml
conda activate pypetrograph
pip install -e .
```

To skip SegmentEveryGrain, comment out `tensorflow`, `keras`, and `segmenteverygrain` in `environment.yml` before you create the env.

## Demo notebooks

- [`pyPetrograph_rgb_porosity.ipynb`](pyPetrograph_rgb_porosity.ipynb) — draw labels, train LightGBM, get porosity
- [`pyPetrograph_mineralmap_count.ipynb`](pyPetrograph_mineralmap_count.ipynb) — count grains on painted mineral maps
- [`pyPetrograph_multimodal_grains.ipynb`](pyPetrograph_multimodal_grains.ipynb) — line up PPL + XPL, outline grains, build an object table

Example thin-section images are in `Test images/`. Analyzed results go to `outputs/`.

## License

GNU Affero General Public License 3.0 or later (`LICENSE`).
