# Demo outputs

Written by the three `pyPetrograph_*.ipynb` notebooks when `set_output_dir("outputs")` is set. Small files here are meant to be committed so you can inspect results without running. Large binaries stay local (see `.gitignore`: `outputs/models/`, `*.npy`, `*.npz`, `*.keras`, `*.pt`, `*.joblib`).

| Folder | From | What you get |
|--------|------|----------------|
| `mineralmap/` | `pyPetrograph_mineralmap_count.ipynb` | Grain counts, polygons, class overlays for 10A / 10C / 10E |
| `labels/` | `pyPetrograph_rgb_porosity.ipynb` | Example pore / grain polygons (AV03, AV11; AV12 is empty) |
| `predictions/` | RGB notebook after train/predict | Class maps, fractions CSV, preview figures |
| `align/` `channels/` `objects/` | `pyPetrograph_multimodal_grains.ipynb` | Lineup, channel stack, object table / stats |
| `models/` | multimodal (gitignored) | SEG U-Net (~26 MB). SAM 2.1 weights are **not** written unless `USE_SAM2 = True` |

See `mineralmap/README.txt` for the count-table columns.
