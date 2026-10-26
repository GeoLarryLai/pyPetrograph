import sys
OUT=sys.argv[1]

from pathlib import Path
import json
import sys

ROOT = Path(".")
if str(ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(ROOT.resolve()))

FIG_DPI = 150

from pyPetrograph import check_and_install_packages
_ = check_and_install_packages(
    install=False, include_seg=True, include_sam2=True, include_amics=True,
    include_app=True,  # reports PySide6 for the notebook pop-ups (pip install PySide6-Essentials); never installs it
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["savefig.dpi"] = FIG_DPI

from pyPetrograph import set_output_dir
from pyPetrograph.amics import (
    MEASURE_COLUMNS,
    add_size_metrics,
    amics_paths,
    click_points,
    count_table,
    dark_mask,
    detect_footer,
    draw_scale,
    legend_to_text,
    load_bse,
    load_mineral_map,
    make_knolling,
    match_map_colors,
    measure_grains,
    mineral_gsd_png,
    pick_amics_files,
    plot_gsd,
    plot_gsd_by_mineral,
    plot_legend,
    plot_missing_colors,
    plot_composition,
    plot_mixed,
    plot_overlay,
    plot_threshold_preview,
    plot_unmapped,
    read_footer_pixelsize,
    read_grains_geojson,
    read_legend,
    recolor_from_matches,
    run_seg_bse,
    sample_name,
    save_figure,
    save_knolling,
    stretch,
    threshold_from_points,
    valid_mask,
    vote_grains,
    write_grains_geojson,
    write_mineral_csvs,
)

set_output_dir(OUT)

FIG_DPI = 150
plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["savefig.dpi"] = FIG_DPI

START_DIR = Path("example_data")
LEGEND_PATH = Path("example_data/10-Mineral_Legend.tiff")  # or Path("example_data/Con-1_Mineral_Legend.tiff")
BSE_PATH = Path("example_data/10B_BSE_Image.tiff")     # or Path("example_data/Con-1_BSE_Image.tiff")
MINERAL_PATH = Path("example_data/10B_Mineral_Image.tiff") # or Path("example_data/Con-1_Mineral_Image.tiff")
# any None opens the file window (prefilled with the ones you set)
SAMPLE = None

SKIP_NAMES = ["Unknown", "Low_Counts", "Un_x_rayed", "Shadows", "Pores"]
COLOR_TOL = 10
WHITE_TOL = 3
AUTO_MATCH_COLORS = True
MATCH_MAX_DIST = 25
FOOTER_ROW = None

THR_RADIUS = 5
THR_PCT = 99
BSE_THRESHOLD = 62

SCALE_LENGTH = 500
SCALE_UNIT = "µm"
UM_PER_PX = 1.0  # a number (µm per pixel) skips the scale window

MIN_GRAIN_PX = 10
DARK_DROP_FRAC = 0.5
USE_SAM2 = True
EDIT_GRAINS = False
UNMAPPED_FRAC = 0.5
MIXED_FRAC = 0.30
OVERLAY_ALPHA = 0.5
KNOLL_ALPHA = 0.5
KNOLL_WIDTH_PX = None
MIN_GRAINS_CURVE = 5

LOAD_SAVED_FILES = True
LOAD_SAVED_THRESHOLD = True
LOAD_SAVED_SCALE = True
LOAD_SAVED_GRAINS = True

print("output folder: outputs/amics/{sample}/")
print(f"SAM 2.1 {'on' if USE_SAM2 else 'off'}; GrainPlot {'on' if EDIT_GRAINS else 'off'}")

FILES = pick_amics_files(LEGEND_PATH, BSE_PATH, MINERAL_PATH, START_DIR, use_saved=LOAD_SAVED_FILES)
LEGEND_FILE, BSE_FILE, MINERAL_FILE = FILES["legend"], FILES["bse"], FILES["mineral"]
for _k, _v in FILES.items():
    print(f"{_k:8s} {_v}")



LEGEND_AUTO, LEGEND_REPORT = read_legend(LEGEND_FILE)
legend_img = np.asarray(Image.open(LEGEND_FILE))
fig, ax = plt.subplots(figsize=(8, max(3, legend_img.shape[0] / 80)))
ax.imshow(legend_img)
ax.set_title(LEGEND_FILE.name)
ax.axis("off")
plt.show()

print("\n".join(LEGEND_REPORT))
print()
print("Paste the dict below into LEGEND_EDIT if a name is wrong. Leave LEGEND_EDIT = None to keep it.")
print(legend_to_text(LEGEND_AUTO))

LEGEND_EDIT = None

LEGEND = dict(LEGEND_EDIT) if LEGEND_EDIT else dict(LEGEND_AUTO)
fig = plot_legend(LEGEND, SKIP_NAMES)
plt.show()
print(f"{len(LEGEND)} legend entries")

SAMPLE = SAMPLE or sample_name(BSE_FILE)
P = amics_paths(SAMPLE, BSE_FILE)

P["legend_json"].write_text(
    json.dumps({name: [int(c) for c in rgb] for name, rgb in LEGEND.items()}, indent=1),
    encoding="utf-8",
)
fig = plot_legend(LEGEND, SKIP_NAMES)
save_figure(fig, P["legend_png"], FIG_DPI)
plt.close(fig)

BSE_GRAY, BSE_RGBA = load_bse(BSE_FILE)
FOOTER = int(FOOTER_ROW) if FOOTER_ROW is not None else int(detect_footer(BSE_RGBA))
BSE_VALID = valid_mask(BSE_RGBA, FOOTER)
BSE_STRETCH = stretch(BSE_GRAY, BSE_VALID)

print(f"sample {SAMPLE}")
print(f"image {BSE_GRAY.shape[1]} x {BSE_GRAY.shape[0]} px; data rows end at {FOOTER}")
print("files:", P["folder"])

h, w = BSE_GRAY.shape[:2]
fig, ax = plt.subplots(figsize=(10, 10 * h / w))
ax.imshow(BSE_GRAY, cmap="gray", vmin=0, vmax=255)
if FOOTER < h:
    ax.axhline(FOOTER - 0.5, color="red", linestyle="--", linewidth=1)
ax.set_title(f"{BSE_FILE.name}  footer row {FOOTER}")
ax.axis("off")
plt.show()

saved_thr = None
if LOAD_SAVED_THRESHOLD and P["threshold_json"].exists():
    saved_thr = json.loads(P["threshold_json"].read_text(encoding="utf-8"))
SAVED_POINTS = [tuple(p) for p in ((saved_thr or {}).get("points") or [])]

POINTS = []
if BSE_THRESHOLD is not None:
    THRESHOLD = float(BSE_THRESHOLD)
    print(f"using BSE_THRESHOLD = {THRESHOLD}")
else:
    print("Threshold window: left click adds | right click removes nearest | C clears all | Enter or close = done.")
    POINTS = click_points(
        BSE_STRETCH, BSE_VALID, SAVED_POINTS, radius=THR_RADIUS, pct=THR_PCT, footer_row=FOOTER
    )
    if not POINTS:
        raise RuntimeError("No points. Click at least one epoxy / hole pixel, or set BSE_THRESHOLD in Settings.")
    THRESHOLD = threshold_from_points(
        BSE_STRETCH, POINTS, BSE_VALID, radius=THR_RADIUS, pct=THR_PCT
    )

DARK = dark_mask(BSE_STRETCH, THRESHOLD, BSE_VALID)
P["threshold_json"].write_text(
    json.dumps({"threshold": THRESHOLD, "points": [[float(x), float(y)] for x, y in POINTS]}, indent=1),
    encoding="utf-8",
)
fig = plot_threshold_preview(BSE_STRETCH, DARK, FOOTER, POINTS)
save_figure(fig, P["threshold_png"], FIG_DPI)
plt.show()
print(f"threshold {THRESHOLD:.1f}; dark or invalid {100 * float(DARK.mean()):.1f}% of the image")

SAVED_SCALE = None
if LOAD_SAVED_SCALE and P["scale_json"].exists():
    SAVED_SCALE = json.loads(P["scale_json"].read_text(encoding="utf-8"))

if UM_PER_PX is not None:
    SCALE = {"um_per_px": float(UM_PER_PX)}
    print(f"using UM_PER_PX = {UM_PER_PX}")
else:
    print("Scale window: click 2 points to draw a new line | Enter = type length | close = keep current scale.")
    NEW_SCALE = draw_scale(BSE_GRAY, saved=SAVED_SCALE, default_length=SCALE_LENGTH, default_unit=SCALE_UNIT)
    if NEW_SCALE is not None:
        SCALE = NEW_SCALE
        P["scale_json"].write_text(json.dumps(SCALE, indent=1), encoding="utf-8")
    elif SAVED_SCALE:
        SCALE = SAVED_SCALE
        print(f"kept the saved scale from {P['scale_json'].name}")
    else:
        print("No scale. Lengths stay in pixels.")
        SCALE = {}

UM_PER_PX_USED = SCALE.get("um_per_px")
SIZE_UNIT = "um" if UM_PER_PX_USED else "px"
if UM_PER_PX_USED and SCALE.get("px_length"):
    print(f"scale {SCALE.get('length')} {SCALE.get('unit')} = {SCALE.get('px_length')} px -> {float(UM_PER_PX_USED):.4f} µm/px")
elif UM_PER_PX_USED:
    print(f"scale {float(UM_PER_PX_USED):.4f} µm/px")
else:
    print("scale: none")

ocr_um = read_footer_pixelsize(BSE_RGBA, FOOTER)
print(f"footer text cross-check: {ocr_um} µm/px" if ocr_um else "footer text cross-check: not read")

def _rings(geom):
    if geom is None or getattr(geom, "is_empty", True):
        return
    if geom.geom_type == "Polygon":
        yield np.asarray(geom.exterior.coords)
        return
    for part in getattr(geom, "geoms", ()):
        yield from _rings(part)


if LOAD_SAVED_GRAINS and P["grains"].exists():
    if EDIT_GRAINS:
        print(f"Opening GrainPlot on the saved grains in {P['grains'].name}.")
        GRAIN_POLYS = run_seg_bse(
            BSE_STRETCH,
            DARK,
            P["grains"],
            use_sam=USE_SAM2,
            edit=True,
            load_from=P["grains"],
        )
    else:
        GRAIN_POLYS, _saved, _meta = read_grains_geojson(P["grains"])
    print(f"{len(GRAIN_POLYS)} grains from {P['grains'].name}")
else:
    note = " GrainPlot opens when outlines are ready; close it to save." if EDIT_GRAINS else ""
    print("Running grain segmentation." + note)
    GRAIN_POLYS = run_seg_bse(
        BSE_STRETCH,
        DARK,
        P["grains"],
        use_sam=USE_SAM2,
        edit=EDIT_GRAINS,
        min_area=MIN_GRAIN_PX / 2,
        drop_frac=DARK_DROP_FRAC,
    )
    print(f"segmentation returned {len(GRAIN_POLYS)} grains")

GRAINS = measure_grains(GRAIN_POLYS, um_per_px=UM_PER_PX_USED)
keep = GRAINS["area_px"].fillna(0) >= (MIN_GRAIN_PX / 2)
GRAIN_POLYS = [poly for poly, ok in zip(GRAIN_POLYS, keep) if ok]
GRAINS = GRAINS.loc[keep].reset_index(drop=True)
GRAINS["grain_id"] = np.arange(1, len(GRAINS) + 1)
write_grains_geojson(
    P["grains"],
    GRAIN_POLYS,
    GRAINS,
    {
        "sample": SAMPLE,
        "um_per_px": UM_PER_PX_USED,
        "scale": SCALE,
        "threshold": THRESHOLD,
        "source_image": str(Path(BSE_FILE)),
    },
)
print(f"kept {len(GRAIN_POLYS)} grains with area >= {MIN_GRAIN_PX / 2:g} px -> {P['grains'].name}")

_h, _w = BSE_STRETCH.shape[:2]
_rows = min(FOOTER, _h)
fig, ax = plt.subplots(figsize=(10, 10 * _rows / _w))
ax.imshow(BSE_STRETCH[:FOOTER], cmap="gray", vmin=0, vmax=255)
segs = [ring for poly in GRAIN_POLYS for ring in _rings(poly)]
if segs:
    ax.add_collection(LineCollection(segs, colors="yellow", linewidths=0.4))
ax.set_xlim(-0.5, _w - 0.5)
ax.set_ylim(_rows - 0.5, -0.5)
ax.axis("off")
ax.set_title(f"{SAMPLE}: {len(GRAIN_POLYS)} grains")
plt.show()

GRAINS = add_size_metrics(GRAINS, UM_PER_PX_USED)
write_grains_geojson(
    P["grains"],
    GRAIN_POLYS,
    GRAINS,
    {
        "sample": SAMPLE,
        "um_per_px": UM_PER_PX_USED,
        "scale": SCALE,
        "threshold": THRESHOLD,
        "source_image": str(Path(BSE_FILE)),
    },
)
_unit = "µm" if SIZE_UNIT == "um" else "px"
_big = GRAINS["area_px"] >= MIN_GRAIN_PX
print(f"{int(_big.sum())} grains >= {MIN_GRAIN_PX:g} px. Medians ({_unit}):")
print("  " + ", ".join(f"{_n} {GRAINS.loc[_big, f'{_n}_{SIZE_UNIT}'].median():.2f}" for _n in ("a", "b", "D", "ECD")))
print(f"rewrote {P['grains'].name} with a, b, D, ECD")

big = GRAINS["area_px"] >= MIN_GRAIN_PX
fig = plot_gsd(GRAINS.loc[big], unit=SIZE_UNIT)
save_figure(fig, P["gsd"], FIG_DPI)
plt.show()
print(f"n = {int(big.sum())} grains >= {MIN_GRAIN_PX:g} px")

map_sample = sample_name(MINERAL_FILE)
if map_sample != SAMPLE:
    print(f"warning: map file starts with {map_sample!r}, sample is {SAMPLE!r}")

MAP_RGB, CMAP, MAP_VALID, MISSING = load_mineral_map(
    MINERAL_FILE, LEGEND, footer_row=FOOTER, color_tol=COLOR_TOL, white_tol=WHITE_TOL
)
if MAP_RGB.shape[:2] != BSE_GRAY.shape[:2]:
    raise ValueError(
        f"mineral map is {MAP_RGB.shape[1]}x{MAP_RGB.shape[0]}, BSE is {BSE_GRAY.shape[1]}x{BSE_GRAY.shape[0]}"
    )

fig = plot_missing_colors(MISSING)
plt.show()
if len(MISSING):
    print(MISSING.head(15).to_string(index=False))
    print("Name these in EXTRA_LEGEND (next cell), then run that cell. Unnamed colors count as Unknown.")
else:
    print("Every map color is in the legend.")

EXTRA_LEGEND = {}

VOTE_LEGEND = {**LEGEND, **EXTRA_LEGEND}
MAP_RGB, CMAP, MAP_VALID, MISSING = load_mineral_map(
    MINERAL_FILE, VOTE_LEGEND, footer_row=FOOTER, color_tol=COLOR_TOL, white_tol=WHITE_TOL
)

COLOR_MATCHES = None
if AUTO_MATCH_COLORS and len(MISSING):
    COLOR_MATCHES = match_map_colors(MISSING, VOTE_LEGEND, CMAP, SKIP_NAMES, max_dist=MATCH_MAX_DIST)
    print("Map colors with no exact legend swatch:")
    for _, _m in COLOR_MATCHES.iterrows():
        _c = f"({_m.map_red}, {_m.map_green}, {_m.map_blue})"
        if _m.status == "free name":
            _how = f"{_m.assigned}  (unused legend name, color difference {_m.distance:.0f})"
        elif _m.status == "shared name":
            _how = f"{_m.assigned}  (nearest mineral, name also used by its exact color; difference {_m.distance:.0f})"
        else:
            _how = "left unnamed (gray, no gray legend entry): counts as Unknown"
        print(f"  {_c:18s} {_m.pixels:>9,d} px -> {_how}")
    _recolor = recolor_from_matches(COLOR_MATCHES, VOTE_LEGEND)
    if _recolor:
        MAP_RGB, CMAP, MAP_VALID, MISSING = load_mineral_map(
            MINERAL_FILE, VOTE_LEGEND, footer_row=FOOTER, color_tol=COLOR_TOL, white_tol=WHITE_TOL, recolor=_recolor
        )
print(f"vote legend: {len(VOTE_LEGEND)} names; colors still unnamed: {len(MISSING)}")

fig = plot_overlay(
    BSE_STRETCH[:FOOTER], MAP_RGB[:FOOTER], MAP_VALID[:FOOTER], GRAIN_POLYS, alpha=OVERLAY_ALPHA
)
save_figure(fig, P["overlay"], FIG_DPI)
plt.show()

votes = vote_grains(
    GRAIN_POLYS, CMAP, VOTE_LEGEND, SKIP_NAMES, mixed_frac=MIXED_FRAC, unmapped_frac=UNMAPPED_FRAC
)
base_cols = [c for c in MEASURE_COLUMNS if c in GRAINS.columns]
GRAINS = pd.concat(
    [GRAINS.reindex(columns=base_cols).reset_index(drop=True), votes.drop(columns=["grain_id"]).reset_index(drop=True)],
    axis=1,
)

fig = plot_unmapped(
    BSE_STRETCH[:FOOTER], MAP_RGB[:FOOTER], MAP_VALID[:FOOTER], GRAIN_POLYS, GRAINS, alpha=OVERLAY_ALPHA
)
save_figure(fig, P["unmapped"], FIG_DPI)
plt.show()

write_grains_geojson(
    P["minerals"],
    GRAIN_POLYS,
    GRAINS,
    {"sample": SAMPLE, "um_per_px": UM_PER_PX_USED, "source_image": str(Path(MINERAL_FILE))},
)
print(f"wrote {P['minerals'].name} (all {len(GRAINS)} grains, Unmapped included)")

is_mapped = (GRAINS["mineral"] != "Unmapped").to_numpy()
MAPPED = GRAINS.loc[is_mapped].reset_index(drop=True)
MAPPED_POLYS = [poly for poly, ok in zip(GRAIN_POLYS, is_mapped) if ok]
print(f"dropped {int((~is_mapped).sum())} unmapped grains (no map color under them); {len(MAPPED)} grains kept")

fig = plot_mixed(
    BSE_STRETCH[:FOOTER], MAP_RGB[:FOOTER], MAP_VALID[:FOOTER], MAPPED_POLYS, MAPPED, alpha=OVERLAY_ALPHA
)
save_figure(fig, P["mixed"], FIG_DPI)
plt.show()
print(MAPPED["mineral"].value_counts().to_string())

COUNTS = count_table(
    MAPPED, VOTE_LEGEND, SKIP_NAMES, sample=SAMPLE, min_grain_px=MIN_GRAIN_PX, unit=SIZE_UNIT
)
COUNTS

_keep = (MAPPED["area_px"] >= MIN_GRAIN_PX).to_numpy()
fig = plot_composition(
    [poly for poly, ok in zip(MAPPED_POLYS, _keep) if ok],
    MAPPED.loc[_keep].reset_index(drop=True),
    VOTE_LEGEND,
    COUNTS,
    sample=SAMPLE,
    shape=BSE_STRETCH[:FOOTER].shape[:2],
)
save_figure(fig, P["composition"], FIG_DPI)
plt.show()
_share = COUNTS.loc[COUNTS["n"] > 0, ["class", "n", "share_pct"]].copy()
_share["share_pct"] = _share["share_pct"].round(1)
_share

big = MAPPED["area_px"] >= MIN_GRAIN_PX
for _mineral, fig in plot_gsd_by_mineral(
    MAPPED.loc[big], unit=SIZE_UNIT, colors=VOTE_LEGEND, min_grains=MIN_GRAINS_CURVE
):
    save_figure(fig, mineral_gsd_png(P["folder"], SAMPLE, _mineral), FIG_DPI)
    plt.show()

counted = (MAPPED["area_px"] >= MIN_GRAIN_PX).to_numpy()
COUNTED = MAPPED.loc[counted].reset_index(drop=True)
COUNTED_POLYS = [poly for poly, ok in zip(MAPPED_POLYS, counted) if ok]

COUNTS.to_csv(P["counts"], index=False)
csv_paths = write_mineral_csvs(COUNTED, SAMPLE, P["folder"])
print(f"wrote {P['counts'].name}")
print(f"wrote {len(csv_paths)} mineral CSVs in {P['folder']}")
COUNTS

knoll = make_knolling(
    BSE_STRETCH,
    COUNTED_POLYS,
    COUNTED,
    VOTE_LEGEND,
    alpha=KNOLL_ALPHA,
    width_px=KNOLL_WIDTH_PX,
)
save_knolling(knoll, P["knolling"], FIG_DPI)
fig_w = 40.0
fig_h = fig_w * knoll.shape[0] / max(knoll.shape[1], 1)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.imshow(knoll)
ax.set_title(P["knolling"].name)
ax.axis("off")
plt.show()
print(f"knolling {knoll.shape[1]} x {knoll.shape[0]} px -> {P['knolling'].name}")