Mineral-map outputs
===================

Made by pyPetrograph_mineralmap_count.ipynb. Demo maps: 10A, 10C, 10E.
White on the TIFF is background, not a grain.

Grains are colored blobs. Black between grains is dirt/glue, not a grain.
A black patch fully inside one grain stays inside that outline. A colour
change is always a grain boundary; same-colour grains that touch are split
at thin bridges (opening radius 4 px), at necks where the distance map dips
≥ 3 px, and between two deep notches when both halves are compact
(solidity ≥ 0.88). A grain is named by its largest known color; it is
Unknown only if ≥ 95% of its pixels are black / off-legend.

Grain IDs live in the CSV and GeoJSON; there is no separate label image.

Open the CSVs in Excel, Numbers, or pandas. Open the GeoJSON in QGIS
(or Python: geopandas.read_file). Open the PNG in any image viewer.


Per image (replace {stem} with e.g. 10B_Mineral_Image)
------------------------------------------------------

{stem}_counts.csv
    Start here for “how many of each mineral.”
    n = grain count (one colour per grain; touching same-colour grains split
    at thin bridges, necks, and deep notches; grains ≥ 100 px).
    poisson_sd = √n. Report as n ± poisson_sd.
    n_lo–n_hi = method range (no split = colour regions + black rule only;
    min size 50 px and 200 px).
    share_pct = n / all grains in that image, with Wilson 95% bounds
    (share_lo95, share_hi95). area_px / area_pct = filled pixels, not count.
    Last row is TOTAL.

{stem}_grains.csv
    One row per grain: grain_id, class, area, shape, centroid, bbox.
    class_frac and frac_* = color mix inside that outline.
    Use this for grain-size lists or to join attributes onto the GeoJSON
    by grain_id.

{stem}_grains.geojson
    One polygon per grain (pixel coordinates: x = column, y = row).
    Properties: grain_id, class, area_px.
    Plot outlines or fill by class. Same IDs as the grains CSV.

{stem}_class_rgb.png
    Quick look: each grain painted its legend color, dark edges at splits.
    Biotite is mid-blue here only so it is not confused with black Unknown.
    This is a preview, not a data table.

{stem}_unlisted.csv
    Audit only — not a mineral class.
    Map colors farther than 40 RGB units from every legend swatch; those
    pixels were grouped into Unknown. Columns: r g b n_px nearest dist.


Folder
------

all_counts.csv
    Every image’s counts.csv stacked. Same columns, plus image.
    Use this for a table or chart across 10A–19E.
