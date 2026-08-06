"""Notebook cell helpers (summary, preview, train, predict)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyPetrograph.common.constants import (
    DEFAULT_FIGURE_DPI,
    DEFAULT_POLY_FILL_ALPHA,
    DEFAULT_PREVIEW_DPI,
    DEFAULT_SURE_PROBA,
    FEATURE_PREVIEW_CMAPS,
    LABELS_SUBDIR,
    LEGEND_FONTSIZE,
    PathLike,
    PREDICTIONS_SUBDIR,
)
from pyPetrograph.common.paths import _subdir_rel, project_models_dir, relpath_display, stem_paths
from pyPetrograph.common.session import Session
from pyPetrograph.common.ui import _pick_files_native
from pyPetrograph.image_processing.brightness import relative_luminance, to_display_uint8
from pyPetrograph.image_processing.features import (
    build_feature_stack,
    downsample_to_approx_mp,
    resolve_feature_toggles,
    resolve_view_target_mp,
)
from pyPetrograph.image_processing.viz import (
    add_class_legend,
    class_color_rgba,
    colorize_labels_hsv,
)
from pyPetrograph.labeling_ml.train import (
    _format_accuracy_pct,
    format_classification_report_pct,
)

def print_label_summary(session: Session) -> None:
    """Print label stats; one gray+polygon figure per queued image (figure_dpi)."""
    import matplotlib.pyplot as plt

    if not session.image_queue:
        print("No images queued.")
        return

    # Do NOT save_all() here — notebook memory is often stale after launch_app
    # (Qt child already wrote labels/). Re-read each image from disk instead.
    if session.image_path is not None:
        session.reload_current_from_disk()

    n = len(session.image_queue)
    dpi = int(session.figure_dpi) or DEFAULT_FIGURE_DPI
    print("Label summary")
    print("=" * 60)
    for i, path in enumerate(session.image_queue):
        img, labels, stem, polys, poly_ids = session._load_labels_for_path(path)
        n_poly = len(polys)
        n_pix = int(np.sum(labels > 0))
        classes = []
        for cid in sorted(int(c) for c in np.unique(labels) if c > 0):
            cnt = int(np.sum(labels == cid))
            name = session.class_names.get(cid, f"class_{cid}")
            classes.append(f"{cid}:{name}={cnt}")
        print(f"[{i + 1}/{n}] {relpath_display(path)}")
        print(f"  polygons: {n_poly} | labeled pixels: {n_pix}")
        print(f"  classes: {', '.join(classes) if classes else '(none)'}")
        sp = stem_paths(path, stem=stem)
        outs = []
        if sp.polygons.exists():
            outs.append(_subdir_rel(sp.polygons, path.parent))
        if sp.session.exists():
            outs.append(_subdir_rel(sp.session, path.parent))
        if outs:
            print("  outputs:")
            for o in outs:
                print(f"    {o}")

        gray = relative_luminance(img)
        fig, ax = plt.subplots(1, 1, figsize=(7, 6.8), dpi=dpi)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        legend_ids: List[int] = []
        if polys:
            ids = list(poly_ids) if len(poly_ids) == len(polys) else [1] * len(polys)
            for poly, cid in zip(polys, ids):
                xs, ys = poly.exterior.xy
                color = class_color_rgba(int(cid))[:3]
                ax.fill(xs, ys, color=color, alpha=DEFAULT_POLY_FILL_ALPHA)
                ax.plot(xs, ys, color=color, lw=2.0)
            legend_ids = sorted({int(c) for c in ids if int(c) > 0})
        elif n_pix > 0:
            overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
            for cid in sorted(int(c) for c in np.unique(labels) if c > 0):
                r, g, b, _ = class_color_rgba(cid)
                m = labels == cid
                overlay[m, 0] = r
                overlay[m, 1] = g
                overlay[m, 2] = b
                overlay[m, 3] = DEFAULT_POLY_FILL_ALPHA
            ax.imshow(overlay)
            legend_ids = sorted(int(c) for c in np.unique(labels) if c > 0)
        ax.set_title(stem, fontsize=11)
        ax.axis("off")
        if legend_ids:
            add_class_legend(ax, legend_ids, session.class_names)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.14 if legend_ids else 0.05)
        # Summary figure IS labels/{stem}_labels.png (not a separate *_summary.png).
        _show_figure_inline(fig, save_path=sp.labels, dpi=dpi)
        print(f"  → {relpath_display(sp.labels)}")

    print("=" * 60)
    parents = session.queue_parent_dirs()
    if len(parents) == 1:
        print(f"outputs under: {relpath_display(parents[0])}/")
    else:
        print("outputs under each image folder:")
        for p in parents:
            print(f"  {relpath_display(p)}/")
    print(f"  {LABELS_SUBDIR}/  {PREDICTIONS_SUBDIR}/")
    print(f"models: {relpath_display(project_models_dir())}/")


def _show_figure_inline(
    fig, save_path: Optional[PathLike] = None, dpi: Optional[int] = None
) -> None:
    """Show a matplotlib figure inline in Jupyter; optionally save PNG to disk."""
    import matplotlib.pyplot as plt

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dpi = int(dpi) if dpi is not None else int(fig.dpi)
        fig.savefig(path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    try:
        from IPython.display import display

        display(fig)
        plt.close(fig)
    except Exception:
        fig.show()


def run_feature_preview(session: Session) -> None:
    """
    Inline preview of toggled feature maps for every queued image (1 column).

    Uses rgb_cache when present. Training still uses full working resolution.
    """
    import matplotlib.pyplot as plt

    if not session.image_queue:
        print("Select images first.")
        return

    toggles = resolve_feature_toggles(session.feature_toggles)
    view_mp = session.view_target_mp()
    dpi = int(session.figure_dpi) or DEFAULT_FIGURE_DPI
    print("Feature preview (downsampled; train uses full res)")
    print(f"view_mp≈{view_mp:.2f} | dpi={dpi} (figure_dpi)")
    print("toggles:", {k: v for k, v in toggles.items() if v})

    for qi, path in enumerate(session.image_queue, 1):
        img, _labels, stem, polys, _ids = session._load_labels_for_path(path)
        small = downsample_to_approx_mp(img, view_mp)
        scale_y = small.shape[0] / img.shape[0]
        scale_x = small.shape[1] / img.shape[1]
        polys_small: List[Polygon] = []
        if polys and abs(scale_x - 1.0) > 1e-6:
            for poly in polys:
                coords = [(x * scale_x, y * scale_y) for x, y in poly.exterior.coords]
                try:
                    polys_small.append(Polygon(coords))
                except Exception:
                    pass
        else:
            polys_small = list(polys)

        stack, names = build_feature_stack(
            small,
            toggles=toggles,
            y_target=session.y_target,
            apply_norm=session.apply_norm,
            texture_window=session.texture_window,
            lbp_p=session.lbp_p,
            lbp_r=session.lbp_r,
            polygons=polys_small,
        )
        nfeat = len(names)
        if nfeat == 0:
            print(f"[{qi}] {stem}: no features toggled on.")
            continue
        fig, axes = plt.subplots(nfeat, 1, figsize=(4.5, 2.6 * nfeat), dpi=dpi)
        axes_arr = np.atleast_1d(axes).ravel()
        print(f"[{qi}/{len(session.image_queue)}] {stem} shape={small.shape} | {names}")
        for i, name in enumerate(names):
            ax = axes_arr[i]
            ch = stack[..., i]
            cmap = FEATURE_PREVIEW_CMAPS.get(name, "viridis")
            if name in ("r", "g", "b"):
                ax.imshow(ch, cmap=cmap, vmin=0, vmax=255)
            elif name == "y":
                ax.imshow(ch, cmap="gray")
            else:
                ax.imshow(ch, cmap=cmap)
            ax.set_title(name, fontsize=10)
            ax.axis("off")
        fig.suptitle(stem, fontsize=12)
        fig.tight_layout()
        out_png = Path(path).resolve().parent / PREDICTIONS_SUBDIR / f"{stem}_features.png"
        _show_figure_inline(fig, save_path=out_png, dpi=dpi)
        print(f"  → {relpath_display(out_png)}")



def run_train_cell(
    session: Session,
    mode: str = "retrain",
    scope: str = "universal",
    model_path: Optional[PathLike] = None,
    prompt_load: bool = True,
) -> Dict[str, Any]:
    """
    Notebook Train cell helper.

    mode: "load" | "retrain"
    scope: used when retrain — "per_image" | "universal"
    Full CV report is printed by print_train_metrics (next cell).
    """
    mode = mode.lower().strip()
    print("Train")
    print("=" * 60)
    print(f"mode: {mode}")
    if mode == "load":
        path = model_path
        if path is None and prompt_load:
            try:
                session.ensure_output_dirs()
                initial = (
                    str(session.models_dir)
                    if session.models_dir is not None and Path(session.models_dir).is_dir()
                    else str(Path(".").resolve())
                )
                picked = _pick_files_native(
                    initial,
                    multi=False,
                    caption="Select model (.joblib)",
                    name_filter="Models (*.joblib);;All (*)",
                )
                path = picked[0] if picked else None
            except Exception as exc:
                raise RuntimeError(
                    f"Could not open file picker ({exc}).\n"
                    "Set model_path=... in this cell."
                ) from exc
        if not path:
            raise RuntimeError(
                "No model selected.\n"
                "Pick a .joblib file, or set mode='retrain'."
            )
        bundle = session.load_model_from_path(path)
        print(f"Loaded: {Path(path).name}")
        print(f"method: {bundle.get('method')} | accuracy: {_format_accuracy_pct(bundle.get('accuracy'))}")
        print("Run the Accuracy (CV) cell for the full report.")
        print("=" * 60)
        return {
            "mode": "load",
            "path": str(path),
            "accuracy": bundle.get("accuracy"),
            "report": bundle.get("report"),
            "cv_folds": bundle.get("cv_folds"),
            "bundle_keys": list(bundle.keys()),
        }

    if mode != "retrain":
        raise ValueError('mode must be "load" or "retrain".')

    print(f"scope: {scope}")
    results = session.train_queue(scope=scope)
    print(f"models: {relpath_display(project_models_dir())}/")
    if results.get("universal_paths"):
        for up in results["universal_paths"]:
            print(f"  universal → {relpath_display(up)}")
        print(f"pixels: {results.get('n_pixels')}")
    elif results.get("universal_path"):
        print(f"  universal → {relpath_display(results['universal_path'])}")
        print(f"pixels: {results.get('n_pixels')}")
    for row in results.get("trained", []):
        if "path" in row:
            print(f"  {row['image']} → {relpath_display(row['path'])}")
        else:
            print(f"  {row['image']}: n_pixels={row.get('n_pixels')}")
    if results.get("skipped"):
        print("Skipped (no labels):", ", ".join(results["skipped"]))
    print("Run the Accuracy (CV) cell for metrics.")
    print("=" * 60)
    return results


def print_train_metrics(
    train_results: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None,
) -> None:
    """Print stratified CV accuracy (%) + classification report (Accuracy notebook cell)."""
    print("Accuracy (CV)")
    print("=" * 60)
    print(
        "Note: rows are mineral classes (ids), not CV folds. "
        "Metrics are out-of-fold over stratified k-fold CV."
    )
    results = train_results or {}

    def _print_block(acc, folds, report, title: Optional[str] = None) -> None:
        if title:
            print(title)
        if folds:
            print(
                f"accuracy (stratified {folds}-fold CV): {_format_accuracy_pct(acc)}"
            )
        elif acc is not None:
            print(f"accuracy: {_format_accuracy_pct(acc)}")
        if report:
            print(report)

    if results.get("report") is not None or results.get("accuracy") is not None:
        _print_block(
            results.get("accuracy"),
            results.get("cv_folds"),
            results.get("report"),
        )
        print("=" * 60)
        return

    trained = results.get("trained") or []
    printed = False
    for row in trained:
        if row.get("report") is not None or row.get("accuracy") is not None:
            printed = True
            folds = row.get("cv_folds")
            fold_note = f"stratified {folds}-fold CV" if folds else "score"
            _print_block(
                row.get("accuracy"),
                folds,
                row.get("report"),
                title=f"{row.get('image')} ({fold_note})",
            )
            print("-" * 40)
    if printed:
        print("=" * 60)
        return

    if session is not None and session.model_bundle:
        b = session.model_bundle
        _print_block(b.get("accuracy"), b.get("cv_folds"), b.get("report"))
        print("=" * 60)
        return

    print("No train metrics found. Run Train (retrain) first.")
    print("=" * 60)


def run_predict_cell(
    session: Session,
    model_source: str = "universal",
) -> Dict[str, Any]:
    """Notebook Predict cell: predict queue; pred + confidence figures at figure_dpi."""
    import matplotlib.pyplot as plt

    print("Predict")
    print("=" * 60)
    print(f"model_source: {model_source}")
    print(
        "P = model class probability from predict_proba "
        "(per-pixel soft score in [0, 1])."
    )
    print(
        f"sure_threshold: {DEFAULT_SURE_PROBA}  "
        f"(area_pct_low = pixels with assigned class & P(class) ≥ {DEFAULT_SURE_PROBA})"
    )
    print(
        "area_pct = hard (argmax) %;  "
        "area_pct_high = max(hard, mean soft P);  "
        "+/− from those bounds."
    )
    results = session.predict_queue(model_source=model_source)
    print("outputs (beside each image folder):")
    for d in results.get("predictions_dirs") or []:
        print(f"  {relpath_display(d)}/")

    images = results["images"]
    n = len(images)
    dpi = int(session.figure_dpi) or DEFAULT_FIGURE_DPI

    for i, entry in enumerate(images):
        print(f"[{i + 1}/{n}] {entry['image']}")
        print(f"  model: {relpath_display(entry['model_used'])}")
        print(f"  → {relpath_display(entry['pred'])}")
        print(f"  → {relpath_display(entry['pred_rgb'])}")
        print(f"  → {relpath_display(entry['confidence'])}")
        print(f"  → {relpath_display(entry['fractions_csv'])}")
        for cid, info in entry["fractions"].items():
            print(
                f"    {info['name']}: {info['area_pct']:.2f}%  "
                f"+{info['area_pct_plus']:.2f}%/−{info['area_pct_minus']:.2f}%  "
                f"(n={info['count']})"
            )
        rgb = np.array(Image.open(entry["pred_rgb"]))
        conf = np.load(entry["confidence"])
        pred_path = entry.get("pred")
        legend_ids = sorted(int(c) for c in entry.get("fractions", {}).keys())
        if pred_path and Path(pred_path).exists():
            opened = np.array(Image.open(pred_path))
            class_rgb = opened if opened.ndim == 3 else colorize_labels_hsv(opened)
        else:
            class_rgb = rgb

        stem = Path(entry["image"]).stem
        out_dir = Path(entry["output_dir"])
        # Figure 1: prediction class map
        fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6.8), dpi=dpi)
        ax1.imshow(class_rgb)
        ax1.set_title(f"{stem} — predicted class map", fontsize=11)
        ax1.axis("off")
        if legend_ids:
            add_class_legend(ax1, legend_ids, session.class_names)
        fig1.tight_layout()
        fig1.subplots_adjust(bottom=0.14 if legend_ids else 0.05)
        fig1_path = out_dir / f"{stem}_fig_pred.png"
        _show_figure_inline(fig1, save_path=fig1_path, dpi=dpi)

        # Figure 2: confidence (inferno, horizontal colorbar below)
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6.2), dpi=dpi)
        im = ax2.imshow(conf, cmap="inferno", vmin=0.0, vmax=1.0)
        ax2.set_title(
            f"{stem} — confidence map: P(assigned class) per pixel",
            fontsize=11,
        )
        ax2.axis("off")
        cbar = fig2.colorbar(
            im, ax=ax2, orientation="horizontal", fraction=0.046, pad=0.08
        )
        cbar.set_label(
            "P(assigned class) = model probability of the predicted label [0–1]",
            fontsize=LEGEND_FONTSIZE,
        )
        cbar.ax.tick_params(labelsize=LEGEND_FONTSIZE)
        fig2.tight_layout()
        fig2_path = out_dir / f"{stem}_fig_confidence.png"
        _show_figure_inline(fig2, save_path=fig2_path, dpi=dpi)
        print(f"  → {relpath_display(fig1_path)}")
        print(f"  → {relpath_display(fig2_path)}")

    print("=" * 60)
    return results
