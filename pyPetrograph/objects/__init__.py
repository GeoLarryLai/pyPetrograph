"""Stage 3–6: grain/pore outlines, object table, grouping, classification."""

from pyPetrograph.objects.classify import (
    load_classes,
    object_model_path,
    predict_object_classes,
    save_classes,
    train_object_classifier,
)
from pyPetrograph.objects.cluster import (
    apply_object_labels,
    cluster_objects,
    launch_montage_labeler,
    load_object_labels,
)
from pyPetrograph.objects.outlines import (
    compare_outlines,
    grain_unet_path,
    init_grain_unet,
    predict_grain_unet,
    run_outlines,
    train_grain_unet,
)
from pyPetrograph.objects.qc import (
    ensure_seg_weights,
    grain_qc_paths,
    launch_grain_qc,
    launch_ppl_grain_qc,
    load_grain_mask,
    load_grain_qc,
    load_ppl_grain_qc,
    overlay_labels,
    seg_setup_hint,
)
from pyPetrograph.objects.stats import (
    compute_object_stats,
    load_object_stats,
    save_object_stats,
)
from pyPetrograph.objects.table import (
    build_object_table,
    build_object_table_from_stack,
    load_object_table,
    save_object_table,
)
from pyPetrograph.objects.watershed import (
    gradient_magnitude,
    instance_to_three_class,
    run_watershed,
    three_class_to_probs,
    watershed_labels,
)

__all__ = [
    "run_watershed",
    "watershed_labels",
    "gradient_magnitude",
    "instance_to_three_class",
    "three_class_to_probs",
    "init_grain_unet",
    "train_grain_unet",
    "predict_grain_unet",
    "grain_unet_path",
    "run_outlines",
    "compare_outlines",
    "launch_grain_qc",
    "launch_ppl_grain_qc",
    "load_grain_qc",
    "load_ppl_grain_qc",
    "grain_qc_paths",
    "load_grain_mask",
    "overlay_labels",
    "ensure_seg_weights",
    "seg_setup_hint",
    "build_object_table",
    "build_object_table_from_stack",
    "save_object_table",
    "load_object_table",
    "cluster_objects",
    "launch_montage_labeler",
    "load_object_labels",
    "apply_object_labels",
    "object_model_path",
    "train_object_classifier",
    "predict_object_classes",
    "save_classes",
    "load_classes",
    "compute_object_stats",
    "save_object_stats",
    "load_object_stats",
]
