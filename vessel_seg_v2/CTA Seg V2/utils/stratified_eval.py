"""
Stratified per-vessel-class evaluation for TopCoW.

Uses the 13-class ground truth labels to measure how well a *binary*
prediction captures each individual vessel component. This lets us
answer: "Does clDice help more on the tiny Acom than on the large MCA?"

The model still trains binary (vessel vs. background). At evaluation
time, we mask both the prediction and ground truth to each vessel class
region and compute metrics within that region only.
"""

import numpy as np
from typing import Dict, List, Tuple
from utils.metrics import compute_dice, compute_cldice, compute_betti0_error


# ── TopCoW class mapping ─────────────────────────────────────────────────────
# These IDs correspond to the integer labels in TopCoW 2024 ground truth.
# Verify against the dataset README after downloading; IDs may shift between
# TopCoW 2023 and 2024 versions.

VESSEL_CLASSES = {
    1:  "BA",           # Basilar artery
    2:  "R-PCA",        # Right posterior cerebral artery
    3:  "L-PCA",        # Left posterior cerebral artery
    4:  "R-ICA",        # Right internal carotid artery
    5:  "L-ICA",        # Left internal carotid artery
    6:  "R-MCA",        # Right middle cerebral artery
    7:  "L-MCA",        # Left middle cerebral artery
    8:  "R-ACA",        # Right anterior cerebral artery (A1/A2)
    9:  "L-ACA",        # Left anterior cerebral artery (A1/A2)
    10: "R-Pcom",       # Right posterior communicating artery
    11: "L-Pcom",       # Left posterior communicating artery
    12: "Acom",         # Anterior communicating artery
    13: "3rd-A2",       # Third A2 segment (variant)
}

# Clinical grouping: large named arteries vs. small communicating segments
LARGE_VESSELS = {1, 2, 3, 4, 5, 6, 7, 8, 9}       # BA, PCA, ICA, MCA, ACA
SMALL_VESSELS = {10, 11, 12, 13}                     # Pcom, Acom, 3rd-A2

GROUP_NAMES = {
    "large": "Large named arteries (BA, ICA, MCA, PCA, ACA)",
    "small": "Communicating arteries (Acom, Pcom, 3rd-A2)",
}


def evaluate_per_class(
    pred_binary: np.ndarray,
    gt_multiclass: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each vessel class separately.

    Args:
        pred_binary: (D, H, W) binary prediction (0 = background, 1 = vessel)
        gt_multiclass: (D, H, W) integer labels with TopCoW class IDs (0 = bg)

    Returns:
        Dict mapping vessel name to {"dice": float, "cldice": float, "present": bool}
        "present" indicates whether this vessel class exists in the ground truth
        (communicating arteries are absent in some patients).
    """
    results = {}

    for class_id, name in VESSEL_CLASSES.items():
        # Create binary masks for this specific vessel class
        gt_mask = (gt_multiclass == class_id).astype(np.uint8)
        n_gt_voxels = gt_mask.sum()

        if n_gt_voxels == 0:
            # This vessel is absent in this patient (common for Pcom, Acom, 3rd-A2)
            results[name] = {
                "dice": float("nan"),
                "cldice": float("nan"),
                "betti0_error": float("nan"),
                "present": False,
                "gt_voxels": 0,
            }
            continue

        # Mask the binary prediction to the region where this vessel exists
        # plus a small dilation margin to catch near-misses
        pred_in_region = (pred_binary > 0).astype(np.uint8) * _dilate_mask(gt_mask, radius=3)

        results[name] = {
            "dice": compute_dice(pred_in_region, gt_mask),
            "cldice": compute_cldice(pred_in_region, gt_mask),
            "betti0_error": compute_betti0_error(pred_in_region, gt_mask),
            "present": True,
            "gt_voxels": int(n_gt_voxels),
        }

    return results


def _dilate_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Binary dilation to create an evaluation region around the GT vessel."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(3, 2)  # 18-connectivity
    dilated = binary_dilation(mask, structure=struct, iterations=radius)
    return dilated.astype(np.uint8)


def aggregate_by_group(
    per_class_results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-class metrics into large vs. small vessel groups.

    Averages only over vessel classes that are present (non-NaN).
    """
    groups = {"large": [], "small": []}

    for class_id, name in VESSEL_CLASSES.items():
        if name not in per_class_results:
            continue
        r = per_class_results[name]
        if not r["present"]:
            continue

        group = "large" if class_id in LARGE_VESSELS else "small"
        groups[group].append(r)

    aggregated = {}
    for group_name, entries in groups.items():
        if not entries:
            aggregated[group_name] = {
                "dice": float("nan"),
                "cldice": float("nan"),
                "n_classes": 0,
            }
            continue

        aggregated[group_name] = {
            "dice": float(np.nanmean([e["dice"] for e in entries])),
            "cldice": float(np.nanmean([e["cldice"] for e in entries])),
            "n_classes": len(entries),
        }

    return aggregated


def evaluate_volume_stratified(
    pred_binary: np.ndarray,
    gt_multiclass: np.ndarray,
) -> Dict:
    """
    Full stratified evaluation for one volume.

    Returns:
        {
            "per_class": {vessel_name: {dice, cldice, ...}, ...},
            "grouped": {"large": {dice, cldice, n}, "small": {dice, cldice, n}},
            "delta": {"dice": large - small, "cldice": large - small}
        }

    The "delta" field is the key result: a large positive delta means
    the model does much worse on small vessels than large ones.
    Topology-aware losses should shrink this gap.
    """
    per_class = evaluate_per_class(pred_binary, gt_multiclass)
    grouped = aggregate_by_group(per_class)

    # Compute the gap between large and small vessel performance
    delta = {}
    for metric in ["dice", "cldice"]:
        large_val = grouped.get("large", {}).get(metric, float("nan"))
        small_val = grouped.get("small", {}).get(metric, float("nan"))
        if not (np.isnan(large_val) or np.isnan(small_val)):
            delta[metric] = float(large_val - small_val)
        else:
            delta[metric] = float("nan")

    return {
        "per_class": per_class,
        "grouped": grouped,
        "delta": delta,
    }


def format_stratified_results(results: Dict) -> str:
    """Pretty-print stratified results for logging."""
    lines = []
    lines.append("  Per-class results:")
    lines.append(f"  {'Vessel':<12} {'DSC':>8} {'clDice':>8} {'Present':>8}")
    lines.append(f"  {'-'*40}")

    for name in VESSEL_CLASSES.values():
        if name not in results["per_class"]:
            continue
        r = results["per_class"][name]
        if r["present"]:
            lines.append(
                f"  {name:<12} {r['dice']:>8.4f} {r['cldice']:>8.4f} {'yes':>8}"
            )
        else:
            lines.append(f"  {name:<12} {'n/a':>8} {'n/a':>8} {'no':>8}")

    lines.append("")
    lines.append("  Grouped results:")
    for group, vals in results["grouped"].items():
        lines.append(
            f"  {GROUP_NAMES.get(group, group)}: "
            f"DSC={vals['dice']:.4f}, clDice={vals['cldice']:.4f} "
            f"(n={vals['n_classes']} classes)"
        )

    delta = results["delta"]
    lines.append(f"  Gap (large - small): DSC={delta['dice']:.4f}, clDice={delta['cldice']:.4f}")

    return "\n".join(lines)
