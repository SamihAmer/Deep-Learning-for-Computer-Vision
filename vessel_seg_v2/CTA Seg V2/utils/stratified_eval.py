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


# ── TopBrain / TopCoW class mapping ──────────────────────────────────────────
# IDs 1-12 and 15 are shared between TopCoW 2024 and TopBrain 2025.
# IDs 13-14, 16-40 are TopBrain extensions (distal branches, posterior fossa,
# small arteries, and venous sinuses).
# Label IDs match the TopCoW 2024 README and TopBrain ITK-Snap labelmap.

VESSEL_CLASSES = {
    # ── Circle of Willis (TopCoW 2024, retained in TopBrain) ──
    1:  "BA",           # Basilar artery
    2:  "R-PCA",        # Right P1/P2 posterior cerebral artery
    3:  "L-PCA",        # Left P1/P2 posterior cerebral artery
    4:  "R-ICA",        # Right internal carotid artery
    5:  "R-MCA",        # Right M1 middle cerebral artery
    6:  "L-ICA",        # Left internal carotid artery
    7:  "L-MCA",        # Left M1 middle cerebral artery
    8:  "R-Pcom",       # Right posterior communicating artery
    9:  "L-Pcom",       # Left posterior communicating artery
    10: "Acom",         # Anterior communicating artery
    11: "R-ACA",        # Right A1/A2 anterior cerebral artery
    12: "L-ACA",        # Left A1/A2 anterior cerebral artery
    15: "3rd-A2",       # Third A2 segment (variant)
    # ── TopBrain extensions: distal cerebral arteries ──
    13: "R-A3",         # Right A3 anterior cerebral artery
    14: "L-A3",         # Left A3 anterior cerebral artery
    16: "3rd-A3",       # Third A3 anterior cerebral artery
    17: "R-M2",         # Right M2 middle cerebral artery
    18: "R-M3",         # Right M3 middle cerebral artery
    19: "L-M2",         # Left M2 middle cerebral artery
    20: "L-M3",         # Left M3 middle cerebral artery
    21: "R-P3P4",       # Right P3/P4 posterior cerebral artery
    22: "L-P3P4",       # Left P3/P4 posterior cerebral artery
    # ── TopBrain extensions: posterior fossa ──
    23: "R-VA",         # Right vertebral artery
    24: "L-VA",         # Left vertebral artery
    25: "R-SCA",        # Right superior cerebellar artery
    26: "L-SCA",        # Left superior cerebellar artery
    27: "R-AICA",       # Right anterior inferior cerebellar artery
    28: "L-AICA",       # Left anterior inferior cerebellar artery
    29: "R-PICA",       # Right posterior inferior cerebellar artery
    30: "L-PICA",       # Left posterior inferior cerebellar artery
    # ── TopBrain extensions: small arteries ──
    31: "R-AChA",       # Right anterior choroidal artery
    32: "L-AChA",       # Left anterior choroidal artery
    33: "R-OA",         # Right ophthalmic artery
    34: "L-OA",         # Left ophthalmic artery
    # ── TopBrain extensions: venous sinuses (CTA only) ──
    35: "VoG",          # Vein of Galen
    36: "StS",          # Straight sinus
    37: "ICVs",         # Internal cerebral veins
    38: "R-BVR",        # Right basal vein of Rosenthal
    39: "L-BVR",        # Left basal vein of Rosenthal
    40: "SSS",          # Superior sagittal sinus
}

# Clinical grouping
LARGE_VESSELS = {1, 2, 3, 4, 5, 6, 7, 11, 12}     # BA, PCA, ICA, M1, A1/A2
SMALL_VESSELS = {8, 9, 10, 15}                       # Pcom, Acom, 3rd-A2
DISTAL_VESSELS = {13, 14, 16, 17, 18, 19, 20, 21, 22}  # A3, M2, M3, P3/P4
POSTERIOR_FOSSA = {23, 24, 25, 26, 27, 28, 29, 30}  # VA, SCA, AICA, PICA
SMALL_ARTERIES = {31, 32, 33, 34}                    # AChA, OA
VENOUS = {35, 36, 37, 38, 39, 40}                    # Sinuses and veins

GROUP_NAMES = {
    "large": "Large CoW arteries (BA, ICA, M1, PCA, ACA)",
    "small": "Communicating arteries (Acom, Pcom, 3rd-A2)",
    "distal": "Distal branches (A3, M2, M3, P3/P4)",
    "posterior_fossa": "Posterior fossa (VA, SCA, AICA, PICA)",
    "small_arteries": "Small arteries (AChA, OA)",
    "venous": "Venous sinuses (VoG, StS, ICVs, BVR, SSS)",
}


def _eval_one_class(class_id, name, pred_binary, gt_multiclass):
    """Evaluate a single vessel class. Used for parallel dispatch."""
    gt_mask = (gt_multiclass == class_id).astype(np.uint8)
    n_gt_voxels = gt_mask.sum()

    if n_gt_voxels == 0:
        return name, {
            "dice": float("nan"),
            "cldice": float("nan"),
            "betti0_error": float("nan"),
            "present": False,
            "gt_voxels": 0,
        }

    pred_in_region = (pred_binary > 0).astype(np.uint8) * _dilate_mask(gt_mask, radius=3)

    return name, {
        "dice": compute_dice(pred_in_region, gt_mask),
        "cldice": compute_cldice(pred_in_region, gt_mask),
        "betti0_error": compute_betti0_error(pred_in_region, gt_mask),
        "present": True,
        "gt_voxels": int(n_gt_voxels),
    }


def evaluate_per_class(
    pred_binary: np.ndarray,
    gt_multiclass: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each vessel class separately.
    Classes are evaluated in parallel using a thread pool.

    Args:
        pred_binary: (D, H, W) binary prediction (0 = background, 1 = vessel)
        gt_multiclass: (D, H, W) integer labels with TopCoW class IDs (0 = bg)

    Returns:
        Dict mapping vessel name to {"dice": float, "cldice": float, "present": bool}
        "present" indicates whether this vessel class exists in the ground truth
        (communicating arteries are absent in some patients).
    """
    from concurrent.futures import ThreadPoolExecutor

    # Only evaluate classes that exist in this volume's GT
    present_ids = set(np.unique(gt_multiclass).astype(int)) - {0}

    results = {}

    # Mark absent classes immediately (no computation needed)
    for class_id, name in VESSEL_CLASSES.items():
        if class_id not in present_ids:
            results[name] = {
                "dice": float("nan"),
                "cldice": float("nan"),
                "betti0_error": float("nan"),
                "present": False,
                "gt_voxels": 0,
            }

    # Evaluate present classes in parallel
    present_classes = [(cid, name) for cid, name in VESSEL_CLASSES.items() if cid in present_ids]

    with ThreadPoolExecutor(max_workers=min(8, len(present_classes) or 1)) as pool:
        futures = [
            pool.submit(_eval_one_class, cid, name, pred_binary, gt_multiclass)
            for cid, name in present_classes
        ]
        for fut in futures:
            name, result = fut.result()
            results[name] = result

    return results


def _dilate_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Binary dilation to create an evaluation region around the GT vessel."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(3, 2)  # 18-connectivity
    dilated = binary_dilation(mask, structure=struct, iterations=radius)
    return dilated.astype(np.uint8)


def _classify_vessel(class_id: int) -> str:
    """Map a vessel class ID to its clinical group name."""
    if class_id in LARGE_VESSELS:
        return "large"
    elif class_id in SMALL_VESSELS:
        return "small"
    elif class_id in DISTAL_VESSELS:
        return "distal"
    elif class_id in POSTERIOR_FOSSA:
        return "posterior_fossa"
    elif class_id in SMALL_ARTERIES:
        return "small_arteries"
    elif class_id in VENOUS:
        return "venous"
    return "other"


def aggregate_by_group(
    per_class_results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-class metrics into vessel groups.

    Averages only over vessel classes that are present (non-NaN).
    """
    groups = {name: [] for name in GROUP_NAMES}

    for class_id, name in VESSEL_CLASSES.items():
        if name not in per_class_results:
            continue
        r = per_class_results[name]
        if not r["present"]:
            continue

        group = _classify_vessel(class_id)
        if group in groups:
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

    # Compute the gap between large and each other vessel group
    delta = {}
    large_vals = grouped.get("large", {})
    for group_name in GROUP_NAMES:
        if group_name == "large":
            continue
        for metric in ["dice", "cldice"]:
            large_val = large_vals.get(metric, float("nan"))
            other_val = grouped.get(group_name, {}).get(metric, float("nan"))
            key = f"{metric}_large_vs_{group_name}"
            if not (np.isnan(large_val) or np.isnan(other_val)):
                delta[key] = float(large_val - other_val)
            else:
                delta[key] = float("nan")

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
        if vals["n_classes"] == 0:
            continue
        lines.append(
            f"  {GROUP_NAMES.get(group, group)}: "
            f"DSC={vals['dice']:.4f}, clDice={vals['cldice']:.4f} "
            f"(n={vals['n_classes']} classes)"
        )

    lines.append("")
    lines.append("  Gaps (large vs other groups):")
    delta = results["delta"]
    for key, val in delta.items():
        if not np.isnan(val):
            lines.append(f"  {key}: {val:.4f}")

    return "\n".join(lines)
