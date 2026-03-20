"""
Evaluation metrics for vessel segmentation.

Computes:
  - Dice Similarity Coefficient (DSC)
  - Centerline Dice (clDice) via hard skeletonization
  - 95th percentile Hausdorff Distance (HD95)
  - Betti-0 error (connected component mismatch)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label
from typing import Dict


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard Dice coefficient between two binary masks."""
    intersection = (pred * gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0  # both empty = perfect
    return float(2.0 * intersection / (pred.sum() + gt.sum()))


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Centerline Dice using hard skeletonization.

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    where:
        Tprec = |skel(pred) AND gt| / |skel(pred)|
        Tsens = |pred AND skel(gt)| / |skel(gt)|
    """
    try:
        from skimage.morphology import skeletonize_3d
    except ImportError:
        print("Warning: skimage not available, returning DSC as clDice fallback")
        return compute_dice(pred, gt)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    skel_pred = skeletonize_3d(pred.astype(np.uint8)).astype(bool)
    skel_gt = skeletonize_3d(gt.astype(np.uint8)).astype(bool)

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    tprec = (skel_pred & gt.astype(bool)).sum() / skel_pred.sum()
    tsens = (pred.astype(bool) & skel_gt).sum() / skel_gt.sum()

    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def compute_hd95(pred: np.ndarray, gt: np.ndarray, voxel_spacing: tuple = (1, 1, 1)) -> float:
    """
    95th percentile Hausdorff distance.

    Returns distance in mm (or voxels if spacing is (1,1,1)).
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    # Surface voxels (boundary detection via erosion)
    from scipy.ndimage import binary_erosion

    pred_border = pred.astype(bool) & ~binary_erosion(pred.astype(bool))
    gt_border = gt.astype(bool) & ~binary_erosion(gt.astype(bool))

    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return float("inf")

    # Distance transforms
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)

    # Surface-to-surface distances
    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_betti0_error(pred: np.ndarray, gt: np.ndarray) -> int:
    """
    Betti-0 error = |num_components(pred) - num_components(gt)|.

    Betti-0 counts connected components. A perfect vessel segmentation should
    have the same number of connected components as ground truth.
    """
    _, n_pred = nd_label(pred.astype(bool))
    _, n_gt = nd_label(gt.astype(bool))
    return abs(int(n_pred) - int(n_gt))


def evaluate_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: tuple = (1, 1, 1),
) -> Dict[str, float]:
    """
    Run all metrics on a single predicted/ground-truth volume pair.

    Args:
        pred: binary prediction mask (D, H, W)
        gt: binary ground truth mask (D, H, W)
        voxel_spacing: (D, H, W) spacing in mm for HD95

    Returns:
        dict with keys: "dice", "cldice", "hd95", "betti0_error"
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    return {
        "dice": compute_dice(pred_bin, gt_bin),
        "cldice": compute_cldice(pred_bin, gt_bin),
        "hd95": compute_hd95(pred_bin, gt_bin, voxel_spacing),
        "betti0_error": compute_betti0_error(pred_bin, gt_bin),
    }
