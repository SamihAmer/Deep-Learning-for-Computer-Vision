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
from concurrent.futures import ThreadPoolExecutor

try:
    from skimage.morphology import skeletonize_3d
    HAS_SKIMAGE = True
except ImportError:
    try:
        from skimage.morphology import skeletonize
        skeletonize_3d = skeletonize  # newer skimage merged skeletonize_3d into skeletonize
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False


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
    if not HAS_SKIMAGE:
        print("Warning: skimage not available, returning DSC as clDice fallback")
        return compute_dice(pred, gt)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    # Parallelize the two independent skeletonizations
    pred_u8 = pred.astype(np.uint8)
    gt_u8 = gt.astype(np.uint8)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_pred = pool.submit(skeletonize_3d, pred_u8)
        fut_gt = pool.submit(skeletonize_3d, gt_u8)
        skel_pred = fut_pred.result().astype(bool)
        skel_gt = fut_gt.result().astype(bool)

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

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    # Parallelize the two independent erosion + distance transform operations
    def _surface_dt(mask, spacing):
        border = mask & ~binary_erosion(mask)
        if border.sum() == 0:
            return border, None
        dt = distance_transform_edt(~border, sampling=spacing)
        return border, dt

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_pred = pool.submit(_surface_dt, pred_bool, voxel_spacing)
        fut_gt = pool.submit(_surface_dt, gt_bool, voxel_spacing)
        pred_border, dt_pred = fut_pred.result()
        gt_border, dt_gt = fut_gt.result()

    if dt_pred is None or dt_gt is None:
        return float("inf")

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
    quick: bool = False,
) -> Dict[str, float]:
    """
    Run metrics on a single predicted/ground-truth volume pair.

    Args:
        pred: binary prediction mask (D, H, W)
        gt: binary ground truth mask (D, H, W)
        voxel_spacing: (D, H, W) spacing in mm for HD95
        quick: if True, only compute Dice (skips clDice/HD95/Betti which are
               slow due to skeletonize_3d and distance_transform_edt).
               Use during training; full metrics for final evaluation.

    Returns:
        dict with keys: "dice" (always), plus "cldice", "hd95", "betti0_error" if not quick
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if quick:
        return {"dice": compute_dice(pred_bin, gt_bin)}

    # Run all four metrics concurrently (they're independent and CPU-bound)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_dice = pool.submit(compute_dice, pred_bin, gt_bin)
        fut_cldice = pool.submit(compute_cldice, pred_bin, gt_bin)
        fut_hd95 = pool.submit(compute_hd95, pred_bin, gt_bin, voxel_spacing)
        fut_betti = pool.submit(compute_betti0_error, pred_bin, gt_bin)

    return {
        "dice": fut_dice.result(),
        "cldice": fut_cldice.result(),
        "hd95": fut_hd95.result(),
        "betti0_error": fut_betti.result(),
    }
