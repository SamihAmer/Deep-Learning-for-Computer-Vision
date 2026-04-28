"""
Evaluation metrics for vessel segmentation.

Midterm metrics (topology / overlap):
  - Dice Similarity Coefficient (DSC)
  - Centerline Dice (clDice) via hard skeletonization
  - 95th percentile Hausdorff Distance (HD95)
  - Betti-0 error (connected component mismatch)

Final-report metrics (reconstruction-aware fidelity):
  - 3D-SSIM on the binary prediction vs. binary GT
  - PSNR (in dB) on the binary prediction vs. binary GT
  - Fréchet Feature Distance (F-FID) over 2D slice features

The reconstruction-aware metrics tend to score very high on binary masks
(background dominates), so they are useful primarily for *relative* ranking
across loss configurations, not as absolute fidelity numbers.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label, gaussian_filter
from typing import Dict, List, Optional, Tuple
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


def compute_ssim3d(
    pred: np.ndarray,
    gt: np.ndarray,
    window_sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    3D Structural Similarity Index between two binary volumes.

    Implemented via separable Gaussian smoothing (scipy) on the float-cast
    masks, computing SSIM at every voxel and averaging. Returns a scalar in
    roughly [-1, 1]; higher is better.

    Note: on binary masks SSIM tends to be very high (~0.99) because most
    voxels are background-vs-background = perfect agreement. Use for
    *relative* ranking across loss configurations, not absolute fidelity.
    """
    x = pred.astype(np.float32)
    y = gt.astype(np.float32)

    mu_x = gaussian_filter(x, sigma=window_sigma)
    mu_y = gaussian_filter(y, sigma=window_sigma)
    mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sig_x = gaussian_filter(x * x, sigma=window_sigma) - mu_x_sq
    sig_y = gaussian_filter(y * y, sigma=window_sigma) - mu_y_sq
    sig_xy = gaussian_filter(x * y, sigma=window_sigma) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sig_x + sig_y + C2)
    return float(np.mean(num / den))


def compute_psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak signal-to-noise ratio (dB) between prediction and GT.

    For perfect agreement (MSE → 0) we cap at 100 dB so the metric stays
    finite when summarized across volumes.
    """
    mse = float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


# ──────────────────────────────────────────────────────────────────────────────
# Fréchet Feature Distance (F-FID): set-level, called once over the val set,
# not per-volume. Lives in metrics.py for cohesion but is invoked from
# evaluate.py — see compute_feature_fid() below.
# ──────────────────────────────────────────────────────────────────────────────

def _slice_features_vgg(
    volumes: List[np.ndarray],
    slice_stride: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract VGG-16 relu3_3 features over axial slices of each volume.

    Returns: (N_total_slices, 256) feature matrix on CPU.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.models as tv_models

    try:
        weights = tv_models.VGG16_Weights.IMAGENET1K_V1
        vgg = tv_models.vgg16(weights=weights).features[:16]   # through relu3_3
    except Exception:
        vgg = tv_models.vgg16(pretrained=True).features[:16]
    vgg = vgg.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    with torch.no_grad():
        for vol in volumes:
            v = vol.astype(np.float32)
            D = v.shape[0]
            idx = list(range(0, D, slice_stride)) or [D // 2]
            slices = v[idx]                                       # (n, H, W)
            x = torch.from_numpy(slices).to(device).unsqueeze(1)   # (n, 1, H, W)
            x = x.expand(-1, 3, -1, -1)
            x = (x - mean) / std
            f = vgg(x)                                             # (n, 256, h, w)
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)             # (n, 256)
            feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 256))


def _frechet_distance(mu1: np.ndarray, sig1: np.ndarray,
                      mu2: np.ndarray, sig2: np.ndarray,
                      eps: float = 1e-6) -> float:
    """
    Closed-form Fréchet distance between two multivariate Gaussians.

    Standard FID-stable implementation: regularize each covariance with
    `eps*I` before sqrtm to avoid numerical issues on near-singular
    matrices (which is the typical regime for small N).
    """
    from scipy import linalg
    diff = mu1 - mu2
    offset = np.eye(sig1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((sig1 + offset).dot(sig2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean)
    return float(fid)


def compute_feature_fid(
    pred_volumes: List[np.ndarray],
    gt_volumes: List[np.ndarray],
    slice_stride: int = 16,
    device: str = "cuda",
) -> float:
    """
    Fréchet Feature Distance over a set of prediction/GT volume pairs.

    NOT a converged Inception-FID. We compute VGG-16 relu3_3 features over
    axial slices, fit a Gaussian to each set, and return the Fréchet
    distance between them. With N=25 cases this is statistically noisy
    and should be reported as 'F-FID' with an explicit caveat — useful for
    ranking, not as an absolute fidelity number.
    """
    pred_feats = _slice_features_vgg(pred_volumes, slice_stride, device)
    gt_feats = _slice_features_vgg(gt_volumes, slice_stride, device)

    if pred_feats.shape[0] < 2 or gt_feats.shape[0] < 2:
        return float("nan")

    mu_p, sig_p = pred_feats.mean(0), np.cov(pred_feats, rowvar=False)
    mu_g, sig_g = gt_feats.mean(0), np.cov(gt_feats, rowvar=False)
    return _frechet_distance(mu_p, sig_p, mu_g, sig_g)


def evaluate_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: tuple = (1, 1, 1),
    quick: bool = False,
    extended: bool = False,
) -> Dict[str, float]:
    """
    Run metrics on a single predicted/ground-truth volume pair.

    Args:
        pred: binary prediction mask (D, H, W)
        gt: binary ground truth mask (D, H, W)
        voxel_spacing: (D, H, W) spacing in mm for HD95
        quick: if True, only compute Dice (skips clDice/HD95/Betti).
        extended: if True, also compute 3D-SSIM and PSNR. F-FID is
                  set-level and computed separately by the caller.

    Returns:
        dict with keys: "dice" (always), plus "cldice", "hd95",
        "betti0_error" if not quick, plus "ssim3d", "psnr" if extended.
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if quick:
        return {"dice": compute_dice(pred_bin, gt_bin)}

    # Run topology metrics concurrently (independent and CPU-bound)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_dice = pool.submit(compute_dice, pred_bin, gt_bin)
        fut_cldice = pool.submit(compute_cldice, pred_bin, gt_bin)
        fut_hd95 = pool.submit(compute_hd95, pred_bin, gt_bin, voxel_spacing)
        fut_betti = pool.submit(compute_betti0_error, pred_bin, gt_bin)

    out = {
        "dice": fut_dice.result(),
        "cldice": fut_cldice.result(),
        "hd95": fut_hd95.result(),
        "betti0_error": fut_betti.result(),
    }

    if extended:
        out["ssim3d"] = compute_ssim3d(pred_bin, gt_bin)
        out["psnr"] = compute_psnr(pred_bin, gt_bin)

    return out


def evaluate_volume_multiclass(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: Dict[int, str],
    voxel_spacing: tuple = (1, 1, 1),
    quick: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Per-class metrics for multi-class segmentation (e.g. artery/vein).

    Args:
        pred: integer-labeled prediction (D, H, W)
        gt:   integer-labeled ground truth (D, H, W)
        class_ids: e.g. {1: "artery", 2: "vein"} — keys are label ids, values are names

    Returns:
        {class_name: {metric: value, ...}, ..., "overall": {...}}
        "overall" collapses all foreground to binary vessel.
    """
    out = {}
    for cid, name in class_ids.items():
        p_c = (pred == cid).astype(np.uint8)
        g_c = (gt == cid).astype(np.uint8)
        out[name] = evaluate_volume(p_c, g_c, voxel_spacing, quick=quick)
    out["overall"] = evaluate_volume(
        (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8),
        voxel_spacing, quick=quick,
    )
    return out
