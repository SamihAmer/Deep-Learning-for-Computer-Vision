"""
Loss functions for vessel segmentation.

Midterm configurations (topology-aware family):
  1. dice_ce            : Dice + Cross-Entropy (baseline)
  2. dice_ce_cldice     : Dice + CE + soft-clDice
  3. dice_ce_skeleton   : Dice + CE + Skeleton Recall

Final-report configurations (reconstruction-aware family + combination):
  4. dice_ce_ssim         : Dice + CE + 3D Structural Similarity
  5. dice_ce_mse_dt       : Dice + CE + MSE on Gaussian-decayed distance-transform target
  6. dice_ce_perceptual   : Dice + CE + 2D slice-wise VGG-16 perceptual loss
  7. dice_ce_cldice_ssim  : Dice + CE + clDice + SSIM (combination experiment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DICE + CROSS-ENTROPY (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftDiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-class segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) integer class labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # one-hot encode targets: (B, C, D, H, W)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        # compute per-class Dice, skip background (class 0)
        dice_sum = 0.0
        for c in range(1, num_classes):
            p = probs[:, c]
            g = targets_oh[:, c]
            intersection = (p * g).sum()
            union = p.sum() + g.sum()
            dice_sum += (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_sum / (num_classes - 1)


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss (nnU-Net default)."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.ce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SOFT-CLDICE (topology-preserving)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reference: Shit et al., "clDice -- A Novel Topology-Preserving Loss Function
# for Tubular Structure Segmentation," CVPR 2021.
#
# The key idea: compute Dice on the *skeletons* of prediction and ground truth,
# using a differentiable soft-skeletonization via iterative min/max pooling.

def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological erosion via 3D min-pooling."""
    # max-pool the negative is equivalent to min-pool
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological dilation via 3D max-pooling."""
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    """Morphological opening = erosion then dilation."""
    return soft_dilate(soft_erode(x))


def soft_skeleton(x: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """
    Differentiable soft-skeletonization.

    Iteratively peels layers from the foreground using morphological opening,
    accumulating the topological skeleton at each scale.
    """
    skeleton = F.relu(x - soft_open(x))
    for _ in range(num_iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skeleton = skeleton + delta
    return skeleton


class SoftClDiceLoss(nn.Module):
    """
    Soft centerline Dice loss.

    Computes:
        tprec = |skel(pred) * gt| / |skel(pred)|   (topology precision)
        tsens = |pred * skel(gt)| / |skel(gt)|      (topology sensitivity)
        clDice = 2 * tprec * tsens / (tprec + tsens)
        loss = 1 - clDice

    Applied per foreground class and averaged — supports both binary (C=2)
    and multi-class (e.g. C=3 for background/artery/vein).
    """

    def __init__(self, num_iters: int = 10, smooth: float = 1e-5):
        super().__init__()
        self.num_iters = num_iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, D, H, W) integer labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        total = 0.0
        for c in range(1, num_classes):
            pred = probs[:, c:c + 1]                             # (B, 1, D, H, W)
            gt = (targets == c).float().unsqueeze(1)             # (B, 1, D, H, W)

            skel_pred = soft_skeleton(pred, self.num_iters)
            skel_gt = soft_skeleton(gt, self.num_iters)

            tprec = ((skel_pred * gt).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
            tsens = ((pred * skel_gt).sum() + self.smooth) / (skel_gt.sum() + self.smooth)

            cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
            total = total + (1.0 - cl_dice)

        return total / (num_classes - 1)


class DiceCEClDiceLoss(nn.Module):
    """Dice + CE + clDice combined loss."""

    def __init__(self, alpha: float = 0.5, cldice_iters: int = 10):
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.cldice = SoftClDiceLoss(num_iters=cldice_iters)
        self.alpha = alpha  # weight for topology term

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.dice_ce(logits, targets)
        topo = self.cldice(logits, targets)
        return (1 - self.alpha) * base + self.alpha * topo


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SKELETON RECALL LOSS (efficient topology-preserving)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reference: Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving
# and Resource Efficient Segmentation of Thin Tubular Structures," ECCV 2024.
#
# Key insight: instead of differentiable skeletonization on GPU (expensive, noisy
# in 3D), precompute binary skeletons on CPU and use them as attention masks for
# a weighted cross-entropy term that focuses on centerline voxels.

from scipy.ndimage import distance_transform_edt

def skeletonize_batch_cpu(targets: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D skeletons of binary targets on CPU.

    Uses morphological thinning via skimage. Falls back to distance-transform
    medial axis if skimage is unavailable.

    Args:
        targets: (B, D, H, W) integer labels on GPU/CPU

    Returns:
        skeleton: (B, D, H, W) binary skeleton tensor (same device as input)
    """
    device = targets.device
    targets_np = targets.cpu().numpy()
    skeletons = np.zeros_like(targets_np, dtype=np.float32)

    try:
        try:
            from skimage.morphology import skeletonize_3d
        except ImportError:
            from skimage.morphology import skeletonize as skeletonize_3d

        for b in range(targets_np.shape[0]):
            binary = (targets_np[b] > 0).astype(np.uint8)
            if binary.sum() > 0:
                skeletons[b] = skeletonize_3d(binary).astype(np.float32)
    except ImportError:
        # Fallback: medial axis approximation via distance transform
        # Less accurate but has no extra dependencies
        print("Warning: skimage not found, using distance transform skeleton approximation")
        for b in range(targets_np.shape[0]):
            binary = (targets_np[b] > 0).astype(np.uint8)
            if binary.sum() > 0:
                dt = distance_transform_edt(binary)
                # Approximate skeleton as local maxima of distance transform
                from scipy.ndimage import maximum_filter
                local_max = maximum_filter(dt, size=3)
                skeletons[b] = ((dt == local_max) & (binary > 0)).astype(np.float32)

    return torch.from_numpy(skeletons).to(device)


class SkeletonRecallLoss(nn.Module):
    """
    Skeleton Recall: weighted CE focusing on ground-truth skeleton voxels.

    The loss upweights centerline voxels so the network is penalized heavily
    for missing the core connectivity of the vessel tree.

    For multi-class (e.g. artery + vein), skeletonizes each class separately
    and averages the per-class recall.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, D, H, W) integer labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        total = 0.0
        for c in range(1, num_classes):
            pred_c = probs[:, c]                                 # (B, D, H, W)
            targets_c = (targets == c).long()                    # (B, D, H, W) binary
            with torch.no_grad():
                skel = skeletonize_batch_cpu(targets_c)
            skel_sum = skel.sum() + self.smooth
            recall = (pred_c * skel).sum() / skel_sum
            total = total + (1.0 - recall)

        return total / (num_classes - 1)


class DiceCESkeletonLoss(nn.Module):
    """Dice + CE + Skeleton Recall combined loss."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.skel_recall = SkeletonRecallLoss()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.dice_ce(logits, targets)
        topo = self.skel_recall(logits, targets)
        return (1 - self.alpha) * base + self.alpha * topo


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SSIM3D (reconstruction-aware, structural similarity)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reference: Wang et al., "Image quality assessment: from error visibility to
# structural similarity," IEEE TIP 2004; Qin et al., "BASNet: Boundary-Aware
# Salient Object Detection," CVPR 2019 (BCE+IoU+SSIM template for binary masks).
#
# We compute 3D-SSIM between the softmax foreground probability and the
# one-hot ground truth using a separable Gaussian window. Loss = 1 − SSIM.

class SSIM3DLoss(nn.Module):
    """3D Structural Similarity loss on the softmax foreground probability."""

    def __init__(self, window_size: int = 7, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.register_buffer("window", self._gaussian3d(window_size, sigma))

    @staticmethod
    def _gaussian3d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        return kernel[None, None]                        # (1, 1, S, S, S)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Run SSIM math in fp32 even when the surrounding model is under
        # autocast (fp16/bf16). The stabilizers C1=1e-4, C2=9e-4 sit at
        # the edge of fp16's normal range, and (mu_xy * mu_xy) for
        # background voxels with prob ~0 can underflow → NaN. Also
        # co-locate the Gaussian window with the input device defensively.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            y = y.float()
            window = self.window.to(device=x.device, dtype=torch.float32)
            pad = self.window_size // 2

            mu_x = F.conv3d(x, window, padding=pad)
            mu_y = F.conv3d(y, window, padding=pad)
            mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

            sig_x = F.conv3d(x * x, window, padding=pad) - mu_x_sq
            sig_y = F.conv3d(y * y, window, padding=pad) - mu_y_sq
            sig_xy = F.conv3d(x * y, window, padding=pad) - mu_xy

            num = (2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)
            den = (mu_x_sq + mu_y_sq + self.C1) * (sig_x + sig_y + self.C2)
            return (num / den).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        total = 0.0
        for c in range(1, num_classes):
            p = probs[:, c:c + 1]
            g = (targets == c).float().unsqueeze(1)
            total = total + (1.0 - self._ssim(p, g))
        return total / (num_classes - 1)


class DiceCESSIMLoss(nn.Module):
    """Dice + CE + 3D-SSIM combined loss."""

    def __init__(self, alpha: float = 0.5, window_size: int = 7):
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.ssim = SSIM3DLoss(window_size=window_size)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * self.dice_ce(logits, targets) \
             + self.alpha * self.ssim(logits, targets)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MSE-ON-DISTANCE-TRANSFORM (reconstruction-aware, regression-style)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reference: Kervadec et al., "Boundary loss for highly unbalanced segmentation,"
# MedIA 2021; Karimi & Salcudean, "Reducing the Hausdorff Distance in Medical
# Image Segmentation with CNNs," IEEE TMI 2020.
#
# Plain MSE on a binary mask collapses on imbalanced data (vessels are ~1-3%
# of voxels), so we regress the prediction against a Gaussian-decayed
# distance-transform target: 1 inside the GT, decaying smoothly outside.

class MSEDistanceTransformLoss(nn.Module):
    """MSE between softmax foreground probability and a smooth DT target."""

    def __init__(self, sigma: float = 5.0):
        super().__init__()
        self.sigma = sigma

    @torch.no_grad()
    def _make_target(self, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B, D, H, W) integer labels
        device = targets.device
        targets_np = targets.detach().cpu().numpy()
        soft = np.zeros_like(targets_np, dtype=np.float32)
        for b in range(targets_np.shape[0]):
            binary = (targets_np[b] > 0).astype(np.uint8)
            if binary.sum() == 0:
                continue
            # EDT of the background → distance to nearest foreground voxel
            dt = distance_transform_edt(1 - binary)
            soft[b] = np.exp(-dt / self.sigma).astype(np.float32)
        return torch.from_numpy(soft).to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        # Foreground probability: collapse all non-background classes
        fg_prob = probs[:, 1:].sum(dim=1)               # (B, D, H, W)
        soft_target = self._make_target(targets).to(fg_prob.dtype)
        return F.mse_loss(fg_prob, soft_target)


class DiceCEMSEDistLoss(nn.Module):
    """Dice + CE + MSE-on-distance-transform combined loss."""

    def __init__(self, alpha: float = 0.5, sigma: float = 5.0):
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.mse_dt = MSEDistanceTransformLoss(sigma=sigma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * self.dice_ce(logits, targets) \
             + self.alpha * self.mse_dt(logits, targets)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PERCEPTUAL LOSS (2D slice-wise VGG-16, reconstruction-aware)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reference: Mosinska et al., "Beyond the Pixel-Wise Loss for Topology-Aware
# Delineation," CVPR 2018 (VGG perceptual for thin tubular structures);
# Johnson et al., "Perceptual Losses for Real-Time Style Transfer," ECCV 2016.
#
# Implementation choice: 2D slice-wise VGG-16 (ImageNet-pretrained) on
# subsampled axial slices of the soft probability map. We replicate the
# single-channel slice to 3 channels and ImageNet-normalize. We do NOT
# modulate by the CTA input (a more advanced variant) because it would
# require threading the input image through the loss interface; direct
# probability-map perceptual is the BASNet-style baseline and is well
# documented in the literature.

class PerceptualLoss2DSlice(nn.Module):
    """2D slice-wise VGG-16 perceptual loss on the foreground probability."""

    def __init__(self, layer_indices=(8, 15), slice_stride: int = 16):
        super().__init__()
        # Lazy import so torchvision is only required when this loss is used
        import torchvision.models as tv_models
        try:
            weights = tv_models.VGG16_Weights.IMAGENET1K_V1
            vgg = tv_models.vgg16(weights=weights).features
        except Exception:
            # Older torchvision API
            vgg = tv_models.vgg16(pretrained=True).features

        # Truncate VGG at each requested layer index
        self.feature_blocks = nn.ModuleList(
            [vgg[: idx + 1] for idx in layer_indices]
        )
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # ImageNet normalization buffers
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.slice_stride = slice_stride

    def _to_vgg_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W) → (N, 3, H, W) ImageNet-normalized
        x3 = x.expand(-1, 3, -1, -1)
        return (x3 - self.mean.to(x3.dtype)) / self.std.to(x3.dtype)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        fg_prob = probs[:, 1:].sum(dim=1)               # (B, D, H, W)
        gt_binary = (targets > 0).float()                # (B, D, H, W)

        B, D, H, W = fg_prob.shape
        # Subsample axial slices to keep memory and compute manageable
        idx = torch.arange(0, D, self.slice_stride, device=fg_prob.device)
        if idx.numel() == 0:
            idx = torch.tensor([D // 2], device=fg_prob.device)

        p_slices = fg_prob[:, idx].reshape(-1, 1, H, W)
        g_slices = gt_binary[:, idx].reshape(-1, 1, H, W)

        # Run VGG in fp32 to avoid bf16/fp16 precision issues with frozen weights
        with torch.amp.autocast("cuda", enabled=False):
            p_in = self._to_vgg_input(p_slices.float())
            g_in = self._to_vgg_input(g_slices.float())

            loss = 0.0
            for block in self.feature_blocks:
                with torch.no_grad():
                    g_feats = block(g_in)
                p_feats = block(p_in)
                loss = loss + F.mse_loss(p_feats, g_feats)
            loss = loss / len(self.feature_blocks)

        return loss


class DiceCEPerceptualLoss(nn.Module):
    """Dice + CE + 2D slice-wise VGG perceptual combined loss."""

    def __init__(self, alpha: float = 0.5, slice_stride: int = 16):
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.perceptual = PerceptualLoss2DSlice(slice_stride=slice_stride)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * self.dice_ce(logits, targets) \
             + self.alpha * self.perceptual(logits, targets)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMBINATION: Dice + CE + clDice + SSIM
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests whether topology-aware (clDice) and reconstruction-aware (SSIM)
# supervision combine constructively. Default weights: 0.5 / 0.25 / 0.25.

class DiceCEClDiceSSIMLoss(nn.Module):
    """Quad-loss: Dice + CE + clDice + 3D-SSIM."""

    def __init__(self, dice_ce_w: float = 0.5,
                 cldice_w: float = 0.25, ssim_w: float = 0.25,
                 cldice_iters: int = 10, ssim_window_size: int = 7):
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.cldice = SoftClDiceLoss(num_iters=cldice_iters)
        self.ssim = SSIM3DLoss(window_size=ssim_window_size)
        self.dice_ce_w = dice_ce_w
        self.cldice_w = cldice_w
        self.ssim_w = ssim_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_ce_w * self.dice_ce(logits, targets)
            + self.cldice_w * self.cldice(logits, targets)
            + self.ssim_w * self.ssim(logits, targets)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP SUPERVISION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSupervisionLoss(nn.Module):
    """
    Applies a base loss at multiple decoder scales with exponentially
    decaying weights (nnU-Net convention).

    The highest-resolution output gets weight 1.0, and each subsequent
    (lower-res) output gets weight *= 0.5.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self, outputs: List[torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: list of logit tensors, index 0 = full resolution
            targets: (B, D, H, W) at full resolution
        """
        total_loss = 0.0
        weight = 1.0
        weight_sum = 0.0

        for out in outputs:
            # Downsample targets to match this output's spatial size
            if out.shape[2:] != targets.shape[1:]:
                t = F.interpolate(
                    targets.float().unsqueeze(1),
                    size=out.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()
            else:
                t = targets

            total_loss += weight * self.base_loss(out, t)
            weight_sum += weight
            weight *= 0.5

        return total_loss / weight_sum


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    loss_name = cfg["loss"]

    if loss_name == "dice_ce":
        base = DiceCELoss()
    elif loss_name == "dice_ce_cldice":
        base = DiceCEClDiceLoss(alpha=cfg.get("cldice_alpha", 0.5))
    elif loss_name == "dice_ce_skeleton":
        base = DiceCESkeletonLoss(alpha=cfg.get("skeleton_recall_alpha", 0.5))
    elif loss_name == "dice_ce_ssim":
        base = DiceCESSIMLoss(
            alpha=cfg.get("ssim_alpha", 0.5),
            window_size=cfg.get("ssim_window_size", 7),
        )
    elif loss_name == "dice_ce_mse_dt":
        base = DiceCEMSEDistLoss(
            alpha=cfg.get("mse_dt_alpha", 0.5),
            sigma=cfg.get("mse_dt_sigma", 5.0),
        )
    elif loss_name == "dice_ce_perceptual":
        base = DiceCEPerceptualLoss(
            alpha=cfg.get("perceptual_alpha", 0.5),
            slice_stride=cfg.get("perceptual_slice_stride", 16),
        )
    elif loss_name == "dice_ce_cldice_ssim":
        base = DiceCEClDiceSSIMLoss(
            dice_ce_w=cfg.get("combo_dice_ce_w", 0.5),
            cldice_w=cfg.get("combo_cldice_w", 0.25),
            ssim_w=cfg.get("combo_ssim_w", 0.25),
            ssim_window_size=cfg.get("ssim_window_size", 7),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    if cfg.get("deep_supervision", True):
        return DeepSupervisionLoss(base)
    return base
