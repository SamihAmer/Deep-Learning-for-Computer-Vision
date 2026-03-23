"""
Loss functions for vessel segmentation.

Three configurations:
  1. dice_ce        : Dice + Cross-Entropy (baseline)
  2. dice_ce_cldice : Dice + CE + soft-clDice (topology-aware)
  3. dice_ce_skeleton : Dice + CE + Skeleton Recall (efficient topology-aware)
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

    Note: This operates on the vessel (foreground) channel probability map.
    For multi-class, apply per-class and average.
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
        probs = F.softmax(logits, dim=1)

        # For binary case, use vessel channel (index 1)
        pred = probs[:, 1:2]  # (B, 1, D, H, W)
        gt = (targets > 0).float().unsqueeze(1)  # (B, 1, D, H, W)

        # Compute soft skeletons
        skel_pred = soft_skeleton(pred, self.num_iters)
        skel_gt = soft_skeleton(gt, self.num_iters)

        # Topology precision and sensitivity
        tprec = ((skel_pred * gt).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        tsens = ((pred * skel_gt).sum() + self.smooth) / (skel_gt.sum() + self.smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice


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
        from skimage.morphology import skeletonize_3d

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
        probs = F.softmax(logits, dim=1)
        pred_vessel = probs[:, 1]  # (B, D, H, W)

        # Compute skeleton of ground truth (on CPU, move back to device)
        with torch.no_grad():
            skel = skeletonize_batch_cpu(targets)  # (B, D, H, W) binary

        # Skeleton recall = how much of the GT skeleton is captured by prediction
        skel_sum = skel.sum() + self.smooth
        recall = (pred_vessel * skel).sum() / skel_sum

        return 1.0 - recall


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
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    if cfg.get("deep_supervision", True):
        return DeepSupervisionLoss(base)
    return base
