"""
Sliding window inference for 3D volumes.

At test time, we can't feed the full volume through the network.
This module tiles the volume into overlapping patches, runs each
through the model, and stitches predictions using Gaussian weighting
to reduce boundary artifacts.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def get_gaussian_weight(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> np.ndarray:
    """
    Create a 3D Gaussian importance map for blending overlapping patches.
    Center voxels get higher weight, edges taper off.
    """
    weight = np.ones(patch_size, dtype=np.float32)
    center = np.array(patch_size) / 2.0
    sigmas = np.array(patch_size) * sigma_scale

    for d in range(patch_size[0]):
        for h in range(patch_size[1]):
            for w in range(patch_size[2]):
                dist = ((np.array([d, h, w]) - center) / sigmas) ** 2
                weight[d, h, w] = np.exp(-0.5 * dist.sum())

    # Normalize so max = 1
    weight = weight / weight.max()
    weight = np.clip(weight, 1e-4, 1.0)
    return weight


def sliding_window_inference(
    volume: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    device: str = "cuda",
    num_classes: int = 2,
    use_gaussian: bool = True,
) -> np.ndarray:
    """
    Run inference on a full 3D volume using overlapping sliding windows.

    Args:
        volume: preprocessed input array (D, H, W), values in [0, 1]
        model: trained 3D U-Net (returns list; we use index 0 = full-res)
        patch_size: (pD, pH, pW) matching training patch size
        overlap: fraction of overlap between adjacent patches
        device: "cuda" or "cpu"
        num_classes: number of output channels
        use_gaussian: blend with Gaussian weighting to reduce seam artifacts

    Returns:
        pred_labels: (D, H, W) integer prediction map
    """
    model.eval()
    vol_shape = volume.shape  # (D, H, W)
    pD, pH, pW = patch_size
    step = tuple(int(p * (1 - overlap)) for p in patch_size)

    # Initialize accumulators
    pred_sum = np.zeros((num_classes, *vol_shape), dtype=np.float32)
    weight_sum = np.zeros(vol_shape, dtype=np.float32)

    # Gaussian blending kernel
    if use_gaussian:
        gaussian = get_gaussian_weight(patch_size)
    else:
        gaussian = np.ones(patch_size, dtype=np.float32)

    # Generate patch start positions
    starts = []
    for dim_size, p_size, s in zip(vol_shape, patch_size, step):
        positions = list(range(0, dim_size - p_size + 1, s))
        # ensure we cover the last edge
        if positions[-1] + p_size < dim_size:
            positions.append(dim_size - p_size)
        starts.append(positions)

    with torch.no_grad():
        for d_start in starts[0]:
            for h_start in starts[1]:
                for w_start in starts[2]:
                    # Extract patch
                    patch = volume[
                        d_start : d_start + pD,
                        h_start : h_start + pH,
                        w_start : w_start + pW,
                    ]

                    # Pad if patch is smaller than expected (shouldn't happen with above logic)
                    if patch.shape != patch_size:
                        padded = np.zeros(patch_size, dtype=np.float32)
                        padded[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
                        patch = padded

                    # To tensor: (1, 1, D, H, W)
                    x = torch.from_numpy(patch[np.newaxis, np.newaxis]).float().to(device)

                    # Forward pass (take highest-res output)
                    outputs = model(x)
                    logits = outputs[0]  # (1, C, D, H, W)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (C, D, H, W)

                    # Accumulate weighted predictions
                    for c in range(num_classes):
                        pred_sum[
                            c,
                            d_start : d_start + pD,
                            h_start : h_start + pH,
                            w_start : w_start + pW,
                        ] += probs[c] * gaussian

                    weight_sum[
                        d_start : d_start + pD,
                        h_start : h_start + pH,
                        w_start : w_start + pW,
                    ] += gaussian

    # Average and argmax
    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(num_classes):
        pred_sum[c] /= weight_sum

    pred_labels = np.argmax(pred_sum, axis=0).astype(np.uint8)
    return pred_labels
