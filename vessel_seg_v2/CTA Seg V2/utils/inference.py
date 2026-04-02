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
from functools import lru_cache


@lru_cache(maxsize=4)
def get_gaussian_weight(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> np.ndarray:
    """
    Create a 3D Gaussian importance map for blending overlapping patches.
    Center voxels get higher weight, edges taper off.

    Cached so it's computed once and reused across all volumes.
    """
    # Vectorized: build 1D Gaussians per axis, then outer product
    axes = []
    for size in patch_size:
        center = size / 2.0
        sigma = size * sigma_scale
        coords = np.arange(size, dtype=np.float32)
        axes.append(np.exp(-0.5 * ((coords - center) / sigma) ** 2))

    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
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
    batch_size: int = 16,
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
        batch_size: number of patches to process in parallel per forward pass

    Returns:
        pred_labels: (D, H, W) integer prediction map
    """
    model.eval()
    vol_shape = volume.shape  # (D, H, W)
    pD, pH, pW = patch_size
    step = tuple(int(p * (1 - overlap)) for p in patch_size)
    dev = torch.device(device)

    # Initialize accumulators on GPU
    pred_sum = torch.zeros((num_classes, *vol_shape), dtype=torch.float32, device=dev)
    weight_sum = torch.zeros(vol_shape, dtype=torch.float32, device=dev)

    # Gaussian blending kernel on GPU (cached across calls)
    if use_gaussian:
        gaussian = torch.from_numpy(get_gaussian_weight(patch_size)).to(dev)
    else:
        gaussian = torch.ones(patch_size, dtype=torch.float32, device=dev)

    # Generate all patch start positions upfront
    starts = []
    for dim_size, p_size, s in zip(vol_shape, patch_size, step):
        positions = list(range(0, dim_size - p_size + 1, s))
        # ensure we cover the last edge
        if positions[-1] + p_size < dim_size:
            positions.append(dim_size - p_size)
        starts.append(positions)

    # Collect all (d, h, w) start coords
    all_coords = [
        (d, h, w)
        for d in starts[0]
        for h in starts[1]
        for w in starts[2]
    ]

    # Pre-extract all patches into a contiguous array to minimize per-batch overhead
    all_patches = np.empty((len(all_coords), 1, pD, pH, pW), dtype=np.float32)
    for i, (d_start, h_start, w_start) in enumerate(all_coords):
        patch = volume[
            d_start : d_start + pD,
            h_start : h_start + pH,
            w_start : w_start + pW,
        ]
        if patch.shape != patch_size:
            padded = np.zeros(patch_size, dtype=np.float32)
            padded[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
            patch = padded
        all_patches[i, 0] = patch

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(all_coords), batch_size):
            batch_coords = all_coords[i : i + batch_size]
            x = torch.from_numpy(all_patches[i : i + batch_size]).to(dev, non_blocking=True)

            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            probs = F.softmax(logits, dim=1)

            # Accumulate each patch in the batch
            for j, (d_start, h_start, w_start) in enumerate(batch_coords):
                pred_sum[
                    :,
                    d_start : d_start + pD,
                    h_start : h_start + pH,
                    w_start : w_start + pW,
                ] += probs[j] * gaussian

                weight_sum[
                    d_start : d_start + pD,
                    h_start : h_start + pH,
                    w_start : w_start + pW,
                ] += gaussian

    # Average, argmax, then move to CPU once at the end
    weight_sum = weight_sum.clamp(min=1e-8)
    pred_sum /= weight_sum.unsqueeze(0)
    pred_labels = pred_sum.argmax(dim=0).byte().cpu().numpy()
    return pred_labels
