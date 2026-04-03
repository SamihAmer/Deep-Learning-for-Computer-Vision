"""
predict.py -- Run vessel segmentation on a CTA NIfTI and save a binary mask.

Produces a NIfTI segmentation with the same header/affine as the input,
ready for import into RedBrick AI (or any annotation tool) as pre-labels.

Usage:
    # Single file
    python predict.py --checkpoint runs/.../best_model.pth --input scan.nii.gz --output mask.nii.gz

    # Entire folder of CTA scans
    python predict.py --checkpoint runs/.../best_model.pth --input_dir /path/to/scans/ --output_dir /path/to/masks/

    # Adjust threshold (default 0.5)
    python predict.py --checkpoint runs/.../best_model.pth --input scan.nii.gz --output mask.nii.gz --threshold 0.3
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import SimpleITK as sitk
import torch

from configs.default import CONFIG
from models.unet3d import UNet3D
from utils.inference import sliding_window_inference
from data.dataset import preprocess_ct


def parse_args():
    parser = argparse.ArgumentParser(description="Run vessel segmentation on CTA NIfTI(s)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--input", type=str, default=None, help="Path to a single input NIfTI (.nii.gz)")
    parser.add_argument("--output", type=str, default=None, help="Path for output segmentation NIfTI")
    parser.add_argument("--input_dir", type=str, default=None, help="Folder of input NIfTIs (batch mode)")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder for output segmentations (batch mode)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for vessel (default 0.5)")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint, return (model, config)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", CONFIG)

    model = UNet3D(
        in_channels=cfg.get("in_channels", 1),
        num_classes=cfg.get("num_classes", 2),
        base_filters=cfg.get("base_filters", 32),
        num_stages=cfg.get("num_stages", 5),
        deep_supervision=cfg.get("deep_supervision", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Best Dice: {ckpt.get('best_dice', '?')}")
    print(f"  HU window: {cfg.get('hu_window', (0, 600))}")
    return model, cfg


def predict_volume(input_path, output_path, model, cfg, device, threshold=0.5):
    """Run inference on a single NIfTI and save the binary mask."""
    t0 = time.time()

    # Load with SimpleITK to preserve full header
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)

    # Preprocess
    hu_window = tuple(cfg.get("hu_window", (0, 600)))
    vol = preprocess_ct(arr, hu_window)

    # Inference
    pred = sliding_window_inference(
        vol, model,
        patch_size=tuple(cfg.get("patch_size", (128, 128, 128))),
        overlap=cfg.get("sliding_window_overlap", 0.5),
        device=str(device),
        num_classes=cfg.get("num_classes", 2),
    )

    # Binary mask: pred is already argmax output (0 or 1)
    mask = (pred > 0).astype(np.uint8)

    # Save as NIfTI with identical header to input
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)  # preserves spacing, origin, direction
    sitk.WriteImage(mask_img, output_path)

    elapsed = time.time() - t0
    n_vessel = mask.sum()
    print(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)}"
          f" | {elapsed:.0f}s | {n_vessel:,} vessel voxels"
          f" | {100 * n_vessel / mask.size:.2f}% foreground")


def main():
    args = parse_args()

    # Validate args
    if args.input is None and args.input_dir is None:
        print("Error: provide --input (single file) or --input_dir (batch mode)")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)

    # Single file mode
    if args.input:
        output = args.output or args.input.replace(".nii.gz", "_vessels.nii.gz")
        print(f"\nPredicting: {args.input}")
        predict_volume(args.input, output, model, cfg, device, args.threshold)
        print(f"\nDone. Output: {output}")
        return

    # Batch mode
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "vessel_masks")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, "*.nii")))
    if not files:
        print(f"No .nii.gz or .nii files found in {input_dir}")
        sys.exit(1)

    print(f"\nBatch mode: {len(files)} files")
    print(f"Output dir: {output_dir}\n")

    for i, fpath in enumerate(files):
        basename = os.path.basename(fpath)
        out_name = basename.replace(".nii.gz", "_vessels.nii.gz").replace(".nii", "_vessels.nii")
        out_path = os.path.join(output_dir, out_name)
        print(f"[{i+1}/{len(files)}]", end="")
        predict_volume(fpath, out_path, model, cfg, device, args.threshold)

    print(f"\nDone. {len(files)} masks saved to {output_dir}")


if __name__ == "__main__":
    main()
