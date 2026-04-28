"""
compare_dynavessel_vs_unet3d.py -- Compare DynaVessel (alceballosa Model 241)
against our best UNet3D on TopBrain validation cases.

Both models produce binary vessel masks (for DynaVessel, arteries ∪ veins).
Ground truth is the TopBrain 40-class labelmap, binarized to any foreground.

Inputs (defaults):
    --cta_dir      : original TopBrain CT volumes (imagesTr style, _0000.nii.gz)
    --gt_dir       : TopBrain 40-class labels (same basename without _0000)
    --dyna_dir     : DynaVessel 3-class outputs
    --unet_dir     : our UNet3D binary predictions
    --cases        : case IDs to compare (e.g. topcow_ct_001)

Usage:
    python compare_dynavessel_vs_unet3d.py \
        --gt_dir   "data/TopCoW2024_Data_Release/labelsTr" \
        --dyna_dir "D:/vessel_seg_v2/compare_dynavessel" \
        --unet_dir "D:/vessel_seg_v2/compare_unet3d" \
        --cases topcow_ct_001 topcow_ct_004 topcow_ct_008
"""

import argparse
import os
import sys
import time

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import compute_dice, compute_cldice, compute_hd95, compute_betti0_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True,
                   help="Directory with TopBrain ground-truth labels")
    p.add_argument("--dyna_dir", required=True,
                   help="Directory with DynaVessel outputs")
    p.add_argument("--unet_dir", required=True,
                   help="Directory with UNet3D predictions")
    p.add_argument("--cases", nargs="+", required=True,
                   help="Case stems, e.g. topcow_ct_001")
    p.add_argument("--unet_suffix", default="_0000_vessels.nii.gz")
    p.add_argument("--dyna_suffix", default="_0000.nii.gz")
    p.add_argument("--gt_suffix", default=".nii.gz")
    return p.parse_args()


def load_mask(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = tuple(img.GetSpacing()[::-1])  # SITK returns (x,y,z); array is (z,y,x)
    return arr, spacing


def evaluate_pair(pred_bin, gt_bin, spacing):
    """Run Dice / clDice / HD95 / Betti-0 on a binary pair."""
    out = {}
    out["dice"] = compute_dice(pred_bin, gt_bin)
    out["cldice"] = compute_cldice(pred_bin, gt_bin)
    out["hd95"] = compute_hd95(pred_bin, gt_bin, voxel_spacing=spacing)
    out["betti0"] = compute_betti0_error(pred_bin, gt_bin)
    return out


def fmt_row(name, m, voxcount=None):
    vc = f"  vox={voxcount:>9,}" if voxcount is not None else ""
    return (f"  {name:<12s}  Dice={m['dice']:.4f}  "
            f"clDice={m['cldice']:.4f}  HD95={m['hd95']:.2f}  "
            f"Betti0={m['betti0']:>4d}{vc}")


def main():
    args = parse_args()

    rows = []
    per_case_unet = []
    per_case_dyna = []
    per_case_dyna_art = []
    per_case_dyna_vein = []
    for case in args.cases:
        gt_path = os.path.join(args.gt_dir, f"{case}{args.gt_suffix}")
        dyna_path = os.path.join(args.dyna_dir, f"{case}{args.dyna_suffix}")
        unet_path = os.path.join(args.unet_dir, f"{case}{args.unet_suffix}")

        missing = [p for p in [gt_path, dyna_path, unet_path] if not os.path.isfile(p)]
        if missing:
            print(f"[{case}] Missing files, skipping:")
            for p in missing:
                print(f"    {p}")
            continue

        t0 = time.time()
        gt_arr, gt_spacing = load_mask(gt_path)
        dyna_arr, _ = load_mask(dyna_path)
        unet_arr, _ = load_mask(unet_path)

        if gt_arr.shape != dyna_arr.shape or gt_arr.shape != unet_arr.shape:
            print(f"[{case}] Shape mismatch: "
                  f"gt={gt_arr.shape} dyna={dyna_arr.shape} unet={unet_arr.shape}")
            continue

        gt_bin = (gt_arr > 0).astype(np.uint8)
        dyna_bin = (dyna_arr > 0).astype(np.uint8)  # artery ∪ vein
        unet_bin = (unet_arr > 0).astype(np.uint8)

        gt_vox = int(gt_bin.sum())
        dyna_vox = int(dyna_bin.sum())
        unet_vox = int(unet_bin.sum())

        # Also report DynaVessel per-class vs GT (artery class alone, vein class alone)
        dyna_art = (dyna_arr == 1).astype(np.uint8)
        dyna_vein = (dyna_arr == 2).astype(np.uint8)

        print(f"\n=== {case} ===  shape={gt_arr.shape}  spacing={tuple(round(s,3) for s in gt_spacing)}")
        print(f"  gt  foreground vox = {gt_vox:,}")
        print(f"  dyna              = {dyna_vox:,}  (A={int(dyna_art.sum()):,}  V={int(dyna_vein.sum()):,})")
        print(f"  unet              = {unet_vox:,}")

        m_unet = evaluate_pair(unet_bin, gt_bin, gt_spacing)
        m_dyna = evaluate_pair(dyna_bin, gt_bin, gt_spacing)
        m_dyna_art = evaluate_pair(dyna_art, gt_bin, gt_spacing)
        m_dyna_vein = evaluate_pair(dyna_vein, gt_bin, gt_spacing)

        print(fmt_row("UNet3D",         m_unet,         unet_vox))
        print(fmt_row("DynaVes(A+V)",   m_dyna,         dyna_vox))
        print(fmt_row("DynaVes-A",      m_dyna_art))
        print(fmt_row("DynaVes-V",      m_dyna_vein))

        per_case_unet.append(m_unet)
        per_case_dyna.append(m_dyna)
        per_case_dyna_art.append(m_dyna_art)
        per_case_dyna_vein.append(m_dyna_vein)
        rows.append((case, m_unet, m_dyna, m_dyna_art, m_dyna_vein,
                     unet_vox, dyna_vox, gt_vox))

        print(f"  (elapsed {time.time()-t0:.1f}s)")

    if not rows:
        print("No cases evaluated.")
        return

    def mean(metrics_list, key):
        vals = [m[key] for m in metrics_list if not np.isinf(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 100)
    print("Summary (means across cases)")
    print("=" * 100)
    header = f"  {'Model':<20s}  {'Dice':>7s}  {'clDice':>7s}  {'HD95':>7s}  {'Betti0':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, lst in [
        ("UNet3D",           per_case_unet),
        ("DynaVessel (A+V)", per_case_dyna),
        ("DynaVessel-A",     per_case_dyna_art),
        ("DynaVessel-V",     per_case_dyna_vein),
    ]:
        print(f"  {name:<20s}  {mean(lst, 'dice'):>7.4f}  "
              f"{mean(lst, 'cldice'):>7.4f}  {mean(lst, 'hd95'):>7.2f}  "
              f"{mean(lst, 'betti0'):>7.1f}")


if __name__ == "__main__":
    main()
