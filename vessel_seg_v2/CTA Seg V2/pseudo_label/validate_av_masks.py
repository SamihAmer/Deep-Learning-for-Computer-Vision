"""
validate_av_masks.py -- Sanity-check alceballosa Model 241 outputs before upload.

For each mask in --masks_dir, reports:
    - shape, spacing, dtype
    - per-class voxel counts (0=bg, 1=artery, 2=vein)
    - artery:vein ratio (preprint reports ~1:2)
    - any unexpected label values

Flags any mask that is empty, missing a vessel class, or has labels outside {0,1,2}.
Optionally checks that the mask grid matches the source CTA grid.

Usage:
    python validate_av_masks.py --masks_dir D:/vessel_seg_v2/av_masks
    python validate_av_masks.py --masks_dir D:/vessel_seg_v2/av_masks \
        --cta_dir D:/vessel_seg_v2/CTA_nifti_inference
"""

import argparse
import glob
import os
import sys

import numpy as np
import SimpleITK as sitk


EXPECTED_LABELS = {0, 1, 2}
MIN_ARTERY_VOX = 500     # empirical floor — a real brain scan has way more
MIN_VEIN_VOX = 500
RATIO_RANGE = (0.2, 3.0)  # artery:vein; preprint says ~1:2, allow slack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--masks_dir", required=True)
    p.add_argument("--cta_dir", default=None,
                   help="If given, verify each mask's grid matches the source CTA.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on any flag.")
    return p.parse_args()


def source_cta_path(cta_dir, mask_name):
    """Map mask filename back to the source CTA nifti."""
    return os.path.join(cta_dir, mask_name)


def geometry_matches(mask_img, cta_img, tol=1e-3):
    if mask_img.GetSize() != cta_img.GetSize():
        return False, f"size {mask_img.GetSize()} vs {cta_img.GetSize()}"
    for a, b in zip(mask_img.GetSpacing(), cta_img.GetSpacing()):
        if abs(a - b) > tol:
            return False, f"spacing {mask_img.GetSpacing()} vs {cta_img.GetSpacing()}"
    for a, b in zip(mask_img.GetOrigin(), cta_img.GetOrigin()):
        if abs(a - b) > tol:
            return False, f"origin {mask_img.GetOrigin()} vs {cta_img.GetOrigin()}"
    return True, ""


def validate(mask_path, cta_dir):
    name = os.path.basename(mask_path)
    flags = []

    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    size = img.GetSize()
    spacing = img.GetSpacing()

    uniq, counts = np.unique(arr, return_counts=True)
    label_counts = dict(zip(uniq.tolist(), counts.tolist()))
    bg = label_counts.get(0, 0)
    art = label_counts.get(1, 0)
    vein = label_counts.get(2, 0)

    bad_labels = set(uniq.tolist()) - EXPECTED_LABELS
    if bad_labels:
        flags.append(f"unexpected labels {sorted(bad_labels)}")

    if art < MIN_ARTERY_VOX:
        flags.append(f"artery voxels {art} < {MIN_ARTERY_VOX}")
    if vein < MIN_VEIN_VOX:
        flags.append(f"vein voxels {vein} < {MIN_VEIN_VOX}")

    ratio = None
    if art > 0 and vein > 0:
        ratio = art / vein
        if not (RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]):
            flags.append(f"A:V ratio {ratio:.2f} outside {RATIO_RANGE}")

    if cta_dir:
        cta_path = source_cta_path(cta_dir, name)
        if os.path.isfile(cta_path):
            cta_img = sitk.ReadImage(cta_path)
            ok, reason = geometry_matches(img, cta_img)
            if not ok:
                flags.append(f"grid mismatch: {reason}")
        else:
            flags.append(f"source CTA missing at {cta_path}")

    return {
        "name": name,
        "size": size,
        "spacing": tuple(round(s, 4) for s in spacing),
        "bg": bg,
        "artery": art,
        "vein": vein,
        "ratio": ratio,
        "flags": flags,
    }


def main():
    args = parse_args()

    masks = sorted(glob.glob(os.path.join(args.masks_dir, "CTA_*.nii.gz")))
    if not masks:
        print(f"No masks found under {args.masks_dir}")
        sys.exit(1)

    print(f"Validating {len(masks)} masks from {args.masks_dir}\n")
    print(f"{'name':<55s} {'size':<18s} {'artery':>10s} {'vein':>10s} {'A:V':>6s}  flags")
    print("-" * 120)

    bad = []
    for mp in masks:
        r = validate(mp, args.cta_dir)
        ratio_s = f"{r['ratio']:.2f}" if r["ratio"] is not None else "  -- "
        size_s = "x".join(str(s) for s in r["size"])
        flag_s = "; ".join(r["flags"]) if r["flags"] else "OK"
        print(f"{r['name']:<55s} {size_s:<18s} "
              f"{r['artery']:>10d} {r['vein']:>10d} {ratio_s:>6s}  {flag_s}")
        if r["flags"]:
            bad.append(r["name"])

    print("-" * 120)
    print(f"\nFlagged {len(bad)}/{len(masks)} masks.")
    if bad:
        for n in bad:
            print(f"  {n}")
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
