"""
Convert CTA DICOM series to NIfTI for alceballosa/robust-vessel-segmentation inference.

Walks the TopCoW DICOM tree and writes one .nii.gz per "Thins with contrast" series.
Names follow the convention used elsewhere in this project:
    CTA_001_Thins_with_contrast.nii.gz
    CTA_013_Thins_with_contrast_full_field.nii.gz

Usage:
    python dicom_to_nifti.py --dicom_root "D:/vessel_seg_v2/TopCoW_DICOM" \
                             --output_dir "D:/vessel_seg_v2/CTA_nifti"
"""

import argparse
import os
import re
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


SERIES_SUBSTR = "thins with contrast"


def find_thins_with_contrast_series(cta_dir):
    """Return list of (series_name, series_dir) for every 'Thins with contrast*' folder."""
    out = []
    for entry in sorted(os.listdir(cta_dir)):
        full = os.path.join(cta_dir, entry)
        if not os.path.isdir(full):
            continue
        if SERIES_SUBSTR in entry.lower():
            out.append((entry, full))
    return out


def slugify(name):
    s = name.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s


def read_dicom_series_to_nifti(series_dir, output_path):
    """Read a DICOM series directory with SimpleITK and write NIfTI."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {series_dir}")

    # Pick the longest series (the actual scan, not scouts)
    best_sid, best_files = None, []
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(series_dir, sid)
        if len(files) > len(best_files):
            best_sid, best_files = sid, files

    reader.SetFileNames(best_files)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return len(best_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True,
                    help="Root folder containing CTA_XXX subdirectories.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write .nii.gz files.")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover every CTA_XXX (possibly nested under CTA_1-10-001/CTA_1-10/CTA_001)
    cta_dirs = []
    for root, dirs, files in os.walk(args.dicom_root):
        name = os.path.basename(root)
        m = re.match(r"^CTA_\d+$", name)
        if m:
            cta_dirs.append(root)

    cta_dirs = sorted(set(cta_dirs))
    print(f"Found {len(cta_dirs)} CTA_XXX directories under {args.dicom_root}")

    written, skipped, failed = 0, 0, 0
    for cta_dir in tqdm(cta_dirs):
        cta_id = os.path.basename(cta_dir)  # e.g. CTA_001

        series = find_thins_with_contrast_series(cta_dir)
        if not series:
            tqdm.write(f"  [!] No 'Thins with contrast' under {cta_id}")
            continue

        for series_name, series_dir in series:
            # Build output name: CTA_001_Thins_with_contrast.nii.gz
            # or CTA_013_Thins_with_contrast_full_field.nii.gz
            tail_slug = slugify(series_name)
            out_name = f"{cta_id}_{tail_slug}.nii.gz"
            out_path = Path(args.output_dir) / out_name

            if args.skip_existing and out_path.exists():
                skipped += 1
                continue

            try:
                n_slices = read_dicom_series_to_nifti(series_dir, out_path)
                tqdm.write(f"  {out_name}  ({n_slices} slices)")
                written += 1
            except Exception as e:
                tqdm.write(f"  [FAIL] {cta_id}/{series_name}: {e}")
                failed += 1

    print(f"\nDone. Written: {written}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
