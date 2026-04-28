"""
upload_av_masks.py -- Upload artery/vein (multi-class) masks to RedBrick AI.

Extends the existing single-class upload_masks.py to handle 3-class
segmentations produced by the alceballosa Model 241:
    0 = background
    1 = artery
    2 = vein

Matches masks to tasks by patient ID + qualifier (same logic as upload_masks.py).

Usage:
    # Dry run first
    python upload_av_masks.py --masks_dir D:/vessel_seg_v2/av_masks/Predictions --dry_run

    # Upload for real (finalize=True → Review stage)
    python upload_av_masks.py --masks_dir D:/vessel_seg_v2/av_masks/Predictions
"""

import os
import sys
import glob
import argparse
import re

import redbrick


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--org_id", default=os.environ.get("REDBRICK_ORG_ID"))
    p.add_argument("--project_id", default=os.environ.get("REDBRICK_PROJECT_ID"))
    p.add_argument("--api_key", default=os.environ.get("REDBRICK_API_KEY"))
    p.add_argument("--masks_dir", required=False)
    p.add_argument("--stage", default="Label")
    p.add_argument("--artery_category", default="Arteries")
    p.add_argument("--vein_category", default="Veins")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--force", action="store_true",
                   help="Upload even to tasks not currently in --stage "
                        "(default: skip mismatched-stage tasks).")
    return p.parse_args()


def connect(args):
    if not all([args.org_id, args.project_id, args.api_key]):
        print("Error: need --org_id, --project_id, --api_key (or env vars)")
        sys.exit(1)
    return redbrick.get_project(
        org_id=args.org_id, project_id=args.project_id, api_key=args.api_key
    )


def mask_to_patient_and_qualifier(mask_path):
    """Extract CTA_XXX and optional qualifier from the mask filename.

    The alceballosa pipeline preserves the input filename, so:
        CTA_001_Thins_with_contrast.nii.gz          -> ('cta_001', None)
        CTA_013_Thins_with_contrast_full_field.nii.gz -> ('cta_013', 'full field')
    """
    n = os.path.basename(mask_path)
    for suffix in [".nii.gz", ".nii"]:
        if n.lower().endswith(suffix):
            n = n[: -len(suffix)]

    # Strip the "_Thins_with_contrast" stem; keep any qualifier after it
    n = re.sub(r"_?Thins_with_contrast", "", n, flags=re.IGNORECASE)
    n = re.sub(r"_vessels$", "", n, flags=re.IGNORECASE)

    m = re.match(r"(CTA_\d+)(.*)", n, re.IGNORECASE)
    if not m:
        return n.lower(), None
    patient_id = m.group(1).lower()
    remainder = m.group(2).strip("_").replace("_", " ").strip()
    qualifier = remainder if remainder else None
    return patient_id, qualifier


def match_mask_to_task(patient_id, qualifier, tasks):
    """Find the matching RedBrick task for a mask (same logic as upload_masks.py)."""
    patient_tasks = [t for t in tasks if patient_id in t.get("name", "").lower()]
    if not patient_tasks:
        return None
    if qualifier:
        for t in patient_tasks:
            if qualifier.lower() in t["name"].lower():
                return t
        for t in patient_tasks:
            if "thins with contrast" in t["name"].lower():
                return t
    else:
        for t in patient_tasks:
            name_lower = t["name"].lower()
            if "thins with contrast" in name_lower and "(" not in name_lower:
                return t
        for t in patient_tasks:
            if "with contrast" in t["name"].lower():
                return t
    return None


def build_update(task, mask_path, artery_cat, vein_cat):
    return {
        "taskId": task["taskId"],
        "series": [
            {
                "segmentations": os.path.abspath(mask_path),
                "segmentMap": {
                    "1": {"category": artery_cat},
                    "2": {"category": vein_cat},
                },
            }
        ],
    }


def main():
    args = parse_args()
    project = connect(args)

    print("Fetching tasks from RedBrick...")
    tasks = list(project.export.list_tasks(limit=args.limit))
    print(f"Found {len(tasks)} tasks\n")

    if args.list:
        for i, t in enumerate(tasks):
            print(
                f"  {i+1:3d}. {t.get('name','?'):<40s} | "
                f"{t.get('currentStageName','?'):<10s} | {t.get('taskId','?')}"
            )
        return

    if not args.masks_dir:
        print("Error: --masks_dir required")
        sys.exit(1)

    masks = sorted(glob.glob(os.path.join(args.masks_dir, "*.nii.gz")))
    if not masks:
        masks = sorted(glob.glob(os.path.join(args.masks_dir, "*.nii")))
    print(f"Found {len(masks)} mask files in {args.masks_dir}")

    updates, matched, unmatched, skipped_stage = [], [], [], []
    for mp in masks:
        pid, qual = mask_to_patient_and_qualifier(mp)
        t = match_mask_to_task(pid, qual, tasks)
        if not t:
            unmatched.append(os.path.basename(mp))
            continue
        current_stage = t.get("currentStageName", "?")
        if current_stage != args.stage and not args.force:
            skipped_stage.append((os.path.basename(mp), t["name"], current_stage))
            continue
        updates.append(build_update(t, mp, args.artery_category, args.vein_category))
        matched.append((os.path.basename(mp), t["name"]))

    print(f"\nMatched: {len(matched)}/{len(masks)}")
    for mf, tn in matched:
        print(f"    {mf} -> {tn}")
    if unmatched:
        print(f"\nUnmatched ({len(unmatched)}):")
        for m in unmatched:
            print(f"    {m}")
    if skipped_stage:
        print(f"\nSkipped (not in stage '{args.stage}', use --force to override):")
        for mf, tn, cs in skipped_stage:
            print(f"    {mf} -> {tn} [currently in '{cs}']")

    if not updates:
        print("\nNo matches. Run with --list to inspect task names.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would upload {len(updates)} A/V masks "
              f"(Artery=1, Vein=2) to stage '{args.stage}'")
        return

    print(f"\nUploading {len(updates)} masks to stage '{args.stage}' "
          f"(finalize=True -> Review)...")
    project.labeling.put_tasks(args.stage, updates, finalize=True)
    print("Done.")


if __name__ == "__main__":
    main()
