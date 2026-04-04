"""
upload_masks.py -- Upload vessel segmentation masks to existing RedBrick AI tasks.

Matches mask files to existing tasks by name, uploads as draft pre-labels
so you can review and correct in the RedBrick UI before submitting.

Usage:
    # Step 1: List tasks to see what names RedBrick has
    python upload_masks.py --list --org_id ORG --project_id PROJ --api_key KEY

    # Step 2: Upload masks (dry run first to check matching)
    python upload_masks.py --masks_dir /path/to/masks --dry_run \
        --org_id ORG --project_id PROJ --api_key KEY

    # Step 3: Upload for real
    python upload_masks.py --masks_dir /path/to/masks \
        --org_id ORG --project_id PROJ --api_key KEY

    # You can also set env vars instead of CLI args:
    export REDBRICK_ORG_ID=...
    export REDBRICK_PROJECT_ID=...
    export REDBRICK_API_KEY=...
    python upload_masks.py --masks_dir /path/to/masks
"""

import os
import sys
import glob
import argparse

import redbrick


def parse_args():
    parser = argparse.ArgumentParser(description="Upload vessel masks to RedBrick AI")
    parser.add_argument("--org_id", type=str, default=os.environ.get("REDBRICK_ORG_ID"),
                        help="RedBrick org ID (or set REDBRICK_ORG_ID env var)")
    parser.add_argument("--project_id", type=str, default=os.environ.get("REDBRICK_PROJECT_ID"),
                        help="RedBrick project ID (or set REDBRICK_PROJECT_ID env var)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("REDBRICK_API_KEY"),
                        help="RedBrick API key (or set REDBRICK_API_KEY env var)")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Directory containing *_vessels.nii.gz mask files")
    parser.add_argument("--stage", type=str, default="Label",
                        help="RedBrick stage name (default: 'Label')")
    parser.add_argument("--category", type=str, default="Vessels",
                        help="Taxonomy category name for vessel label (default: 'Vessels')")
    parser.add_argument("--list", action="store_true",
                        help="Just list existing tasks and exit")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be uploaded without uploading")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max tasks to fetch (default: 500)")
    return parser.parse_args()


def connect(args):
    """Connect to RedBrick project."""
    if not all([args.org_id, args.project_id, args.api_key]):
        print("Error: need --org_id, --project_id, --api_key (or set env vars)")
        sys.exit(1)
    return redbrick.get_project(
        org_id=args.org_id,
        project_id=args.project_id,
        api_key=args.api_key,
    )


def mask_to_patient_and_qualifier(mask_path):
    """Extract patient ID (e.g. 'CTA_001') and optional qualifier from mask filename.

    Examples:
        CTA_001_vessels.nii.gz -> ('cta_001', None)
        CTA_013_full_field_vessels.nii.gz -> ('cta_013', 'full field')
        CTA_050_bone_kernal_vessels.nii.gz -> ('cta_050', 'bone kernal')
    """
    n = os.path.basename(mask_path)
    for suffix in [".nii.gz", ".nii"]:
        if n.lower().endswith(suffix):
            n = n[:-len(suffix)]
    n = n.replace("_vessels", "")

    # Extract patient ID: CTA_XXX (3 digits)
    import re
    match = re.match(r"(CTA_\d+)(.*)", n, re.IGNORECASE)
    if not match:
        return n.lower(), None

    patient_id = match.group(1).lower()
    remainder = match.group(2).strip("_").replace("_", " ").strip()
    qualifier = remainder if remainder else None
    return patient_id, qualifier


def match_mask_to_task(patient_id, qualifier, tasks):
    """Find the matching RedBrick task for a mask.

    For most masks: match "Thins with contrast" task for that patient.
    For masks with qualifiers: match the task whose name contains the qualifier.
    """
    patient_tasks = [
        t for t in tasks
        if patient_id in t.get("name", "").lower()
    ]

    if not patient_tasks:
        return None

    if qualifier:
        # Match qualifier to task name (e.g. "full field" in "Thins with contrast (full field)")
        for t in patient_tasks:
            if qualifier.lower() in t["name"].lower():
                return t
        # Fallback: try "Thins with contrast" anyway
        for t in patient_tasks:
            if "thins with contrast" in t["name"].lower():
                return t
    else:
        # Default: match "Thins with contrast" (not "Thicker", not "without")
        for t in patient_tasks:
            name_lower = t["name"].lower()
            if "thins with contrast" in name_lower and "(" not in name_lower:
                return t
        # Fallback: any "with contrast"
        for t in patient_tasks:
            if "with contrast" in t["name"].lower():
                return t

    return None


def main():
    args = parse_args()
    project = connect(args)

    # Fetch existing tasks
    print("Fetching tasks from RedBrick...")
    tasks = list(project.export.list_tasks(limit=args.limit))
    print(f"Found {len(tasks)} tasks\n")

    # List mode
    if args.list:
        for i, t in enumerate(tasks):
            task_id = t.get("taskId", "?")
            name = t.get("name", "?")
            status = t.get("currentStageName", "?")
            print(f"  {i+1:3d}. {name:<40s} | {status:<10s} | {task_id}")
        return

    # Upload mode
    if not args.masks_dir:
        print("Error: --masks_dir required for upload")
        sys.exit(1)

    # Find mask files
    masks = sorted(glob.glob(os.path.join(args.masks_dir, "*.nii.gz")))
    if not masks:
        masks = sorted(glob.glob(os.path.join(args.masks_dir, "*.nii")))
    if not masks:
        print(f"No .nii.gz or .nii files found in {args.masks_dir}")
        sys.exit(1)
    print(f"Found {len(masks)} mask files in {args.masks_dir}")

    # Match masks to tasks
    updates = []
    matched = []
    unmatched_masks = []

    for mask_path in masks:
        patient_id, qualifier = mask_to_patient_and_qualifier(mask_path)
        t = match_mask_to_task(patient_id, qualifier, tasks)

        if t:
            updates.append({
                "taskId": t["taskId"],
                "series": [{
                    "segmentations": os.path.abspath(mask_path),
                    "segmentMap": {
                        "1": {"category": args.category},
                    }
                }]
            })
            matched.append((os.path.basename(mask_path), t["name"], t["taskId"]))
        else:
            unmatched_masks.append(os.path.basename(mask_path))

    # Report matching
    print(f"\nMatched: {len(matched)}/{len(masks)}")
    if matched:
        print("\n  Matches:")
        for mask_file, task_name, task_id in matched:
            print(f"    {mask_file} -> {task_name} ({task_id[:8]}...)")

    if unmatched_masks:
        print(f"\n  Unmatched masks ({len(unmatched_masks)}):")
        for m in unmatched_masks:
            print(f"    {m}")
        print("\n  These masks don't match any task name. Check naming.")

    if not updates:
        print("\nNo matches found. Nothing to upload.")
        print("Run with --list to see task names, then check your mask filenames match.")
        return

    # Dry run or upload
    if args.dry_run:
        print(f"\n[DRY RUN] Would upload {len(updates)} masks to stage '{args.stage}'")
        print("Run without --dry_run to upload.")
        return

    print(f"\nUploading {len(updates)} masks to stage '{args.stage}' (finalize=True, submits to Review)...")
    project.labeling.put_tasks(args.stage, updates, finalize=True)
    print("Done! Tasks moved to Review stage — open RedBrick to correct the segmentations.")


if __name__ == "__main__":
    main()
