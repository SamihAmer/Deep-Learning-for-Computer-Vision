"""
download_corrected_labels.py — Export corrected A/V labels from RedBrick.

Pulls the latest segmentations from tasks that have moved past the Label stage
(i.e. radiologist has finalized corrections) and writes 3-class NIfTI masks
into --output_dir, preserving the original patient+qualifier naming.

Usage:
    python download_corrected_labels.py \
        --output_dir D:/vessel_seg_v2/av_masks_corrected \
        --artery_category Artery \
        --vein_category Vein
"""

import argparse
import os
import sys

import redbrick


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--org_id", default=os.environ.get("REDBRICK_ORG_ID"))
    p.add_argument("--project_id", default=os.environ.get("REDBRICK_PROJECT_ID"))
    p.add_argument("--api_key", default=os.environ.get("REDBRICK_API_KEY"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--artery_category", default="Arteries")
    p.add_argument("--vein_category", default="Veins")
    p.add_argument("--only_completed", action="store_true",
                   help="Only export tasks that have reached the End stage")
    p.add_argument("--limit", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    if not all([args.org_id, args.project_id, args.api_key]):
        print("Error: REDBRICK_ORG_ID / REDBRICK_PROJECT_ID / REDBRICK_API_KEY required")
        sys.exit(1)

    project = redbrick.get_project(
        org_id=args.org_id, project_id=args.project_id, api_key=args.api_key
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("Exporting tasks from RedBrick ...")
    # RedBrick SDK: export.export_tasks writes to a folder and returns task metadata
    exported = project.export.export_tasks(
        task_id=None,
        only_ground_truth=args.only_completed,
        destination=args.output_dir,
        rt_struct=False,
        semantic_mask=False,   # we want instance IDs preserved (1=artery, 2=vein)
        binary_mask=False,
        limit=args.limit,
    )
    print(f"Exported {len(exported) if hasattr(exported, '__len__') else '?'} tasks to {args.output_dir}")
    print("\nCheck that the produced NIfTIs encode Artery=1 and Vein=2.")
    print("If RedBrick writes per-category NIfTIs, merge them into one 3-class mask")
    print("before fine-tuning.")


if __name__ == "__main__":
    main()
