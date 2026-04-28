"""
check_redbrick_taxonomy.py — Inspect the current RedBrick project taxonomy.

Prints every segmentation/label category so you can verify whether "Artery"
and "Vein" need to be added before multi-class upload.

Usage:
    python check_redbrick_taxonomy.py

Reads credentials from env vars REDBRICK_ORG_ID / REDBRICK_PROJECT_ID / REDBRICK_API_KEY,
or pass as CLI flags.
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
    return p.parse_args()


def main():
    args = parse_args()
    if not all([args.org_id, args.project_id, args.api_key]):
        print("Error: need REDBRICK_ORG_ID, REDBRICK_PROJECT_ID, REDBRICK_API_KEY")
        sys.exit(1)

    project = redbrick.get_project(
        org_id=args.org_id, project_id=args.project_id, api_key=args.api_key
    )

    print(f"Project: {project.name}  ({project.project_id})")
    tax = project.taxonomy
    print(f"Taxonomy: {getattr(tax, 'name', '?')} (v{getattr(tax, 'taxonomy_version', '?')})")

    categories = []
    try:
        # redbrick-sdk ≥ 2.x exposes object_types / categories
        for c in getattr(tax, "object_types", []) or []:
            categories.append((c.get("category", "?"), c.get("classId", "?"), c.get("labelType", "?")))
    except Exception:
        pass
    # Fallback: iterate categories attribute
    if not categories and hasattr(tax, "categories"):
        for c in tax.categories:
            categories.append((getattr(c, "name", "?"), getattr(c, "class_id", "?"), getattr(c, "label_type", "?")))

    if not categories:
        print("\nCouldn't extract categories from SDK; inspect taxonomy object manually:")
        print(vars(tax))
        return

    print("\nCurrent categories:")
    for name, cid, ltype in categories:
        print(f"  [{cid}] {name}   (type: {ltype})")

    names = {n.lower() for n, _, _ in categories}
    # Accept singular or plural spelling
    has_artery = any(n in names for n in ("artery", "arteries"))
    has_vein = any(n in names for n in ("vein", "veins"))
    missing = []
    if not has_artery:
        missing.append("Artery/Arteries")
    if not has_vein:
        missing.append("Vein/Veins")

    if missing:
        print(f"\nMissing for multi-class upload: {missing}")
        print("Add them in the RedBrick admin UI under the project's taxonomy,")
        print("using segmentation/label type. Suggested colors: red + blue.")
    else:
        print("\nArtery + Vein categories present. Ready to upload multi-class masks.")


if __name__ == "__main__":
    main()
