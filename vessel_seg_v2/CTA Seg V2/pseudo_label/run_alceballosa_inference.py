"""
Run alceballosa/robust-vessel-segmentation Model 241 on a folder of CTA NIfTIs.

This invokes the repo's `extractVessels.py` with `-m Prediction -v 241`. In
sequential mode each scan is processed end-to-end and the final 3-class mask is
written to <output_dir>/<scan_name>.nii.gz (no intermediate Predictions/ dir).

Expects:
    - Repo at ../robust-vessel-segmentation/ (relative to this script's parent)
    - Weights at ../robust-vessel-segmentation/atlases_and_weights/weights/Dataset241_.../
    - Atlases at ../robust-vessel-segmentation/atlases_and_weights/atlases/

The model outputs 3-class labels (0=background, 1=artery, 2=vein) per the preprint.

Usage:
    python run_alceballosa_inference.py \
        --input_dir  "D:/vessel_seg_v2/CTA_nifti" \
        --output_dir "D:/vessel_seg_v2/av_masks"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--repo_dir", default=None,
                    help="Path to robust-vessel-segmentation repo. "
                         "Default: sibling of this script's parent.")
    ap.add_argument("--weights_base", default=None,
                    help="Subdir of --repo_dir holding weights/, atlases/, ants-*/. "
                         "Auto-detects 'dynavessel' or 'atlases_and_weights' if omitted.")
    ap.add_argument("--version", type=int, default=241)
    ap.add_argument("--mode", default="Prediction",
                    help="Prediction (no registration) or Full (with registration).")
    ap.add_argument("--sliding_window", type=float, default=0.5)
    args = ap.parse_args()

    if args.repo_dir is None:
        here = Path(__file__).resolve().parent
        # this script: .../CTA Seg V2/pseudo_label/run_alceballosa_inference.py
        # repo:        .../robust-vessel-segmentation
        args.repo_dir = str(here.parent.parent / "robust-vessel-segmentation")

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.is_dir():
        print(f"[ERR] Repo not found at {repo_dir}", file=sys.stderr)
        sys.exit(1)

    if args.weights_base:
        bases = [args.weights_base]
    else:
        bases = ["dynavessel", "atlases_and_weights"]
    base_dir = None
    for b in bases:
        candidate = repo_dir / b
        if (candidate / "weights").is_dir() and (candidate / "atlases").is_dir():
            base_dir = candidate
            break
    if base_dir is None:
        print(f"[ERR] Could not find weights/atlases under any of "
              f"{[str(repo_dir / b) for b in bases]}. "
              "Extract the Google Drive download so that <base>/weights/Dataset241_*/ "
              "and <base>/atlases/ exist.",
              file=sys.stderr)
        sys.exit(1)

    weights_dir = base_dir / "weights"
    atlases_dir = base_dir / "atlases"
    print(f"Using weights base: {base_dir}")

    script = repo_dir / "scripts" / "inference" / "extractVessels.py"

    os.makedirs(args.output_dir, exist_ok=True)

    # extractVessels.py blindly copies `antspy_registration_settings.json` from
    # atlases/rectangle_neck_scene_RegistrationMask/ on startup, but the current
    # Google Drive download ships a different atlas layout (ants_template_rois/)
    # with no settings JSON. In Prediction mode the file is never actually read,
    # so pre-placing a dummy in the output dir makes the `is_file()` check pass
    # and skips the copy.
    settings_local = Path(args.output_dir) / "antspy_registration_settings.json"
    if not settings_local.exists():
        settings_local.write_text(
            json.dumps({"type_of_transform": "Affine"}, indent=2)
        )
        print(f"Wrote dummy registration settings to {settings_local}")

    cmd = [
        sys.executable, str(script),
        "-d", args.input_dir,
        args.output_dir,
        "-m", args.mode,
        "-s", str(args.sliding_window),
        "-g", str(args.num_gpus),
        "-v", str(args.version),
    ]
    print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["nnUNet_results"] = str(weights_dir)

    ret = subprocess.call(cmd, env=env, cwd=str(repo_dir))
    sys.exit(ret)


if __name__ == "__main__":
    main()
