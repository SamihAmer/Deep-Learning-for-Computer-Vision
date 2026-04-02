"""
evaluate.py -- Stratified evaluation of a trained vessel segmentation model.

Loads the best checkpoint, runs sliding-window inference on the validation set,
and reports per-vessel-class and grouped (large vs small) metrics.

Usage:
    python evaluate.py --checkpoint runs/dice_ce_.../best_model.pth \
                       --data_dir "C:/path/to/TopCoW2024_Data_Release"
"""

import os
import json
import argparse
import time

import numpy as np
import torch

from configs.default import CONFIG
from data.dataset import discover_cases, discover_topbrain_cases, train_val_split, load_nifti, preprocess_ct
from models.unet3d import UNet3D
from utils.inference import sliding_window_inference
from utils.metrics import evaluate_volume
from utils.stratified_eval import (
    evaluate_volume_stratified,
    format_stratified_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TopCoW dataset root")
    parser.add_argument("--topbrain_dir", type=str, default=None,
                        help="Path to TopBrain dataset root (evaluate on TopBrain labels instead of TopCoW)")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint and recover config
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", CONFIG)
    cfg["data_dir"] = args.data_dir

    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"Checkpoint best_dice: {ckpt.get('best_dice', '?')}")

    # Build model and load weights
    model = UNet3D(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_filters=cfg["base_filters"],
        num_stages=cfg["num_stages"],
        deep_supervision=cfg["deep_supervision"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded.\n")

    # Discover cases and get the same val split used during training
    modality = cfg.get("modality", "ct")
    if args.topbrain_dir:
        all_cases = discover_topbrain_cases(args.topbrain_dir, modality=modality)
        print(f"Using TopBrain data ({len(all_cases)} cases, 40-class labels)")
    else:
        all_cases = discover_cases(cfg["data_dir"], modality=modality)
    _, val_cases = train_val_split(all_cases, cfg["train_val_split"])
    ct_count = sum(1 for c in val_cases if "topcow_ct_" in os.path.basename(c["image"]))
    mr_count = sum(1 for c in val_cases if "topcow_mr_" in os.path.basename(c["image"]))
    print(f"Evaluating on {len(val_cases)} validation cases (modality filter: {modality}, CT: {ct_count}, MR: {mr_count})\n")

    # Run inference + evaluation
    global_metrics = {"dice": [], "cldice": [], "hd95": [], "betti0_error": []}
    all_stratified = []

    for i, case in enumerate(val_cases):
        t0 = time.time()
        vol, meta = load_nifti(case["image"])
        lbl, _ = load_nifti(case["label"])
        vol_preprocessed = preprocess_ct(vol, tuple(cfg["hu_window"]))

        # Sliding window inference -> binary prediction
        pred = sliding_window_inference(
            vol_preprocessed,
            model,
            patch_size=tuple(cfg["patch_size"]),
            overlap=cfg["sliding_window_overlap"],
            device=str(device),
            num_classes=cfg["num_classes"],
        )

        # Global metrics (binary)
        gt_binary = (lbl > 0).astype(np.uint8)
        spacing = meta.get("spacing", (1, 1, 1))
        vol_metrics = evaluate_volume(pred, gt_binary, voxel_spacing=spacing)
        for k, v in vol_metrics.items():
            global_metrics[k].append(v)

        # Stratified metrics (per vessel class, using multiclass GT)
        strat = evaluate_volume_stratified(pred, lbl.astype(np.uint8))
        all_stratified.append(strat)

        elapsed = time.time() - t0
        case_name = os.path.basename(case["image"])
        print(f"  [{i+1}/{len(val_cases)}] {case_name} | DSC: {vol_metrics['dice']:.4f} | clDice: {vol_metrics['cldice']:.4f} | {elapsed:.0f}s")

    # Aggregate global metrics
    print(f"\n{'='*60}")
    print("GLOBAL METRICS (binary, averaged over validation set)")
    print(f"{'='*60}")
    for k, v in global_metrics.items():
        print(f"  {k:>15s}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

    # Aggregate stratified metrics
    print(f"\n{'='*60}")
    print("STRATIFIED METRICS (per vessel class, averaged over validation set)")
    print(f"{'='*60}")

    from utils.stratified_eval import VESSEL_CLASSES

    # Collect per-class across all volumes
    class_dice = {name: [] for name in VESSEL_CLASSES.values()}
    class_cldice = {name: [] for name in VESSEL_CLASSES.values()}
    class_b0 = {name: [] for name in VESSEL_CLASSES.values()}

    for strat in all_stratified:
        for name in VESSEL_CLASSES.values():
            r = strat["per_class"].get(name, {})
            if r.get("present", False):
                class_dice[name].append(r["dice"])
                class_cldice[name].append(r["cldice"])
                class_b0[name].append(r["betti0_error"])

    print(f"\n  {'Vessel':<12} {'DSC':>10} {'clDice':>10} {'B0err':>10} {'N':>5}")
    print(f"  {'-'*50}")
    for name in VESSEL_CLASSES.values():
        n = len(class_dice[name])
        if n > 0:
            d = np.mean(class_dice[name])
            c = np.mean(class_cldice[name])
            b = np.mean(class_b0[name])
            print(f"  {name:<12} {d:>10.4f} {c:>10.4f} {b:>10.1f} {n:>5}")
        else:
            print(f"  {name:<12} {'n/a':>10} {'n/a':>10} {'n/a':>10} {0:>5}")

    # Grouped: all 6 vessel groups
    from utils.stratified_eval import (
        LARGE_VESSELS, SMALL_VESSELS, DISTAL_VESSELS,
        POSTERIOR_FOSSA, SMALL_ARTERIES, VENOUS, GROUP_NAMES,
    )

    group_sets = {
        "large": LARGE_VESSELS,
        "small": SMALL_VESSELS,
        "distal": DISTAL_VESSELS,
        "posterior_fossa": POSTERIOR_FOSSA,
        "small_arteries": SMALL_ARTERIES,
        "venous": VENOUS,
    }

    group_dice = {}
    group_cldice = {}
    for gname, gids in group_sets.items():
        group_dice[gname] = [d for cid, name in VESSEL_CLASSES.items() if cid in gids for d in class_dice[name]]
        group_cldice[gname] = [d for cid, name in VESSEL_CLASSES.items() if cid in gids for d in class_cldice[name]]

    print(f"\n  {'Group':<50} {'DSC':>8} {'clDice':>8} {'N':>5}")
    print(f"  {'-'*74}")
    for gname in group_sets:
        d_vals = group_dice[gname]
        c_vals = group_cldice[gname]
        label = GROUP_NAMES.get(gname, gname)
        if d_vals:
            print(f"  {label:<50} {np.mean(d_vals):>8.4f} {np.mean(c_vals):>8.4f} {len(d_vals):>5}")
        else:
            print(f"  {label:<50} {'n/a':>8} {'n/a':>8} {0:>5}")

    # Gap: large vs each other group
    if group_dice["large"]:
        print(f"\n  Gaps (large vessel DSC - other group DSC):")
        large_mean = np.mean(group_dice["large"])
        for gname in ["small", "distal", "posterior_fossa", "small_arteries", "venous"]:
            if group_dice[gname]:
                gap = large_mean - np.mean(group_dice[gname])
                print(f"    vs {GROUP_NAMES.get(gname, gname)}: {gap:+.4f}")

    # Save results to JSON
    out_dir = os.path.dirname(args.checkpoint)
    results = {
        "global": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in global_metrics.items()},
        "per_class": {
            name: {
                "dice": float(np.mean(class_dice[name])) if class_dice[name] else None,
                "cldice": float(np.mean(class_cldice[name])) if class_cldice[name] else None,
                "betti0_error": float(np.mean(class_b0[name])) if class_b0[name] else None,
                "n_volumes": len(class_dice[name]),
            }
            for name in VESSEL_CLASSES.values()
        },
        "grouped": {
            gname: {
                "dice": float(np.mean(group_dice[gname])) if group_dice[gname] else None,
                "cldice": float(np.mean(group_cldice[gname])) if group_cldice[gname] else None,
                "n": len(group_dice[gname]),
            }
            for gname in group_sets
        },
    }

    results_path = os.path.join(out_dir, "stratified_eval.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
