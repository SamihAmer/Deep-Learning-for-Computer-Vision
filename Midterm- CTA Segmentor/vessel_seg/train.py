"""
train.py -- Main training loop for cerebral vessel segmentation.

Usage:
    python train.py --data_dir /path/to/topcow2024
    python train.py --data_dir /path/to/topcow2024 --loss dice_ce_cldice
    python train.py --data_dir /path/to/topcow2024 --loss dice_ce_skeleton --epochs 300
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# ── Project imports ───────────────────────────────────────────────────────────
from configs.default import CONFIG
from data.dataset import discover_cases, train_val_split, TopCoWPatchDataset
from models.unet3d import UNet3D
from losses.losses import build_loss
from utils.metrics import evaluate_volume
from utils.inference import sliding_window_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for vessel segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TopCoW dataset root")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None, choices=["dice_ce", "dice_ce_cldice", "dice_ce_skeleton"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patch_size", type=int, nargs=3, default=None)
    parser.add_argument("--base_filters", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    """Override config values with CLI arguments."""
    cfg = cfg.copy()
    cfg["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.loss:
        cfg["loss"] = args.loss
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["learning_rate"] = args.lr
    if args.patch_size:
        cfg["patch_size"] = tuple(args.patch_size)
    if args.base_filters:
        cfg["base_filters"] = args.base_filters
    if args.no_amp:
        cfg["use_amp"] = False
    return cfg


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Build learning rate scheduler."""
    total_steps = cfg["epochs"] * steps_per_epoch

    if cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
    elif cfg["scheduler"] == "poly":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_steps, power=0.9
        )
    elif cfg["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg['scheduler']}")


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
) -> dict:
    """Run one training epoch. Returns dict of averaged metrics."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg["use_amp"]:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            optimizer.step()

        if cfg["scheduler"] != "plateau":
            scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    return {"train_loss": running_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_cases: list,
    criterion: nn.Module,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Run full-volume inference on validation set and compute metrics.
    This is slower than patch-based validation but gives true performance.
    """
    from data.dataset import load_nifti, preprocess_ct, binarize_labels

    model.eval()
    all_metrics = {"dice": [], "cldice": [], "hd95": [], "betti0_error": []}

    for case in val_cases:
        vol, meta = load_nifti(case["image"])
        lbl, _ = load_nifti(case["label"])
        vol = preprocess_ct(vol, tuple(cfg["hu_window"]))
        if cfg["num_classes"] == 2:
            lbl = binarize_labels(lbl)

        pred = sliding_window_inference(
            vol, model,
            patch_size=tuple(cfg["patch_size"]),
            overlap=cfg["sliding_window_overlap"],
            device=str(device),
            num_classes=cfg["num_classes"],
        )

        spacing = meta.get("spacing", (1, 1, 1))
        metrics = evaluate_volume(pred, lbl.astype(np.uint8), voxel_spacing=spacing)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    # Average
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['loss']}_{timestamp}"
    run_dir = os.path.join(cfg["output_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_mem / 1e9:.1f} GB")

    # Data
    print(f"\nDiscovering cases in {cfg['data_dir']}...")
    all_cases = discover_cases(cfg["data_dir"])
    print(f"Found {len(all_cases)} cases")
    train_cases, val_cases = train_val_split(all_cases, cfg["train_val_split"])
    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}")

    train_dataset = TopCoWPatchDataset(train_cases, cfg, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = UNet3D(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_filters=cfg["base_filters"],
        num_stages=cfg["num_stages"],
        deep_supervision=cfg["deep_supervision"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: 3D U-Net with {n_params:.1f}M parameters")
    print(f"Loss: {cfg['loss']}")
    print(f"Patch size: {cfg['patch_size']}")
    print(f"Deep supervision: {cfg['deep_supervision']}")

    # Loss, optimizer, scheduler
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg["use_amp"])

    # Training loop
    best_dice = 0.0
    log_path = os.path.join(run_dir, "training_log.json")
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg
        )

        # Validate every 10 epochs (full-volume inference is slow)
        val_metrics = {}
        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            print(f"  Running validation (full-volume inference)...")
            val_metrics = validate(model, val_cases, criterion, cfg, device)

            # Save best model
            if val_metrics.get("dice", 0) > best_dice:
                best_dice = val_metrics["dice"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "config": cfg,
                    },
                    os.path.join(run_dir, "best_model.pth"),
                )
                print(f"  New best model saved (Dice: {best_dice:.4f})")

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log
        entry = {"epoch": epoch, "lr": lr, "time": elapsed, **train_metrics, **val_metrics}
        history.append(entry)

        # Print
        line = f"Epoch {epoch:3d}/{cfg['epochs']} | Loss: {train_metrics['train_loss']:.4f} | LR: {lr:.2e} | {elapsed:.0f}s"
        if val_metrics:
            line += (
                f" | DSC: {val_metrics['dice']:.4f}"
                f" | clDice: {val_metrics['cldice']:.4f}"
                f" | HD95: {val_metrics['hd95']:.1f}"
                f" | B0err: {val_metrics['betti0_error']:.1f}"
            )
        print(line)

        # Save log periodically
        if epoch % 10 == 0:
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)

    # Save final model
    torch.save(
        {
            "epoch": cfg["epochs"],
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        os.path.join(run_dir, "final_model.pth"),
    )

    # Save final log
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best validation Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
