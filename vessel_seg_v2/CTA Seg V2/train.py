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
from torch.cuda.amp import GradScaler

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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume training from")
    parser.add_argument("--modality", type=str, default=None, choices=["ct", "mr", "all"],
                        help="Filter dataset by modality (default from config: ct)")
    parser.add_argument("--val_interval", type=int, default=None,
                        help="Validate every N epochs (default from config: 5)")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Stop after N val cycles without improvement (0=disabled, default: 5)")
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
    if args.modality:
        cfg["modality"] = args.modality
    if args.val_interval:
        cfg["val_interval"] = args.val_interval
    if args.early_stopping_patience is not None:
        cfg["early_stopping_patience"] = args.early_stopping_patience
    return cfg


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Build learning rate scheduler with optional linear warmup."""
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg.get("warmup_epochs", 0) * steps_per_epoch

    if cfg["scheduler"] == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
        )
    elif cfg["scheduler"] == "poly":
        main_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_steps - warmup_steps, power=0.9
        )
    elif cfg["scheduler"] == "plateau":
        # Plateau doesn't support SequentialLR, return as-is
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg['scheduler']}")

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )

    return main_scheduler


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
    running_dice = 0.0
    n_batches = 0
    nan_count = 0
    total_forward_ms = 0.0
    total_backward_ms = 0.0
    grad_norms = []

    # Reset peak GPU memory tracking for this epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg["use_amp"]:
            # Forward pass
            torch.cuda.synchronize()
            t_fwd = time.time()
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            torch.cuda.synchronize()
            total_forward_ms += (time.time() - t_fwd) * 1000

            # Skip batch if loss is NaN/inf to prevent poisoning weights
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                nan_count += 1
                continue

            # Backward pass
            t_bwd = time.time()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            total_backward_ms += (time.time() - t_bwd) * 1000
        else:
            # Forward pass
            torch.cuda.synchronize()
            t_fwd = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)
            torch.cuda.synchronize()
            total_forward_ms += (time.time() - t_fwd) * 1000

            # Skip batch if loss is NaN/inf
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                nan_count += 1
                continue

            # Backward pass
            t_bwd = time.time()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()
            total_backward_ms += (time.time() - t_bwd) * 1000

        if cfg["scheduler"] != "plateau":
            scheduler.step()

        running_loss += loss.item()
        grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

        # Quick training Dice from the full-res output (no deep supervision heads)
        with torch.no_grad():
            preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred_mask = preds.argmax(dim=1)  # (B, D, H, W)
            lbl = labels
            # Binary Dice on foreground
            intersection = ((pred_mask == 1) & (lbl == 1)).sum().float()
            union = (pred_mask == 1).sum().float() + (lbl == 1).sum().float()
            batch_dice = (2.0 * intersection / (union + 1e-7)).item()
            running_dice += batch_dice

        n_batches += 1

    # Peak GPU memory
    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

    return {
        "train_loss": running_loss / max(n_batches, 1),
        "train_dice": running_dice / max(n_batches, 1),
        "avg_forward_ms": total_forward_ms / max(n_batches, 1),
        "avg_backward_ms": total_backward_ms / max(n_batches, 1),
        "avg_grad_norm": avg_grad_norm,
        "peak_vram_gb": round(peak_mem_gb, 2),
        "nan_batches": nan_count,
    }


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
        print(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark: enabled")

    # Data
    print(f"\nDiscovering cases in {cfg['data_dir']}...")
    modality = cfg.get("modality", "ct")
    all_cases = discover_cases(cfg["data_dir"], modality=modality)
    print(f"Found {len(all_cases)} cases (modality filter: {modality})")

    # Sanity check: log modality breakdown of discovered cases
    ct_count = sum(1 for c in all_cases if "topcow_ct_" in os.path.basename(c["image"]))
    mr_count = sum(1 for c in all_cases if "topcow_mr_" in os.path.basename(c["image"]))
    print(f"  CT cases: {ct_count}, MR cases: {mr_count}")
    if modality == "ct" and mr_count > 0:
        raise RuntimeError(f"Modality filter is 'ct' but {mr_count} MR cases slipped through — check discover_cases()")

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
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["use_amp"])

    # Resume from checkpoint if specified
    start_epoch = 1
    best_dice = 0.0
    patience_counter = 0
    history = []
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"Resuming from epoch {start_epoch}, best Dice so far: {best_dice:.4f}")

    # Training loop
    log_path = os.path.join(run_dir, "training_log.json")

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg['epochs']} epochs (from epoch {start_epoch})")
    print(f"{'='*60}\n")

    epoch_times = []  # for ETA calculation
    training_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg
        )

        # Validate periodically (full-volume inference is slow)
        val_metrics = {}
        val_interval = cfg.get("val_interval", 10)
        if epoch % val_interval == 0 or epoch == cfg["epochs"]:
            print(f"  Running validation (full-volume inference)...")
            val_metrics = validate(model, val_cases, criterion, cfg, device)

            # Save best model
            if val_metrics.get("dice", 0) > best_dice:
                best_dice = val_metrics["dice"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_dice": best_dice,
                        "patience_counter": patience_counter,
                        "history": history,
                        "config": cfg,
                    },
                    os.path.join(run_dir, "best_model.pth"),
                )
                print(f"  New best model saved (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                early_stopping_patience = cfg.get("early_stopping_patience", 0)
                if early_stopping_patience > 0:
                    print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # ETA calculation (based on training-only epochs, excludes validation time)
        epoch_times.append(elapsed)
        epochs_remaining = cfg["epochs"] - epoch
        # Use rolling average of last 10 epochs for more stable ETA
        recent = epoch_times[-10:]
        avg_epoch_time = sum(recent) / len(recent)
        eta_seconds = avg_epoch_time * epochs_remaining
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)

        # Log
        entry = {
            "epoch": epoch, "lr": lr, "time": elapsed,
            **train_metrics, **val_metrics,
        }
        history.append(entry)

        # Print main line
        line = (
            f"Epoch {epoch:3d}/{cfg['epochs']}"
            f" | Loss: {train_metrics['train_loss']:.4f}"
            f" | TrDice: {train_metrics['train_dice']:.4f}"
            f" | LR: {lr:.2e}"
            f" | {elapsed:.0f}s"
            f" | ETA: {eta_h}h{eta_m:02d}m"
        )
        if train_metrics["nan_batches"] > 0:
            line += f" | NaN: {train_metrics['nan_batches']}"
        if val_metrics:
            line += (
                f"\n  Val  DSC: {val_metrics['dice']:.4f}"
                f" | clDice: {val_metrics['cldice']:.4f}"
                f" | HD95: {val_metrics['hd95']:.1f}"
                f" | B0err: {val_metrics['betti0_error']:.1f}"
            )
        # Print GPU/timing details at validation epochs
        if epoch % val_interval == 0 or epoch == start_epoch:
            line += (
                f"\n  GPU  VRAM: {train_metrics['peak_vram_gb']:.1f}GB"
                f" | Fwd: {train_metrics['avg_forward_ms']:.0f}ms"
                f" | Bwd: {train_metrics['avg_backward_ms']:.0f}ms"
                f" | GradNorm: {train_metrics['avg_grad_norm']:.2f}"
            )
        print(line)

        # Save log and resumable checkpoint at validation epochs
        if epoch % val_interval == 0:
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_dice": best_dice,
                    "patience_counter": patience_counter,
                    "history": history,
                    "config": cfg,
                },
                os.path.join(run_dir, "latest_checkpoint.pth"),
            )

        # Early stopping check
        early_stopping_patience = cfg.get("early_stopping_patience", 0)
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping: no improvement for {patience_counter} validation cycles.")
            break

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

    total_time = time.time() - training_start
    total_h = int(total_time // 3600)
    total_m = int((total_time % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"Training complete in {total_h}h{total_m:02d}m")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
