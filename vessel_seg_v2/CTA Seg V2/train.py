"""
train.py -- Main training loop for cerebral vessel segmentation.

Supports single-GPU and multi-GPU (DDP) training transparently.

Usage:
    # Single GPU
    python train.py --data_dir /path/to/topcow2024

    # Multi-GPU (auto-detected via torchrun)
    torchrun --nproc_per_node=4 train.py --data_dir /path/to/topcow2024

    # Examples
    torchrun --nproc_per_node=4 train.py --data_dir /data/topcow2024 --loss dice_ce_cldice
    python train.py --data_dir /data/topcow2024 --resume runs/<run>/latest_checkpoint.pth

    # Fine-tune on TopBrain (40-class labels, separate dataset)
    torchrun --nproc_per_node=8 train.py --data_dir /data/topcow2024 \
        --finetune runs/<run>/best_model.pth --topbrain_dir /data/topbrain
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

# ── Project imports ───────────────────────────────────────────────────────────
from configs.default import CONFIG
from data.dataset import (
    discover_cases,
    discover_topbrain_cases,
    discover_av_pseudo_cases,
    train_val_split,
    TopCoWPatchDataset,
)
from models.unet3d import UNet3D
from losses.losses import build_loss
from utils.metrics import evaluate_volume
from utils.inference import sliding_window_inference


# ── DDP helpers ──────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (local_rank, world_size)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1  # single-GPU fallback

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    """True on rank 0 or single-GPU."""
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model):
    """Get raw model from DDP wrapper."""
    return model.module if isinstance(model, DDP) else model


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for vessel segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TopCoW dataset root")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None, choices=[
        # Midterm topology-aware family
        "dice_ce", "dice_ce_cldice", "dice_ce_skeleton",
        # Final-report reconstruction-aware + combination family
        "dice_ce_ssim", "dice_ce_mse_dt", "dice_ce_perceptual",
        "dice_ce_cldice_ssim",
    ])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patch_size", type=int, nargs=3, default=None)
    parser.add_argument("--base_filters", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume training from")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint .pth to fine-tune from (loads model weights only, fresh optimizer)")
    parser.add_argument("--topbrain_dir", type=str, default=None,
                        help="Path to TopBrain dataset root (uses 40-class labels instead of TopCoW 13-class)")
    parser.add_argument("--av_pseudo_images", type=str, default=None,
                        help="Directory of CTA images for 3-class artery/vein pseudo-label fine-tuning")
    parser.add_argument("--av_pseudo_labels", type=str, default=None,
                        help="Directory of 3-class A/V labels (0=bg, 1=artery, 2=vein)")
    parser.add_argument("--include_patients", type=str, default=None,
                        help="Comma-separated patient IDs to include (e.g. 001,002,010). Others excluded from training.")
    parser.add_argument("--patches_per_volume", type=int, default=None,
                        help="Override patches sampled per volume per epoch")
    parser.add_argument("--foreground_ratio", type=float, default=None,
                        help="Override fraction of patches centered on foreground voxels (0.0-1.0)")
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
    if args.patches_per_volume:
        cfg["patches_per_volume"] = args.patches_per_volume
    if args.foreground_ratio is not None:
        cfg["foreground_ratio"] = args.foreground_ratio
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

            # On NaN/inf loss, replace with zero but keep the computation graph
            # so backward() still runs and DDP gradient all-reduce stays in sync.
            # Skipping backward with `continue` causes rank desync and NCCL deadlocks.
            is_nan = not torch.isfinite(loss)
            if is_nan:
                nan_count += 1
                loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

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

            is_nan = not torch.isfinite(loss)
            if is_nan:
                nan_count += 1
                loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

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
    cfg: dict,
    device: torch.device,
    quick: bool = True,
) -> dict:
    """
    Run full-volume inference on validation set and compute metrics.
    Runs on rank 0 only — other ranks wait at a barrier in the caller.

    Pipelines GPU inference with CPU metric computation: while metrics
    run on volume N, volume N+1 is already being inferred on the GPU.

    Args:
        quick: if True (default), only compute Dice during training.
               Skips clDice (skeletonize_3d ~30-60s/vol) and HD95
               (distance_transform_edt ~10-20s/vol). Full metrics
               are computed at the final epoch or via evaluate.py.
    """
    from data.dataset import load_nifti, preprocess_ct, binarize_labels
    from concurrent.futures import ThreadPoolExecutor

    raw_model = unwrap(model)
    raw_model.eval()

    all_metrics = {"dice": []}
    if not quick:
        all_metrics.update({"cldice": [], "hd95": [], "betti0_error": []})

    # Thread pool for CPU-bound metric computation (runs in background
    # while GPU processes the next volume)
    metrics_pool = ThreadPoolExecutor(max_workers=1)
    pending_future = None

    def _collect(future):
        """Collect results from a completed metrics future."""
        metrics = future.result()
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

    for case in val_cases:
        vol, meta = load_nifti(case["image"])
        lbl, _ = load_nifti(case["label"])
        vol = preprocess_ct(vol, tuple(cfg["hu_window"]))
        if cfg["num_classes"] == 2:
            lbl = binarize_labels(lbl)

        pred = sliding_window_inference(
            vol, raw_model,
            patch_size=tuple(cfg["patch_size"]),
            overlap=cfg["sliding_window_overlap"],
            device=str(device),
            num_classes=cfg["num_classes"],
        )

        # Collect previous volume's metrics if ready
        if pending_future is not None:
            _collect(pending_future)

        # Submit this volume's metrics to run on CPU while GPU starts next volume
        spacing = meta.get("spacing", (1, 1, 1))
        pending_future = metrics_pool.submit(
            evaluate_volume, pred, lbl.astype(np.uint8),
            voxel_spacing=spacing, quick=quick,
        )

    # Collect the last volume's metrics
    if pending_future is not None:
        _collect(pending_future)

    metrics_pool.shutdown(wait=True)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)

    # Fine-tuning mode: apply defaults before LR scaling
    is_finetune = args.finetune is not None
    if is_finetune:
        if not args.epochs:
            cfg["epochs"] = cfg.get("finetune_epochs", 150)
        if not args.lr:
            cfg["learning_rate"] = cfg.get("finetune_lr", 1e-4)
        cfg["warmup_epochs"] = cfg.get("finetune_warmup_epochs", 5)

    # DDP setup (no-op for single GPU)
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    # Note: LR is NOT scaled with world size. The effective batch is larger
    # (batch_size * world_size) but the base LR (1e-3) works well as-is.
    # Linear/sqrt scaling caused instability with topology-aware losses.

    # Setup output directory (rank 0 only)
    if is_main():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "finetune_topbrain_" if args.topbrain_dir else ""
        run_name = f"{prefix}{cfg['loss']}_{timestamp}"
        run_dir = os.path.join(cfg["output_dir"], run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
    else:
        run_dir = None

    # Broadcast run_dir from rank 0 to all ranks
    if world_size > 1:
        if is_main():
            run_dir_bytes = run_dir.encode("utf-8")
            size_tensor = torch.tensor([len(run_dir_bytes)], dtype=torch.long, device=device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(size_tensor, src=0)
        if is_main():
            dir_tensor = torch.tensor(
                list(run_dir_bytes), dtype=torch.uint8, device=device
            )
        else:
            dir_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=device)
        dist.broadcast(dir_tensor, src=0)
        if not is_main():
            run_dir = bytes(dir_tensor.cpu().tolist()).decode("utf-8")

    # Logging
    if is_main():
        print(f"Device: {device} (world_size={world_size})")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(rank)}")
            print(f"VRAM: {torch.cuda.get_device_properties(rank).total_memory / 1e9:.1f} GB")
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark: enabled")
    else:
        torch.backends.cudnn.benchmark = True

    # Data: AV pseudo-label → TopBrain → TopCoW precedence
    modality = cfg.get("modality", "ct")
    if args.av_pseudo_images and args.av_pseudo_labels:
        # AV pseudo-label mode implies 3-class (bg/artery/vein); auto-set if caller
        # didn't override, and hard-fail on mismatch so a stale 2-class config
        # doesn't silently truncate vein labels.
        if cfg["num_classes"] == 2:
            if is_main():
                print("AV pseudo mode: overriding num_classes 2 → 3")
            cfg["num_classes"] = 3
        elif cfg["num_classes"] != 3:
            raise RuntimeError(
                f"AV pseudo mode requires num_classes=3, got {cfg['num_classes']}"
            )
        if is_main():
            print(f"\nDiscovering AV pseudo-label cases")
            print(f"  images: {args.av_pseudo_images}")
            print(f"  labels: {args.av_pseudo_labels}")
        all_cases = discover_av_pseudo_cases(args.av_pseudo_images, args.av_pseudo_labels)
        if is_main():
            print(f"Found {len(all_cases)} AV pseudo-label cases (3-class: bg/artery/vein)")
    elif args.topbrain_dir:
        if is_main():
            print(f"\nDiscovering TopBrain cases in {args.topbrain_dir}...")
        all_cases = discover_topbrain_cases(args.topbrain_dir, modality=modality)
        if is_main():
            print(f"Found {len(all_cases)} TopBrain cases (modality: {modality})")
            print(f"  TopBrain labels: 40 vessel classes (vs 13 in TopCoW)")
    else:
        if is_main():
            print(f"\nDiscovering cases in {cfg['data_dir']}...")
        all_cases = discover_cases(cfg["data_dir"], modality=modality)
        if is_main():
            print(f"Found {len(all_cases)} cases (modality filter: {modality})")

    # Sanity check: modality breakdown (only meaningful for TopCoW/TopBrain naming)
    if not (args.av_pseudo_images and args.av_pseudo_labels):
        ct_count = sum(1 for c in all_cases if "topcow_ct_" in os.path.basename(c["image"]))
        mr_count = sum(1 for c in all_cases if "topcow_mr_" in os.path.basename(c["image"]))
        if is_main():
            print(f"  CT cases: {ct_count}, MR cases: {mr_count}")
        if modality == "ct" and mr_count > 0:
            raise RuntimeError(f"Modality filter is 'ct' but {mr_count} MR cases slipped through — check discover_cases()")

    train_cases, val_cases = train_val_split(all_cases, cfg["train_val_split"])
    if is_main():
        print(f"Train: {len(train_cases)}, Val: {len(val_cases)}")

    # Filter to specific patient IDs if requested (e.g. TopBrain-only fine-tuning)
    if args.include_patients:
        patient_ids = set(args.include_patients.split(","))
        def _has_patient_id(case, ids):
            basename = os.path.basename(case["image"])
            for pid in ids:
                if f"_{pid}_" in basename:
                    return True
            return False

        train_before = len(train_cases)
        train_cases = [c for c in train_cases if _has_patient_id(c, patient_ids)]
        val_cases = [c for c in val_cases if _has_patient_id(c, patient_ids)]
        if is_main():
            print(f"Patient filter: {train_before} -> {len(train_cases)} train, {len(val_cases)} val")

    train_dataset = TopCoWPatchDataset(train_cases, cfg, augment=True)

    # DDP: DistributedSampler ensures each rank gets different patches
    sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
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

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main():
        print(f"\nModel: 3D U-Net with {n_params:.1f}M parameters")
        print(f"Loss: {cfg['loss']}")
        print(f"Patch size: {cfg['patch_size']}")
        print(f"Deep supervision: {cfg['deep_supervision']}")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs, effective batch size = {cfg['batch_size'] * world_size}")
            print(f"LR: {cfg['learning_rate']:.1e} (NOT scaled with world_size by design — see comment above)")

    # Loss, optimizer, scheduler. Move criterion to device so any nn.Module
    # buffers (SSIM Gaussian window) or sub-modules (VGG perceptual backbone)
    # land on GPU; without this, SSIM/perceptual losses crash on first
    # forward pass with a CPU/GPU device-mismatch error.
    criterion = build_loss(cfg).to(device)
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
    if args.finetune:
        # Fine-tune: load model weights only, fresh optimizer/scheduler
        if is_main():
            print(f"\nFine-tuning from checkpoint: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=device)
        unwrap(model).load_state_dict(ckpt["model_state_dict"])
        pretrain_dice = ckpt.get("best_dice", 0.0)
        pretrain_epoch = ckpt.get("epoch", "?")
        if is_main():
            print(f"Loaded weights from epoch {pretrain_epoch} (pretrain Dice: {pretrain_dice:.4f})")
            print(f"Starting fresh optimizer at LR={cfg['learning_rate']:.1e}")
    elif args.resume:
        if is_main():
            print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        unwrap(model).load_state_dict(ckpt["model_state_dict"])
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
        if is_main():
            print(f"Resuming from epoch {start_epoch}, best Dice so far: {best_dice:.4f}")

    # Training loop
    log_path = os.path.join(run_dir, "training_log.json")

    if is_main():
        print(f"\n{'='*60}")
        print(f"Starting training for {cfg['epochs']} epochs (from epoch {start_epoch})")
        print(f"{'='*60}\n")

    epoch_times = []  # for ETA calculation
    training_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # DDP: set epoch for sampler to re-shuffle
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg
        )

        # Validate periodically (full-volume inference is slow)
        val_metrics = {}
        val_interval = cfg.get("val_interval", 10)
        if epoch % val_interval == 0 or epoch == cfg["epochs"]:
            is_final = (epoch == cfg["epochs"])

            # Validate on rank 0 only — other ranks wait at the barrier below.
            # Distributing val across ranks causes NCCL timeouts when some
            # ranks have 0 cases and finish instantly while others are still
            # doing sliding window inference.
            if is_main():
                mode = "full metrics" if is_final else "quick (Dice only)"
                print(f"  Running validation ({mode})...")
                val_metrics = validate(
                    model, val_cases, cfg, device,
                    quick=not is_final,
                )

                if val_metrics.get("dice", 0) > best_dice:
                    best_dice = val_metrics["dice"]
                    patience_counter = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": unwrap(model).state_dict(),
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

            # Sync all ranks — other ranks wait here while rank 0 validates
            if world_size > 1:
                # Broadcast best_dice and patience_counter from rank 0
                state = torch.tensor(
                    [best_dice, float(patience_counter)], device=device
                )
                dist.broadcast(state, src=0)
                best_dice = state[0].item()
                patience_counter = int(state[1].item())

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if is_main():
            # ETA calculation
            epoch_times.append(elapsed)
            epochs_remaining = cfg["epochs"] - epoch
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
                line += f"\n  Val  DSC: {val_metrics['dice']:.4f}"
                if "cldice" in val_metrics:
                    line += (
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

            # Save resumable checkpoint every epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap(model).state_dict(),
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

            # Save log at validation epochs
            if epoch % val_interval == 0:
                with open(log_path, "w") as f:
                    json.dump(history, f, indent=2)

        # Early stopping check (all ranks must agree)
        early_stopping_patience = cfg.get("early_stopping_patience", 0)
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            if is_main():
                print(f"\nEarly stopping: no improvement for {patience_counter} validation cycles.")
            break

    # Save final model (rank 0 only)
    if is_main():
        torch.save(
            {
                "epoch": cfg["epochs"],
                "model_state_dict": unwrap(model).state_dict(),
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

    cleanup_ddp()


if __name__ == "__main__":
    main()
