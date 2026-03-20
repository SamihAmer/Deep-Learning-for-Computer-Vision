"""
Default configuration for cerebral vessel segmentation.
Override values by importing and updating the dict, or by passing CLI args.
"""

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    "data_dir": "/path/to/topcow2024",       # root of extracted TopCoW dataset
    "output_dir": "./runs",                    # checkpoints + logs
    "num_workers": 4,
    "train_val_split": 0.8,                    # 80/20 random split

    # ── Patch sampling ────────────────────────────────────────────────────
    "patch_size": (128, 128, 128),             # voxel crop dimensions
    "patches_per_volume": 8,                   # patches sampled per volume per epoch
    "foreground_ratio": 0.33,                  # fraction of patches centered on vessel

    # ── Preprocessing ─────────────────────────────────────────────────────
    "hu_window": (0, 600),                     # HU clipping range for CTA
    "target_spacing": None,                    # resample to isotropic (e.g. [0.5,0.5,0.5]) or None to keep native

    # ── Model ─────────────────────────────────────────────────────────────
    "in_channels": 1,
    "num_classes": 2,                          # binary: background + vessel (set 14 for multiclass CoW)
    "base_filters": 32,                        # first conv layer filters (doubles each stage)
    "num_stages": 5,                           # encoder depth
    "deep_supervision": True,

    # ── Loss ──────────────────────────────────────────────────────────────
    "loss": "dice_ce",                         # "dice_ce" | "dice_ce_cldice" | "dice_ce_skeleton"
    "cldice_alpha": 0.5,                       # weight for topology loss when combined
    "skeleton_recall_alpha": 0.5,

    # ── Training ──────────────────────────────────────────────────────────
    "epochs": 300,
    "batch_size": 2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": "cosine",                     # "cosine" | "poly" | "plateau"
    "warmup_epochs": 10,
    "use_amp": True,                           # mixed precision

    # ── Augmentation ──────────────────────────────────────────────────────
    "aug_rotation_range": 15,                  # degrees
    "aug_scale_range": (0.85, 1.15),
    "aug_elastic": True,
    "aug_gamma_range": (0.7, 1.5),
    "aug_mirror": False,                       # disabled for left/right anatomical labels

    # ── Inference ─────────────────────────────────────────────────────────
    "sliding_window_overlap": 0.5,             # overlap ratio for patch stitching
    "tta": False,                              # test-time augmentation (flips)
}
