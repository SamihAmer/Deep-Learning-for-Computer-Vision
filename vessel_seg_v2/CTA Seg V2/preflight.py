"""
preflight.py -- Pre-training sanity checks for vessel segmentation.

Run this on a new machine (e.g. AWS GPU instance) before launching train.py
to catch issues early: missing data, wrong modality, spacing outliers,
broken dependencies, GPU problems, etc.

Usage:
    python preflight.py --data_dir /path/to/topcow2024
    python preflight.py --data_dir /path/to/topcow2024 --quick   # skip forward pass test
"""

import os
import sys
import argparse
import time
from collections import defaultdict

# ─── Helpers ────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

_fail_count = 0
_warn_count = 0


def check(ok: bool, msg: str, fatal: bool = True):
    global _fail_count
    if ok:
        print(f"  {PASS} {msg}")
    else:
        tag = FAIL if fatal else WARN
        print(f"  {tag} {msg}")
        if fatal:
            _fail_count += 1
        else:
            global _warn_count
            _warn_count += 1


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─── Checks ────────────────────────────────────────────────────────────────

def check_dependencies():
    section("1. Dependencies")
    deps = {
        "numpy": "numpy",
        "torch": "torch",
        "SimpleITK": "SimpleITK",
        "scipy": "scipy",
        "skimage": "skimage",
    }
    for name, module in deps.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            print(f"  {PASS} {name} ({ver})")
        except ImportError:
            print(f"  {FAIL} {name} not installed")
            global _fail_count
            _fail_count += 1

    # Check project imports
    try:
        from configs.default import CONFIG
        from data.dataset import discover_cases, train_val_split, TopCoWPatchDataset
        from models.unet3d import UNet3D
        from losses.losses import build_loss
        from utils.metrics import evaluate_volume
        from utils.inference import sliding_window_inference
        print(f"  {PASS} All project modules importable")
    except Exception as e:
        print(f"  {FAIL} Project import error: {e}")
        _fail_count += 1


def check_gpu():
    section("2. GPU & CUDA")
    import torch

    has_cuda = torch.cuda.is_available()
    check(has_cuda, f"CUDA available: {has_cuda}")
    if not has_cuda:
        return

    n_gpus = torch.cuda.device_count()
    print(f"  {INFO} GPU count: {n_gpus}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  {INFO} GPU {i}: {name} ({vram_gb:.1f} GB)")
        check(vram_gb >= 8.0, f"GPU {i} VRAM >= 8 GB (have {vram_gb:.1f} GB)", fatal=False)

    # cuDNN
    cudnn = torch.backends.cudnn.is_available()
    print(f"  {INFO} cuDNN available: {cudnn}")
    if cudnn:
        print(f"  {INFO} cuDNN version: {torch.backends.cudnn.version()}")

    # AMP support
    try:
        with torch.amp.autocast("cuda"):
            pass
        print(f"  {PASS} AMP (mixed precision) supported")
    except Exception as e:
        print(f"  {WARN} AMP not available: {e}")


def check_data(data_dir: str):
    section("3. Data Directory")
    check(os.path.isdir(data_dir), f"Data dir exists: {data_dir}")

    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    check(os.path.isdir(images_dir), f"imagesTr/ exists")
    check(os.path.isdir(labels_dir), f"labelsTr/ exists")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return [], {}

    # Discover cases
    from data.dataset import discover_cases
    all_cases = discover_cases(data_dir, modality="all")
    ct_cases = discover_cases(data_dir, modality="ct")
    mr_cases = discover_cases(data_dir, modality="mr")

    print(f"  {INFO} Total cases (all): {len(all_cases)}")
    print(f"  {INFO} CT cases:          {len(ct_cases)}")
    print(f"  {INFO} MR cases:          {len(mr_cases)}")

    check(len(ct_cases) > 0, "At least 1 CT case found")
    check(len(mr_cases) == 0 or len(ct_cases) > 0,
          "CT cases exist (MR will be filtered out during training)")

    # Verify no MR leaks into CT set
    mr_leak = [c for c in ct_cases if "topcow_mr_" in os.path.basename(c["image"])]
    check(len(mr_leak) == 0,
          f"No MRA cases in CT-filtered set (found {len(mr_leak)} leaks)")

    # Check all CT images have labels
    missing_labels = []
    for c in ct_cases:
        if not os.path.isfile(c["label"]):
            missing_labels.append(c["image"])
    check(len(missing_labels) == 0,
          f"All CT images have matching labels ({len(missing_labels)} missing)")
    for m in missing_labels[:5]:
        print(f"    Missing label for: {os.path.basename(m)}")

    return ct_cases, {"ct": len(ct_cases), "mr": len(mr_cases)}


def check_volumes(ct_cases: list, n_sample: int = 5):
    """Load a sample of volumes and check HU ranges, spacing, shapes."""
    section("4. Volume Integrity (sampling)")
    if not ct_cases:
        print(f"  {FAIL} No cases to check")
        return

    import numpy as np
    from data.dataset import load_nifti

    n_sample = min(n_sample, len(ct_cases))
    # Sample evenly across dataset
    indices = [int(i * len(ct_cases) / n_sample) for i in range(n_sample)]

    spacings = []
    shapes = []
    hu_ranges = []
    label_classes_all = []

    print(f"  {INFO} Sampling {n_sample}/{len(ct_cases)} volumes...")

    for idx in indices:
        case = ct_cases[idx]
        basename = os.path.basename(case["image"])
        try:
            vol, meta = load_nifti(case["image"])
            lbl, _ = load_nifti(case["label"])
        except Exception as e:
            print(f"  {FAIL} Failed to load {basename}: {e}")
            global _fail_count
            _fail_count += 1
            continue

        sp = meta["spacing"]
        spacings.append(sp)
        shapes.append(vol.shape)
        hu_ranges.append((float(vol.min()), float(vol.max())))
        label_classes = np.unique(lbl).astype(int).tolist()
        label_classes_all.append(label_classes)

        # Check HU range looks like CT (not MRA)
        is_ct_range = vol.min() < -500 and vol.max() > 200
        check(is_ct_range,
              f"{basename}: HU=[{vol.min():.0f}, {vol.max():.0f}], "
              f"spacing={tuple(round(s, 3) for s in sp)}, "
              f"shape={vol.shape}, labels={label_classes}",
              fatal=False)

        # Check labels have foreground
        fg_ratio = (lbl > 0).sum() / lbl.size
        check(fg_ratio > 0.0001,
              f"  Foreground ratio: {fg_ratio:.4f} ({(lbl > 0).sum()} voxels)",
              fatal=False)

    # Spacing analysis across sampled volumes
    section("5. Spacing Consistency")
    if len(spacings) < 2:
        print(f"  {INFO} Not enough volumes to analyze spacing consistency")
        return

    spacings_arr = np.array(spacings)
    sp_mean = spacings_arr.mean(axis=0)
    sp_std = spacings_arr.std(axis=0)
    sp_min = spacings_arr.min(axis=0)
    sp_max = spacings_arr.max(axis=0)
    sp_ratio = sp_max / (sp_min + 1e-8)  # max/min ratio per axis

    print(f"  {INFO} Spacing stats (D, H, W) across {len(spacings)} volumes:")
    print(f"         Mean:  ({sp_mean[0]:.3f}, {sp_mean[1]:.3f}, {sp_mean[2]:.3f}) mm")
    print(f"         Std:   ({sp_std[0]:.3f}, {sp_std[1]:.3f}, {sp_std[2]:.3f}) mm")
    print(f"         Min:   ({sp_min[0]:.3f}, {sp_min[1]:.3f}, {sp_min[2]:.3f}) mm")
    print(f"         Max:   ({sp_max[0]:.3f}, {sp_max[1]:.3f}, {sp_max[2]:.3f}) mm")
    print(f"         Ratio: ({sp_ratio[0]:.2f}x, {sp_ratio[1]:.2f}x, {sp_ratio[2]:.2f}x)")

    # Flag if spacing varies more than 2x on any axis
    for ax, name in enumerate(["D (slice)", "H", "W"]):
        check(sp_ratio[ax] < 2.0,
              f"Spacing variation on {name} axis < 2x (ratio: {sp_ratio[ax]:.2f}x)",
              fatal=False)

    # Flag anisotropy
    for i, sp in enumerate(spacings):
        aniso = max(sp) / (min(sp) + 1e-8)
        if aniso > 3.0:
            basename = os.path.basename(ct_cases[indices[i]]["image"])
            print(f"  {WARN} {basename} is highly anisotropic: "
                  f"spacing={tuple(round(s, 3) for s in sp)}, ratio={aniso:.1f}x")


def check_full_spacing(ct_cases: list):
    """Load ALL volumes to get complete spacing distribution."""
    section("5b. Full Spacing Survey (all volumes)")
    if len(ct_cases) <= 5:
        print(f"  {INFO} Skipped (already checked all {len(ct_cases)} volumes above)")
        return

    import numpy as np
    from data.dataset import load_nifti
    import SimpleITK as sitk

    print(f"  {INFO} Reading headers from all {len(ct_cases)} volumes (metadata only)...")
    spacings = []
    for case in ct_cases:
        try:
            # Read only header, not full volume data
            reader = sitk.ImageFileReader()
            reader.SetFileName(case["image"])
            reader.ReadImageInformation()
            sp = reader.GetSpacing()[::-1]  # (D, H, W)
            spacings.append(sp)
        except Exception as e:
            print(f"  {WARN} Could not read header: {os.path.basename(case['image'])}: {e}")

    if not spacings:
        return

    spacings_arr = np.array(spacings)
    sp_min = spacings_arr.min(axis=0)
    sp_max = spacings_arr.max(axis=0)
    sp_mean = spacings_arr.mean(axis=0)
    sp_median = np.median(spacings_arr, axis=0)
    sp_ratio = sp_max / (sp_min + 1e-8)

    print(f"  {INFO} Full spacing stats (D, H, W) across {len(spacings)} volumes:")
    print(f"         Mean:   ({sp_mean[0]:.3f}, {sp_mean[1]:.3f}, {sp_mean[2]:.3f}) mm")
    print(f"         Median: ({sp_median[0]:.3f}, {sp_median[1]:.3f}, {sp_median[2]:.3f}) mm")
    print(f"         Min:    ({sp_min[0]:.3f}, {sp_min[1]:.3f}, {sp_min[2]:.3f}) mm")
    print(f"         Max:    ({sp_max[0]:.3f}, {sp_max[1]:.3f}, {sp_max[2]:.3f}) mm")
    print(f"         Ratio:  ({sp_ratio[0]:.2f}x, {sp_ratio[1]:.2f}x, {sp_ratio[2]:.2f}x)")

    for ax, name in enumerate(["D (slice)", "H", "W"]):
        check(sp_ratio[ax] < 2.0,
              f"Full dataset spacing variation on {name} < 2x (ratio: {sp_ratio[ax]:.2f}x)",
              fatal=False)

    # Flag outliers (> 2 std from mean)
    outliers = []
    for i, sp in enumerate(spacings):
        z_scores = np.abs((sp - sp_mean) / (spacings_arr.std(axis=0) + 1e-8))
        if np.any(z_scores > 2.0):
            outliers.append((ct_cases[i]["image"], sp, z_scores))
    if outliers:
        print(f"  {WARN} {len(outliers)} spacing outlier(s) (>2 std from mean):")
        for path, sp, z in outliers[:10]:
            print(f"         {os.path.basename(path)}: "
                  f"spacing=({sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f})")
    else:
        print(f"  {PASS} No spacing outliers detected")


def check_train_val_split(ct_cases: list):
    section("6. Train/Val Split")
    from data.dataset import train_val_split
    from configs.default import CONFIG

    ratio = CONFIG["train_val_split"]
    train, val = train_val_split(ct_cases, ratio)
    print(f"  {INFO} Split ratio: {ratio}")
    print(f"  {INFO} Train: {len(train)}, Val: {len(val)}")

    check(len(train) > 0, "Training set is non-empty")
    check(len(val) > 0, "Validation set is non-empty")

    # Check no overlap
    train_imgs = {c["image"] for c in train}
    val_imgs = {c["image"] for c in val}
    overlap = train_imgs & val_imgs
    check(len(overlap) == 0, f"No train/val overlap ({len(overlap)} shared)")


def check_forward_pass():
    section("7. Model Forward Pass")
    import torch
    from configs.default import CONFIG
    from models.unet3d import UNet3D
    from losses.losses import build_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ps = CONFIG["patch_size"]

    model = UNet3D(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG["base_filters"],
        num_stages=CONFIG["num_stages"],
        deep_supervision=CONFIG["deep_supervision"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {INFO} Model params: {n_params:.1f}M")

    # Forward pass with dummy data
    dummy = torch.randn(1, 1, *ps, device=device)
    dummy_lbl = torch.zeros(1, *ps, dtype=torch.long, device=device)

    try:
        with torch.amp.autocast("cuda", enabled=CONFIG["use_amp"]):
            t0 = time.time()
            outputs = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_ms = (time.time() - t0) * 1000

        if isinstance(outputs, (list, tuple)):
            print(f"  {INFO} Deep supervision outputs: {len(outputs)} heads")
            for i, o in enumerate(outputs):
                print(f"         Head {i}: {tuple(o.shape)}")
            # Check full-res head matches input spatial dims
            full_res = outputs[0]
            check(full_res.shape[2:] == dummy.shape[2:],
                  f"Full-res output matches input spatial dims: {tuple(full_res.shape[2:])}")
        else:
            full_res = outputs
            print(f"  {INFO} Single output: {tuple(full_res.shape)}")

        print(f"  {PASS} Forward pass OK ({fwd_ms:.0f}ms)")

        # Test loss computation
        criterion = build_loss(CONFIG)
        with torch.amp.autocast("cuda", enabled=CONFIG["use_amp"]):
            loss = criterion(outputs, dummy_lbl)
        check(torch.isfinite(loss), f"Loss is finite: {loss.item():.4f}")

        # Check VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated(device) / 1e9
            vram_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            bs = CONFIG["batch_size"]
            estimated_train = vram_used * bs * 2.5  # rough: bs scaling + gradients + optimizer
            print(f"  {INFO} VRAM for bs=1 forward: {vram_used:.2f} GB")
            print(f"  {INFO} Estimated training (bs={bs}): ~{estimated_train:.1f} GB")
            check(estimated_train < vram_total * 0.95,
                  f"Estimated VRAM ({estimated_train:.1f} GB) fits in GPU ({vram_total:.1f} GB)",
                  fatal=False)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  {FAIL} OOM on forward pass with patch_size={ps} — reduce patch_size or batch_size")
        else:
            print(f"  {FAIL} Forward pass failed: {e}")
        global _fail_count
        _fail_count += 1
    finally:
        # Cleanup
        del model, dummy, dummy_lbl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_config():
    section("8. Config Sanity")
    from configs.default import CONFIG

    check(CONFIG["modality"] == "ct",
          f"Modality is 'ct' (got '{CONFIG['modality']}')")
    check(CONFIG["hu_window"] == (0, 600),
          f"HU window is (0, 600) for CTA (got {CONFIG['hu_window']})")
    check(CONFIG["num_classes"] == 2,
          f"Binary segmentation (num_classes={CONFIG['num_classes']})")
    check(CONFIG["aug_mirror"] is False,
          f"Mirror augmentation disabled (preserves L/R anatomy)")
    check(CONFIG["deep_supervision"] is True,
          f"Deep supervision enabled")
    check(CONFIG["use_amp"] is True,
          f"Mixed precision enabled")
    check(CONFIG["data_dir"] != "/path/to/topcow2024",
          f"data_dir has been set (not placeholder)", fatal=False)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-training sanity checks")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to TopCoW dataset root")
    parser.add_argument("--quick", action="store_true",
                        help="Skip forward pass and full spacing survey (faster)")
    parser.add_argument("--n_sample", type=int, default=5,
                        help="Number of volumes to load for spot-check (default 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  VESSEL SEG V2 — PRE-TRAINING PREFLIGHT CHECK")
    print("=" * 60)

    check_dependencies()
    check_gpu()
    check_config()
    ct_cases, counts = check_data(args.data_dir)
    check_volumes(ct_cases, n_sample=args.n_sample)
    if not args.quick:
        check_full_spacing(ct_cases)
    check_train_val_split(ct_cases)
    if not args.quick:
        check_forward_pass()

    # Summary
    print(f"\n{'='*60}")
    if _fail_count == 0 and _warn_count == 0:
        print("  ALL CHECKS PASSED — ready to train!")
    elif _fail_count == 0:
        print(f"  PASSED with {_warn_count} warning(s) — review above before training")
    else:
        print(f"  {_fail_count} FAILED, {_warn_count} warning(s) — fix before training")
    print(f"{'='*60}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
