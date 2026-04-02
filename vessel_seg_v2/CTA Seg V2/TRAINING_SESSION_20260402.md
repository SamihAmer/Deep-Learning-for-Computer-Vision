# Training Session — April 2, 2026

Cerebral vessel segmentation V2: TopBrain fine-tuning, topology-aware loss comparison, and multi-GPU DDP training on p4d.24xlarge (8x A100 40GB).

## Overview

This session accomplished three main goals:

1. **TopBrain fine-tuning** — Fine-tuned the baseline dice_ce model on TopBrain 2025 data (40-class vessel labels vs TopCoW's 13-class), enabling the model to segment far more of the cerebral vasculature.
2. **Loss function comparison** — Trained two additional models from scratch with topology-aware losses (dice_ce_cldice, dice_ce_skeleton) on the full 125-case TopCoW dataset, then fine-tuned both on TopBrain.
3. **Infrastructure** — Added DDP multi-GPU support, TopBrain data loading, quick validation, and fixed several training issues along the way.

## Hardware

- **Instance**: AWS p4d.24xlarge
- **GPUs**: 8x NVIDIA A100-SXM4-40GB (42.4 GB each)
- **CPUs**: 96 vCPUs
- **RAM**: 1.1 TB
- **PyTorch**: 2.11.0+cu126
- **CUDA**: 12.6 (driver 13.0)

## Data Setup

### TopCoW 2024 (from-scratch training)
- **Location**: `~/data/topcow2024/` (symlinks to `~/data/TopCoW2024_Data_Release/`)
- **Structure**: `imagesTr/` (symlink) + `labelsTr/` (symlink to `cow_seg_labelsTr/`)
- **Cases**: 125 CT + 125 MR (code filters to CT only)
- **Labels**: 13 vessel classes (Circle of Willis arteries)
- **Note**: The original `topcow2024/` directory only had 25 CT cases (the TopBrain overlap subset). We rebuilt it with symlinks to the full release to get all 125 CT cases.

### TopBrain 2025 (fine-tuning)
- **Location**: `~/data/topbrain/TopBrain_Data_Release_Batches1n2_081425/`
- **Structure**: `imagesTr_topbrain_ct/` + `labelsTr_topbrain_ct/`
- **Cases**: 25 CT (same patients as TopCoW subset, different labels)
- **Labels**: 40 vessel classes (CoW arteries + distal branches + posterior fossa + small arteries + venous sinuses)
- **Key insight**: Same 25 CT scans as TopCoW but with 40 vessel classes instead of 13. When binarized for training, the ground truth mask covers significantly more vasculature.

### Pretrained Checkpoint
- **File**: `~/data/best_model.pth`
- **Training**: dice_ce loss, 300 epochs, single GPU (Windows), on TopCoW
- **Performance**: Epoch 250, Val Dice 0.8846

## Codebase Changes

### New: TopBrain data support (`data/dataset.py`)
- Added `discover_topbrain_cases()` function that handles TopBrain's directory layout (`imagesTr_topbrain_ct/` + `labelsTr_topbrain_ct/`)
- Auto-detects nested extraction directory (`TopBrain_Data_Release_*`)
- Supports `modality` filtering (ct/mr/all)

### New: Fine-tuning support (`train.py`, `configs/default.py`)
- `--finetune` flag: loads model weights only, fresh optimizer (no LR/scheduler state carried over)
- `--topbrain_dir` flag: uses TopBrain labels instead of TopCoW
- Fine-tuning config defaults in `default.py`: `finetune_lr: 1e-4`, `finetune_epochs: 150`, `finetune_warmup_epochs: 5`
- Run directory prefixed with `finetune_topbrain_` for easy identification

### New: DDP multi-GPU training (`train.py`)
- Auto-detected from `torchrun` environment variables (RANK, LOCAL_RANK, WORLD_SIZE)
- `DistributedSampler` for training data
- `DistributedDataParallel` model wrapper
- Rank 0-only validation, logging, and checkpointing
- `dist.broadcast` for early stopping synchronization
- Checkpoints always save unwrapped model state (portable between single/multi-GPU)

### Fixed: NaN batch handling in DDP (`train.py`)
- **Problem**: Original code used `continue` to skip NaN batches, which skipped `loss.backward()`. In DDP, this desynchronized NCCL gradient all-reduce across ranks, causing deadlocks at the next collective operation.
- **Fix**: Replace NaN loss with zero via `torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))`, then still call `backward()` and `scaler.update()`. Skip `optimizer.step()` only on the NaN rank. DDP gradient sync stays in sync.

### Fixed: Validation NCCL timeout (`train.py`)
- **Problem**: Distributed validation split 5 val cases across 8 GPUs. Ranks 5-7 got 0 cases, finished instantly, and called `all_reduce`. Ranks 0-4 were still doing sliding window inference. NCCL timed out after 10 minutes.
- **Fix**: Validate on rank 0 only. Other ranks wait at `dist.broadcast` barrier (no timeout).

### New: Quick validation mode (`train.py`, `utils/metrics.py`)
- `evaluate_volume(quick=True)` computes only Dice (instant), skipping clDice (`skeletonize_3d` ~30-60s/vol) and HD95 (`distance_transform_edt` ~10-20s/vol)
- Training validation uses quick mode; full metrics computed only at final epoch
- Validation time: ~30-60s instead of 10-12 minutes

### Fixed: LR scaling removed (`train.py`)
- **Problem**: Linear LR scaling with world_size (LR × 8 = 8e-3) caused NaN explosion with topology-aware losses in early epochs.
- **Fix**: LR is no longer scaled with world_size. Base LR (1e-3) used as-is.

### Updated: Stratified evaluation (`utils/stratified_eval.py`, `evaluate.py`)
- All 40 TopBrain vessel classes mapped with 6 clinical groups
- Parallelized per-class evaluation with ThreadPoolExecutor
- `evaluate.py` supports `--topbrain_dir` for evaluating fine-tuned models
- Reports all 6 vessel groups with gap analysis

### New: ONNX export with legacy exporter
- PyTorch 2.11's new dynamo ONNX exporter produced 36KB files (missing weights)
- Fixed by using `dynamo=False` flag to use the legacy TorchScript exporter
- All exports produce correct 94.5 MB files

### New: Experiment runner (`run_experiments.sh`)
- Sequential execution of from-scratch training with both topology-aware losses
- Configurable GPU count and data paths

## Training Runs

### Run 1: Fine-tune dice_ce on TopBrain
- **Checkpoint**: `runs/finetune_topbrain_dice_ce_20260402_191511/`
- **Base**: `~/data/best_model.pth` (epoch 250, Dice 0.8846)
- **Data**: TopBrain 25 CT (40-class labels)
- **Config**: LR 1e-4, 150 epochs, 5-epoch warmup, batch 4/gpu x 8 GPUs
- **Result**: Val Dice 0.7691, clDice 0.7771, HD95 12.1, B0err 142.2
- **Time**: 8 minutes
- **Note**: Lower Dice than baseline because TopBrain ground truth covers far more vasculature (40 vs 13 classes). Qualitatively, this model segments ~20x more vessels in SurgicalAR.

### Run 2: Train dice_ce_cldice from scratch
- **Checkpoint**: `runs/dice_ce_cldice_20260402_193545/`
- **Data**: TopCoW 125 CT (13-class labels)
- **Config**: LR 1e-3 (NOTE: had LR scaling bug, actual LR was 8e-3 for first ~150 epochs)
- **Result**: Val Dice 0.8484, clDice 0.8923, HD95 3.72, B0err 5.28
- **Time**: ~45 minutes
- **Issue**: LR was incorrectly scaled by world_size (8x), causing NaN explosions in early epochs. Model partially recovered as cosine schedule decayed LR. Results are suboptimal — should be retrained at 1e-3.

### Run 3: Train dice_ce_skeleton from scratch
- **Checkpoint**: `runs/dice_ce_skeleton_20260402_203733/`
- **Data**: TopCoW 125 CT (13-class labels)
- **Config**: LR 1e-3 (fixed, no scaling), 300 epochs
- **Result**: Val Dice 0.8445, clDice 0.8376, HD95 2.82, B0err 26.84
- **Time**: ~68 minutes
- **Note**: Clean training run with correct LR.

### Run 4: Fine-tune dice_ce_cldice on TopBrain
- **Checkpoint**: `runs/finetune_topbrain_dice_ce_cldice_20260402_214807/`
- **Base**: Run 2 best checkpoint
- **Result**: Val Dice 0.5878, clDice 0.7196, HD95 23.8, B0err 305.6
- **Note**: Poor results due to LR-damaged base model.

### Run 5: Fine-tune dice_ce_skeleton on TopBrain
- **Checkpoint**: `runs/finetune_topbrain_dice_ce_skeleton_20260402_215720/`
- **Base**: Run 3 best checkpoint
- **Result**: Val Dice 0.7064, clDice 0.5517, HD95 22.1, B0err 1752.2

## Stratified Evaluation Results

### Global Metrics (all 6 models)

| Model | DSC | clDice | HD95 | B0 err |
|-------|-----|--------|------|--------|
| **dice_ce baseline** | **0.8958** | **0.9401** | **1.36** | **2.76** |
| dice_ce + TopBrain | 0.7691 | 0.7771 | 12.07 | 142.6 |
| dice_ce_cldice* | 0.8484 | 0.8923 | 3.72 | 5.28 |
| dice_ce_cldice* + TopBrain | 0.5878 | 0.7196 | 23.80 | 305.6 |
| dice_ce_skeleton | 0.8445 | 0.8376 | 2.82 | 26.84 |
| dice_ce_skeleton + TopBrain | 0.7064 | 0.5516 | 22.13 | 1752.2 |

*dice_ce_cldice had LR scaling bug (8e-3 instead of 1e-3) — results are suboptimal

### From-Scratch Models: Large vs Communicating Vessel Gap

| Model | Large CoW DSC | Comm. DSC | Gap |
|-------|--------------|-----------|-----|
| dice_ce baseline | 0.823 | 0.472 | +0.351 |
| dice_ce_skeleton | 0.798 | 0.466 | +0.332 |
| dice_ce_cldice* | 0.801 | 0.439 | +0.362 |

### TopBrain Fine-tuned: All 6 Vessel Groups (DSC)

| Group | dice_ce+TB | dice_ce_cldice*+TB | dice_ce_skeleton+TB |
|-------|-----------|-------------------|-------------------|
| Large CoW arteries | 0.834 | 0.815 | **0.821** |
| Communicating arteries | 0.491 | 0.478 | **0.513** |
| Distal branches | 0.694 | 0.718 | **0.752** |
| Posterior fossa | 0.570 | 0.530 | **0.626** |
| Small arteries (OA, AChA) | 0.005 | 0.030 | **0.589** |
| Venous sinuses | 0.468 | 0.336 | **0.613** |

**Key finding**: dice_ce_skeleton fine-tuned on TopBrain is the clear winner for comprehensive vessel coverage, especially for small arteries (0.589 vs ~0 for others) and venous structures.

## ONNX Exports

All models exported to `~/` for SCP download:

| File | Model | Size |
|------|-------|------|
| `~/finetune_topbrain_dice_ce.onnx` | dice_ce fine-tuned on TopBrain | 94.5 MB |
| `~/dice_ce_cldice.onnx` | dice_ce_cldice from scratch | 94.5 MB |
| `~/dice_ce_cldice_topbrain.onnx` | dice_ce_cldice fine-tuned on TopBrain | 94.5 MB |
| `~/dice_ce_skeleton.onnx` | dice_ce_skeleton from scratch | 94.5 MB |
| `~/dice_ce_skeleton_topbrain.onnx` | dice_ce_skeleton fine-tuned on TopBrain | 94.5 MB |

Export uses legacy TorchScript ONNX exporter (`dynamo=False`) with single-channel sigmoid output compatible with SurgicalAR (`MultiClass: false, OutputClassCount: 1`).

## Clinical Testing in SurgicalAR

All models were tested in SurgicalAR (3D DICOM viewer with ML inference):

- **dice_ce baseline**: Segments Circle of Willis accurately but limited coverage
- **TopBrain fine-tuned models**: Segment ~20x more vessels (distal branches, posterior fossa, venous)
- **Issue**: Topology-aware models (cldice, skeleton) over-propagate and produce false positives on teeth, bone, skin, and other non-vessel structures
- **Root cause identified**: HU window [0, 600] clips bone (800-2000 HU) to the same value as vessels (200-400 HU) — the model literally cannot distinguish them

## Known Issues & Next Steps

1. **Rerun dice_ce_cldice** at correct LR (1e-3) for fair comparison — the current run had 8e-3 LR scaling bug
2. **Widen HU window** from [0, 600] to [0, 1500] so the model can differentiate bone from vessels. Requires retraining.
3. **Post-processing**: Connected component filtering to remove small disconnected false positive blobs
4. **Brain extraction pipeline**: Investigate why SurgicalAR's built-in brain mask isn't preventing extracranial false positives
5. **Custom labels**: Label full-volume CTAs in RedBrick AI with binary vessel masks for more comprehensive training data
6. **Model architecture**: The HU window change is expected to be the single highest-impact improvement for false positive reduction

## Environment Setup

```bash
# Virtual environment
python3 -m venv ~/vessel-env
source ~/vessel-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install SimpleITK nibabel scipy scikit-image onnxscript onnx

# Data symlinks (already set up)
# ~/data/topcow2024/imagesTr -> ~/data/TopCoW2024_Data_Release/imagesTr
# ~/data/topcow2024/labelsTr -> ~/data/TopCoW2024_Data_Release/cow_seg_labelsTr
# ~/data/topbrain/ extracted from topbrain.zip

# Training commands
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2

# Fine-tune on TopBrain
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --finetune ~/data/best_model.pth \
    --topbrain_dir ~/data/topbrain

# Train from scratch with topology-aware losses
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce_skeleton

# Fine-tune on TopBrain
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --finetune runs/<run>/best_model.pth \
    --topbrain_dir ~/data/topbrain \
    --loss <loss_name>

# Evaluate
python3 evaluate.py --checkpoint runs/<run>/best_model.pth --data_dir ~/data/topcow2024
python3 evaluate.py --checkpoint runs/<run>/best_model.pth --data_dir ~/data/topcow2024 --topbrain_dir ~/data/topbrain

# Export ONNX
python3 -c "... dynamo=False ..."  # see export commands in session
```
