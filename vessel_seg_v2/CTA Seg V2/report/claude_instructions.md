# EC2 Data Extraction for Midterm Report

## Context

We're writing a 5-page IEEE-style midterm report for a deep learning class. The report compares three loss functions (Dice+CE, Dice+CE+clDice, Dice+CE+Skeleton Recall) for cerebral CTA vessel segmentation using a 3D U-Net. The report LaTeX source is at `report/midterm_report.tex` and already compiles with the data we have locally. But we're missing training logs from the April 2-4 AWS training session that we need for complete learning curve figures.

## What We Need

### 1. Training Log JSONs (CRITICAL)

We need the `training_log.json` files from these three runs. These contain per-epoch training loss, training dice, learning rate, and periodic validation metrics that we need to plot overlaid learning curves for all three loss configurations.

**Files to find and copy:**

```
# Run 6 — dice_ce_cldice (fixed LR, clean run) — April 4
~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA Seg V2/runs/dice_ce_cldice_20260404_053853/training_log.json

# Run 3 — dice_ce_skeleton — April 2
~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA Seg V2/runs/dice_ce_skeleton_20260402_203733/training_log.json
```

If those exact paths don't exist, search for them:
```bash
find ~/Deep-Learning-for-Computer-Vision -name "training_log.json" -path "*cldice*" 2>/dev/null
find ~/Deep-Learning-for-Computer-Vision -name "training_log.json" -path "*skeleton*" 2>/dev/null
```

Or more broadly:
```bash
find ~/Deep-Learning-for-Computer-Vision -name "training_log.json" 2>/dev/null
find ~/data -name "training_log.json" 2>/dev/null
```

**Expected format** (each file is a JSON array, one object per epoch):
```json
[
  {
    "epoch": 1,
    "lr": 0.0001,
    "time": 88.2,
    "train_loss": 0.956,
    "train_dice": 0.303,
    "avg_forward_ms": 192.2,
    "avg_backward_ms": 447.2,
    "avg_grad_norm": 3.02,
    "peak_vram_gb": 14.0,
    "nan_batches": 0
  },
  ...
  {
    "epoch": 250,
    "lr": 7.16e-05,
    "train_loss": 0.125,
    "train_dice": 0.867,
    ...
    "dice": 0.870,        // <-- validation metrics appear at val_interval epochs
    "cldice": 0.918,
    "hd95": 5.92,
    "betti0_error": 3.88
  },
  ...
]
```

Each file should be ~100-110 KB for a 300-epoch run.

### 2. Per-Case Evaluation Results (NICE TO HAVE)

If full `evaluate.py` was run on the cldice_v2 and skeleton models, there may be evaluation output files or terminal logs with per-case DSC/clDice for each of the 25 validation cases. We already have per-case results for the baseline but not the other two. Check:

```bash
find ~/Deep-Learning-for-Computer-Vision -name "eval_results*" -o -name "evaluation*" 2>/dev/null
```

### 3. Stratified Evaluation Logs (NICE TO HAVE)

The stratified per-vessel-class results for all from-scratch models. We have the summary in TRAINING_SESSION_20260402.md but if there are more detailed logs (per-vessel per-case), those would be useful. Check:

```bash
find ~/Deep-Learning-for-Computer-Vision -name "stratified*" 2>/dev/null
```

## How to Get the Files Back

Option A — SCP from EC2 to local:
```bash
# From your local machine (Git Bash / terminal):
scp -i <key.pem> ubuntu@<ec2-ip>:~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2/runs/dice_ce_cldice_20260404_053853/training_log.json ./cldice_training_log.json
scp -i <key.pem> ubuntu@<ec2-ip>:~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2/runs/dice_ce_skeleton_20260402_203733/training_log.json ./skeleton_training_log.json
```

Option B — Cat and copy-paste (files are ~100KB, feasible):
```bash
cat ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2/runs/dice_ce_cldice_20260404_053853/training_log.json
```

Option C — Have Claude on EC2 create a tar:
```bash
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2/runs
tar czf ~/report_data.tar.gz \
  dice_ce_cldice_20260404_053853/training_log.json \
  dice_ce_skeleton_20260402_203733/training_log.json \
  dice_ce_cldice_20260404_053853/config.json \
  dice_ce_skeleton_20260402_203733/config.json
# Then SCP ~/report_data.tar.gz to local
```

## Where to Put Them Locally

Place the training logs here:
```
vessel_seg_v2/CTA Seg V2/runs/dice_ce_cldice_20260404_053853/training_log.json
vessel_seg_v2/CTA Seg V2/runs/dice_ce_skeleton_20260402_203733/training_log.json
```

Then re-run the figure generation script to get overlaid learning curves:
```bash
cd "vessel_seg_v2/CTA Seg V2/report"
python generate_figures.py \
  --cldice_log_dir "../runs/dice_ce_cldice_20260404_053853" \
  --skeleton_log_dir "../runs/dice_ce_skeleton_20260402_203733"
```

And recompile:
```bash
pdflatex -interaction=nonstopmode midterm_report.tex
```

## Summary of All Training Runs (for reference)

| Run | Dir Name | Loss | Data | Status | Notes |
|-----|----------|------|------|--------|-------|
| Baseline | `dice_ce_20260331_222218` (root level) | dice_ce | TopCoW 125 | **Have log locally** | 300 epochs, single A10G |
| Run 3 | `dice_ce_skeleton_20260402_203733` | dice_ce_skeleton | TopCoW 125 | **Need log from EC2** | 300 epochs, 8x A100, clean |
| Run 6 | `dice_ce_cldice_20260404_053853` | dice_ce_cldice | TopCoW 125 | **Need log from EC2** | 300 epochs, 8x A100, clean (fixed LR) |
| Run 2 | `dice_ce_cldice_20260402_193545` | dice_ce_cldice | TopCoW 125 | Skip | Had LR bug, superseded by Run 6 |
