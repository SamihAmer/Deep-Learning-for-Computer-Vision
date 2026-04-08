# EC2 Retraining & Data Extraction for Midterm Report

## Background

We're writing a midterm report for a deep learning class comparing three loss functions for cerebral CTA vessel segmentation:

1. **Dice+CE** (baseline)
2. **Dice+CE+clDice** (topology-aware, Shit et al. CVPR 2021)
3. **Dice+CE+Skeleton Recall** (topology-aware, Kirchhoff et al. ECCV 2024)

The core hypothesis from our proposal is: **topology-aware losses disproportionately improve segmentation of thin communicating arteries** (Acom, Pcom, ACA) that are most clinically significant yet most prone to topological failure.

## The Problem: Hardware Confound

Our current results are NOT a valid controlled comparison. The three from-scratch models were trained on different hardware:

| Model | Hardware | Effective batch size | Wall-clock |
|-------|----------|---------------------|------------|
| Dice+CE baseline | 1x A10G 24GB | 4 | ~6.5h |
| Dice+CE+clDice | 8x A100 40GB | 32 (4x8 DDP) | ~60min |
| Dice+CE+Skeleton | 8x A100 40GB | 32 (4x8 DDP) | ~68min |

The baseline had 8x more gradient steps per epoch, different GPU hardware, and different effective batch size. Any performance difference could be from these factors, not the loss function. We need to retrain the baseline on the SAME config as the other two.

## What We Need

### Task 1: Retrain Dice+CE Baseline on 8x A100 (CRITICAL)

Retrain the Dice+CE baseline from scratch using the exact same hardware, config, and DDP setup as the topology-aware models:

```bash
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2

torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --loss dice_ce
```

This should produce a run in `runs/dice_ce_<timestamp>/` with:
- `training_log.json` (300 epochs)
- `best_model.pth`
- `config.json`

Expected time: ~45-60 minutes based on prior runs.

**Important:** Make sure LR is NOT scaled by world_size. The code was fixed for this (see `train.py`), but verify that base LR = 1e-3 is used as-is.

### Task 2: Fine-tune All Three on TopBrain (CRITICAL)

Once the new baseline is trained, fine-tune all three models on TopBrain:

```bash
# Fine-tune new baseline
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --finetune runs/<new_dice_ce_run>/best_model.pth \
    --topbrain_dir ~/data/topbrain \
    --loss dice_ce

# Fine-tune clDice (use Run 6 — the clean cldice_v2 model)
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --finetune runs/dice_ce_cldice_20260404_053853/best_model.pth \
    --topbrain_dir ~/data/topbrain \
    --loss dice_ce_cldice

# Fine-tune Skeleton (use Run 3)
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --finetune runs/dice_ce_skeleton_20260402_203733/best_model.pth \
    --topbrain_dir ~/data/topbrain \
    --loss dice_ce_skeleton
```

Note: The clDice and Skeleton fine-tuning may already exist from the April 2-4 session (Runs 5 and 7). If their base models match the from-scratch runs above, those can be reused. But if we're re-doing the baseline, we need to re-fine-tune it.

Expected time: ~8 minutes each, ~25 minutes total.

### Task 3: Evaluate All Models (CRITICAL)

Run full evaluation (DSC, clDice, HD95, Betti-0) with stratified per-vessel-class breakdown on all models:

```bash
# From-scratch models (evaluated against TopCoW 13-class labels)
python3 evaluate.py --checkpoint runs/<new_dice_ce>/best_model.pth --data_dir ~/data/topcow2024
python3 evaluate.py --checkpoint runs/dice_ce_cldice_20260404_053853/best_model.pth --data_dir ~/data/topcow2024
python3 evaluate.py --checkpoint runs/dice_ce_skeleton_20260402_203733/best_model.pth --data_dir ~/data/topcow2024

# Fine-tuned models (evaluated against TopBrain 40-class labels)
python3 evaluate.py --checkpoint runs/<new_dice_ce_ft>/best_model.pth --data_dir ~/data/topcow2024 --topbrain_dir ~/data/topbrain
python3 evaluate.py --checkpoint runs/<cldice_ft>/best_model.pth --data_dir ~/data/topcow2024 --topbrain_dir ~/data/topbrain
python3 evaluate.py --checkpoint runs/<skeleton_ft>/best_model.pth --data_dir ~/data/topcow2024 --topbrain_dir ~/data/topbrain
```

Expected time: Evaluation takes ~30-60s per case, 25 val cases = ~15-25 min per model, ~2-3 hours total for all 6.

### Task 4: Collect Training Logs (CRITICAL)

We need `training_log.json` from ALL three from-scratch runs for learning curve plots:

```bash
# New baseline
runs/<new_dice_ce_run>/training_log.json

# clDice (already trained, Run 6)
runs/dice_ce_cldice_20260404_053853/training_log.json

# Skeleton (already trained, Run 3)
runs/dice_ce_skeleton_20260402_203733/training_log.json
```

### Task 5: Bundle Everything for Download

```bash
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2

# Create a tar with all the data we need
tar czf ~/report_data.tar.gz \
    runs/<new_dice_ce>/training_log.json \
    runs/<new_dice_ce>/config.json \
    runs/dice_ce_cldice_20260404_053853/training_log.json \
    runs/dice_ce_cldice_20260404_053853/config.json \
    runs/dice_ce_skeleton_20260402_203733/training_log.json \
    runs/dice_ce_skeleton_20260402_203733/config.json

echo "Bundle ready at ~/report_data.tar.gz"
echo "SCP it to local with:"
echo "scp -i <key.pem> ubuntu@<ip>:~/report_data.tar.gz ."
```

## What Results We're Looking For

The central question: **Do topology-aware losses disproportionately improve thin communicating artery segmentation?**

The key table we need (from-scratch models, all trained on identical 8x A100 config):

| Model | Large CoW DSC | Communicating DSC | Gap |
|-------|--------------|-------------------|-----|
| Dice+CE | ? | ? | ? |
| +clDice | ? | ? | ? |
| +Skeleton | ? | ? | ? |

And for TopBrain fine-tuned (all 6 vessel groups):

| Group | Dice+CE+TB | clDice+TB | Skeleton+TB |
|-------|-----------|-----------|-------------|
| Large CoW arteries | ? | ? | ? |
| Communicating | ? | ? | ? |
| Distal branches | ? | ? | ? |
| Posterior fossa | ? | ? | ? |
| Small arteries | ? | ? | ? |
| Venous sinuses | ? | ? | ? |

Also per-case DSC/clDice for all three from-scratch models (25 val cases each).

## Evaluation Output Format

`evaluate.py` prints results to stdout and also saves to the run directory. Please capture the full terminal output for each evaluation run. The stratified evaluation should show per-vessel-class DSC, clDice, Betti-0 error, and N (number of cases where that vessel class is present).

## Data Paths (verify these exist)

```bash
# TopCoW data
ls ~/data/topcow2024/imagesTr/  # should have 125+ CT NIfTI files
ls ~/data/topcow2024/labelsTr/  # should have matching label files

# TopBrain data
ls ~/data/topbrain/  # should have TopBrain_Data_Release_* subdirectory
```

## How These Results Will Be Used

These results feed directly into an IEEE-style LaTeX report (`report/midterm_report.tex`). The report already exists with placeholder/old data that will be replaced. Specifically:

**Tables that need new numbers:**
- **Table I** — Global metrics (DSC, clDice, HD95, B0 err) for the 3 from-scratch models
- **Table II** — Per-vessel-class DSC/clDice for the new baseline (and ideally all 3 models)
- **Table III** — Large vs Communicating vessel gap for all 3 from-scratch models
- **Table IV** — TopBrain fine-tuned DSC by vessel group (6 groups x 3 models)
- **Table V** — Computational cost (training time, VRAM) — now all on same hardware

**Figures that need new data:**
- **Learning curves** — We need `training_log.json` from all 3 from-scratch runs to overlay training loss and validation Dice curves

**To make plugging in results easy, please save evaluation output in a structured format.** After each evaluation run, save a summary like this to a text file in the run directory:

```
# Example: runs/<run_name>/eval_summary.txt
Model: dice_ce (from-scratch, 8xA100)
Global: DSC=0.XXX, clDice=0.XXX, HD95=X.XX, B0err=X.XX

Per-vessel:
BA: DSC=0.XXX, clDice=0.XXX, B0err=X.X, N=25
R-PCA: DSC=0.XXX, clDice=0.XXX, B0err=X.X, N=25
...

Groups:
Large CoW: DSC=0.XXX
Communicating: DSC=0.XXX
Gap: 0.XXX
```

This makes it straightforward to update the report tables without parsing raw terminal output.

## Summary

| Priority | Task | Time estimate |
|----------|------|---------------|
| CRITICAL | Retrain dice_ce from scratch on 8x A100 | ~60 min |
| CRITICAL | Fine-tune new baseline on TopBrain | ~8 min |
| CRITICAL | Evaluate all 6 models (3 from-scratch + 3 fine-tuned) | ~2-3 hours |
| CRITICAL | Collect all training_log.json files | ~1 min |
| CRITICAL | Save eval_summary.txt for each model | ~5 min |
| CRITICAL | Bundle into tar for SCP | ~1 min |

Total EC2 time: ~4-5 hours. This gives us a clean, controlled comparison with no hardware confound.
