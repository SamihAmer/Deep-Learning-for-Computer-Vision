#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_experiments.sh — Train from scratch on full TopCoW (125 CT) with
# the two topology-aware loss functions, for comparison against the
# baseline dice_ce model.
#
# Usage (from tmux):
#   source ~/vessel-env/bin/activate
#   cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2
#   bash run_experiments.sh
# ──────────────────────────────────────────────────────────────────────

set -e  # stop on first error

DATA_DIR="$HOME/data/topcow2024"
NGPUS=8

echo "============================================================"
echo "  VESSEL SEG V2 — LOSS FUNCTION COMPARISON"
echo "============================================================"
echo ""
echo "TopCoW data: $DATA_DIR (125 CT cases)"
echo "GPUs:        $NGPUS"
echo ""

# ── 1. Train from scratch with dice_ce_cldice ───────────────────────
echo "============================================================"
echo "  [1/2] Train from scratch: dice_ce_cldice (Dice+CE+clDice)"
echo "============================================================"
torchrun --nproc_per_node=$NGPUS train.py \
    --data_dir "$DATA_DIR" \
    --loss dice_ce_cldice

echo ""

# ── 2. Train from scratch with dice_ce_skeleton ─────────────────────
echo "============================================================"
echo "  [2/2] Train from scratch: dice_ce_skeleton (Dice+CE+SkeletonRecall)"
echo "============================================================"
torchrun --nproc_per_node=$NGPUS train.py \
    --data_dir "$DATA_DIR" \
    --loss dice_ce_skeleton

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results in ./runs/:"
ls -dt runs/*/ 2>/dev/null | head -10
