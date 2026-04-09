#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_ablation.sh — Complete ablation study pipeline for midterm report.
#
# 1. Retrain dice_ce baseline on 8x A100 (matching cldice/skeleton config)
# 2. Fine-tune the new baseline on TopBrain
# 3. Evaluate all 6 models (3 from-scratch + 3 TopBrain fine-tuned)
# 4. Save eval summaries and bundle training logs
#
# Usage (SSH-safe):
#   source ~/vessel-env/bin/activate
#   cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2
#   nohup bash run_ablation.sh > ~/ablation.log 2>&1 &
# ──────────────────────────────────────────────────────────────────────

set -e

DATA_DIR="$HOME/data/topcow2024"
TOPBRAIN_DIR="$HOME/data/topbrain"
NGPUS=8
PROJECT_DIR="$HOME/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA Seg V2"

cd "$PROJECT_DIR"

echo "============================================================"
echo "  ABLATION STUDY — $(date)"
echo "============================================================"
echo ""

# ── Step 1: Retrain dice_ce baseline on 8x A100 ────────────────────
echo "============================================================"
echo "  [Step 1/4] Train dice_ce from scratch (8x A100, 300 epochs)"
echo "============================================================"
torchrun --nproc_per_node=$NGPUS train.py \
    --data_dir "$DATA_DIR" \
    --loss dice_ce

# Find the new dice_ce run directory
DICE_CE_RUN=$(ls -dt runs/dice_ce_2* 2>/dev/null | head -1)
echo "  New baseline: $DICE_CE_RUN"
echo ""

# ── Step 2: Fine-tune new baseline on TopBrain ──────────────────────
echo "============================================================"
echo "  [Step 2/4] Fine-tune dice_ce on TopBrain"
echo "============================================================"
torchrun --nproc_per_node=$NGPUS train.py \
    --data_dir "$DATA_DIR" \
    --finetune "$DICE_CE_RUN/best_model.pth" \
    --topbrain_dir "$TOPBRAIN_DIR" \
    --loss dice_ce

DICE_CE_FT_RUN=$(ls -dt runs/finetune_topbrain_dice_ce_2* 2>/dev/null | head -1)
echo "  Fine-tuned: $DICE_CE_FT_RUN"
echo ""

# ── Existing runs (already trained on 8x A100) ─────────────────────
CLDICE_RUN="runs/dice_ce_cldice_20260404_053853"
SKELETON_RUN="runs/dice_ce_skeleton_20260402_203733"
CLDICE_FT_RUN="runs/finetune_topbrain_dice_ce_cldice_20260404_064204"
SKELETON_FT_RUN="runs/finetune_topbrain_dice_ce_skeleton_20260402_215720"

# ── Step 3: Evaluate all 6 models ──────────────────────────────────
echo "============================================================"
echo "  [Step 3/4] Evaluating all 6 models"
echo "============================================================"

evaluate_model() {
    local CKPT=$1
    local DATA=$2
    local TOPBRAIN=$3
    local GPU=$4
    local LABEL=$5
    local RUN_DIR=$(dirname "$CKPT")
    local SUMMARY="$RUN_DIR/eval_summary.txt"

    echo ""
    echo "--- Evaluating: $LABEL ---"
    echo "  Checkpoint: $CKPT"
    echo "  Output: $SUMMARY"

    if [ -n "$TOPBRAIN" ]; then
        python3 evaluate.py --checkpoint "$CKPT" --data_dir "$DATA" --topbrain_dir "$TOPBRAIN" --gpu "$GPU" 2>&1 | tee "$SUMMARY"
    else
        python3 evaluate.py --checkpoint "$CKPT" --data_dir "$DATA" --gpu "$GPU" 2>&1 | tee "$SUMMARY"
    fi
    echo ""
}

# From-scratch models (TopCoW 13-class eval)
evaluate_model "$DICE_CE_RUN/best_model.pth"  "$DATA_DIR" "" 0 "dice_ce (from-scratch, 8xA100)"
evaluate_model "$CLDICE_RUN/best_model.pth"   "$DATA_DIR" "" 0 "dice_ce_cldice (from-scratch, 8xA100)"
evaluate_model "$SKELETON_RUN/best_model.pth"  "$DATA_DIR" "" 0 "dice_ce_skeleton (from-scratch, 8xA100)"

# TopBrain fine-tuned models (TopBrain 40-class eval)
evaluate_model "$DICE_CE_FT_RUN/best_model.pth"  "$DATA_DIR" "$TOPBRAIN_DIR" 0 "dice_ce+TopBrain"
evaluate_model "$CLDICE_FT_RUN/best_model.pth"   "$DATA_DIR" "$TOPBRAIN_DIR" 0 "dice_ce_cldice+TopBrain"
evaluate_model "$SKELETON_FT_RUN/best_model.pth"  "$DATA_DIR" "$TOPBRAIN_DIR" 0 "dice_ce_skeleton+TopBrain"

# ── Step 4: Bundle everything ───────────────────────────────────────
echo "============================================================"
echo "  [Step 4/4] Bundling results"
echo "============================================================"

tar czf ~/report_data.tar.gz \
    "$DICE_CE_RUN/training_log.json" \
    "$DICE_CE_RUN/config.json" \
    "$DICE_CE_RUN/eval_summary.txt" \
    "$CLDICE_RUN/training_log.json" \
    "$CLDICE_RUN/config.json" \
    "$CLDICE_RUN/eval_summary.txt" \
    "$SKELETON_RUN/training_log.json" \
    "$SKELETON_RUN/config.json" \
    "$SKELETON_RUN/eval_summary.txt" \
    "$DICE_CE_FT_RUN/eval_summary.txt" \
    "$CLDICE_FT_RUN/eval_summary.txt" \
    "$SKELETON_FT_RUN/eval_summary.txt" \
    2>/dev/null

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
echo ""
echo "Bundle: ~/report_data.tar.gz"
echo "SCP: scp -i <key.pem> ubuntu@<ip>:~/report_data.tar.gz ."
echo ""
echo "Runs:"
echo "  dice_ce:          $DICE_CE_RUN"
echo "  dice_ce_cldice:   $CLDICE_RUN"
echo "  dice_ce_skeleton: $SKELETON_RUN"
echo "  dice_ce+TB:       $DICE_CE_FT_RUN"
echo "  cldice+TB:        $CLDICE_FT_RUN"
echo "  skeleton+TB:      $SKELETON_FT_RUN"
