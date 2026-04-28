#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_final_ablation.sh — Final-report ablation pipeline (Sections VI–VII).
#
# Trains four new loss configurations from scratch on TopCoW, fine-tunes
# each on TopBrain, re-evaluates all 7 models with the extended metric
# suite (DSC/clDice/HD95/B0err + 3D-SSIM/PSNR/F-FID), and bundles logs.
#
# The four new losses:
#   4. dice_ce_ssim         (3D structural similarity)
#   5. dice_ce_mse_dt       (MSE on Gaussian-decayed distance transform)
#   6. dice_ce_perceptual   (2D slice-wise VGG-16 perceptual)
#   7. dice_ce_cldice_ssim  (combination: topology + reconstruction)
#
# Usage (SSH-safe):
#   source ~/vessel-env/bin/activate
#   cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2
#   # one-time: pre-download VGG weights so DDP ranks don't race-download
#   python -c "import torchvision.models as M; M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1)"
#   nohup bash run_final_ablation.sh > ~/final_ablation.log 2>&1 &
#
# Expected wall time on 8x A100 (p4d.24xlarge): ~5-7h total.
# ──────────────────────────────────────────────────────────────────────

# Do NOT use `set -e`. Per-step error trapping below logs failures and
# continues — a transient OOM on run 3 of 7 should not discard runs 4-7.
set +e

DATA_DIR="$HOME/data/topcow2024"
TOPBRAIN_DIR="$HOME/data/topbrain"
NGPUS=8
PROJECT_DIR="$HOME/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA Seg V2"
FAIL_LOG="$HOME/final_ablation_failures.log"

cd "$PROJECT_DIR"
: > "$FAIL_LOG"   # truncate the failure log at the start of each run

# Run a single step with error trapping. Logs failure to FAIL_LOG and
# returns the failing exit code; the orchestrator continues regardless.
run_step() {
    local LABEL="$1"
    shift
    echo ""
    echo ">>> STEP: $LABEL  ($(date +%H:%M:%S))"
    "$@"
    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAIL ($STATUS): $LABEL" | tee -a "$FAIL_LOG"
    else
        echo "[$(date +%H:%M:%S)] OK: $LABEL"
    fi
    return $STATUS
}

echo "============================================================"
echo "  FINAL ABLATION (Sections VI-VII) — $(date)"
echo "============================================================"
echo "  Failure log: $FAIL_LOG"
echo ""

# ── Existing midterm runs (already trained, used in Page 6 table) ──
DICE_CE_RUN="runs/dice_ce_20260409_001230"
CLDICE_RUN="runs/dice_ce_cldice_20260404_053853"
SKELETON_RUN="runs/dice_ce_skeleton_20260402_203733"
DICE_CE_FT_RUN="runs/finetune_topbrain_dice_ce_20260409_011117"
CLDICE_FT_RUN="runs/finetune_topbrain_dice_ce_cldice_20260404_064204"
SKELETON_FT_RUN="runs/finetune_topbrain_dice_ce_skeleton_20260402_215720"

# ── Step 1: Train each new loss from scratch on TopCoW ─────────────
NEW_LOSSES=("dice_ce_ssim" "dice_ce_mse_dt" "dice_ce_perceptual" "dice_ce_cldice_ssim")
declare -A NEW_RUN_DIRS

for LOSS in "${NEW_LOSSES[@]}"; do
    echo "============================================================"
    echo "  [Step 1/3] Train $LOSS from scratch (8x A100, 300 epochs)"
    echo "============================================================"
    run_step "train $LOSS" \
        torchrun --nproc_per_node=$NGPUS train.py \
            --data_dir "$DATA_DIR" \
            --loss "$LOSS"

    RUN_DIR=$(ls -dt runs/${LOSS}_2* 2>/dev/null | head -1)
    NEW_RUN_DIRS[$LOSS]=$RUN_DIR
    echo "  Trained: $RUN_DIR"
    echo ""
done

# ── Step 2: TopBrain fine-tune each new loss ───────────────────────
declare -A NEW_FT_DIRS

for LOSS in "${NEW_LOSSES[@]}"; do
    echo "============================================================"
    echo "  [Step 2/3] Fine-tune $LOSS on TopBrain (8x A100, 150 epochs)"
    echo "============================================================"
    SOURCE_RUN=${NEW_RUN_DIRS[$LOSS]}
    if [ -z "$SOURCE_RUN" ] || [ ! -f "$SOURCE_RUN/best_model.pth" ]; then
        echo "[$(date +%H:%M:%S)] SKIP: fine-tune $LOSS — from-scratch run missing or failed" \
            | tee -a "$FAIL_LOG"
        continue
    fi

    run_step "fine-tune $LOSS on TopBrain" \
        torchrun --nproc_per_node=$NGPUS train.py \
            --data_dir "$DATA_DIR" \
            --finetune "$SOURCE_RUN/best_model.pth" \
            --topbrain_dir "$TOPBRAIN_DIR" \
            --loss "$LOSS"

    FT_DIR=$(ls -dt runs/finetune_topbrain_${LOSS}_2* 2>/dev/null | head -1)
    NEW_FT_DIRS[$LOSS]=$FT_DIR
    echo "  Fine-tuned: $FT_DIR"
    echo ""
done

# ── Step 3: Extended evaluation of all 14 models ───────────────────
echo "============================================================"
echo "  [Step 3/3] Extended evaluation of all models"
echo "============================================================"

evaluate_model() {
    local CKPT=$1
    local DATA=$2
    local TOPBRAIN=$3
    local GPU=$4
    local LABEL=$5
    local RUN_DIR=$(dirname "$CKPT")
    local SUMMARY="$RUN_DIR/eval_summary_extended.txt"

    if [ ! -f "$CKPT" ]; then
        echo "[$(date +%H:%M:%S)] SKIP eval $LABEL — checkpoint missing: $CKPT" \
            | tee -a "$FAIL_LOG"
        return 0
    fi

    if [ -n "$TOPBRAIN" ]; then
        run_step "eval $LABEL" \
            bash -c "python3 evaluate.py --extended_metrics \
                --checkpoint '$CKPT' --data_dir '$DATA' \
                --topbrain_dir '$TOPBRAIN' --gpu '$GPU' \
                2>&1 | tee '$SUMMARY'"
    else
        run_step "eval $LABEL" \
            bash -c "python3 evaluate.py --extended_metrics \
                --checkpoint '$CKPT' --data_dir '$DATA' --gpu '$GPU' \
                2>&1 | tee '$SUMMARY'"
    fi
    echo ""
}

# Midterm models — re-eval with extended metrics so the Page 6 table is complete
evaluate_model "$DICE_CE_RUN/best_model.pth"   "$DATA_DIR" "" 0 "dice_ce (midterm)"
evaluate_model "$CLDICE_RUN/best_model.pth"    "$DATA_DIR" "" 0 "dice_ce_cldice (midterm)"
evaluate_model "$SKELETON_RUN/best_model.pth"  "$DATA_DIR" "" 0 "dice_ce_skeleton (midterm)"
evaluate_model "$DICE_CE_FT_RUN/best_model.pth"  "$DATA_DIR" "$TOPBRAIN_DIR" 0 "dice_ce + TB (midterm)"
evaluate_model "$CLDICE_FT_RUN/best_model.pth"   "$DATA_DIR" "$TOPBRAIN_DIR" 0 "cldice + TB (midterm)"
evaluate_model "$SKELETON_FT_RUN/best_model.pth" "$DATA_DIR" "$TOPBRAIN_DIR" 0 "skeleton + TB (midterm)"

# New from-scratch models
for LOSS in "${NEW_LOSSES[@]}"; do
    evaluate_model "${NEW_RUN_DIRS[$LOSS]}/best_model.pth" "$DATA_DIR" "" 0 "$LOSS (new, from-scratch)"
done

# New TopBrain-fine-tuned models
for LOSS in "${NEW_LOSSES[@]}"; do
    evaluate_model "${NEW_FT_DIRS[$LOSS]}/best_model.pth" "$DATA_DIR" "$TOPBRAIN_DIR" 0 "$LOSS + TB (new)"
done

# ── Bundle logs and summaries for download ─────────────────────────
echo "============================================================"
echo "  Bundling final-ablation results"
echo "============================================================"

BUNDLE=~/report_data_final.tar.gz
TAR_TARGETS=(
    "$DICE_CE_RUN/eval_summary_extended.txt"
    "$CLDICE_RUN/eval_summary_extended.txt"
    "$SKELETON_RUN/eval_summary_extended.txt"
    "$DICE_CE_FT_RUN/eval_summary_extended.txt"
    "$CLDICE_FT_RUN/eval_summary_extended.txt"
    "$SKELETON_FT_RUN/eval_summary_extended.txt"
)
for LOSS in "${NEW_LOSSES[@]}"; do
    TAR_TARGETS+=(
        "${NEW_RUN_DIRS[$LOSS]}/training_log.json"
        "${NEW_RUN_DIRS[$LOSS]}/config.json"
        "${NEW_RUN_DIRS[$LOSS]}/eval_summary_extended.txt"
        "${NEW_RUN_DIRS[$LOSS]}/stratified_eval.json"
        "${NEW_FT_DIRS[$LOSS]}/eval_summary_extended.txt"
        "${NEW_FT_DIRS[$LOSS]}/stratified_eval.json"
    )
done

# Filter to only existing files so a missing run doesn't break the tar
EXISTING_TARGETS=()
for f in "${TAR_TARGETS[@]}"; do
    [ -f "$f" ] && EXISTING_TARGETS+=("$f")
done
tar czf "$BUNDLE" "${EXISTING_TARGETS[@]}" 2>/dev/null

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
echo ""
echo "Bundle: $BUNDLE"
echo "Pull with: scp -i <key.pem> ubuntu@<ec2-ip>:$BUNDLE ."
echo ""
if [ -s "$FAIL_LOG" ]; then
    echo "WARNING — some steps failed. Failure log:"
    cat "$FAIL_LOG"
    echo ""
fi
echo "From-scratch runs:"
for LOSS in "${NEW_LOSSES[@]}"; do
    echo "  $LOSS:  ${NEW_RUN_DIRS[$LOSS]:-MISSING}"
done
echo ""
echo "Fine-tuned runs:"
for LOSS in "${NEW_LOSSES[@]}"; do
    echo "  $LOSS + TB:  ${NEW_FT_DIRS[$LOSS]:-MISSING}"
done
