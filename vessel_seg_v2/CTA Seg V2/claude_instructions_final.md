# Final Report — Plan, Training Procedure, and Reviewer Instructions

**Author:** Samih Tharwat Amer
**Date:** 2026-04-27
**Document purpose:** lock the plan for the 2 additional pages of the final report (extending the existing 4-page midterm), describe every code/training change so a reviewer (Codex or human) can validate it, and act as the reproducible runbook for the EC2 run.

---

## 1. Executive summary

The midterm report (4 pages, IEEE conference style) presented a controlled three-way ablation of **topology-aware** loss functions for cerebral CTA vessel segmentation: Dice+CE vs. +clDice vs. +Skeleton Recall, with TopBrain transfer learning. The final report adds **2 pages** (pages 5–6) that extend this work along the axis the professor flagged in his email — **image-quality / reconstruction-style losses (MSE, SSIM, perceptual)** — plus an **expanded evaluation methodology** with image-quality *metrics* (3D-SSIM, PSNR, Fréchet Feature Distance).

The new content is **non-redundant** with the midterm: every new training run uses a different objective, every new metric measures a different fidelity dimension, and the closing experiment (combination loss) directly tests the orthogonality of topology-aware vs. reconstruction-aware supervision. **No DynaVessel cross-architecture comparison** — that comparison was collected for a separate downstream artery/vein project (Phase 7) and would conflate architecture, data, and loss; including it here would dilute the loss-ablation thesis.

---

## 2. What the midterm already covers (do not repeat)

| Already in midterm | Section | Re-do? |
|---|---|---|
| Dice+CE / +clDice / +Skeleton Recall | IV | No |
| DSC, clDice, HD95, Betti-0 on TopCoW val | IV.B | No |
| 13-class CoW per-vessel stratified DSC | IV.C | No |
| Large-vs-Communicating gap | IV.C, Table III | No |
| TopBrain fine-tune, 6 vessel groups | IV.D, Table IV | No |
| Compute-cost table on 8× A100 | IV.E, Table V | No |
| 3D U-Net architecture description | III.A | No |
| HU windowing / patch-sampling preprocessing | III.C | No |

---

## 3. Final pages 5–6: scope, narrative, and content

### 3.1 Page 5 — Section VI: Extended Loss Function Investigation

**Narrative:** the midterm asked whether *topology-aware* losses help. This section asks the dual question — whether *reconstruction-aware* losses (operating on the soft probability map rather than its skeleton) help, and how they trade off against the topology-aware family.

**Three new training configurations** (all share the midterm's 3D U-Net, hyperparameters, and 8× A100 setup — only the loss differs):

| # | Loss tag (CLI / config) | Equation | Reference |
|---|---|---|---|
| 4 | `dice_ce_ssim` | `L = 0.5·L_DiceCE + 0.5·(1 − SSIM₃D(p, g_oh))` | Wang et al. TIP 2004; Qin et al. CVPR 2019 (BASNet) |
| 5 | `dice_ce_mse_dt` | `L = 0.5·L_DiceCE + 0.5·MSE(p, exp(−DT(¬g)/σ))`, σ=5 vox | Kervadec et al. MedIA 2021; Karimi & Salcudean TMI 2019 |
| 6 | `dice_ce_perceptual` | `L = 0.5·L_DiceCE + 0.5·MSE(φ_VGG(p̂_2D), φ_VGG(g_2D))` at relu2_2 + relu3_3 | Mosinska et al. CVPR 2018; Johnson et al. ECCV 2016 |

**Implementation honesty (state in the paper):**
- Plain MSE on a binary mask collapses on imbalanced data; we therefore regress the prediction against a Gaussian-decayed distance-transform target (the published recipe).
- Perceptual loss uses **2D slice-wise VGG-16 (ImageNet-pretrained)** on subsampled axial slices of the soft probability map — not 3D MedicalNet — to keep dependencies minimal and reproducible. The probability map is treated as a single-channel image replicated to 3 channels and ImageNet-normalized. We do **not** modulate by the input CTA (a more advanced variant) because doing so would require threading the input image through the loss interface; the BASNet-style direct probability-map perceptual is documented in the literature.
- All α weights = 0.5 across new and existing variants for fairness.

### 3.2 Page 6 — Section VII: Reconstruction-Aware Evaluation + Combination Loss

**Three new evaluation metrics** computed for **all 7 models** (3 midterm + 3 new + 1 combination):

| Metric | Definition | Honesty caveat |
|---|---|---|
| 3D-SSIM | `monai`-style 3D SSIM on the binary prediction vs. binary GT | Tends to be high (~0.99) because background dominates — useful for **relative** ranking, not absolute |
| PSNR | `−10·log₁₀(MSE(p̂, g))` in dB | Same caveat as SSIM; informative as a relative metric |
| Fréchet Feature Distance (F-FID) | Fréchet distance between Gaussians fit to VGG-16 relu3_3 features over the val set | We label this **F-FID, not FID**, because N=25 is far below the ~10k samples required for converged Inception-FID. This is a feature-space proxy, not the original Heusel-et-al. metric. |

**Combination experiment** — one additional configuration:

| # | Loss tag | Weights | Question |
|---|---|---|---|
| 7 | `dice_ce_cldice_ssim` | 0.5 / 0.25 / 0.25 (Dice+CE / clDice / SSIM) | Do topology-aware and reconstruction-aware losses combine constructively, or cancel? |

**Page 6 layout:**
1. Paragraph introducing the three new metrics + the F-FID caveat.
2. **Table VI** (extended): all 7 models × 7 metrics (DSC, clDice, HD95, B0err, 3D-SSIM, PSNR, F-FID).
3. Combination-loss subsection: report whether `dice_ce_cldice_ssim` matches or beats the best single-objective model on each metric family.
4. **Figure 3** (qualitative): three example axial slices (large CoW artery, communicating artery, distal branch) showing soft probability maps from each loss family, side-by-side. Built from saved validation predictions — no retraining needed.
5. Updated discussion paragraph: which loss family aligns with which clinical fidelity dimension.

---

## 4. Code changes (already applied — see git diff for verification)

### 4.1 `losses/losses.py` (extended)

Added four new loss classes plus their factory entries:

| Class | `loss` config string |
|---|---|
| `DiceCESSIMLoss` (uses `SSIM3DLoss`) | `dice_ce_ssim` |
| `DiceCEMSEDistLoss` (uses `MSEDistanceTransformLoss`) | `dice_ce_mse_dt` |
| `DiceCEPerceptualLoss` (uses `PerceptualLoss2DSlice`) | `dice_ce_perceptual` |
| `DiceCEClDiceSSIMLoss` | `dice_ce_cldice_ssim` |

`build_loss(cfg)` updated; `DeepSupervisionLoss` wrapper unchanged (all new losses match the existing `(logits, targets)` interface, so deep supervision works automatically at every decoder scale).

### 4.2 `configs/default.py`

Added these keys (all default to no-op for the 3 midterm losses):

```python
"ssim_alpha": 0.5,
"ssim_window_size": 7,
"mse_dt_alpha": 0.5,
"mse_dt_sigma": 5.0,
"perceptual_alpha": 0.5,
"perceptual_slice_stride": 16,    # subsample axial slices for VGG forward
"combo_dice_ce_w": 0.5,
"combo_cldice_w": 0.25,
"combo_ssim_w": 0.25,
```

### 4.3 `train.py`

`--loss` argparse choices extended to include the four new tags. No other change — the training loop is loss-agnostic.

### 4.4 `utils/metrics.py`

Added three new metric functions:

| Function | Output |
|---|---|
| `compute_ssim3d(pred, gt, window_size=7)` | float in [0, 1] |
| `compute_psnr(pred, gt)` | float in dB (capped at 100) |
| `compute_feature_fid(pred_set, gt_set)` | float (lower = closer) |

`evaluate_volume()` now optionally runs the new metrics when `quick=False`. The new metrics operate on the same binary prediction the existing four metrics use — no change to the inference pipeline.

### 4.5 `run_final_ablation.sh` (new)

Orchestration script that:
1. Trains the 3 new from-scratch losses on TopCoW (300 ep, 8× A100).
2. Fine-tunes each on TopBrain (150 ep).
3. Trains the combination loss (300 ep + 150 ep TopBrain fine-tune).
4. Re-evaluates **all 7 models** with the extended metric suite.
5. Bundles training logs and eval summaries into `~/report_data_final.tar.gz`.

---

## 5. Training procedure (EC2 8× A100 — same as midterm)

### 5.0 Autonomous handoff sequence (Claude Code on EC2, follow in order)

Each step has a **check** command and an **expected** outcome. Proceed only if the check passes; otherwise stop and surface the failure.

| # | Step | Section | Stop on failure? |
|---|---|---|---|
| 0 | Environment + 8 GPUs available | §5.0.1 below | yes |
| 1 | Pull latest code, clean tree | §5.0.2 below | yes (ask user about conflicts) |
| 2 | Pre-flight: VGG download + 4-loss finite-scalar check | §5.1 | yes |
| 3 | DDP smoke test (1 epoch on slowest loss) | §5.1.5 | yes |
| 4 | Verify midterm checkpoints exist (orchestration Step 3 needs them) | §5.0.3 below | no, just note |
| 5 | Launch orchestration in background | §5.2 | — |
| 6 | Monitor progress over 5–7 hours | §5.2.1 below | — |
| 7 | Completion check + final-state assessment | §5.7 below | — |

#### 5.0.1 Environment + GPU check

```bash
source ~/vessel-env/bin/activate
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2
python -c "import torch; n = torch.cuda.device_count(); assert n == 8, f'expected 8 GPUs, got {n}'; print(f'OK: {n} GPUs')"
```
Expected: `OK: 8 GPUs`. If fewer, surface to user and stop — the loss alphas were tuned for the 8× A100 effective batch size.

#### 5.0.2 Pull latest code

```bash
git fetch origin
git status                # if dirty, ask user before stashing
git pull --ff-only origin main
```
Expected: fast-forward. If `git pull` reports merge conflicts or non-fast-forward, **stop** and ask the user; do not `git reset --hard` to "fix" it.

#### 5.0.3 Verify midterm checkpoints (so Step 3 of the orchestration has something to re-evaluate)

```bash
for d in dice_ce_20260409_001230 dice_ce_cldice_20260404_053853 \
         dice_ce_skeleton_20260402_203733 \
         finetune_topbrain_dice_ce_20260409_011117 \
         finetune_topbrain_dice_ce_cldice_20260404_064204 \
         finetune_topbrain_dice_ce_skeleton_20260402_215720; do
    if [ ! -f "runs/$d/best_model.pth" ]; then echo "MISSING: runs/$d/best_model.pth"; fi
done
```
Expected: empty output. If anything is MISSING, the orchestration will skip just that one eval (handled gracefully by the new `evaluate_model()` skip logic). Do **not** retrain to recover — those numbers already exist in the midterm; the missing artifact is only the *extended-metrics* re-eval, which can be filled in manually later if needed.

### 5.1 Pre-flight (one-time, on the EC2 instance)

```bash
source ~/vessel-env/bin/activate
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2

# Pull the new code
git pull

# Pre-download VGG-16 weights once, on rank-0 only, before launching DDP.
# Otherwise 8 ranks race-download the same file and may corrupt the cache.
python -c "import torchvision.models as M; M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1)"

# Sanity-check the new losses build and forward without OOM. We mirror
# train.py's pattern exactly: build_loss(cfg).to(device), then forward
# under autocast — this would have caught the device-placement bug
# before consuming EC2 hours.
python -c "
import torch
from losses.losses import build_loss
device = torch.device('cuda')
for tag in ['dice_ce_ssim', 'dice_ce_mse_dt', 'dice_ce_perceptual', 'dice_ce_cldice_ssim']:
    cfg = {'loss': tag, 'deep_supervision': True}
    crit = build_loss(cfg).to(device)
    logits = [torch.randn(1, 2, s, s, s, device=device) for s in (128, 64, 32, 16, 8)]
    targets = torch.randint(0, 2, (1, 128, 128, 128), device=device)
    with torch.amp.autocast('cuda'):
        loss = crit(logits, targets)
    assert torch.isfinite(loss), f'{tag} produced non-finite loss'
    print(f'{tag}: {loss.item():.4f}')
"
```

If any of the four prints fails, **stop** and report the error before consuming GPU time. Most likely failure mode is a torchvision-version mismatch on the VGG download — `pip install --upgrade torchvision` and retry, do **not** edit the loss code.

### 5.1.5 DDP smoke test (1 epoch, all 8 GPUs)

The single-process pre-flight does **not** exercise DDP. Run a 1-epoch multi-rank pass on the slowest new loss before committing to 5–7 hours. This catches DDP-specific issues (NCCL init, VGG broadcast race, gradient all-reduce mismatches) that the single-process pre-flight cannot.

```bash
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --loss dice_ce_perceptual \
    --epochs 1 --val_interval 1 \
    --output_dir ~/smoke_test
```

Expected: completes in **<10 min**, reports `nan_batches: 0`, peak VRAM `<30 GB` per rank, and validation Dice prints (will be terrible after 1 epoch — that's fine, we're checking plumbing). If OOM, edit `configs/default.py` and increase `perceptual_slice_stride` from 16 to 32, then retry. Once clean, delete the smoke-test run dir:

```bash
rm -rf ~/smoke_test
```

### 5.2 Run the full final ablation

```bash
nohup bash run_final_ablation.sh > ~/final_ablation.log 2>&1 &
echo $! > ~/final_ablation.pid
```

Expected wall time on 8× A100: **~5–7 hours total** (4 from-scratch × ~60–80 min + 4 fine-tunes × ~10 min + eval ~30 min). Perceptual is the slowest because of the per-batch VGG forward pass.

### 5.2.1 Monitoring during the long run

A Claude Code session does not need to poll continuously. Sample every 30–60 minutes is sufficient. Useful commands:

```bash
# Live progress (Ctrl-C to detach — does not kill the run)
tail -f ~/final_ablation.log

# Per-step success/failure log (empty = no failures so far)
cat ~/final_ablation_failures.log 2>/dev/null

# GPU utilization snapshot
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# Recent run directories as they appear
ls -lat runs/ | head -20

# Confirm the orchestrator is still alive
ps -p $(cat ~/final_ablation.pid) -o pid,etime,cmd
```

Symptoms that warrant intervention:
- `nvidia-smi` shows 0% utilization on all GPUs for >10 min → orchestrator is hung or between phases (check `tail` of log)
- `ps -p` returns "no such process" but bundle does not exist → orchestrator died, see §5.6
- `final_ablation_failures.log` grows past 2 lines → multiple losses are failing, likely a systemic issue (OOM, disk full, NCCL); stop and investigate rather than letting the run continue

### 5.3 Per-loss training command equivalents

If running individually instead of via the orchestration script:

```bash
# From-scratch on TopCoW
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce_ssim
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce_mse_dt
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce_perceptual
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice_ssim

# TopBrain fine-tune (after each from-scratch run completes)
torchrun --nproc_per_node=8 train.py \
    --data_dir ~/data/topcow2024 \
    --finetune runs/dice_ce_ssim_<TS>/best_model.pth \
    --topbrain_dir ~/data/topbrain \
    --loss dice_ce_ssim
# (repeat for the other 3 new losses)
```

### 5.4 Extended evaluation

```bash
python evaluate.py --checkpoint runs/<run>/best_model.pth \
    --data_dir ~/data/topcow2024 --extended_metrics
```

Add `--topbrain_dir ~/data/topbrain` for the fine-tuned models. The `--extended_metrics` flag toggles the SSIM / PSNR / F-FID computation (default off for back-compat with midterm eval scripts).

### 5.5 Bundling for download

The orchestration script tars the relevant logs and summaries into `~/report_data_final.tar.gz`. Pull with:

```bash
scp -i <key.pem> ubuntu@<ec2-ip>:~/report_data_final.tar.gz .
```

### 5.6 Recovery from partial failure

The orchestration is designed to keep going on failure, so a missing run is recoverable without re-running the whole pipeline.

**Case A — one loss failed at training time** (typical: transient OOM, NaN). Re-run that loss's three steps individually:

```bash
LOSS=dice_ce_perceptual    # set to whichever failed

# 1. From scratch
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss $LOSS

# 2. Fine-tune (use the run dir from step 1)
NEW_RUN=$(ls -dt runs/${LOSS}_2* | head -1)
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 \
    --finetune "$NEW_RUN/best_model.pth" \
    --topbrain_dir ~/data/topbrain --loss $LOSS

# 3. Evaluate both checkpoints with extended metrics
NEW_FT=$(ls -dt runs/finetune_topbrain_${LOSS}_2* | head -1)
python3 evaluate.py --extended_metrics --checkpoint "$NEW_RUN/best_model.pth" \
    --data_dir ~/data/topcow2024 2>&1 | tee "$NEW_RUN/eval_summary_extended.txt"
python3 evaluate.py --extended_metrics --checkpoint "$NEW_FT/best_model.pth" \
    --data_dir ~/data/topcow2024 --topbrain_dir ~/data/topbrain \
    2>&1 | tee "$NEW_FT/eval_summary_extended.txt"
```

**Case B — training crashed mid-epoch but `latest_checkpoint.pth` exists**. Use `--resume` to pick up where it stopped (state_dict, optimizer, scheduler, scaler, history all restored):

```bash
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 \
    --loss dice_ce_perceptual \
    --resume runs/dice_ce_perceptual_<TS>/latest_checkpoint.pth
```

**Case C — orchestrator died entirely**. Examine `tail -100 ~/final_ablation.log` for the cause. Most common is disk-full on `/tmp` (NCCL temp files) — clean up and re-launch only the steps that didn't complete by editing `NEW_LOSSES=(...)` at the top of `run_final_ablation.sh` to drop the already-finished losses, then re-run.

### 5.7 Success criteria for the final report

The handoff is complete when **all four** of the following are true:

1. **Bundle exists:** `ls -la ~/report_data_final.tar.gz` shows a fresh file under ~50 MB (logs and summaries only, not weights).
2. **Failure log is empty or only contains midterm-checkpoint skips:**
    ```bash
    test -s ~/final_ablation_failures.log && cat ~/final_ablation_failures.log || echo "NO FAILURES"
    ```
    Any "SKIP eval ... checkpoint missing" lines are acceptable; "FAIL (...): train ..." lines are not.
3. **Each new run dir has all four artifacts:**
    ```bash
    for L in dice_ce_ssim dice_ce_mse_dt dice_ce_perceptual dice_ce_cldice_ssim; do
        for SUFFIX in "" "finetune_topbrain_"; do
            DIR=$(ls -dt runs/${SUFFIX}${L}_2* 2>/dev/null | head -1)
            for F in best_model.pth training_log.json config.json eval_summary_extended.txt; do
                [ -f "$DIR/$F" ] || echo "MISSING: $DIR/$F"
            done
        done
    done
    ```
    Expected: empty output.
4. **Table VI in `report/final_report_draft.tex` is fillable** — every `\tbd` cell can be replaced with a number from `runs/<run>/eval_summary_extended.txt`.

If all four pass, copy `~/report_data_final.tar.gz` back to local and proceed to fill in the LaTeX placeholders. If any fail, see §5.6.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Perceptual loss slow / OOM on 128³ patches × 8 ranks | Subsample axial slices (`perceptual_slice_stride=16`); register VGG with `requires_grad=False`; run in `eval()` mode so BN buffers don't update |
| MSE-on-DT slow due to per-batch CPU `distance_transform_edt` | DT is computed inside `torch.no_grad()` on small 128³ patches — measured at <100 ms/batch on the dataset's typical foreground density. If this becomes a bottleneck, precompute DT in the dataloader. |
| AMP / fp16 numerical issues with SSIM Gaussian conv | The SSIM3D buffer is dynamically cast to the input's dtype at forward time |
| F-FID on N=25 is statistically meaningless | Reported as **F-FID** with explicit caveat; only used for relative ranking |
| Combo loss may NaN if components disagree | Train loop already replaces non-finite losses with zero (see `train.py` line 228–231); monitor `nan_batches` in the log |
| VGG weight download race under DDP | Pre-download once on rank 0 before launching `torchrun` (see §5.1) |

---

## 7. What the reviewer (Codex) should verify

1. **Non-redundancy:** every new section adds content not already in the midterm. ✅ See §2.
2. **Professor's feedback addressed:** at least 2 of the 5 listed losses (MSE, SSIM, perceptual) are added as **training objectives**, not just metrics. ✅ See §3.1.
3. **Honesty:** F-FID caveat stated; perceptual-loss simplification (no image modulation) stated; SSIM/PSNR on binary masks reported with their high-baseline caveat. ✅ See §3.2.
4. **Reproducibility:** all configs are seeded, all hyperparameters logged to `run_dir/config.json`, all training logs to `training_log.json`. ✅ See `train.py` lines 408–410, 728–731.
5. **Identical training conditions:** same architecture, optimizer, scheduler, augmentation, seed, hardware as midterm — only the loss differs. ✅ See §5.3 (no `--epochs` / `--lr` overrides).
6. **Page budget:** 2 pages, IEEE conference style — no figure-table sprawl. ✅ See `final_report_draft.tex`.

---

## 8. Citations to add to the bibliography

```
[17] Wang, Bovik, Sheikh, Simoncelli. "Image quality assessment: from error visibility
     to structural similarity." IEEE TIP 13(4), 2004.
[18] Qin, Zhang, Huang, Gao, Dehghan, Jagersand. "BASNet: Boundary-Aware Salient
     Object Detection." CVPR 2019.
[19] Mosinska, Marquez-Neila, Koziński, Fua. "Beyond the Pixel-Wise Loss for
     Topology-Aware Delineation." CVPR 2018.
[20] Kervadec, Bouchtiba, Desrosiers, Granger, Dolz, Ben Ayed. "Boundary loss for
     highly unbalanced segmentation." MedIA 67, 2021.
[21] Karimi, Salcudean. "Reducing the Hausdorff Distance in Medical Image Segmentation
     with CNNs." IEEE TMI 39(2), 2020.
[22] Johnson, Alahi, Fei-Fei. "Perceptual Losses for Real-Time Style Transfer and
     Super-Resolution." ECCV 2016.
[23] Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter. "GANs Trained by a Two
     Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS 2017.
[24] Ma et al. "Loss odyssey in medical image segmentation." MedIA 71, 2021.
```

---

## 9. Commit hygiene

Group the changes into three commits so the diff is reviewable:

1. **feat(losses):** add SSIM3D, MSE-on-DT, perceptual, and combo losses + factory entries
2. **feat(metrics):** add 3D-SSIM, PSNR, F-FID to evaluation suite + `--extended_metrics` flag
3. **chore(scripts):** add `run_final_ablation.sh` orchestrator + update `--loss` argparse choices

Do **not** commit `final_report_draft.tex` PDF builds, EC2 logs, or downloaded VGG weights. The existing `.gitignore` covers the first two; double-check before pushing.
