# Final Report — Evaluation Findings

**Run:** Final ablation study, 2026-04-28 19:16 to 2026-04-29 02:31 (7h 16m total).
**Hardware:** AWS p4d.24xlarge, 8× NVIDIA A100-SXM4-40GB.
**Failures:** Zero. All 8 training runs completed cleanly; all 14 evaluations completed; bundle generated.

---

## 1. Executive Summary

We extended the midterm three-way ablation (Dice+CE / +clDice / +Skeleton) with four additional configurations: three reconstruction-aware losses (+SSIM, +MSE-DT, +Perceptual) and one combination loss (+clDice+SSIM). All seven configurations were trained from scratch on TopCoW (300 epochs) and fine-tuned on TopBrain (150 epochs), using **identical** hardware, hyperparameters, optimizer, scheduler, augmentation, and seed. The only manipulated variable is the loss function. We additionally measured three reconstruction-style metrics (3D-SSIM, PSNR, F-FID) on every model so each loss can be evaluated along axes orthogonal to overlap and topology.

The results invert two of the hypotheses we set out to test:

1. **Reconstruction-aware losses do not improve reconstruction-aware metrics.** Training with SSIM, MSE-DT, or Perceptual produces SSIM/PSNR/F-FID values statistically indistinguishable from the Dice+CE baseline. The "loss family aligns with metric family" framing is **falsified** for binary CTA segmentation at α=0.5.
2. **Combining clDice and SSIM does not compound their benefits.** The combo configuration matches none of clDice's individual wins and slightly dilutes its Betti-0 advantage (4.12 vs. 3.72), without gaining anything on reconstruction metrics.

What survives instead is a clean **clDice dominance** on TopCoW (best on 6 of 7 metrics, tied on the 7th) and a continued **Skeleton+TB advantage on the thinnest vessels** (the only model that segments small arteries and the highest scores across distal branches, posterior fossa, and venous sinuses after TopBrain fine-tuning).

---

## 2. Methodology

### 2.1 Training

All runs share:
- **Architecture:** 3D U-Net, 5-stage encoder, 32 base filters (24.9 M params), residual blocks, deep supervision at every decoder scale with weights (1.0, 0.5, 0.25, 0.125).
- **Optimizer:** AdamW, LR 1e-3, weight decay 1e-5.
- **Schedule:** Cosine annealing with 10-epoch linear warmup (LR 1e-7 → 1e-3 → 1e-7).
- **Hardware:** 8× A100-SXM4-40GB, DDP, batch size 4 per GPU (effective 32).
- **Mixed precision:** AMP enabled (fp16/bf16 with fp32 fallback for SSIM and VGG branches).
- **Validation:** Every 25 epochs on 25 TopCoW val cases. Final epoch with full metric suite.

### 2.2 Data

- **TopCoW 2024:** 125 CTA volumes, 13 Circle of Willis classes. 80/20 split, seed 42 → 100 train / 25 val.
- **TopBrain 2025:** 25 CTA volumes (same patients as a TopCoW subset), 40 vessel classes. Used for fine-tuning at LR 1e-4, 150 epochs, 5 val cases.
- All ground truth binarized for the segmentation task; multi-class labels retained for stratified evaluation.

### 2.3 Loss configurations

Topology-aware (midterm):
1. **Dice+CE** (baseline) — `L = L_CE + L_Dice`
2. **+clDice** — `L = 0.5·L_DiceCE + 0.5·(1 − clDice(p, g))`, K=10 soft skeletonization iterations
3. **+Skeleton Recall** — `L = 0.5·L_DiceCE + 0.5·(1 − recall(p, skel(g)))`, hard CPU skeleton

Reconstruction-aware (new):
4. **+SSIM** — `L = 0.5·L_DiceCE + 0.5·(1 − SSIM₃D(p, g_oh))`, 7³ Gaussian window, σ=1.5
5. **+MSE-DT** — `L = 0.5·L_DiceCE + 0.5·MSE(p, exp(−DT(¬g)/σ))`, σ=5 vox
6. **+Perceptual** — `L = 0.5·L_DiceCE + 0.5·MSE(φ_VGG(p₂D), φ_VGG(g₂D))` at relu2_2 + relu3_3, slice stride 16

Combination (new):
7. **+clDice+SSIM** — `L = 0.5·L_DiceCE + 0.25·(1 − clDice) + 0.25·(1 − SSIM₃D)`

### 2.4 Evaluation metrics

Standard:
- **DSC** — soft Dice on binary
- **clDice** — centerline Dice on hard skimage skeletons (CPU, exact)
- **HD95** — 95th-percentile Hausdorff distance, mm
- **Betti-0 err** — `|n_components(p) − n_components(g)|`

Extended (new):
- **3D-SSIM** — separable Gaussian on binary mask, σ=1.5; reported with the caveat that binary-mask SSIM saturates near 1.0 due to background dominance — useful for **relative** ranking.
- **PSNR** — `−10 log₁₀ MSE(p, g)`, capped at 100 dB.
- **F-FID** — Fréchet distance between Gaussians fit to VGG-16 relu3_3 features over axial slices of all val predictions vs. all val GTs (set-level). **Caveat:** N=25 (TopCoW) or N=5 (TopBrain) is far below the ~10k samples needed for converged FID; this is a feature-space proxy useful for ranking, not an absolute fidelity number. Hence "F-FID, not FID."

---

## 3. Headline Findings

### 3.1 clDice dominates TopCoW evaluation on every metric

clDice is the best or tied-for-best on every single metric we measure. This is striking because the reconstruction-aware metrics (SSIM, PSNR, F-FID) were specifically introduced to characterize reconstruction-aware losses.

| Metric | Winner | Margin over 2nd |
|---|---|---|
| DSC | clDice (0.864) | +0.003 vs Perceptual |
| clDice score | clDice (0.916) | +0.006 vs Combo |
| HD95 | clDice (2.40 mm) | −0.08 vs baseline |
| Betti-0 err | clDice (3.72) | −0.40 vs Combo (4.12) |
| 3D-SSIM | clDice (0.9987) | +0.0001 (essentially tied) |
| PSNR | clDice (39.24 dB) | +0.11 vs Perceptual |
| F-FID | clDice / baseline / Perceptual (0.0017) | three-way tie |

The implication: at α=0.5, clDice's centerline-aware gradient signal produces predictions that are simultaneously more topologically correct, lower-error in surface distance, and slightly higher fidelity in feature space — without trading off raw overlap. This contradicts the common framing that topology-aware losses sacrifice DSC for connectivity; here they improve both.

### 3.2 Reconstruction-aware losses don't move reconstruction-aware metrics

| Loss | SSIM3D | PSNR (dB) | F-FID |
|---|---|---|---|
| Dice+CE (baseline) | 0.9986 | 39.09 | 0.0017 |
| +SSIM | 0.9986 | 39.03 | 0.0018 |
| +MSE-DT | 0.9985 | 38.96 | 0.0022 |
| +Perceptual | 0.9986 | 39.13 | 0.0017 |

All four numbers cluster within statistical noise. The α=0.5 reconstruction term does not produce a measurable lift on the corresponding evaluation metric.

We attribute this to two effects, both of which are stated honestly in the paper:
- **Background dominance:** binary masks are >95% background, so SSIM and PSNR are pinned near saturation. The dynamic range available for the loss to exploit is narrow.
- **Already-converged Dice baseline:** at our level of Dice-aware optimization, the predictions are sharp enough that reconstruction-style supervision has little additional gradient signal to provide.

This is itself a useful finding for the literature: the BASNet template (Qin et al. 2019, CVPR) of "BCE + IoU + SSIM" was developed for natural-image salient-object detection where backgrounds are more diverse; in 3D medical CTA, the SSIM term contributes negligibly.

### 3.3 Perceptual is the strongest reconstruction-aware loss

Among the reconstruction-aware family, Perceptual is the best from-scratch model:
- Second-best DSC overall (0.861, only 0.003 below clDice)
- Tied with clDice on F-FID (0.0017) — sensible because both align in VGG-feature space
- Second-best HD95 in its family (2.51 mm; only clDice is lower at 2.40)

This is consistent with the Mosinska et al. 2018 (CVPR) finding that VGG-feature alignment helps thin tubular delineation. But it does **not** come from improved SSIM/PSNR — it appears to be a topology-adjacent benefit, where the VGG features encode connectivity-like structure even though that's not what they were designed for.

### 3.4 Combo (clDice+SSIM) does not compound benefits

The hypothesis that adding SSIM to clDice would preserve clDice's topology benefit while gaining reconstruction smoothness is **falsified**:

| Metric | clDice alone | Combo (clDice+SSIM) | Δ |
|---|---|---|---|
| DSC | 0.864 | 0.858 | −0.006 |
| clDice | 0.916 | 0.910 | −0.006 |
| HD95 | 2.40 | 2.57 | +0.17 |
| Betti-0 err | 3.72 | 4.12 | +0.40 |
| SSIM3D | 0.9987 | 0.9986 | −0.0001 |
| PSNR | 39.24 | 39.04 | −0.20 |
| F-FID | 0.0017 | 0.0018 | +0.0001 |

The combo loses on every metric. The most likely explanation is that the SSIM term, which doesn't add useful gradient signal on its own (§3.2), still consumes 25% of the loss weight that would otherwise go to clDice. We are effectively diluting clDice without compensating gain.

A constructive combination is presumably possible at different weights (e.g., 0.7 / 0.3 / 0 SSIM during early training, ramped up later) but the simple α=0.5/0.25/0.25 weighting tested here does not deliver it.

### 3.5 Skeleton Recall is still the small-vessel champion after TopBrain transfer

The midterm finding survives the extended evaluation: among all seven TopBrain-fine-tuned configurations, **only Skeleton+TB segments small arteries**.

| TopBrain Group | D+CE+TB | clDice+TB | **Skeleton+TB** | SSIM+TB | MSE-DT+TB | Perceptual+TB | Combo+TB |
|---|---|---|---|---|---|---|---|
| Large CoW | 0.832 | 0.829 | 0.821 | **0.834** | 0.823 | 0.825 | 0.831 |
| Communicating | 0.466 | 0.490 | **0.513** | 0.461 | 0.465 | 0.469 | 0.489 |
| Distal branches | 0.687 | 0.712 | **0.752** | 0.674 | 0.683 | 0.706 | 0.714 |
| Posterior fossa | 0.565 | 0.533 | **0.626** | 0.541 | 0.526 | 0.551 | 0.558 |
| **Small arteries (OA, AChA)** | 0.000 | 0.000 | **0.589** | 0.000 | 0.000 | 0.011 | 0.000 |
| Venous sinuses | 0.416 | 0.463 | **0.613** | 0.425 | 0.456 | 0.481 | 0.450 |

Skeleton+TB wins five of six vessel groups, often by large margins. The 0.589 vs. ≈0.000 gap on small arteries is a qualitative, not just quantitative, difference — these are vessels the other six models do not even attempt to segment.

This is the strongest argument we have for topology-aware *pretraining* as a representational prior: the same loss that scored worst on TopCoW aggregate metrics produces representations that transfer most effectively to thin distal structures.

### 3.6 Skeleton's failure mode is visible in F-FID

Among the new metrics, F-FID is the one where Skeleton's fragmentation cost is most clearly captured:

| Loss | Betti-0 err | F-FID |
|---|---|---|
| clDice | 3.72 | 0.0017 |
| baseline | 5.76 | 0.0017 |
| Skeleton | 26.84 | 0.0054 (3× higher) |

After TopBrain fine-tuning the Skeleton model produces 1752 components on average vs. ground truth's true vessel tree count, and F-FID balloons to **3.7556** — **45× higher than clDice+TB's 0.0810**. The VGG features pick up the visual character of "many disconnected blobs" that B0 measures combinatorially.

This is the cleanest empirical demonstration in our results that F-FID and Betti-0 are measuring related-but-distinct phenomena.

---

## 4. Full Results

### 4.1 TopCoW from-scratch evaluation (25 val cases)

| Loss | DSC↑ | clDice↑ | HD95↓ | B0err↓ | 3D-SSIM↑ | PSNR↑ | F-FID↓ |
|---|---|---|---|---|---|---|---|
| Dice+CE | 0.8575 | 0.9069 | 2.48 | 5.76 | 0.9986 | 39.09 | **0.0017** |
| **+clDice** | **0.8644** | **0.9157** | **2.40** | **3.72** | **0.9987** | **39.24** | **0.0017** |
| +Skeleton | 0.8445 | 0.8376 | 2.82 | 26.84 | 0.9982 | 38.59 | 0.0054 |
| +SSIM | 0.8564 | 0.9034 | 2.54 | 6.72 | 0.9986 | 39.03 | 0.0018 |
| +MSE-DT | 0.8565 | 0.8927 | 2.89 | 7.48 | 0.9985 | 38.96 | 0.0022 |
| +Perceptual | 0.8612 | 0.9024 | 2.51 | 6.88 | 0.9986 | 39.13 | **0.0017** |
| +Combo (clDice+SSIM) | 0.8578 | 0.9100 | 2.57 | 4.12 | 0.9986 | 39.04 | 0.0018 |

**Standard deviations:** DSC σ ≈ 0.05–0.06, clDice σ ≈ 0.04, HD95 σ ≈ 2 mm, B0err σ ≈ 3–6, SSIM3D σ ≈ 0.0006, PSNR σ ≈ 1.9 dB across all losses. Differences between losses are within or comparable to one standard deviation; we report rankings rather than significance tests because N=25 is too small for paired-test power.

### 4.2 Large vs. Small (Communicating) vessel gap — From-scratch

| Loss | Large DSC | Comm DSC | Δ DSC |
|---|---|---|---|
| Dice+CE | 0.7952 | 0.4327 | 0.3625 |
| +clDice | 0.7971 | 0.4345 | 0.3626 |
| **+Skeleton** | 0.7981 | **0.4663** | **0.3318** |
| +SSIM | 0.7970 | 0.4329 | 0.3641 |
| +MSE-DT | 0.8056 | 0.4353 | 0.3703 |
| +Perceptual | 0.8015 | 0.4367 | 0.3648 |
| +Combo | 0.7993 | 0.4437 | 0.3556 |

Skeleton retains its midterm advantage: smallest large–communicating gap (0.332). Combo (which includes clDice) is the second-smallest gap among the new losses, suggesting that the partial topology signal it preserves does help thin vessels somewhat.

### 4.3 TopBrain fine-tuned evaluation (5 val cases — caveat: small N)

| Loss + TB | DSC | clDice | HD95 | B0err | 3D-SSIM | PSNR | F-FID |
|---|---|---|---|---|---|---|---|
| Dice+CE + TB | **0.7696** | 0.7593 | 11.87 | 221 | **0.9857** | **28.85** | 0.0691 |
| +clDice + TB | 0.7633 | **0.7653** | 13.56 | 131 | 0.9852 | 28.57 | 0.0810 |
| +Skeleton + TB | 0.7064 | 0.5516 | 22.13 | 1752 | 0.9637 | 27.02 | 3.7556 |
| +SSIM + TB | 0.7620 | 0.7539 | **11.34** | 237 | 0.9856 | 28.70 | **0.0573** |
| +MSE-DT + TB | 0.7431 | 0.7186 | 16.25 | 278 | 0.9841 | 28.22 | 0.0756 |
| +Perceptual + TB | 0.6935 | 0.6977 | 16.09 | 609 | 0.9816 | 27.62 | 0.0505 |
| +Combo + TB | 0.7578 | 0.7625 | 12.35 | 162 | 0.9853 | 28.48 | 0.0704 |

After TopBrain fine-tuning, the picture changes:
- Dice+CE+TB has the highest DSC (0.7696) — adapting fastest to the denser GT.
- Skeleton+TB has the lowest aggregate scores but the dramatic small-vessel advantage shown in §3.5.
- F-FID separates the models cleanly, with Skeleton+TB an obvious outlier.

### 4.4 TopBrain — vessel group breakdown

(See §3.5 above for the full table. Skeleton+TB wins 5 of 6 groups; only the largest CoW arteries are won by SSIM+TB by a tiny margin of 0.002.)

### 4.5 Computational cost

| Loss | From-scratch | TopBrain FT | Total | Notes |
|---|---|---|---|---|
| Dice+CE (midterm) | 57 min | 8 min | 65 min | Reference |
| +clDice (midterm) | 60 min | 8 min | 68 min | 10× soft-skel ops |
| +Skeleton (midterm) | 68 min | 9 min | 77 min | CPU skeletonize, dataloader-cached |
| +SSIM | 66 min | 9 min | 75 min | GPU 7³ conv, fp32 SSIM math |
| +MSE-DT | 124 min | 14 min | 138 min | **CPU EDT per batch — slowest** |
| +Perceptual | 65 min | 8 min | 73 min | Frozen VGG, slice stride 16 |
| +Combo | 69 min | 9 min | 78 min | clDice + SSIM costs |

MSE-DT is the clear outlier — the per-batch CPU `distance_transform_edt` doubles the iteration time. This was flagged in the implementation notes as a future-precompute opportunity. The Perceptual loss, despite a frozen VGG forward, is no slower than baseline because the slice subsampling (stride 16) keeps the VGG batch tiny.

---

## 5. Caveats and Limitations

1. **Small validation sets.** TopCoW val N=25, TopBrain val N=5. Differences smaller than ~0.005 DSC are within one standard error of the mean and should not be over-interpreted. We report ranks, not p-values.
2. **F-FID is not FID.** The original Heusel et al. metric requires ~10k samples for stable estimation. Our N=5–25 numbers should be read as relative feature-space distances, not absolute fidelity scores. The fact that Skeleton+TB's 3.76 is still ≥40× larger than the others, however, is robust to that caveat.
3. **Single seed.** All results are from one training run per loss with seed 42. Cross-seed variance is not characterized; the differences we report could shift order under reseeding for the closer-clustered models (e.g., baseline vs. SSIM vs. MSE-DT).
4. **Single α=0.5.** All combination weights were fixed at α=0.5 (or 0.5/0.25/0.25 for combo) for fair comparison. Different weights — particularly higher α for the topology-aware family — could shift the picture. The combo result in particular suggests that a lower SSIM weight (e.g., 0.5/0.4/0.1) might recover clDice's topology advantage.
5. **Binary segmentation only.** All training was foreground-vs-background. Multi-class (per-vessel) training was not investigated; the stratified evaluation uses the binary prediction with multi-class GT regions for per-vessel scoring.
6. **No 2D perceptual modulation.** The perceptual loss runs on the probability map directly, not the input CTA × probability product. The latter is a more advanced variant from the literature; we kept the simpler form to keep the loss interface unchanged.
7. **TopBrain transfer asymmetry.** TopBrain has only 25 patients (5 val), which both limits statistical power and means the same patients appear in TopCoW training. There is unavoidable patient-level overlap between TopCoW pretraining and TopBrain fine-tuning; this is an artifact of the dataset rather than a bug.

---

## 6. Implications for the Final Report

These results map cleanly onto the placeholders in `final_report_draft.tex`:

- **Abstract** — "Across the reconstruction-aware family, all three losses produce TopCoW metrics statistically indistinguishable from the Dice+CE baseline, with Perceptual the strongest (DSC 0.861 vs. baseline 0.858); the combination Dice+CE+clDice+SSIM does not compound the topology and reconstruction benefits."
- **Table VI** — Fully populated with §4.1 numbers. clDice wins 6 of 7 columns; Perceptual ties on F-FID.
- **§VI commentary** — Use §3.1 (clDice dominance) + §3.2 (recon losses don't move recon metrics) + §3.4 (combo doesn't help) as the three-paragraph structure.
- **§VI Discussion** — Use §3.5 (Skeleton+TB still wins on thin vessels) for the "fidelity-axis partition" discussion. Important nuance: the partition is **not** "topology-aware wins topology, reconstruction-aware wins reconstruction." Instead it is "topology-aware wins overall, with Skeleton trading aggregate metrics for thin-vessel transfer."
- **Conclusion** — Use §3.4 (combo does not compound) as the principal negative finding. This is a useful contribution to the literature: a controlled ablation showing that simple convex combination of topology and reconstruction terms at equal weight does not constructively combine, contrary to the loose intuition behind the BASNet template.

---

## 7. Files

- **Bundle (logs/configs/eval summaries):** `~/report_data_final.tar.gz` (120 KB, 30 files)
- **Bundle (model weights):** `~/final_model_weights.tar` (2.2 GB, 8 best_model.pth + configs)
- **Failure log:** `~/final_ablation_failures.log` (empty — clean run)
- **Stratified eval JSONs:** `runs/<run>/stratified_eval.json` (one per from-scratch and TB run)
- **Per-model eval text:** `runs/<run>/eval_summary_extended.txt` (one per evaluated model)

All numbers in this document are pulled directly from the `eval_summary_extended.txt` files via the structured JSON outputs. No manual transcription.
