# Presentation Script — Final Report Additions

**Purpose:** speaker notes for **5 new slides** to insert into the existing 21-slide midterm deck (`presentation_v2_darktheme.pptx`) covering the final-report extension. Slot these in **between Slide 17 (Qualitative Comparison) and Slide 18 (Discussion — What We Learned)** so the new findings are presented before the existing summary, then the existing Slides 18–21 can be lightly updated to reference them.

**Total added time: ~4–5 minutes** (raises a 14-min midterm talk to ~19 min — trim Slides 5 and 17 if a hard 15-min cap applies).

**Slide numbering convention:** I label these 17a–17e so they don't disrupt the existing numbering. After insertion, your Slide 18 becomes Slide 22, etc.

---

## Slide 17a: From Midterm to Final — The Professor's Question (~30 seconds)

**Key points:**
- Midterm tested topology-aware losses; result was clean and supported the hypothesis
- The professor asked whether reconstruction-style losses (MSE, PSNR, SSIM, perceptual, FID) could play a similar role
- This frames the final report as: same controlled-ablation discipline, dual-axis question

**Suggested visual:** Two columns. Left: "Midterm — Topology-aware family" with three logos for Dice+CE, clDice, Skeleton Recall. Right: "Final — Reconstruction-aware family" with three logos for SSIM, MSE-DT, Perceptual. A "+ combination" panel underneath.

**Script:**
> "After the midterm, my professor asked a sharp question: I had shown that *topology-aware* losses help with thin vessels. But what about a different family of losses that exists in the literature — image-quality or *reconstruction-aware* losses — like MSE, SSIM, perceptual loss, FID? These are well-studied for image restoration, super-resolution, and salient-object detection, but they've been almost completely untested for binary medical segmentation."
>
> "So the final report extends the same controlled-ablation methodology to ask: do reconstruction-aware losses help where topology-aware losses don't? And: if I combine the best member of each family, do their benefits compound? That gives us seven loss configurations total — three from the midterm, three new reconstruction-aware ones, and one combination."

---

## Slide 17b: Three New Losses + Three New Metrics (~1 minute)

**Key points:**
- Three new losses, all sharing the Dice+CE base and α=0.5 weight on the new term:
  - **+SSIM**: 3D structural similarity on probability map (BASNet recipe, CVPR 2019)
  - **+MSE-on-distance-transform**: regress prediction against `exp(−DT(¬g)/σ)` (Kervadec MIDL 2019, Karimi TMI 2020)
  - **+Perceptual**: 2D slice-wise VGG-16 ImageNet features at relu2_2 + relu3_3 (Mosinska CVPR 2018)
- Three new evaluation metrics — all reconstruction-style: 3D-SSIM, PSNR, F-FID
- Plus one combination loss: Dice+CE + clDice + SSIM at (0.5, 0.25, 0.25) weights
- Identical training pipeline to midterm: 3D U-Net, 8× A100, 300 epochs from scratch + 150 fine-tune on TopBrain

**Suggested visual:** Three equation cards, then a metric-icons row underneath.

**Script:**
> "I added three new training configurations. The first is **Dice+CE+SSIM** — three-D structural similarity on the soft probability map. This is borrowed from BASNet, a 2019 CVPR paper that showed SSIM as an auxiliary loss helps boundary sharpness on binary salient-object masks."
>
> "The second is **Dice+CE+MSE-on-distance-transform**. Plain MSE on a binary mask doesn't work because vessels are only one to three percent of voxels — the loss is dominated by background. So instead we regress the prediction against a Gaussian-decayed distance transform target, which is a published recipe from Kervadec and Karimi."
>
> "The third is **Dice+CE+Perceptual**. We pass axial slices of the prediction through a frozen ImageNet-pretrained VGG-16 and align features at relu2_2 and relu3_3. This is the Mosinska 2018 recipe for thin tubular structures."
>
> "I also added three new evaluation metrics — three-D SSIM, PSNR, and Fréchet Feature Distance — to evaluate every model on a fidelity axis orthogonal to overlap and topology. Note F-FID is not a converged Inception-FID — N=25 is far below the ten-thousand samples needed for that — so I report it explicitly as a feature-space proxy useful for ranking. Honest naming."
>
> "Same architecture, same hyperparameters, same eight-A100 hardware. Only the loss varies."

---

## Slide 17c: Headline Finding #1 — clDice Wins Every Column (~1 minute)

**Key points:**
- Extended Table V — seven models × seven metrics — clDice is best or tied-for-best on **every single column**
- Includes the three reconstruction-style metrics that were specifically introduced to characterize reconstruction-aware losses
- The reconstruction-aware family (SSIM, MSE-DT, Perceptual) cluster within statistical noise of the baseline on every metric
- This falsifies our "loss family aligns with metric family" hypothesis

**Suggested visual:** The Table V rendered as a heat-map. Bold the clDice row across all columns. Annotation: "expected this row to win the topology metrics; surprised it wins the reconstruction metrics too."

**Script:**
> "Here's the surprising part. I expected reconstruction-aware losses to win on the new reconstruction-aware metrics — SSIM, PSNR, F-FID. They didn't. Look at this table: clDice is the best, or tied-for-best, on every single column. Including the three metrics I specifically introduced to characterize the reconstruction-aware losses."
>
> "The +SSIM model and the Dice+CE baseline produce three-D-SSIM scores within 0.0001 of each other. Same story on PSNR — within 0.06 dB. Same on F-FID. The α=0.5 reconstruction term basically doesn't move the needle on the corresponding metric."
>
> "Why? Two reasons. First, binary CTA masks are over 95% background, so SSIM and PSNR are pinned near saturation — there's just very little dynamic range for the loss to exploit. Second, the Dice-aware baseline is already producing sharp predictions, and the reconstruction-style supervision has very little additional gradient signal to provide."
>
> "This is actually a useful contribution to the literature. The BASNet template — BCE plus IoU plus SSIM — was developed for natural-image salient-object detection where backgrounds are diverse and textured. In 3D medical CTA, that template contributes negligibly. We have a controlled-ablation argument for that claim."

---

## Slide 17d: Headline Finding #2 — Combo Loss Dilutes Instead of Compounds (~45 seconds)

**Key points:**
- Combo (Dice+CE+clDice+SSIM at 0.5/0.25/0.25 weights) loses to clDice alone on every metric
- DSC 0.858 vs clDice's 0.864; Betti-0 error 4.12 vs 3.72; F-FID 0.0018 vs 0.0017
- Mechanism: SSIM term has no useful gradient signal but still consumes 25% of the loss weight that would otherwise reinforce clDice
- A constructive combination is presumably possible at lower SSIM weight or with a curriculum schedule, but the simple equal-weight form doesn't deliver

**Suggested visual:** Bar chart with 7 bars across 4 metrics, highlighting that the combo bar is lower than the clDice bar on every metric.

**Script:**
> "The second finding inverted another hypothesis. I expected that combining clDice with SSIM would preserve clDice's topology benefit and add reconstruction smoothness. So I added one more model — Dice+CE plus clDice plus SSIM at half-quarter-quarter weights."
>
> "It loses on every metric. Slightly worse DSC. Worse Betti-0 error. Worse F-FID. Not by huge amounts, but consistently."
>
> "The likely explanation: the SSIM term, which we already showed doesn't add useful gradient signal on its own, still consumes 25% of the total loss weight. That's 25% of the gradient that would otherwise be reinforcing clDice. So we're effectively diluting the topology supervision without compensating gain."
>
> "Important caveat: this is just one weighting. A constructive combination could exist at lower SSIM weight, or with a curriculum schedule where SSIM ramps in late. But the simple equal-weight form everyone tries first does not work."

---

## Slide 17e: What Survives — Skeleton Still Owns Small Vessels (~45 seconds)

**Key points:**
- Despite winning nothing on TopCoW aggregate metrics, Skeleton Recall + TopBrain remains the only model that segments small arteries (0.589 DSC vs ≤ 0.011 for every other model)
- Wins five of six TopBrain vessel groups
- F-FID separates Skeleton's fragmentation cost more cleanly than any single-volume metric (3.76 vs 0.07–0.08 for the others — that's 45×)
- Cleanest argument that topology-aware *pretraining* shapes the representation, even when fine-tuning happens afterward with rich annotations

**Suggested visual:** Bar chart from final report Fig 4 — the TopBrain vessel-group bar chart with the small-arteries column highlighted in red.

**Script:**
> "The midterm finding survives the extended evaluation. Across all seven configurations after TopBrain fine-tuning, Skeleton Recall is still the only model that segments small arteries — the ophthalmic and anterior choroidal arteries. It scores 0.589 DSC where every other model scores at most 0.011. Among the new reconstruction-aware losses, only Perceptual produces any non-zero output, and that's a single percent."
>
> "There's also a nice auxiliary observation about F-FID. Skeleton has 1752 connected components on average after TopBrain fine-tuning — the others are around 130 to 280. Skeleton's F-FID is 3.76, which is 45 times higher than clDice's 0.08. The VGG features are picking up the visual character of 'many disconnected blobs' that Betti-0 captures combinatorially. So F-FID and Betti-0 measure related but distinct phenomena, and they both agree Skeleton is the topology outlier."
>
> "This is the strongest empirical argument we have: topology-aware pretraining produces representations that transfer more effectively to thin distal structures, even though Skeleton scored worst on TopCoW aggregate metrics."

---

## Edits to existing slides

### Slide 18 (Discussion) — minor update

Add one bullet:
- *"And here's where the final-report findings change the picture: the loss-family-vs-metric-family hypothesis was falsified. clDice dominates every fidelity axis simultaneously. Reconstruction-aware losses don't 'own' a complementary fidelity axis."*

### Slide 19 (Limitations & Future Work) — add bullet

Add one bullet:
- *"All combination weights were tested at α=0.5 (or 0.5/0.25/0.25 for combo). A constructive clDice+SSIM combination at lower SSIM weight or with curriculum scheduling could still exist; we did not search for it."*

### Slide 20 (Conclusion) — replace one sentence

Replace the existing "Topology-aware losses help thin vessels" sentence with:

> *"After extending to seven loss configurations: topology-aware supervision is the dominant beneficial axis along which to tune the loss for binary cerebral CTA. clDice wins on TopCoW; Skeleton Recall on small-vessel transfer. Reconstruction-aware terms imported from natural-image salient-object detection contribute negligibly at α=0.5."*

---

## Optional Slide 17f — only if you have time and want to add depth

### Slide 17f: Compute Cost Comparison (~30 seconds)

**Key points:**
- All four new losses train at 65–138 minutes for 300 epochs on 8× A100
- MSE-DT is the outlier (138 min, 2× slower) because the per-batch CPU distance transform is the bottleneck — flagged as a future-precompute opportunity
- Perceptual is no slower than baseline despite VGG forward, because slice-stride 16 keeps the VGG batch tiny

**Skip this slide unless audience asks about training costs.**

---

## Notes on insertion

- Insert in `presentation_v2_darktheme.pptx` between current Slides 17 and 18.
- Match the existing dark theme: same background gradient, same font (looks like Inter or similar sans), same heading-and-bullet layout.
- For Slide 17b, the equation cards can be screenshots of the LaTeX equations from `final_report_draft.tex` Section V.
- For Slide 17c, the heat-map can be generated from `eval_summary_extended.txt` or just typeset Table V directly.
- For Slide 17e, reuse `figures_v4/topbrain_extended_groups.png` directly — same figure as Fig 4 in the paper.
- Total added time: ~4 minutes. If your venue caps at 15, drop Slide 5 (Related Work — already covered in Slides 9–11) and one of Slides 17 (Qualitative — fine to skip if showing the bar chart in 17e).
