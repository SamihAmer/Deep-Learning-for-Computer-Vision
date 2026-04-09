# Presentation Script — Deep Learning for Vessel Segmentation in Cerebral CTA

**Total time target: 12-13 minutes** (leaves buffer under 15-minute max)

---

## Slide 1: Title Slide (~15 seconds)

**Content:**
- Title: "Deep Learning for Vessel Segmentation in Cerebral CTA: A Controlled Ablation of Topology-Aware Loss Functions"
- Author: Samih Tharwat Amer
- Affiliation: Whiting School of Engineering, Johns Hopkins University
- Date: April 2026

**Visuals:** University logo (optional), clean title layout

**Script:**
> "Hi, my name is Samih Amer. Today I'll be presenting my midterm project on deep learning for vessel segmentation in cerebral CT angiography, specifically a controlled ablation study comparing topology-aware loss functions."

---

## Slide 2: Clinical Motivation (~1 minute)

**Content:**
- What is CTA vessel segmentation and why it matters
- Circle of Willis anatomy — the arterial ring at the base of the brain
- Clinical applications: stroke planning, aneurysm detection, surgical navigation
- The problem: communicating arteries are thin but clinically critical

**Visuals:**
- Diagram or image of the Circle of Willis with labeled arteries (large arteries in blue, communicating arteries highlighted in red)
- Can use a public-domain CoW anatomy diagram

**Script:**
> "Cerebrovascular diseases — stroke, aneurysms, AVMs — are a leading cause of death and disability. Accurate segmentation of the cerebral vasculature from CT angiography is essential for diagnosis and surgical planning."
>
> "The Circle of Willis is the arterial ring at the base of the brain. It has large named arteries — the ICA, MCA, basilar — shown here in blue. But the thin communicating arteries — the Acom and Pcom, shown in red — are the ones that matter most clinically. They provide collateral blood flow when a major vessel is blocked. If your segmentation misses or disconnects these, it's clinically useless — even if the Dice score looks good on paper."

---

## Slide 3: The Problem — Topological Failures (~1 minute)

**Content:**
- Standard Dice+CE loss treats all voxels equally — no connectivity incentive
- A segmentation can have high Dice but broken topology (severed vessels, fragments)
- Aggregate metrics hide failures on thin structures

**Visuals:**
- Conceptual diagram: two segmentations with same Dice score — one connected, one fragmented
- Or: a bar showing "global DSC 0.87" vs per-vessel breakdown showing Pcom at 0.42

**Script:**
> "The standard approach for training segmentation networks uses Dice loss plus cross-entropy. These objectives treat every voxel equally. They have no concept of connectivity or topology. So a model can achieve a Dice score of 0.87 — looks great on paper — while completely severing the communicating arteries."
>
> "When we actually tested our baseline model in a clinical 3D viewer, it missed a significant number of vessels despite the high Dice score. The global metric was dominated by the large, easy arteries, and was hiding the failure on the thin structures that actually matter."

---

## Slide 4: Hypothesis & Approach (~45 seconds)

**Content:**
- Hypothesis: Topology-aware loss functions will disproportionately improve segmentation of thin communicating arteries
- Approach: Controlled ablation — same architecture, same data, same hardware, vary ONLY the loss function
- Three configurations: (1) Dice+CE, (2) Dice+CE+clDice, (3) Dice+CE+Skeleton Recall

**Visuals:**
- Simple diagram showing the three loss configs as branches from the same architecture/data/hardware
- Emphasize "controlled" — everything identical except the loss

**Script:**
> "Our hypothesis is that topology-aware loss functions — losses that explicitly penalize broken connectivity — will disproportionately improve segmentation of these thin communicating arteries."
>
> "To test this cleanly, we designed a controlled ablation study. Same 3D U-Net architecture, same data, same hardware, same hyperparameters — we change only the loss function. Three configurations: the baseline Dice plus cross-entropy, adding soft centerline Dice from Shit et al. at CVPR 2021, and adding Skeleton Recall from Kirchhoff et al. at ECCV 2024."

---

## Slide 5: Datasets (~1 minute)

**Content:**
- TopCoW 2024: 125 CTA volumes, 13 Circle of Willis vessel classes, MICCAI challenge data
- TopBrain 2025: Same 25 CT scans but with 40 vessel classes (distal branches, posterior fossa, small arteries, venous sinuses)
- 80/20 train/val split on TopCoW; TopBrain used for fine-tuning evaluation

**Visuals:**
- Table: TopCoW vs TopBrain comparison (# volumes, # classes, vessel types covered)
- Maybe a visual showing the 13 CoW classes vs the expanded 40-class TopBrain coverage

**Script:**
> "We use two datasets. TopCoW 2024 is a MICCAI challenge dataset with 125 CTA volumes and expert annotations for 13 Circle of Willis vessel classes. We use the CT subset with an 80/20 split — 100 training, 25 validation."
>
> "TopBrain 2025 provides the same 25 CT scans but with dramatically richer labels — 40 vessel classes covering not just the Circle of Willis but also distal branches, posterior fossa vessels like the vertebral arteries and PICAs, small arteries like the ophthalmic, and venous sinuses. When binarized, TopBrain masks cover far more of the cerebral vasculature. We use this for fine-tuning to test how well each loss function's learned representations transfer to comprehensive vessel coverage."

---

## Slide 6: Architecture — 3D U-Net (~45 seconds)

**Content:**
- 3D U-Net following nnU-Net design
- 5 encoder stages (32→512 channels), residual blocks, instance norm, LeakyReLU
- Strided conv downsampling, transposed conv upsampling, skip connections
- Deep supervision at each decoder level
- 24.9M parameters

**Visuals:**
- The U-Net architecture diagram (figures/unet3d_architecture.pdf)

**Script:**
> "We use a 3D U-Net following the nnU-Net design philosophy. Five encoder stages doubling from 32 to 512 channels, with residual connections, instance normalization, and LeakyReLU. Downsampling via strided convolutions — no max pooling. The decoder mirrors the encoder with transposed convolutions and skip connections. We attach deep supervision heads at each decoder level with decaying weights. Total: about 25 million parameters."
>
> "We chose the 3D U-Net specifically because it's the strongest controlled baseline — any performance difference can be attributed to the loss function, not the architecture."

---

## Slide 7: Loss Functions — How They Work (~1.5 minutes)

**Content:**
- Three loss formulations with intuitive explanations (not just equations)
- Dice+CE: measures overlap, treats all voxels equally
- clDice: computes Dice on the *skeletons* — penalizes broken centerlines
- Skeleton Recall: precomputes skeletons on CPU, upweights centerline voxels — "did you find the core of every vessel?"

**Visuals:**
- Conceptual diagram for each loss:
  - Dice+CE: prediction vs ground truth overlap
  - clDice: skeleton of prediction vs skeleton of GT (show the thin centerlines)
  - Skeleton Recall: GT skeleton highlighted, "is this covered by the prediction?"
- All combined with Dice+CE as base, topology term weighted at α=0.5

**Script:**
> "The baseline Dice plus cross-entropy measures voxel overlap. It works well for large structures but has no concept of connectivity."
>
> "Centerline Dice — clDice — takes a different approach. It computes skeletons of both the prediction and the ground truth using differentiable morphological operations, then measures Dice on those skeletons. This directly penalizes topological discontinuities — if your prediction's centerline doesn't match the ground truth's centerline, the loss goes up."
>
> "Skeleton Recall takes a more efficient approach. It precomputes the ground truth skeleton on CPU — a thin one-voxel-wide centerline — and asks: how much of this skeleton did the prediction cover? It's essentially a weighted cross-entropy that focuses optimization on the most topologically important voxels. Both topology terms are combined with the base Dice+CE loss at equal weight."

---

## Slide 8: Training Setup (~30 seconds)

**Content:**
- All models: 300 epochs, 8× A100 GPUs, DDP, identical config
- AdamW optimizer, LR 1e-3, cosine annealing with warmup
- 128³ patches, batch size 4/GPU, mixed precision
- TopBrain fine-tuning: 150 epochs, LR 1e-4, fresh optimizer

**Visuals:**
- Clean summary table of training hyperparameters
- Hardware spec: AWS p4d.24xlarge, 8× A100-SXM4-40GB

**Script:**
> "All three models trained for 300 epochs on 8 A100 GPUs using distributed data parallel — identical hardware, identical config. AdamW optimizer, cosine annealing learning rate schedule with warmup. 128-cubed patches, batch size 4 per GPU, mixed precision. For TopBrain fine-tuning, we load the best checkpoint with a fresh optimizer and train 150 more epochs at a lower learning rate."

---

## Slide 9: Learning Curves (~30 seconds)

**Content:**
- Training loss and validation Dice over 300 epochs for all three models

**Visuals:**
- The overlaid learning curves figure (figures/learning_curves.pdf)

**Script:**
> "Here are the training dynamics for all three models. The top panel shows smoothed training loss, the bottom shows validation Dice. All three converge to similar regions — training loss around 0.10 to 0.13, validation Dice around 0.84 to 0.86. The clDice and baseline curves are nearly identical, while Skeleton Recall runs at slightly higher loss throughout — that's expected since it has an additional penalty on centerline voxels. Convergence is stable across all configurations with no numerical issues."

---

## Slide 10: Global Results — From-Scratch (~1 minute)

**Content:**
- Table I from the report: DSC, clDice, HD95, Betti-0 for all three models
- Key finding: clDice is the best overall model, not the baseline

**Visuals:**
- The global comparison bar chart (figures/global_comparison.pdf) or the table
- Bold/highlight clDice as best DSC, best clDice metric, best B0

**Script:**
> "Here are the global metrics on the TopCoW validation set. The clDice model achieves the best aggregate performance — 0.864 DSC, 0.916 centerline Dice, and critically the lowest Betti-0 error at 3.72, meaning the fewest connected component mismatches. The baseline is close behind at 0.858 DSC."
>
> "Skeleton Recall has lower aggregate metrics at 0.845 DSC, and notably a high Betti-0 error of 26.84 — it's over-segmenting, producing fragments. But as we'll see next, it's doing something the others aren't on the thin vessels."

---

## Slide 11: Stratified Results — The Vessel Gap (~1.5 minutes)

**Content:**
- Per-vessel DSC breakdown showing the large vs communicating gap
- Skeleton Recall's targeted improvement on Pcom arteries
- The gap table: Skeleton has smallest gap at 0.332

**Visuals:**
- The stratified vessels bar chart (figures/stratified_vessels.pdf)
- The vessel gap chart (figures/vessel_gap.pdf) — side by side large vs communicating
- Or Table II from the report showing all three models per-vessel

**Script:**
> "This is where the story gets interesting. When we break down performance by vessel class, there's a dramatic gap. Large arteries — ICA, MCA, basilar — all above 0.79 DSC across all models. But the thin communicating arteries — Acom, Pcom — drop to 0.41 to 0.51."
>
> "Look at the Pcom arteries specifically — the thinnest communicating vessels. Skeleton Recall achieves 0.511 on right Pcom and 0.479 on left Pcom, compared to 0.459 and 0.410 for the baseline. That's a 5 to 7 point improvement targeted at exactly the vessels our hypothesis predicted."
>
> "The gap table shows Skeleton Recall has the smallest large-versus-communicating gap at 0.332, compared to 0.363 for both other models. So our hypothesis is partially supported — Skeleton Recall does disproportionately improve the thinnest communicating arteries, though the effect is modest at the CoW scale."

---

## Slide 12: TopBrain Fine-Tuning — The Big Result (~1.5 minutes)

**Content:**
- TopBrain expands coverage from 13 to 40 vessel classes
- Skeleton Recall dominates on thin structures after fine-tuning
- The key numbers: 0.589 vs 0.000 on small arteries

**Visuals:**
- The TopBrain grouped bar chart (figures/topbrain_comparison.pdf) — this is the most impactful figure
- Highlight the small arteries bar dramatically

**Script:**
> "The most striking results come from TopBrain fine-tuning. When we fine-tune all three models on the 40-class TopBrain labels, they all gain the ability to segment distal branches, posterior fossa vessels, and venous sinuses — structures they'd never seen during from-scratch training."
>
> "But look at the loss function's downstream effect. On large CoW arteries, all three are comparable around 0.82 — the loss doesn't matter much for big vessels. On communicating arteries, Skeleton Recall leads at 0.513. But on small arteries — ophthalmic and anterior choroidal — Skeleton Recall achieves 0.589 DSC while both other models score zero. Zero. They literally cannot find these vessels."
>
> "This is a qualitative difference, not just quantitative. The Skeleton Recall model's centerline-focused pretraining created representations that transfer to thin structures that the other losses' representations simply cannot capture. The same pattern holds for posterior fossa vessels — 0.626 versus 0.565 — and venous sinuses — 0.613 versus 0.416."

---

## Slide 13: Qualitative Comparison — 3D Renderings (~1 minute)

**Content:**
- Side-by-side SurgicalAR screenshots showing Dice+CE+TopBrain vs Skeleton+TopBrain
- Same patient, same camera angle

**Visuals:**
- [YOUR SURGICALAR SCREENSHOTS HERE]
- Left: Dice+CE + TopBrain fine-tuned — good CoW coverage but missing small vessels
- Right: Skeleton + TopBrain fine-tuned — dramatically more comprehensive coverage

**Script:**
> "Here's what this looks like in practice. These are 3D renderings from the same patient, same viewing angle, in our clinical visualization software. On the left is the Dice+CE model fine-tuned on TopBrain. It captures the major arteries well, but notice [point out missing vessels]. On the right is the Skeleton Recall model — you can see [point out additional vessels, small arteries, posterior fossa coverage]. This is the 0.589 versus zero we saw in the numbers, visualized."

---

## Slide 14: Discussion — What We Learned (~1 minute)

**Content:**
- Hypothesis partially supported
- clDice = best aggregate model (improves topology precision)
- Skeleton Recall = best for thin vessels (improves topology recall)
- The trade-off: Skeleton Recall over-segments (high Betti-0) but finds more
- TopBrain fine-tuning is the primary driver of expanded coverage; loss function shapes transfer quality

**Visuals:**
- Summary comparison table or bullet points
- Maybe a 2×2 matrix: {clDice, Skeleton} × {aggregate metrics, thin vessel metrics}

**Script:**
> "So what did we learn? Our hypothesis — that topology-aware losses improve thin communicating artery segmentation — is partially supported. The effect on CoW-scale communicating arteries is modest: a 3 to 7 point DSC improvement from Skeleton Recall. But when we test transfer to truly thin structures via TopBrain, the effect is dramatic."
>
> "The two topology-aware losses serve different purposes. clDice is the best overall model — it improves both overlap and topological precision, with the lowest Betti-0 error. Skeleton Recall sacrifices aggregate performance and over-segments, but it finds vessels that nothing else can. It's a precision versus recall trade-off at the topological level."
>
> "And the key insight: TopBrain fine-tuning drives comprehensive vessel coverage, but the loss function used during pretraining shapes how well that coverage transfers to the thinnest structures."

---

## Slide 15: Limitations & Future Work (~30 seconds)

**Content:**
- Large-small vessel gap persists (>0.33 DSC) across all models
- Skeleton Recall's high Betti-0 error — over-fragmentation needs post-processing
- HU window [0,600] clips bone to same range as vessels — causes false positives
- Future: wider HU window, class-balanced sampling, connected component filtering, retraining on corrected labels from model-assisted annotation pipeline

**Visuals:**
- Bullet list, clean and simple

**Script:**
> "Some limitations. The large-small vessel gap above 0.33 DSC persists across all models — communicating arteries remain fundamentally hard to segment. Skeleton Recall's high Betti-0 error means it produces fragmented predictions that need post-processing. And our HU window clips bone into the same intensity range as vessels, causing false positives."
>
> "Future work includes widening the HU window to separate bone from vessels, class-balanced sampling strategies, and retraining on corrected labels from our model-assisted annotation pipeline — we've already uploaded pre-labeled masks for 50 unlabeled CTA volumes to an annotation platform for correction."

---

## Slide 16: Conclusion (~30 seconds)

**Content:**
- Controlled three-way comparison on identical hardware
- clDice: best aggregate metrics
- Skeleton Recall: best thin-vessel transfer
- Topology-aware pretraining creates better representations for thin tubular structures
- The gap remains — communicating arteries are still the hardest problem in cerebrovascular segmentation

**Visuals:**
- 3-4 key takeaway bullets, large font

**Script:**
> "To conclude: we ran a controlled ablation comparing three loss functions for cerebral CTA vessel segmentation on identical hardware. clDice provides the best aggregate metrics. Skeleton Recall provides the best thin-vessel transfer, achieving 0.589 DSC on small arteries where both alternatives score zero. Topology-aware pretraining creates representations that generalize better to thin tubular structures."
>
> "The large-versus-small vessel gap remains the central challenge in cerebrovascular segmentation — the arteries that matter most clinically are the hardest to find. Thank you."

---

## Slide 17: Thank You / Questions

**Content:**
- "Thank you" or "Questions?"
- Contact info (optional)
- Key references (optional)

---

## Timing Summary

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:15 |
| 2 | Clinical Motivation | 1:00 |
| 3 | The Problem | 1:00 |
| 4 | Hypothesis & Approach | 0:45 |
| 5 | Datasets | 1:00 |
| 6 | Architecture | 0:45 |
| 7 | Loss Functions | 1:30 |
| 8 | Training Setup | 0:30 |
| 9 | Learning Curves | 0:30 |
| 10 | Global Results | 1:00 |
| 11 | Stratified Results | 1:30 |
| 12 | TopBrain Fine-Tuning | 1:30 |
| 13 | Qualitative (Screenshots) | 1:00 |
| 14 | Discussion | 1:00 |
| 15 | Limitations & Future | 0:30 |
| 16 | Conclusion | 0:30 |
| 17 | Thank You | — |
| **Total** | | **~13:15** |

## Figures to Include in Slides

All generated and available in `report/figures/`:
- `unet3d_architecture.pdf` — Slide 6
- `learning_curves.pdf` — Slide 9
- `global_comparison.pdf` — Slide 10
- `stratified_vessels.pdf` — Slide 11
- `vessel_gap.pdf` — Slide 11
- `topbrain_comparison.pdf` — Slide 12
- `per_case_boxplot.pdf` — Slide 10 (optional, shows per-case variance)

**You need to provide:**
- Circle of Willis anatomy diagram — Slide 2 (find a public-domain one or use a textbook figure)
- SurgicalAR screenshots — Slide 13 (Dice+CE+TB vs Skeleton+TB, same patient, same angle)
