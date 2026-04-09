# Presentation Script V2 — Deep Learning for Vessel Segmentation in Cerebral CTA

**Total time target: 14–15 minutes** (within 15-minute max)

---

## Slide 1: Title Slide (~15 seconds)

**Script:**
> "Hi, my name is Samih Amer. Today I'll be presenting my midterm project on deep learning for vessel segmentation in cerebral CT angiography — specifically, a controlled ablation study comparing topology-aware loss functions for improving segmentation of the thinnest, most clinically important cerebral arteries."

---

## Slide 2: Clinical Motivation (~1 minute)

**Key points:**
- What is CTA vessel segmentation and why it matters
- Circle of Willis anatomy — the arterial ring at the base of the brain
- Clinical applications: stroke planning, aneurysm detection, surgical navigation
- The problem: communicating arteries are thin but clinically critical

**Script:**
> "Cerebrovascular diseases — stroke, aneurysms, arteriovenous malformations — are a leading cause of death and disability worldwide. Accurate segmentation of cerebral vasculature from CT angiography is essential for diagnosis and surgical planning."
>
> "The Circle of Willis is the arterial anastomotic ring at the base of the brain. It consists of large named arteries — the internal carotid arteries, middle cerebral arteries, and basilar artery — these are relatively large, several millimeters in diameter, and easy to see on imaging. But connecting them are thin communicating arteries — the anterior communicating artery, or Acom, and the posterior communicating arteries, the Pcoms. These can be under a millimeter in diameter, but they're the ones that matter most clinically. They provide collateral blood flow when a major vessel is blocked — for example during a stroke. If your segmentation misses or disconnects these vessels, the clinical utility is severely compromised."

---

## Slide 3: The Problem — Topological Failures (~1 minute)

**Key points:**
- Standard Dice+CE loss treats all voxels equally — no connectivity incentive
- A segmentation can have high Dice but broken topology
- Aggregate metrics hide failures on thin structures

**Script:**
> "The standard approach for training segmentation networks uses Dice loss plus cross-entropy. These objectives treat every voxel independently and equally — they have no concept of connectivity or topology. So a model can achieve a global Dice score of 0.87, which looks great on paper, while completely severing communicating arteries or producing disconnected vessel fragments."
>
> "When we tested our baseline model in a clinical 3D viewer, we found exactly this: the global Dice was reasonable, but when we looked at individual vessel classes, the right posterior communicating artery had a Dice of only 0.42, and other communicating arteries were similarly low. The aggregate metric was dominated by the large, easy-to-segment arteries and was hiding catastrophic failures on the thin structures that actually matter for clinical decision-making."

---

## Slide 4: The TopCoW Challenge (~1 minute)

**Key points:**
- MICCAI 2024 challenge for Circle of Willis segmentation
- 125 CTA + 125 MRA volumes, expert annotations for 13 vessel classes
- Evaluation protocol: DSC, clDice, HD95, Betti-0 error (connected components)
- Challenge findings: topology remains the unsolved problem

**Script:**
> "Our work builds on the TopCoW 2024 Challenge — a MICCAI challenge specifically designed to benchmark Circle of Willis segmentation. The challenge provides 125 CTA and 125 MRA volumes with expert voxel-level annotations for 13 vessel classes. What makes TopCoW unique among segmentation challenges is its evaluation protocol: beyond standard Dice and Hausdorff distance, it measures topological correctness using the Betti-0 error — which counts connected component mismatches between prediction and ground truth — and centerline Dice, which evaluates skeleton-level overlap."
>
> "The key finding from the TopCoW challenge is that even top-performing teams achieve good aggregate Dice scores while still producing topologically incorrect segmentations. The communicating arteries, which are the most clinically important structures in the Circle of Willis, remain the weakest point across all submitted methods. This motivated our study: can we specifically improve topology through the loss function alone?"

---

## Slide 5: Related Work & State of the Art (~1 minute)

**Key points:**
- nnU-Net as the dominant framework (Isensee et al., Nature Methods 2021)
- Topology-preserving losses: clDice (CVPR 2021), Skeleton Recall (ECCV 2024)
- Other approaches: CBD-Loss (MICCAI 2024), persistent homology, NexToU
- Gap in literature: losses evaluated alongside architecture changes or only via aggregate challenge scores

**Script:**
> "The backbone of modern medical image segmentation is nnU-Net by Isensee et al., published in Nature Methods 2021. It established the Dice plus cross-entropy paradigm that nearly all top methods build on. In 2021, Shit et al. at CVPR introduced centerline Dice — clDice — which computes Dice on morphological skeletons using differentiable soft-skeletonization. This was the first loss function to directly penalize topological disconnection in a differentiable way."
>
> "More recently, Kirchhoff et al. at ECCV 2024 proposed Skeleton Recall, which takes a more computationally efficient approach — precomputing binary skeletons on CPU and using them as attention masks. Other related work includes the Centerline Boundary Dice loss from Shi et al. at MICCAI 2024, methods based on persistent homology from Berger and Stucki, and specialized architectures like NexToU."
>
> "The key limitation we identified is that existing evaluations of these topology-aware losses either introduce them alongside architectural changes — making it impossible to attribute improvements to the loss function alone — or they report only aggregate scores on multi-team challenge leaderboards. Our work addresses this gap through a controlled, single-variable ablation."

---

## Slide 6: Hypothesis & Approach (~45 seconds)

**Key points:**
- Hypothesis: topology-aware losses disproportionately improve thin communicating arteries
- Controlled ablation: same architecture, data, hardware — only loss changes
- Three configurations

**Script:**
> "Our hypothesis is that topology-aware loss functions will disproportionately improve segmentation of thin communicating arteries — the Acom and Pcom — which are most prone to topological failure."
>
> "To test this cleanly, we designed a controlled ablation. We hold everything identical — the same 3D U-Net architecture, the same data splits, the same hardware, the same optimizer and hyperparameters — and vary only the loss function. Three configurations: first, Dice plus cross-entropy as the baseline; second, adding soft centerline Dice from Shit et al.; and third, adding Skeleton Recall from Kirchhoff et al. Both topology-aware terms are combined with the base loss at equal weight, alpha equals 0.5."

---

## Slide 7: Datasets (~1 minute)

**Key points:**
- TopCoW 2024: 125 CTA volumes, 13 CoW classes, 80/20 split
- TopBrain 2025: Same 25 CT scans, 40 classes (3x coverage)
- TopBrain as transfer evaluation: tests whether pretraining transfers to thin structures

**Script:**
> "We use two datasets. TopCoW 2024 provides 125 CTA volumes with expert annotations for 13 Circle of Willis vessel classes. We use the CT subset only to avoid modality confounding, with an 80/20 random split — 100 training, 25 validation."
>
> "TopBrain 2025 provides the same 25 CT scans from TopCoW but with dramatically richer labels — 40 vessel classes covering not just the Circle of Willis but also distal branches like the A3, M2, and M3 segments, posterior fossa vessels like the vertebral arteries, SCA, AICA, and PICA, small arteries like the ophthalmic and anterior choroidal, and venous sinuses. When binarized, TopBrain ground truth masks cover substantially more vasculature. We use this for fine-tuning to test a critical question: does the loss function used during pretraining affect how well the learned representations transfer to thin structures the model has never seen?"

---

## Slide 8: Architecture — 3D U-Net (~1 minute)

**Key points:**
- 3D U-Net following nnU-Net design philosophy
- Encoder: 5 stages, 32→512 channels, residual blocks
- Decoder: transposed convolutions, skip connections, deep supervision
- Mathematical detail on the core operations

**Script:**
> "We use a 3D U-Net following the nnU-Net design philosophy. The encoder has 5 stages, starting from 32 channels and doubling at each level up to 512 channels. Each stage consists of two 3-by-3-by-3 convolutional blocks with instance normalization, LeakyReLU activation with slope 0.01, and residual connections via 1-by-1 projection when channels change. Downsampling uses strided 2-by-2-by-2 convolutions — no max pooling — which is important because strided convolutions are learnable and don't discard information the way max pooling does."
>
> "The decoder mirrors this with transposed convolutions for upsampling and skip connections via concatenation. We attach deep supervision heads — 1-by-1 convolutions producing class predictions — at each decoder level, with exponentially decaying weights: 1.0, 0.5, 0.25, 0.125. Deep supervision provides gradient signal at multiple scales, which helps training converge and is particularly relevant for thin structures that may only be resolved at certain scales. The total model has approximately 24.9 million parameters, initialized with Kaiming normal initialization."

---

## Slide 9: Loss Functions — Dice + Cross-Entropy (~45 seconds)

**Key points:**
- Full Dice loss equation
- Cross-entropy formulation
- Why this is the baseline: nnU-Net default, no topology awareness

**Script:**
> "Let me walk through the mathematics of each loss function. The baseline combines soft Dice loss with cross-entropy. The Dice loss is defined as one minus the ratio of twice the intersection to the total volume of prediction and ground truth. Mathematically, for predicted probabilities p and ground truth g, that's one minus two times the sum of p-i times g-i plus epsilon, divided by the sum of p-i plus the sum of g-i plus epsilon, where epsilon is 1e-5 for numerical stability."
>
> "Cross-entropy provides the per-voxel classification signal. Together, Dice handles the region-level overlap while cross-entropy provides sharp voxel-level gradients. This is the nnU-Net default and our baseline — it works well for large, easily delineated structures, but it treats every voxel equally. A voxel on the surface of the internal carotid artery receives the same gradient signal as a voxel at the center of a 1-millimeter communicating artery."

---

## Slide 10: Loss Functions — Soft clDice (~1.5 minutes)

**Key points:**
- Differentiable soft-skeletonization via iterative morphological operations
- Topology precision and sensitivity
- clDice formula and combination with base loss

**Script:**
> "The centerline Dice, or clDice, takes a fundamentally different approach. Instead of measuring overlap on the full volume, it extracts the topological skeleton — the centerline — of both prediction and ground truth, then measures Dice on those skeletons."
>
> "The key innovation is differentiable soft-skeletonization. A morphological skeleton is traditionally a binary operation, but Shit et al. made it differentiable using iterative soft morphological operations. The soft skeleton is computed as: skeleton of x equals ReLU of x minus the morphological opening of x, plus the sum from k equals 1 to K of ReLU of the k-th erosion of x minus the opening of the k-th erosion. Erosion is implemented as 3D min-pooling, dilation as 3D max-pooling, and opening is erosion followed by dilation. We use K equals 10 iterations."
>
> "From these skeletons, we compute topology precision — how much of the predicted skeleton lies within the ground truth — and topology sensitivity — how much of the ground truth skeleton is covered by the prediction. The clDice is then the harmonic mean of these two, just like traditional F1-score but operating on skeletons. The combined loss is (1 minus alpha) times the base Dice+CE loss plus alpha times (1 minus clDice), with alpha equals 0.5."

---

## Slide 11: Loss Functions — Skeleton Recall (~1 minute)

**Key points:**
- Key difference: precomputes skeleton on CPU using morphological thinning
- Only measures recall (not precision) on the skeleton
- Computational efficiency advantage

**Script:**
> "Skeleton Recall takes a different and more computationally efficient approach. Instead of differentiable skeletonization on GPU, it precomputes the ground truth skeleton on CPU using classical morphological thinning from scikit-image — this produces a binary, one-voxel-wide centerline."
>
> "The loss then simply asks: how much of this precomputed skeleton is covered by the predicted probabilities? Mathematically, it's one minus the sum of the predicted probabilities times the skeleton mask, divided by the total number of skeleton voxels. This is essentially a recall-only metric on the skeleton — it does not measure precision. If the model predicts extra, non-skeleton voxels, this loss doesn't penalize that — only the base Dice+CE handles false positives."
>
> "This design choice has an important implication: Skeleton Recall focuses all its gradient signal on ensuring the network finds every centerline voxel, even at the cost of over-segmentation. The combined loss uses the same alpha equals 0.5 weighting as clDice."

---

## Slide 12: Training Setup (~30 seconds)

**Key points:**
- All models: 300 epochs, 8× A100 GPUs, DDP, identical config
- AdamW, cosine annealing, mixed precision
- TopBrain fine-tuning: 150 epochs, lower LR

**Script:**
> "All three models were trained for 300 epochs on 8 NVIDIA A100 GPUs using distributed data parallel — identical hardware, identical configuration. AdamW optimizer with learning rate 1e-3 and weight decay 1e-5, cosine annealing schedule with 10-epoch linear warmup, 128-cubed patches, batch size 4 per GPU, mixed precision. Validation every 25 epochs with early stopping patience of 5 validation cycles. For TopBrain fine-tuning, we load the best from-scratch checkpoint with a fresh optimizer and train 150 more epochs at learning rate 1e-4."

---

## Slide 13: Learning Curves (~30 seconds)

**Script:**
> "Here are the training dynamics for all three models over 300 epochs. The top panel shows smoothed training loss, the bottom shows validation Dice. All three converge to similar regions — training loss around 0.10 to 0.13, validation Dice around 0.84 to 0.86. clDice and Dice+CE curves are nearly identical, while Skeleton Recall runs at slightly higher loss — expected since it has the additional penalty on centerline voxels. Convergence is stable across all configurations."

---

## Slide 14: Global Results — From Scratch (~1 minute)

**Key points:**
- clDice is best overall: DSC 0.864, clDice 0.916, Betti-0 3.72
- Skeleton Recall lower aggregate but watch the next slide
- Betti-0 error of 26.84 indicates over-fragmentation

**Script:**
> "Now to the results. This shows global metrics on the TopCoW validation set for all three models. On the left, overlap and topology metrics — clDice achieves the best DSC at 0.864 and best centerline Dice at 0.916. In the middle, surface distances are comparable. On the right, the critical Betti-0 error — this measures connected component mismatches. clDice achieves 3.72, meaning very few topological errors. But Skeleton Recall has a Betti-0 error of 26.84 — it's producing many small disconnected fragments."
>
> "So at the aggregate level, clDice is the clear winner. But aggregate metrics are exactly what we argued can be misleading. Let's look at what happens when we stratify by vessel class."

---

## Slide 15: Stratified Results — The Vessel Gap (~1.5 minutes)

**Key points:**
- Per-vessel DSC breakdown: large arteries vs communicating
- Skeleton Recall's targeted improvement on Pcom
- Gap table: smallest gap at 0.332

**Script:**
> "This is where the story changes. On the left, we see the large-versus-communicating vessel gap for each model. Large arteries — ICA, MCA, basilar — are all above 0.79 DSC across all models. But communicating arteries drop to 0.41 to 0.51. Look at the gap annotations: Dice+CE and clDice both have a gap of 0.363, but Skeleton Recall reduces it to 0.332."
>
> "The per-vessel breakdown on the right tells us exactly where. Look at the Pcom arteries — the thinnest communicating vessels. Skeleton Recall achieves 0.511 on right Pcom and 0.479 on left Pcom, compared to 0.459 and 0.410 for the baseline. That's a 5 to 7 point improvement targeted precisely at the vessels our hypothesis predicted would benefit."
>
> "So our hypothesis is partially supported at the CoW scale: Skeleton Recall provides a modest but consistent improvement on the thinnest communicating arteries. But the real test comes next."

---

## Slide 16: TopBrain Fine-Tuning — The Big Result (~1.5 minutes)

**Key points:**
- TopBrain expands to 40 vessel classes
- Skeleton Recall dominates on thin structures
- 0.589 vs 0.000 on small arteries — qualitative difference

**Script:**
> "This is the most important figure in the presentation. When we fine-tune all three models on TopBrain's 40-class labels, they all gain the ability to segment distal branches, posterior fossa vessels, and venous sinuses — structures absent from the 13-class TopCoW training."
>
> "Look at the loss function's downstream effect. On large CoW arteries, all three are comparable around 0.82 — the loss function doesn't matter much for big vessels. On communicating arteries, Skeleton Recall leads at 0.513. On distal branches, Skeleton Recall at 0.752 versus 0.687 for the baseline."
>
> "But look at small arteries — the ophthalmic and anterior choroidal arteries. Skeleton Recall achieves 0.589 DSC. Both other models score zero. Zero. They literally cannot find these vessels, even after identical fine-tuning on the same data. This is a qualitative difference, not merely quantitative. Skeleton Recall's centerline-focused pretraining created feature representations that encode thin tubular geometry in a way that transfers when exposed to richer annotations. The same pattern holds for posterior fossa vessels — 0.626 versus 0.565 — and venous sinuses — 0.613 versus 0.416."

---

## Slide 17: Qualitative Comparison — 3D Renderings (~1 minute)

**Script:**
> "Here's what this looks like in practice. These are 3D renderings from the same patient, same viewing angle, in our clinical visualization platform. On the left is the Dice+CE baseline fine-tuned on TopBrain — it captures the major arteries well, but notice the missing small vessels and discontinuities. On the right is the Skeleton Recall model — you can see dramatically more comprehensive vessel coverage, particularly in the small arteries and posterior fossa. This is the 0.589 versus zero that we saw in the numbers, visualized in 3D."

---

## Slide 18: Discussion — What We Learned (~1 minute)

**Key points:**
- Hypothesis partially supported
- clDice = best aggregate (precision), Skeleton Recall = best thin vessel (recall)
- The trade-off: topological precision vs recall
- Key insight: pretraining loss shapes transfer quality

**Script:**
> "So what did we learn? Our hypothesis is partially supported. At the CoW scale, the improvement on communicating arteries is modest — 3 to 7 points DSC from Skeleton Recall. But when we test transfer to truly thin structures via TopBrain, the effect is dramatic."
>
> "The two topology-aware losses serve different purposes, and this maps to a precision-versus-recall trade-off at the topological level. clDice improves both overlap and topological precision — lowest Betti-0 error, fewest fragments. Skeleton Recall improves topological recall — it finds vessels that nothing else can, but it over-segments, producing fragments that need post-processing."
>
> "The key insight is that TopBrain fine-tuning drives comprehensive vessel coverage for all models, but the loss function used during pretraining shapes the quality of that transfer to the thinnest structures. Skeleton Recall's centerline-focused gradient signal during pretraining apparently shapes the network's feature hierarchy to better encode thin tubular geometry."

---

## Slide 19: Limitations & Future Work (~30 seconds)

**Script:**
> "Some important limitations. The large-small vessel gap above 0.33 DSC persists across all models — communicating arteries remain fundamentally hard. Skeleton Recall's high Betti-0 error of 26.84 means fragmented predictions that need connected-component post-processing. Our HU window of 0 to 600 clips bone into the same intensity range as contrast-enhanced vessels, causing false positives near the skull base."
>
> "Future work: widening the HU window to separate bone from vessels, class-balanced sampling strategies to focus on under-represented thin vessels, and retraining on corrected labels from our model-assisted annotation pipeline — we've already uploaded pre-labeled masks for 50 unlabeled CTA volumes to a RedBrick AI annotation platform for expert correction."

---

## Slide 20: Conclusion (~30 seconds)

**Script:**
> "To conclude: we ran a controlled three-way ablation comparing loss functions for cerebral CTA vessel segmentation on identical hardware. clDice provides the best aggregate metrics — highest DSC, lowest Betti-0 error. Skeleton Recall provides the best thin-vessel transfer — 0.589 DSC on small arteries where both alternatives score zero. Topology-aware pretraining creates representations that generalize better to thin tubular structures."
>
> "The large-versus-small vessel gap remains the central challenge in cerebrovascular segmentation. The arteries that matter most clinically are the hardest to find. Thank you."

---

## Slide 21: Thank You / Questions

---

## Timing Summary

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:15 |
| 2 | Clinical Motivation | 1:00 |
| 3 | The Problem | 1:00 |
| 4 | TopCoW Challenge (NEW) | 1:00 |
| 5 | Related Work (NEW) | 1:00 |
| 6 | Hypothesis & Approach | 0:45 |
| 7 | Datasets | 1:00 |
| 8 | Architecture — 3D U-Net | 1:00 |
| 9 | Loss: Dice+CE (NEW — split) | 0:45 |
| 10 | Loss: clDice (NEW — split) | 1:30 |
| 11 | Loss: Skeleton Recall (NEW — split) | 1:00 |
| 12 | Training Setup | 0:30 |
| 13 | Learning Curves | 0:30 |
| 14 | Global Results | 1:00 |
| 15 | Stratified Results | 1:30 |
| 16 | TopBrain Fine-Tuning | 1:30 |
| 17 | Qualitative (Screenshots) | 1:00 |
| 18 | Discussion | 1:00 |
| 19 | Limitations & Future | 0:30 |
| 20 | Conclusion | 0:30 |
| 21 | Thank You | — |
| **Total** | | **~17:15** |

*Note: The timing above is a maximum estimate. In practice, slides 4, 5, 8–11 can be trimmed by speaking slightly faster or cutting a sentence or two, bringing the total to ~15 minutes. Rehearse and cut as needed.*

## Figures to Include in Slides

All regenerated in `report/figures_v2/`:
- `learning_curves.png` — Slide 13
- `global_comparison.png` — Slide 14
- `stratified_vessels.png` — Slide 15
- `vessel_gap.png` — Slide 15
- `topbrain_comparison.png` — Slide 16
- `per_case_boxplot.png` — Slide 14 (optional)

**You need to provide:**
- Circle of Willis anatomy diagram — Slide 2
- SurgicalAR screenshots — Slide 17 (Dice+CE+TB vs Skeleton+TB, same patient)
