# Base Model Results — Dice+CE (dice_ce_20260331_222218)

## Training Summary

| | |
|---|---|
| Loss | Dice + Cross-Entropy |
| Epochs | 300 |
| Best checkpoint | Epoch 275 |
| Batch size | 4 |
| Patch size | 128³ |
| Scheduler | Cosine annealing + 10 epoch warmup |
| LR | 1e-3 → 1e-7 |
| Peak VRAM | ~10 GB (A10G 24 GB) |
| Epoch time | ~78 s |
| ONNX export | `vessel_seg_best_e275_dsc0871.onnx` |

---

## Validation Metrics (Epoch 300 — Training Log)

| Metric | Value |
|--------|-------|
| DSC | 0.8704 |
| clDice | 0.9254 |
| HD95 | 5.75 mm |
| Betti-0 error | 3.48 |

---

## Full Evaluation Results (`evaluate.py` — 25 CT val cases)

### Global (binary segmentation)

| Metric | Mean | Std |
|--------|------|-----|
| DSC | **0.8713** | ±0.0520 |
| clDice | **0.9209** | ±0.0403 |
| HD95 | 5.76 mm | ±19.78 mm |
| Betti-0 error | 4.24 | ±3.60 |

> Note: HD95 std is inflated by a small number of failure cases (see per-case table below). Cases `ct_005` (DSC 0.666) and `ct_055` (clDice 0.759) are clear outliers.

### Per-Case

| Case | DSC | clDice | Time |
|------|-----|--------|------|
| topcow_ct_072 | 0.8799 | 0.8875 | 37s |
| topcow_ct_148 | 0.8826 | 0.9374 | 32s |
| topcow_ct_078 | 0.8563 | 0.9262 | 37s |
| topcow_ct_065 | 0.9135 | 0.9263 | 37s |
| topcow_ct_030 | 0.8596 | 0.9137 | 36s |
| topcow_ct_028 | 0.8708 | 0.9156 | 33s |
| topcow_ct_152 | 0.9005 | 0.9603 | 33s |
| topcow_ct_163 | 0.8826 | 0.9165 | 27s |
| topcow_ct_005 | **0.6662** | 0.8989 | 42s |
| topcow_ct_055 | 0.8216 | **0.7586** | 39s |
| topcow_ct_076 | 0.8756 | 0.9240 | 49s |
| topcow_ct_012 | 0.9352 | 0.9498 | 50s |
| topcow_ct_070 | 0.8375 | 0.9311 | 45s |
| topcow_ct_157 | 0.9162 | 0.9500 | 60s |
| topcow_ct_087 | 0.8528 | 0.9259 | 47s |
| topcow_ct_014 | 0.8255 | 0.9307 | 31s |
| topcow_ct_162 | 0.8763 | 0.9732 | 30s |
| topcow_ct_018 | 0.9129 | 0.9444 | 38s |
| topcow_ct_029 | 0.8549 | 0.9273 | 45s |
| topcow_ct_032 | 0.8884 | 0.9600 | 47s |
| topcow_ct_036 | 0.8783 | 0.8910 | 59s |
| topcow_ct_135 | 0.9289 | 0.9579 | 54s |
| topcow_ct_004 | 0.8417 | 0.8800 | 54s |
| topcow_ct_015 | 0.9247 | 0.9199 | 44s |
| topcow_ct_082 | 0.8986 | 0.9159 | 54s |

### Stratified by Vessel Class

| Vessel | DSC | clDice | Betti-0 err | N |
|--------|-----|--------|-------------|---|
| BA | 0.8209 | 0.8603 | 0.4 | 25 |
| R-PCA | 0.8151 | 0.9183 | 0.6 | 25 |
| L-PCA | 0.8121 | 0.9086 | 1.3 | 25 |
| R-ICA | 0.8575 | 0.8982 | 0.2 | 25 |
| L-ICA | 0.8141 | 0.9162 | 0.0 | 25 |
| R-MCA | 0.8507 | 0.9009 | 0.5 | 25 |
| L-MCA | 0.8133 | 0.9131 | 0.0 | 25 |
| R-ACA | **0.4896** | 0.7123 | 0.5 | 15 |
| L-ACA | **0.4619** | 0.7033 | 1.1 | 15 |
| R-Pcom | **0.4167** | **0.5272** | 0.1 | 21 |
| L-Pcom | 0.7287 | 0.8044 | 1.9 | 25 |
| Acom | 0.7398 | 0.8403 | 2.3 | 25 |
| 3rd-A2 | n/a | n/a | n/a | 0 |

### Large vs Small Vessel Gap

| Group | DSC | clDice |
|-------|-----|--------|
| Large (BA, ICA, MCA, PCA, ACA) | 0.7749 | 0.8738 |
| Small (Acom, Pcom, 3rd-A2) | 0.6403 | 0.7350 |
| **Gap** | **0.135** | **0.139** |

---

## SurgicalAR Integration Results

Exported ONNX was tested in SurgicalAR. Subjective result: **the model missed a significant number of vessels in clinical use**, despite the validation DSC looking reasonable on paper.

### Likely Contributing Factors

- **Small vessel failure is real.** R-ACA (0.49), L-ACA (0.46), and R-Pcom (0.42) DSC are poor. In a Circle of Willis segmentation these are visible structures — the model is effectively not finding them consistently.
- **Binary DSC is misleading.** The 0.87 global DSC is dominated by large vessels (BA, ICA, MCA) which are easy to segment. Small communicating arteries that complete the CoW are underweighted in the overall score but very visible when missing clinically.
- **Topology not enforced.** Dice+CE has no connectivity loss — broken or fragmented vessels can score reasonably on Dice but look completely wrong anatomically. The Betti-0 error of 4.24 confirms real connectivity issues.
- **No TTA.** Test-time augmentation is disabled (`tta: false`). Enabling it may improve boundary precision on smaller structures.

### Next Steps

- Train `dice_ce_cldice` and `dice_ce_skeleton` variants — both topology losses specifically target small vessel connectivity and are expected to close the large/small gap.
- Compare stratified results across all three loss configurations before making another SurgicalAR export.
- Consider qualitative review of the two outlier cases (`ct_005`, `ct_055`) to understand failure mode.
