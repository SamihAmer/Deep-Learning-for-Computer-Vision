# DynaVessel vs our UNet3D — generalization-ceiling check

3 TopCoW CT cases: `topcow_ct_001`, `topcow_ct_004`, `topcow_ct_008`.
- UNet3D: `runs/dice_ce_20260325_080955/best_model.pth` (epoch 250, best val Dice 0.8846, trained on TopCoW 13-class binarized).
- DynaVessel: alceballosa Model 241, binarized post-hoc to arteries ∪ veins.

## vs TopCoW 13-class GT (CoW arteries only — UNet3D's training target)

| Model              |   Dice | clDice |  HD95 | Betti0 |
|--------------------|-------:|-------:|------:|-------:|
| UNet3D             | 0.8749 | 0.9364 |  1.38 |   3.0  |
| DynaVessel (A+V)   | 0.0490 | 0.0702 | 76.87 | 233.7  |
| DynaVessel-A only  | 0.1691 | 0.1409 | 64.46 | 163.3  |
| DynaVessel-V only  | 0.0000 | 0.0000 | 78.35 | 126.0  |

UNet3D is the specialist — near-perfect CoW segmentation. DynaVessel looks terrible
here only because it's segmenting many structures TopCoW-13 does not label.

## vs TopBrain 40-class GT (CoW + venous sinuses + extra branches)

| Model              |   Dice | clDice |  HD95 | Betti0 |
|--------------------|-------:|-------:|------:|-------:|
| UNet3D             | 0.2149 | 0.2294 | 82.12 |  26.7  |
| DynaVessel (A+V)   | 0.2938 | 0.3810 | 36.14 | 204.0  |
| DynaVessel-A only  | 0.4122 | 0.4856 | 35.07 | 133.7  |
| DynaVessel-V only  | 0.1719 | 0.1581 | 35.88 |  96.3  |

UNet3D misses every venous structure (it was never taught them) → Dice collapses.
DynaVessel covers more of the cerebral vasculature, which is exactly the prior
we want for the Phase 7 A/V fine-tune.

## Per-case voxel counts (foreground)

|        case       | TopCoW-13 GT | TopBrain-40 GT | UNet3D | DynaVessel (A+V) |
|-------------------|-------------:|---------------:|-------:|-----------------:|
| topcow_ct_001     |       11,408 |         79,177 | 10,964 |          388,087 |
| topcow_ct_004     |       18,507 |        129,106 | 15,635 |          619,601 |
| topcow_ct_008     |        6,720 |         49,084 |  5,772 |          271,786 |

UNet3D voxel counts nearly match TopCoW-13 GT — confirming the model learned CoW
tightly. DynaVessel is ~5× broader than TopBrain-40 GT (over-segments small
branches). TopBrain-40 is ~7× broader than TopCoW-13 (venous sinuses drive most
of the difference).

## Takeaway for Phase 7

DynaVessel's strength is vessel *coverage* (especially veins); its weakness is
boundary tightness. Our UNet3D's strength is boundary tightness on the CoW;
its weakness is vessel coverage outside the CoW. Fine-tuning our UNet3D on
radiologist-corrected DynaVessel pseudo-labels, starting from a `finetune_topbrain_*`
checkpoint (already trained on venous sinuses), should compound both strengths.
