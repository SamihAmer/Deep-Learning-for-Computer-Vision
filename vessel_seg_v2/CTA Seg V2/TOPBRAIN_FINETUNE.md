# Fine-Tuning with TopBrain 2025

Fine-tune the existing Dice+CE vessel segmentation model on the 25 TopBrain-labeled CTA volumes, which have **40-class annotations** covering significantly more of the cerebral vasculature than the original TopCoW 13-class labels.

## Background

The baseline model was trained on 125 TopCoW CTA volumes where only 13 Circle of Willis arteries are labeled. TopBrain extends this to 40 vessel classes (distal branches, cerebellar arteries, vertebral arteries, venous sinuses, etc.), resulting in **6–11x more labeled vessel voxels** per volume. However, only 25 of the 125 patients have TopBrain labels.

Training on all 125 volumes together is suboptimal — in the 100 TopCoW-only volumes, vessels like M2/M3 and cerebellar arteries are visible in the CTA image but labeled as background, which creates conflicting supervision. Fine-tuning on only the 25 TopBrain volumes avoids this: the model starts with strong CTA features from pretraining and refines on volumes where every visible vessel is correctly labeled.

## TopBrain Label Classes (CTA)

| ID | Vessel | Group |
|----|--------|-------|
| 1 | BA (Basilar artery) | CoW |
| 2 | R-P1P2 | CoW |
| 3 | L-P1P2 | CoW |
| 4 | R-ICA | CoW |
| 5 | R-M1 | CoW |
| 6 | L-ICA | CoW |
| 7 | L-M1 | CoW |
| 8 | R-Pcom | CoW |
| 9 | L-Pcom | CoW |
| 10 | Acom | CoW |
| 11 | R-A1A2 | CoW |
| 12 | L-A1A2 | CoW |
| 13 | R-A3 | Distal |
| 14 | L-A3 | Distal |
| 15 | 3rd-A2 | CoW |
| 16 | 3rd-A3 | Distal |
| 17–18 | R-M2, R-M3 | Distal |
| 19–20 | L-M2, L-M3 | Distal |
| 21–22 | R-P3P4, L-P3P4 | Distal |
| 23–24 | R-VA, L-VA | Posterior fossa |
| 25–26 | R-SCA, L-SCA | Posterior fossa |
| 27–28 | R-AICA, L-AICA | Posterior fossa |
| 29–30 | R-PICA, L-PICA | Posterior fossa |
| 31–32 | R-AChA, L-AChA | Small arteries |
| 33–34 | R-OA, L-OA | Small arteries |
| 35 | VoG (Vein of Galen) | Venous |
| 36 | StS (Straight sinus) | Venous |
| 37 | ICVs (Internal cerebral veins) | Venous |
| 38–39 | R-BVR, L-BVR | Venous |
| 40 | SSS (Superior sagittal sinus) | Venous |

Training is binary — all labels are collapsed to vessel/background via `label > 0`.

## Prerequisites

- Trained baseline model checkpoint: `runs/dice_ce_20260325_080955/best_model.pth`
- TopBrain labels already merged into `data/TopCoW2024_Data_Release/labelsTr/` (original TopCoW labels backed up in `labelsTr_topcow_backup/`)
- `conda activate vessel_seg`

## Steps

### 1. Verify the TopBrain labels are in place

The 25 TopBrain CTA label files should already be in `data/TopCoW2024_Data_Release/labelsTr/`. Verify by checking that patient 001 has more than 13 label classes:

```bash
python -c "
import nibabel as nib
import numpy as np
lbl = nib.load('data/TopCoW2024_Data_Release/labelsTr/topcow_ct_001.nii.gz').get_fdata()
unique = sorted(np.unique(lbl).astype(int))
print(f'Unique labels: {unique}')
print(f'Vessel voxels: {(lbl > 0).sum():,}')
assert max(unique) > 15, 'TopBrain labels not merged — max label should be >15'
print('TopBrain labels verified.')
"
```

If this fails (max label <= 15), the TopBrain labels need to be merged. See the backup/merge steps below.

### 2. Run fine-tuning

```bash
python train.py \
    --data_dir data/TopCoW2024_Data_Release \
    --finetune runs/dice_ce_20260325_080955/best_model.pth \
    --include_patients 001,002,003,004,005,006,007,008,010,011,012,013,014,015,016,017,018,020,021,022,023,024,025,026,027 \
    --lr 1e-4 \
    --epochs 100 \
    --patches_per_volume 16 \
    --foreground_ratio 0.5 \
    --loss dice_ce \
    --val_interval 10
```

**Parameter choices:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `--finetune` | best_model.pth | Loads pretrained weights, fresh optimizer (no LR/scheduler state carried over) |
| `--include_patients` | 25 TopBrain IDs | Trains only on volumes with 40-class labels — avoids conflicting supervision |
| `--lr 1e-4` | 10x lower than pretraining | Preserves learned features while adapting to new vessels |
| `--epochs 100` | Shorter than pretraining (300) | Warm start converges faster |
| `--patches_per_volume 16` | 4x default (4) | Compensates for fewer volumes (25 vs 100) to maintain epoch size |
| `--foreground_ratio 0.5` | Up from default 0.33 | TopBrain labels have far more vessel voxels — sample them more |
| `--val_interval 10` | More frequent than default (25) | Shorter training needs tighter monitoring |

**Expected runtime:** ~8–10 hours on a 4070 Ti (12 GB VRAM) with batch_size=2.

For faster iteration, try `--patches_per_volume 8` (~4–5 hours) at the cost of less diversity per epoch.

### 3. Evaluate

```bash
python evaluate.py \
    --checkpoint runs/<finetune_run>/best_model.pth \
    --data_dir data/TopCoW2024_Data_Release
```

The updated `stratified_eval.py` now reports per-vessel metrics for all 40 TopBrain classes grouped into:
- **Large CoW arteries** (BA, ICA, M1, PCA, A1/A2)
- **Communicating arteries** (Pcom, Acom, 3rd-A2)
- **Distal branches** (A3, M2, M3, P3/P4)
- **Posterior fossa** (VA, SCA, AICA, PICA)
- **Small arteries** (AChA, OA)
- **Venous sinuses** (VoG, StS, ICVs, BVR, SSS)

### 4. Export to ONNX

```bash
python export_onnx.py \
    --checkpoint runs/<finetune_run>/best_model.pth \
    --output vessel_seg_topbrain.onnx
```

The ONNX export and SurgicalAR integration are unchanged — the model architecture is identical, only the weights differ.

### 5. Deploy to SurgicalAR

Copy the new ONNX model to the SurgicalAR models directory:

```
SurgicalAR/Assets/StreamingAssets/MachineLearning/ConfigurableSegmentationPipelineCore/Models/CerebralVesselSegmentation/dice_ce/model.onnx
```

No configuration changes needed — the pipeline configs (Models.json, Pipelines.json, etc.) remain the same.

## Restoring Original TopCoW Labels

If you need to revert to the original TopCoW labels:

```bash
cp data/TopCoW2024_Data_Release/labelsTr_topcow_backup/topcow_ct_*.nii.gz \
   data/TopCoW2024_Data_Release/labelsTr/
```

## Merging TopBrain Labels (if not already done)

If starting fresh, download and merge TopBrain labels:

```bash
# Download from Zenodo
curl -L -o data/TopBrain_Data_Release.zip \
    "https://zenodo.org/records/16878417/files/TopBrain_Data_Release_Batches1n2_081425.zip?download=1"

# Back up original labels
mkdir -p data/TopCoW2024_Data_Release/labelsTr_topcow_backup
for f in 001 002 003 004 005 006 007 008 010 011 012 013 014 015 016 017 018 020 021 022 023 024 025 026 027; do
    cp "data/TopCoW2024_Data_Release/labelsTr/topcow_ct_${f}.nii.gz" \
       "data/TopCoW2024_Data_Release/labelsTr_topcow_backup/"
done

# Extract TopBrain CTA labels over TopCoW labels
unzip -o -j data/TopBrain_Data_Release.zip \
    "TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct/*.nii.gz" \
    -d "data/TopCoW2024_Data_Release/labelsTr/"
```

## References

- TopBrain 2025 Challenge: https://topbrain2025.grand-challenge.org
- TopBrain Zenodo: https://zenodo.org/records/16878417
- TopCoW pre-print: https://arxiv.org/abs/2312.17670
