# Pseudo-Labeling Pipeline (Approach A)

Goal: use the alceballosa/robust-vessel-segmentation Model 241 to generate
artery/vein pseudo-labels on our 50 CTA volumes, have a radiologist correct
them in RedBrick AI, then fine-tune our 3D U-Net as a 3-class model
(background / artery / vein).

## Status (checkpoint)

- [x] Clone `alceballosa/robust-vessel-segmentation`
- [x] Install nnU-Net v2 + antspyx in the `vessel_seg` conda env
- [x] Extract CTA_11-20, 21-30, 31-50 zips to `D:/vessel_seg_v2/TopCoW_DICOM_staging`
- [x] DICOM → NIfTI conversion done — **51 files / 49 patients** in
      `D:/vessel_seg_v2/CTA_nifti/` (CTA_043 has no "Thins with contrast" series)
- [x] alceballosa inference wrapper (`run_alceballosa_inference.py`)
- [x] Multi-class upload script (`upload_av_masks.py`) — Artery=1, Vein=2
- [x] RedBrick taxonomy inspector (`check_redbrick_taxonomy.py`)
- [x] Corrected-label exporter (`download_corrected_labels.py`)
- [x] Checkpoint 2→3-class expander (`expand_checkpoint_to_3class.py`) —
      validated: expanded heads load strict into 3-class UNet3D, forward OK
- [x] `train.py` / `data/dataset.py` / `losses/losses.py` / `utils/metrics.py`
       updated for 3-class (bg / artery / vein); AV pseudo mode auto-sets
       `num_classes=3` and raises on mismatch
- [x] **USER STEP**: Download Google Drive weights + atlases (manual, see below)
- [x] **USER STEP**: Add Artery + Vein (Arteries/Veins) to the RedBrick taxonomy
- [x] Smoke-test alceballosa inference on CTA_001 (output: 3 classes, 0.5mm spacing,
      artery/vein ratio ~1:2 as expected)
- [x] Run alceballosa inference on all 50 CTAs (5h 1min wall-time on RTX 4070 Ti)
- [x] Validate all 50 masks pass quality gate (A:V ratios 0.23–0.83, grids match sources)
- [x] Upload 49 masks to RedBrick (CTA_001 skipped — already in Review from earlier test)
- [ ] Radiologist correction (external)
- [ ] Download corrected labels
- [ ] Fine-tune 3-class model on EC2

## Files in this folder

| File                               | Phase | Purpose                              |
|------------------------------------|-------|--------------------------------------|
| `dicom_to_nifti.py`                | 1b    | Convert TopCoW DICOM → NIfTI         |
| `run_alceballosa_inference.py`     | 1c    | Wrapper around extractVessels.py -v 241 |
| `check_redbrick_taxonomy.py`       | 2     | Inspect current RedBrick categories  |
| `validate_av_masks.py`             | 2     | Sanity-check alceballosa outputs before upload (shape, class counts, A:V ratio, grid match) |
| `upload_av_masks.py`               | 3     | Upload 3-class masks to RedBrick     |
| `download_corrected_labels.py`     | 5     | Export corrected labels from RedBrick |
| `expand_checkpoint_to_3class.py`   | 6     | Duplicate vessel-channel weights to initialize artery+vein heads |

## Labels (from preprint, Section 2.1)

The alceballosa model outputs **3 classes**:

| ID | Class      |
|----|------------|
| 0  | background |
| 1  | artery     |
| 2  | vein       |

## Phase 1 — Setup

### 1a. Download weights + atlases from Google Drive

The repo README points to:
https://drive.google.com/open?id=1uFTrSajk2oAx4LHctZB_0cg98Ubo1QJ-

Download the folder and place it at:
```
vessel_seg_v2/robust-vessel-segmentation/atlases_and_weights/
├── weights/
│   └── Dataset241_.../           # Z-score normalized model (what we want)
├── atlases/
│   └── rectangle_neck_scene_RegistrationMask/
└── ants-2.6.3/                   # (Linux-only binary — not needed for -m Prediction)
```

We use `-m Prediction -v 241`, so the ANTs Linux binary is not required on
Windows. The atlases directory is still consulted for a registration-settings
JSON.

### 1b. Convert DICOM to NIfTI

```bash
conda activate vessel_seg
cd "vessel_seg_v2/CTA Seg V2"
python pseudo_label/dicom_to_nifti.py \
    --dicom_root "D:/vessel_seg_v2/TopCoW_DICOM" \
    --output_dir "D:/vessel_seg_v2/CTA_nifti"
```

Produces one `.nii.gz` per "Thins with contrast" series, named like:
- `CTA_001_Thins_with_contrast.nii.gz`
- `CTA_013_Thins_with_contrast_full_field.nii.gz`

Note on CTA_050: has both `_bone_kernal` and `_soft_tissue_kernal` variants.
alceballosa Model 241 was trained on soft-tissue CTA — the bone-kernel
reconstruction will produce a lower-quality pseudo-label. Before upload,
keep only the soft-tissue variant for CTA_050 (or keep both and pick one
at upload time).

### 1c. Run alceballosa Model 241 inference

Build a filtered input dir (drops the bone-kernel variant of CTA_050 because
Model 241 is soft-tissue trained):

```bash
mkdir -p /d/vessel_seg_v2/CTA_nifti_inference
for f in /d/vessel_seg_v2/CTA_nifti/*.nii.gz; do
    case "$f" in *bone_kernal*) continue;; esac
    ln "$f" "/d/vessel_seg_v2/CTA_nifti_inference/$(basename "$f")" 2>/dev/null
done
```

Then run inference:

```bash
python pseudo_label/run_alceballosa_inference.py \
    --input_dir  "D:/vessel_seg_v2/CTA_nifti_inference" \
    --output_dir "D:/vessel_seg_v2/av_masks" \
    --num_gpus 1
```

Output: 3-class NIfTI masks in `D:/vessel_seg_v2/av_masks/<scan>.nii.gz`
(sequential mode writes the final patient-space mask directly, so no
`Predictions/` subdirectory). The pipeline is resumable — re-running skips any
scan whose output already exists.

#### Windows patches applied to the vendored repo

The upstream alceballosa repo targets Linux. The following tweaks live in
`vessel_seg_v2/robust-vessel-segmentation/scripts/inference/`:

- `utils.py` `execute_and_log`: replaced `NamedTemporaryFile(delete=True)` +
  `os.system(... > tmp 2>&1)` (which fails on Windows because the temp file is
  held with an exclusive lock) with `subprocess.run(..., capture_output=True)`.
- `extractVessels.py`: quoted `-i/-o` paths with double quotes (cmd.exe treats
  single quotes as literal characters).
- `helper.py` `process_row_resampling`: replaced the `antsApplyTransforms
  -t identity` shell call (binary not available on Windows) with a SimpleITK
  nearest-neighbor resample onto the patient grid — identical result.

The `dataset.json` / `plans.json` files were not included in the Google Drive
download, but nnUNetv2 checkpoints embed them under `init_args`. Extract with:

```bash
python - <<'PY'
import json, shutil, torch
from pathlib import Path
d = Path("vessel_seg_v2/robust-vessel-segmentation/atlases_and_weights/weights/Dataset241_Dyn/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")
ck = torch.load(d / "fold_all/checkpoint_final.pth", map_location="cpu", weights_only=False)
(d / "plans.json").write_text(json.dumps(ck["init_args"]["plans"], indent=2))
(d / "dataset.json").write_text(json.dumps(ck["init_args"]["dataset_json"], indent=2))
shutil.copyfile(d / "fold_all/checkpoint_final.pth", d / "fold_all/checkpoint_best.pth")
PY
```

## Phase 2 — RedBrick taxonomy

Add two categories to the RedBrick project taxonomy:
- **Artery** (red, id 1)
- **Vein**   (blue, id 2)

Keep the existing `Vessels` category unused for new tasks, or deprecate it.

## Phase 3 — Upload multi-class masks

Update `upload_masks.py` to iterate label IDs 1 and 2 and upload one
`instances` entry per class, each pointing at the same NIfTI but with a
different `category`.

## Phase 4 — Radiologist correction

External — in the RedBrick browser.

## Phase 5 — Download corrected labels

Use the RedBrick SDK `export_tasks` to pull corrected NIfTI masks.

## Phase 6 — 3-class training pipeline

Changes required in `CTA Seg V2/`:
- `configs/default.py`: `num_classes: 2 → 3`
- `data/dataset.py`: stop binarizing labels (preserve 1/2)
- `losses/losses.py`: multi-class Dice / clDice / Skeleton already support N-class via softmax
- `utils/metrics.py`: per-class DSC / clDice
- `evaluate.py`: stratified (artery vs vein) reporting

## Phase 7 — Fine-tuning (EC2)

Start from the best Skeleton+TopBrain checkpoint, expand the final 1×1×1 conv
from 2 → 3 output channels (copy existing vessel-channel weights into the
artery channel, init vein channel from scratch). Run a short fine-tune.
