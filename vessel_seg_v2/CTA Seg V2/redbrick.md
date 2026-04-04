# RedBrick AI — Model-Assisted Labeling Workflow

Upload model-generated vessel segmentation masks to RedBrick AI as pre-labels,
then correct them manually in the browser.

## Prerequisites

```bash
pip install redbrick-sdk
```

Set environment variables (PowerShell):
```powershell
$env:REDBRICK_ORG_ID="da664378-a762-4538-8cb5-cb74b5d89345"
$env:REDBRICK_PROJECT_ID="a2b8ba7d-9e5c-42f7-a20d-050a9f5048be"
$env:REDBRICK_API_KEY="your-api-key"
```

## Pipeline

### 1. Run inference on CTA scans

On the EC2 instance (or locally with GPU):

```bash
python predict.py \
    --checkpoint runs/<model>/best_model.pth \
    --input_dir ~/data/cta_scans/ \
    --output_dir ~/data/vessel_masks/
```

This outputs `*_vessels.nii.gz` binary masks with matching NIfTI headers.

### 2. Download masks locally

```bash
scp -i "key.pem" -r ubuntu@<ip>:~/data/vessel_masks/ C:\Users\Samih\Downloads\cta_masks\
```

### 3. List tasks to verify matching

```bash
python upload_masks.py --list
```

### 4. Dry run to check name matching

```bash
python upload_masks.py --masks_dir C:\Users\Samih\Downloads\cta_masks --dry_run
```

Masks are matched to RedBrick tasks by patient ID:
- `CTA_001_vessels.nii.gz` → `/CTA_001_Thins with contrast`
- `CTA_013_full_field_vessels.nii.gz` → `/CTA_013_Thins with contrast (full field)`
- `CTA_050_bone_kernal_vessels.nii.gz` → `/CTA_050_Thins with contrast (bone kernal)`

### 5. Upload masks

```bash
python upload_masks.py --masks_dir C:\Users\Samih\Downloads\cta_masks
```

Tasks are submitted with `finalize=True`, which moves them from **Label → Review_1**.
Open them in the Review stage to see and edit the vessel overlay.

## Important Notes

- **Taxonomy**: The category must match exactly — `Vessels` (capital V)
- **`finalize=True`**: Required. `finalize=False` saves as draft under the API key, invisible to other users
- **Task matching**: Masks match to "Thins with contrast" tasks by default. Special qualifiers (full_field, bone_kernal, etc.) are handled automatically
- **Unmatched masks**: `CTA_043` has no task in RedBrick — this is reported during upload
- **Re-uploading**: To redo, move tasks back from Review to Label in the UI, then re-upload

## Scripts

| Script | Purpose |
|--------|---------|
| `predict.py` | Run model inference on NIfTI(s), output binary vessel masks |
| `upload_masks.py` | Match masks to RedBrick tasks and upload via SDK |
| `test_upload.py` | Debug script for testing single-task upload |
