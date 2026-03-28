# CTA Cerebral Vessel Segmentation — SurgicalAR Integration Plan

## Overview

Integrate our custom-trained 3D U-Net cerebral vessel segmentation models into SurgicalAR's existing ML pipeline. The models are trained on the TopCoW 2024 dataset (250 CTA volumes, 13 vessel classes collapsed to binary) and segment the Circle of Willis vasculature from CT angiography scans.

Three model variants will be integrated, each using a different loss function to study the effect of topology-aware losses on small vessel segmentation:

| Model | Loss Function | Status | Pipeline Name |
|-------|--------------|--------|---------------|
| **Vessel Base Model** | Dice + CE | Trained (best DSC: 0.8846) | `cerebral_vessel_dice_ce` |
| **Vessel clDice Model** | Dice + CE + clDice | Not yet trained | `cerebral_vessel_dice_ce_cldice` |
| **Vessel Skeleton Model** | Dice + CE + Skeleton Recall | Not yet trained | `cerebral_vessel_dice_ce_skeleton` |

---

## Architecture

### Model Spec

- **Architecture**: 3D U-Net (nnU-Net style), 23.6M parameters 
- **Encoder**: 5 stages, residual ConvBlocks, strided conv downsampling, InstanceNorm + LeakyReLU
- **Decoder**: Transposed conv upsampling, skip connections, deep supervision heads
- **Input**: `(B, 1, D, H, W)` single-channel CT, normalized to [0, 1] via HU window (0, 600)
- **Output**: `(B, 2, D, H, W)` softmax logits (background + vessel)
- **Patch size**: 128 x 128 x 128
- **Training**: AdamW, cosine LR schedule, mixed precision, 300 epochs

### ONNX Export

The model is exported using `export_onnx.py` which wraps the U-Net in an `InferenceWrapper` that returns only the full-resolution output (index 0 of deep supervision heads). This is critical because:

- The model was trained with `deep_supervision=True`, so the checkpoint contains `ds_heads` weights
- Creating the model with `deep_supervision=False` and loading with `strict=False` would leave `final_head` randomly initialized (broken output)
- The wrapper preserves the exact trained architecture and just indexes `outputs[0]`

```python
class InferenceWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, x):
        return self.unet(x)[0]
```

Export command:
```bash
cd "vessel_seg_v2/CTA Seg V2"
python export_onnx.py --checkpoint "runs/dice_ce_20260325_080955/best_model.pth"
```

Dynamic axes are set on spatial dimensions (D, H, W) so the native sliding window can use the model with varying patch positions.

### ONNX Input/Output Names

- Input: `"input"` — shape `(1, 1, 128, 128, 128)`
- Output: `"logits"` — shape `(1, 2, 128, 128, 128)`

These names must match `InputName` and `OutputName` in Models.json.

---

## SurgicalAR Integration Points

The integration is **entirely config-driven** — no C# code changes required. Four JSON files and the model file placement are all that's needed.

### File Locations

All configs live in:
```
SurgicalAR/Assets/StreamingAssets/MachineLearning/ConfigurableSegmentationPipelineCore/
```

### 1. Models.json — Model Registration

Each ONNX model gets an entry here. Key fields:

```json
{
    "UniqueModelName": "CerebralVesselDiceCE",
    "ModelFolder": "CerebralVesselSegmentation/dice_ce",
    "InputName": "input",
    "OutputName": "logits",
    "PatchShape": [128, 128, 128],
    "InputPermution": [0, 1, 2, 3, 4],
    "OutputPermution": [0, 1, 2, 3, 4],
    "MultiClass": true,
    "OutputClassCount": 2,
    "LogitsConfidenceThreshold": 0
}
```

**Notes:**
- `InputPermution` / `OutputPermution` (note: field name has typo in codebase, must match) — `[0,1,2,3,4]` means no dimension reordering. Our PyTorch model uses standard `(B, C, D, H, W)` layout which matches the native pipeline's default.
- `MultiClass: true` with `OutputClassCount: 2` tells the native code to apply argmax across 2 channels, yielding labels 0 (background) and 1 (vessel).
- `PatchShape: [128, 128, 128]` must match training patch size.

**For future models**, add similar entries with different `UniqueModelName` and `ModelFolder`:
- `CerebralVesselDiceCEclDice` → `CerebralVesselSegmentation/dice_ce_cldice`
- `CerebralVesselDiceCESkeleton` → `CerebralVesselSegmentation/dice_ce_skeleton`

### 2. Pipelines.json — Preprocessing & Postprocessing

```json
{
    "UniquePipelineName": "cerebral_vessel_dice_ce",
    "DisplayName": "Vessel Base Model",
    "ExposePipeline": true,
    "PipelineType": "ConfigurableSegmentationPipelineGeneral",
    "TargetModelNames": ["CerebralVesselDiceCE"],
    "RecordLogits": false,
    "NormalizationConfig": [
        {
            "Type": "CTNormalization",
            "LowerBound": 0.0,
            "UpperBound": 600.0,
            "Mean": 0.0,
            "Std": 600.0
        }
    ],
    "PreprocessingConfig": [
        {
            "Type": "ReorientVolumeDirection",
            "Direction": "RPI"
        }
    ],
    "PostprocessingConfig": [
        {
            "Type": "FillHoles",
            "BackgroundValue": 0,
            "ForegroundValue": 1,
            "MajorityThreshold": 1,
            "Radius": [1, 1, 1]
        },
        {
            "Type": "ResampleToInput"
        }
    ],
    "DependentPipelineNames": [],
    "OrganSetName": "CerebralVesselOrgans"
}
```

**Normalization details:**

The model was trained with `preprocess_ct()`:
```python
volume = np.clip(volume, 0, 600)
volume = (volume - 0) / (600 - 0)  # → [0, 1]
```

`CTNormalization` with `LowerBound=0, UpperBound=600, Mean=0, Std=600` replicates this exactly:
```
clip(x, 0, 600) → (x - 0) / 600 → [0, 1]
```

**Preprocessing:**
- `ReorientVolumeDirection: RPI` — reorients to Right-Posterior-Inferior, a standard anatomical orientation. The model was trained on TopCoW data in various orientations and processes 3D patches locally, so it is orientation-agnostic.
- No resampling is applied — the model was trained on native voxel spacings.

**Postprocessing:**
- `FillHoles` — morphological closing to fill small holes in the vessel mask.
- `ResampleToInput` — resamples the prediction back to the original volume resolution.

**For future models**, create separate pipeline entries with different `UniquePipelineName`, `DisplayName`, and `TargetModelNames`. The normalization, preprocessing, and postprocessing stay the same since all three models were trained with identical data preprocessing.

### 3. Organs.json — Structure Names

```json
{
    "OrganSetName": "CerebralVesselOrgans",
    "OrganNames": ["Cerebral Vessels"]
}
```

All three model variants share this organ set (they all produce the same binary vessel mask). Only one entry is needed.

### 4. TaskLoadingConfiguration.json — Task Definition & UI Grouping

```json
{
    "TaskName": "cerebral_vessel_dice_ce",
    "SupportedAppTypes": ["SurgicalAR", "RenderX", "Cranial"],
    "SupportedModalities": ["CT"],
    "LoadAsMesh": true,
    "MeshTransparent": true,
    "MeshGenerationSmoothingFactor": 0.3,
    "BackendWin": "Local",
    "BackendMac": "Local",
    "ExecutionProviderWin": "TensorRT",
    "ExecutionProviderMac": "CPU",
    "AlwaysRecompute": false
}
```

**Key settings:**
- `SupportedModalities: ["CT"]` — CTA is a CT modality, so this ensures the task only appears for CT scans.
- `LoadAsMesh: true` — triggers marching cubes mesh generation from the segmentation mask, producing a 3D vessel mesh that appears in the Structures panel.
- `MeshTransparent: true` — vessels render semi-transparent so underlying anatomy is visible.
- `MeshGenerationSmoothingFactor: 0.3` — smoothing applied to the marching cubes output. Adjust if vessel meshes look too jagged (increase) or too blobby (decrease).
- `ExecutionProviderWin: "TensorRT"` — uses TensorRT for GPU-accelerated inference on Windows. Falls back to ONNX Runtime if TensorRT compilation fails. First run will be slow (TensorRT engine compilation), subsequent runs use cached `.trt` engine.
- `AlwaysRecompute: false` — caches results so re-opening the task doesn't re-run inference.

**Add to the "Automatic Segmentation" group** in the `Groups` array:
```json
{
    "GroupName": "Automatic Segmentation",
    "TaskNames": [
        ...existing tasks...,
        "cerebral_vessel_dice_ce"
    ]
}
```

**For future models**, add separate task entries (`cerebral_vessel_dice_ce_cldice`, `cerebral_vessel_dice_ce_skeleton`) and add them to the same group.

### 5. Model File Placement

```
StreamingAssets/MachineLearning/ConfigurableSegmentationPipelineCore/Models/
  CerebralVesselSegmentation/
    dice_ce/
      model.onnx          ← exported from export_onnx.py
```

The native plugin looks for `model.onnx` (and optionally `model.trt`) inside the folder specified by `ModelFolder` in Models.json.

**TensorRT engine**: On first inference with `ExecutionProviderWin: "TensorRT"`, the native plugin compiles the ONNX model into a `.trt` engine and caches it. This first run takes significantly longer (minutes). Subsequent runs load the cached engine. If you want to pre-compile, you can also use the `trtexec` CLI tool from NVIDIA's TensorRT SDK.

---

## UI Flow

After integration, the task appears at:

**Plan → Structures → Automatic Segmentation → Vessel Base Model**

Clicking "Generate" triggers:
1. `ScanSegmentationManager.StartSettingSegmentationStatus("cerebral_vessel_dice_ce", true)`
2. Check cache → miss → `RunInference()`
3. `LocalMachineLearningVolumeSegmentationManager.RunInference()`:
   - Creates ITK workspace from JSON configs
   - Converts ScanViewer volume data to ITK image
   - Native pipeline: CTNormalization → ReorientVolumeDirection → sliding window inference (128^3 patches) → FillHoles → ResampleToInput
4. `ExportSegmentationToMesh()` — marching cubes generates STL
5. STL loaded as `MeshData` → appears in Structures panel as "Cerebral Vessels"

---

## Step-by-Step: Adding the Remaining Two Models

When the dice_ce_cldice and dice_ce_skeleton models are trained:

### 1. Export to ONNX
```bash
python export_onnx.py --checkpoint "runs/dice_ce_cldice_.../best_model.pth" --output vessel_seg_cldice.onnx
python export_onnx.py --checkpoint "runs/dice_ce_skeleton_.../best_model.pth" --output vessel_seg_skeleton.onnx
```

### 2. Place model files
```bash
cp vessel_seg_cldice.onnx   "<SurgicalAR>/Models/CerebralVesselSegmentation/dice_ce_cldice/model.onnx"
cp vessel_seg_skeleton.onnx "<SurgicalAR>/Models/CerebralVesselSegmentation/dice_ce_skeleton/model.onnx"
```

### 3. Models.json — add 2 entries
```json
{
    "UniqueModelName": "CerebralVesselDiceCEclDice",
    "ModelFolder": "CerebralVesselSegmentation/dice_ce_cldice",
    "InputName": "input",
    "OutputName": "logits",
    "PatchShape": [128, 128, 128],
    "InputPermution": [0, 1, 2, 3, 4],
    "OutputPermution": [0, 1, 2, 3, 4],
    "MultiClass": true,
    "OutputClassCount": 2,
    "LogitsConfidenceThreshold": 0
},
{
    "UniqueModelName": "CerebralVesselDiceCESkeleton",
    "ModelFolder": "CerebralVesselSegmentation/dice_ce_skeleton",
    "InputName": "input",
    "OutputName": "logits",
    "PatchShape": [128, 128, 128],
    "InputPermution": [0, 1, 2, 3, 4],
    "OutputPermution": [0, 1, 2, 3, 4],
    "MultiClass": true,
    "OutputClassCount": 2,
    "LogitsConfidenceThreshold": 0
}
```

### 4. Pipelines.json — add 2 entries
Same as the dice_ce pipeline but with different names and model references:
```json
{
    "UniquePipelineName": "cerebral_vessel_dice_ce_cldice",
    "DisplayName": "Vessel clDice Model",
    "TargetModelNames": ["CerebralVesselDiceCEclDice"],
    ...same normalization, preprocessing, postprocessing...
    "OrganSetName": "CerebralVesselOrgans"
},
{
    "UniquePipelineName": "cerebral_vessel_dice_ce_skeleton",
    "DisplayName": "Vessel Skeleton Model",
    "TargetModelNames": ["CerebralVesselDiceCESkeleton"],
    ...same normalization, preprocessing, postprocessing...
    "OrganSetName": "CerebralVesselOrgans"
}
```

### 5. Organs.json — no changes
All three models share `CerebralVesselOrgans`.

### 6. TaskLoadingConfiguration.json — add 2 task entries + update group
```json
{
    "TaskName": "cerebral_vessel_dice_ce_cldice",
    "SupportedAppTypes": ["SurgicalAR", "RenderX", "Cranial"],
    "SupportedModalities": ["CT"],
    "LoadAsMesh": true,
    "MeshTransparent": true,
    "MeshGenerationSmoothingFactor": 0.3,
    "BackendWin": "Local",
    "BackendMac": "Local",
    "ExecutionProviderWin": "TensorRT",
    "ExecutionProviderMac": "CPU",
    "AlwaysRecompute": false
},
{
    "TaskName": "cerebral_vessel_dice_ce_skeleton",
    ...same as above...
}
```

Add both to the group:
```json
"TaskNames": [
    ...existing...,
    "cerebral_vessel_dice_ce",
    "cerebral_vessel_dice_ce_cldice",
    "cerebral_vessel_dice_ce_skeleton"
]
```

---

## Troubleshooting

### Model doesn't appear in UI
- Check `SupportedAppTypes` includes the app you're running (e.g., `"Cranial"`)
- Check `SupportedModalities` includes `"CT"` and you've loaded a CT scan
- Verify `TaskName` in TaskLoadingConfiguration matches `UniquePipelineName` in Pipelines.json

### Inference fails / crashes
- Check `InputName` and `OutputName` match the ONNX model's actual input/output names
- Verify `PatchShape` matches the ONNX model's expected spatial dimensions
- If TensorRT fails, try `"ExecutionProviderWin": "CUDA"` or `"CPU"` as fallback
- Check ONNX opset version compatibility (opset 17 used)

### Bad segmentation quality
- Verify normalization matches training: `CTNormalization` with `LowerBound=0, UpperBound=600, Mean=0, Std=600` gives `clip(x,0,600)/600` which matches `preprocess_ct()`
- Check that the input DICOM is a CTA (contrast-enhanced CT) — the model was only trained on CTA, not non-contrast CT
- If vessels look fragmented, try reducing `MeshGenerationSmoothingFactor` (e.g., 0.15)
- If too many false positives, the model may need fine-tuning on data closer to your clinical CTA protocol

### Permutation issues
- Our model uses standard PyTorch layout `(B, C, D, H, W)` = `[0, 1, 2, 3, 4]`
- If the native pipeline expects a different layout, adjust `InputPermution` / `OutputPermution`
- Common alternative: `[0, 2, 3, 4, 1]` converts `(B, C, D, H, W)` → `(B, D, H, W, C)`

### TensorRT first-run slowness
- First inference compiles the ONNX to a TensorRT engine (can take several minutes)
- The `.trt` file is cached alongside the `.onnx` file for subsequent runs
- To pre-compile: use NVIDIA's `trtexec --onnx=model.onnx --saveEngine=model.trt`

---

## Key File Paths

| What | Path |
|------|------|
| Best checkpoint (dice_ce) | `vessel_seg_v2/CTA Seg V2/runs/dice_ce_20260325_080955/best_model.pth` |
| ONNX export script | `vessel_seg_v2/CTA Seg V2/export_onnx.py` |
| Stratified eval script | `vessel_seg_v2/CTA Seg V2/evaluate.py` |
| SurgicalAR configs | `SurgicalAR/Assets/StreamingAssets/MachineLearning/ConfigurableSegmentationPipelineCore/` |
| Model placement | `...ConfigurableSegmentationPipelineCore/Models/CerebralVesselSegmentation/dice_ce/model.onnx` |
| SurgicalAR branch | `Users/Samih/CTASegmentation` |

---

## Validation Plan

1. **Export ONNX** and place in model directory
2. **Open SurgicalAR**, load a CTA DICOM (either converted from TopCoW NIfTI or clinical CTA)
3. Navigate to **Plan → Structures → Automatic Segmentation → Vessel Base Model**
4. Click Generate and verify:
   - Inference runs without errors
   - A "Cerebral Vessels" mesh appears in the Structures panel
   - The mesh looks anatomically plausible (Circle of Willis region)
5. **Compare** the SurgicalAR segmentation visually against the Python sliding window inference output to confirm consistency
6. If issues arise, check the troubleshooting section above
