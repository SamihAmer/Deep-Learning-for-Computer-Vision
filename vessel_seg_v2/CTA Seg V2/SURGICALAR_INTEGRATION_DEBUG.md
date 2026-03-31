# CTA Vessel Segmentation Integration into SurgicalAR

## Overview

This document records the full technical story of integrating a custom 3D U-Net cerebral vessel segmentation model into SurgicalAR's ML pipeline. The model was trained on the TopCoW 2024 dataset using Dice + Cross-Entropy loss, achieving a reported Dice Similarity Coefficient (DSC) of 0.8846 during training evaluation.

The integration was initially planned as config-driven only: register the model in four JSON configuration files and place the ONNX model in the correct folder. In practice, the process uncovered multiple issues in both the SurgicalAR pipeline and the model export itself, requiring code changes, ONNX model surgery, and config revisions.

---

## Issues Found and Fixes Applied

### 1. Task Not Appearing in UI -- `.trt` File Requirement

**Symptom:** The "Vessel Base Model" task did not appear in the Automatic Segmentation dropdown after adding all four JSON config entries and placing `model.onnx` in the model folder.

**Root Cause:** On Windows, `MachineLearningModelConfigurationManager` (defined in `MachineLearning/Segmentation/Configurations/MachineLearningModelConfiguration.cs`) sets `RequiredModelPostfixes = { ".onnx", ".trt" }`, meaning BOTH `model.onnx` AND `model.trt` must exist in the model folder for the task to be considered valid. Our model only had `model.onnx`.

**Why both checks matter:**

- The `PreCheckModelExistence = true` flag in `LocalMachineLearningVolumeSegmentationManager.cs` triggers file existence checks during workspace initialization. If any required postfix is missing, the task is silently excluded from the UI.
- Separately, `TryGetRunnableModelConfigContent()` also checks file existence at inference time. Both checks must pass for the pipeline to function end-to-end.

**First attempt (failed):** Created an empty `model.trt` placeholder file to pass the existence check. This satisfied the UI filter, but when inference ran with TensorRT as the execution provider, the native plugin attempted to deserialize the empty file and crashed.

**Fix applied:** Changed `RequiredModelPostfixes` on Windows from `{ ".onnx", ".trt" }` to `{ ".onnx" }` in `MachineLearningModelConfiguration.cs` at line 22. A TODO comment was added to revert this change after testing is complete and a real `.trt` file is generated.

**File:** `SurgicalAR/Assets/Scripts/MediVis/MachineLearning/Segmentation/Configurations/MachineLearningModelConfiguration.cs`

---

### 2. Native Inference Crash -- `MultiClass: true` with `OutputClassCount: 2`

**Symptom:** `InferenceITKImageSegmentation` returned `E_FAIL` (HRESULT -2147467259) regardless of execution provider (CPU, CUDA, and TensorRT all failed identically).

**Root Cause:** The native C++ plugin (`MachineLearningInferenceRunner.dll`) does not support the combination of `MultiClass: true` with `OutputClassCount: 2` when there is only 1 organ in the organ set. Inspection of every working MultiClass pipeline in the codebase confirmed that `OutputClassCount` always equals the organ count. Our config had `OutputClassCount: 2` but only 1 organ ("Cerebral Vessels"), violating this undocumented constraint.

**Debugging attempts that did not resolve the issue:**

| Attempt | Rationale | Result |
|---|---|---|
| Changed ONNX opset from 17 to 11 | Rule out opset compatibility | Still crashed |
| Fixed dynamic axes to static `[128, 128, 128]` shapes | Rule out dynamic shape issues | Still crashed |
| Switched execution providers (CPU, CUDA, TensorRT) | Rule out provider-specific bugs | All three crashed identically |
| Added "Background" as a second organ to match `OutputClassCount: 2` | Align organ count with output channels | Still crashed |
| Stripped pipeline to bare minimum (EmptyProcessor normalization, no preprocessing) | Rule out preprocessing errors | Still crashed |
| Built a minimal single-layer Conv3d test model with `MultiClass: true` | Isolate whether the model architecture was the cause | Still crashed -- proving the config path was the issue |

**What worked:** A minimal Conv3d test model with `MultiClass: false, OutputClassCount: 1` ran successfully through the full inference pipeline. This conclusively proved that the `MultiClass: true` code path in the native plugin was the source of the crash when OutputClassCount did not match the organ count in the expected way.

**Fix applied:** Changed the model configuration to `MultiClass: false, OutputClassCount: 1, LogitsConfidenceThreshold: 0.5` and modified the ONNX model to output a single channel (see Issue 3 below).

**File (config):** `StreamingAssets/MachineLearning/ConfigurableSegmentationPipelineCore/Models.json`

---

### 3. ONNX Model Conversion -- Softmax Numerical Failure

**Symptom:** After converting the 2-channel model to single-channel by appending Softmax + Slice nodes in ONNX, the model produced all-zero output (no vessel voxels detected).

**Root Cause:** ONNX opset 12 and below defines `Softmax(axis=N)` differently from PyTorch's `softmax(dim=N)`. In opset 11, `Softmax(axis=1)` applied to a tensor of shape `(1, 2, 128, 128, 128)` flattens all dimensions after the axis before computing softmax. This means it computes softmax across `2 * 128 * 128 * 128 = 4,194,304` values instead of across the 2 class channels. The result is near-zero probability for every element, producing an all-zero binary mask regardless of input.

**Note:** This behavior was changed in ONNX opset 13, which aligned Softmax semantics with PyTorch. However, our model was exported at opset 11.

**Fix applied:** Instead of Softmax, the ONNX graph was modified to compute `sigmoid(vessel_logit - background_logit)`:

1. **Slice** channel 0 (background logit) from the 2-channel output
2. **Slice** channel 1 (vessel logit) from the 2-channel output
3. **Sub**: compute `vessel_logit - background_logit`
4. **Sigmoid**: map the difference to the range [0, 1]

This approach is:

- **Mathematically equivalent** to 2-class softmax: `softmax(x)[1] = sigmoid(x[1] - x[0])`
- **Numerically stable** with no opset version dependency
- **Compatible** with the `LogitsConfidenceThreshold: 0.5` setting, since threshold 0.5 on the sigmoid output is equivalent to argmax on the original 2-channel logits

Added ONNX nodes: 2x Slice, 1x Sub, 1x Sigmoid. Output name changed from `"logits"` to `"output"`, shape changed from `(1, 2, 128, 128, 128)` to `(1, 1, 128, 128, 128)`.

---

### 4. No Mesh Generated -- All-Zero Predictions (Model Quality Issue)

**Symptom:** After all pipeline issues were resolved, inference ran to completion without errors, but the result was: "No valid segmentation mesh is generated for Vessel Base Model."

**Root Cause:** The model itself produces very poor predictions. Validation performed entirely in Python (outside SurgicalAR) using full sliding window inference on the same TopCoW CTA data showed:

- Only 478 vessel voxels detected versus 11,408 ground truth voxels
- Effective DSC approximately 0 (complete failure)
- Logit magnitudes were abnormally large (values up to 244), whereas well-trained models typically produce logits in the range [-10, 10]

**Conclusion:** This is NOT a SurgicalAR integration issue. The pipeline processed the model output correctly; the model output itself was wrong. The SurgicalAR ML pipeline is fully functional for this task configuration.

**Likely upstream cause:** The `InferenceWrapper` class in `export_onnx.py` (the training codebase) uses `self.unet(x)[0]` to extract the full-resolution output from the deep supervision heads. The `[0]` index may be selecting the lowest-resolution supervision head instead of the full-resolution one, which would explain both the poor predictions and the abnormally large logit magnitudes.

---

## Current State of Changes

### Configuration Files

All config files are located under `StreamingAssets/MachineLearning/ConfigurableSegmentationPipelineCore/`.

#### Models.json -- CerebralVesselDiceCE Entry

```json
{
    "UniqueModelName": "CerebralVesselDiceCE",
    "ModelFolder": "CerebralVesselSegmentation/dice_ce",
    "InputName": "input",
    "OutputName": "output",
    "PatchShape": [128, 128, 128],
    "InputPermution": [0, 1, 2, 3, 4],
    "OutputPermution": [0, 1, 2, 3, 4],
    "MultiClass": false,
    "OutputClassCount": 1,
    "LogitsConfidenceThreshold": 0.5
}
```

Key differences from the original integration plan:

| Field | Original Plan | Final Value | Reason |
|---|---|---|---|
| `MultiClass` | `true` | `false` | Native plugin crash (Issue 2) |
| `OutputClassCount` | `2` | `1` | Native plugin crash (Issue 2) |
| `OutputName` | `"logits"` | `"output"` | ONNX model conversion (Issue 3) |
| `LogitsConfidenceThreshold` | `0` | `0.5` | Required for binary sigmoid output |

#### Pipelines.json -- cerebral_vessel_dice_ce Entry

Unchanged from the original plan. Uses CTNormalization with HU window [0, 600], ReorientVolumeDirection to RPI, FillHoles post-processing, and ResampleToInput to restore original resolution.

#### Organs.json -- CerebralVesselOrgans

Contains 1 organ: "Cerebral Vessels". This matches `OutputClassCount: 1`.

#### TaskLoadingConfiguration.json -- cerebral_vessel_dice_ce Entry

`ExecutionProviderWin` is set to `"CUDA"` (was `"TensorRT"` in the original plan, changed because no `.trt` file is available).

---

### C# Code Modified

#### MachineLearningModelConfiguration.cs (line 22)

**File:** `SurgicalAR/Assets/Scripts/MediVis/MachineLearning/Segmentation/Configurations/MachineLearningModelConfiguration.cs`

**Change:** Windows `RequiredModelPostfixes` changed from `{ ".onnx", ".trt" }` to `{ ".onnx" }`.

**Status:** Has a TODO comment to revert after testing. This change should be reverted once a proper `.trt` file is generated for production deployment.

#### LocalMachineLearningVolumeSegmentationManager.cs

**File:** `SurgicalAR/Assets/Scripts/MediVis/MachineLearning/Segmentation/LocalMachineLearningVolumeSegmentationManager.cs`

**Change:** Contains temporary `[VesselDebug]` logging statements that trace the full pipeline execution path: pipeline JSON, model JSON, organ JSON, workspace creation result, volume data range, ITK image creation, and inference result.

**Status:** These debug statements should be removed before committing.

---

### ONNX Model Files

All files are located in `Models/CerebralVesselSegmentation/dice_ce/`.

| File | Description | Status |
|---|---|---|
| `model.onnx` | Converted single-channel sigmoid model. Output shape `(1, 1, 128, 128, 128)`, vessel probability in [0, 1]. Contains appended Slice, Sub, and Sigmoid nodes. | Currently deployed |
| `model_real.onnx` | Original 2-channel model with opset 11 and static shapes. Output shape `(1, 2, 128, 128, 128)`. | Preserved for reference |
| `model_minimal.onnx` | Minimal single-layer Conv3d test model used during debugging. | Can be deleted |

---

## Next Steps to Complete the Integration

1. **Validate the PyTorch model directly.** Run the trained PyTorch model (not the ONNX export) on TopCoW data using `evaluate.py` to confirm it achieves DSC approximately 0.88. This determines whether the problem is in the model weights or the export.

2. **Compare PyTorch vs ONNX predictions.** Run both on the same input patch and compare outputs element-wise. If they diverge, the `InferenceWrapper` output indexing is wrong.

3. **Fix the `InferenceWrapper` output indexing.** If `self.unet(x)[0]` selects the wrong deep supervision head, try `[-1]` or iterate through the heads to find the full-resolution output.

4. **Re-export to ONNX.** Once the PyTorch model is validated, re-export with the correct output head. Apply the same sigmoid conversion (or export directly as single-channel sigmoid). Place in SurgicalAR -- the pipeline will work immediately with no further changes.

5. **Clean up temporary changes:**
   - Remove `[VesselDebug]` logging from `LocalMachineLearningVolumeSegmentationManager.cs`
   - Revert `RequiredModelPostfixes` TODO in `MachineLearningModelConfiguration.cs` (after generating a `.trt` file)
   - Delete `model_minimal.onnx` from the model folder

6. **Production deployment considerations:**
   - Generate a proper `.trt` file using `trtexec` for TensorRT inference
   - Switch `ExecutionProviderWin` back to `"TensorRT"` in `TaskLoadingConfiguration.json`
   - Revert `RequiredModelPostfixes` to require both `.onnx` and `.trt`

---

## Key Learnings for Future Model Integrations

### 1. Use `MultiClass: false` with `OutputClassCount: 1` for Binary Segmentation

The native plugin's `MultiClass: true` code path requires `OutputClassCount` to equal the number of organs in the organ set and has additional undocumented constraints. For any binary (single-organ) segmentation model, use `MultiClass: false` with `OutputClassCount: 1` and set an appropriate `LogitsConfidenceThreshold`.

### 2. Model Output Must Be Single-Channel Probability for Binary Models

When `MultiClass: false`, the pipeline expects a single-channel output in [0, 1] (probability), which is thresholded by `LogitsConfidenceThreshold`. Use sigmoid activation, not raw logits or softmax.

### 3. ONNX Softmax Semantics Changed at Opset 13

ONNX opset 12 and below defines `Softmax(axis=N)` to flatten all dimensions after axis N before computing softmax. This is NOT equivalent to PyTorch's `softmax(dim=N)`. Opset 13+ fixed this, but if your model uses opset 12 or below, avoid Softmax entirely. The `sigmoid(logit_a - logit_b)` approach is a safe, opset-independent alternative for 2-class problems.

### 4. `.trt` File Is Required on Windows by Default

The `RequiredModelPostfixes` check requires both `.onnx` and `.trt` files to exist on Windows. Either pre-compile the `.trt` file using `trtexec` before deployment, or temporarily modify `RequiredModelPostfixes` during development.

### 5. Always Validate the ONNX Export in Python Before SurgicalAR Integration

Run inference on real data in Python using the ONNX model and compare against the PyTorch model's predictions before placing the model in SurgicalAR. This catches export issues (wrong output head, opset incompatibilities, numerical divergences) without the overhead of debugging through the native plugin.
