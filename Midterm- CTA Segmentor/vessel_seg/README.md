# Cerebral Vessel Segmentation from CTA

3D U-Net with topology-aware loss functions for Circle of Willis segmentation, trained on the TopCoW 2024 dataset.

## Project Structure

```
vessel_seg/
    configs/
        default.py          # All hyperparameters in one place
    data/
        dataset.py           # TopCoW data loading, patch sampling, augmentation
    models/
        unet3d.py            # 3D U-Net with deep supervision
    losses/
        losses.py            # Dice+CE, soft-clDice, Skeleton Recall, deep supervision wrapper
    utils/
        metrics.py           # DSC, clDice, HD95, Betti-0 evaluation
        inference.py         # Sliding window inference with Gaussian stitching
    train.py                 # Main training script
```

## Setup

```bash
# Create environment
conda create -n vessel_seg python=3.10 -y
conda activate vessel_seg

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install SimpleITK nibabel numpy scipy scikit-image

# Download TopCoW 2024 from Zenodo
# https://zenodo.org/records/topcow2024  (check exact DOI on grand-challenge.org)
# Extract to /path/to/topcow2024/ with imagesTr/ and labelsTr/ subdirectories
```

## Running the Three Experiments

The project ablates three loss configurations. Run each as a separate experiment:

```bash
# Experiment 1: Baseline (Dice + Cross-Entropy)
python train.py --data_dir /path/to/topcow2024 --loss dice_ce

# Experiment 2: Dice + CE + soft-clDice (topology-preserving)
python train.py --data_dir /path/to/topcow2024 --loss dice_ce_cldice

# Experiment 3: Dice + CE + Skeleton Recall (efficient topology-preserving)
python train.py --data_dir /path/to/topcow2024 --loss dice_ce_skeleton
```

### Common overrides

```bash
# Smaller patches if GPU memory is tight (e.g., 8GB VRAM)
python train.py --data_dir ... --loss dice_ce --patch_size 96 96 96 --batch_size 1

# Fewer base filters to reduce memory
python train.py --data_dir ... --loss dice_ce --base_filters 16

# Disable mixed precision if encountering NaN
python train.py --data_dir ... --loss dice_ce --no_amp

# Shorter run for debugging
python train.py --data_dir ... --loss dice_ce --epochs 5
```

## Outputs

Each run creates a timestamped directory under `./runs/`:

```
runs/dice_ce_20260319_143022/
    config.json              # Frozen config for reproducibility
    training_log.json        # Per-epoch loss + validation metrics
    best_model.pth           # Checkpoint with highest validation Dice
    final_model.pth          # Checkpoint after last epoch
```

## Evaluation Metrics

Validation runs every 10 epochs with full-volume sliding window inference:

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| DSC | Volumetric overlap | Standard segmentation accuracy |
| clDice | Centerline overlap | Topology / connectivity preservation |
| HD95 | Worst-case boundary error | Surface accuracy |
| Betti-0 error | Connected component mismatch | Structural correctness |

## Key Design Decisions

**Patch-based training**: 3D CTA volumes are too large for GPU memory. We sample
128^3 patches with 33% foreground oversampling to handle the extreme class imbalance
(vessels are less than 3% of voxels).

**Deep supervision**: Loss is computed at multiple decoder scales with exponentially
decaying weights. This stabilizes training in deep 3D networks and was shown to be
critical for topology-aware losses (cbDice paper reported DSC dropping to zero
without it on some datasets).

**Mirror augmentation disabled**: TopCoW has left/right anatomical labels. Random
flips would swap left ICA with right ICA, corrupting the labels.

**Gaussian-weighted stitching**: At inference, overlapping patches are blended with
a Gaussian kernel centered on each patch. This eliminates the visible seam artifacts
that appear with naive averaging at patch boundaries.

## Things to Watch For

1. **GPU memory**: 128^3 patches with 32 base filters and batch size 2 needs roughly
   10-12GB VRAM. Reduce patch size or base filters if you hit OOM.

2. **clDice training instability**: The soft-skeletonization can produce gradients
   with high variance early in training. If loss spikes, try reducing cldice_alpha
   to 0.3, or start with pure Dice+CE for a warmup period before adding clDice.

3. **Skeleton Recall is slower per iteration** than Dice+CE because of CPU-based
   skeletonization, but it converges in fewer epochs. Net wall-clock time is
   usually comparable.

4. **Validation is slow**: Full-volume inference with sliding window takes several
   minutes per case. The 10-epoch interval is a reasonable tradeoff. For faster
   iteration during debugging, set a smaller validation split or skip validation
   with a quick patch-based proxy.

## Extending the Project

If time permits, potential additions (roughly ordered by effort):

- **Multi-class segmentation**: Set `num_classes=14` to segment individual CoW
  components instead of binary vessel mask. Requires adjusting the Dice loss to
  handle per-class averaging and the topology losses for multi-label skeletons.

- **CTA+MRA joint training**: Load both CTA and MRA volumes from TopCoW as
  separate training examples (same labels). This was a winning strategy in the
  TopCoW 2024 challenge.

- **Post-processing**: Connected component filtering to remove small spurious
  detections, largest-component selection, or morphological closing to bridge
  small gaps.

- **cbDice loss**: The centerline-boundary Dice (MICCAI 2024) adds radius
  awareness to clDice. More complex to implement but addresses diameter bias.
