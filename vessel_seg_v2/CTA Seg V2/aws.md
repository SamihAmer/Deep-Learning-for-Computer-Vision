# Training on AWS GPU Instance

## 1. Launch Instance

**AMI**: Deep Learning Base AMI (Ubuntu 22.04) — `ami-0fed8ea08e4253cad`
Has CUDA + cuDNN preinstalled, no bloated frameworks.

**Instance type** (pick one):

| Instance | GPU | VRAM | RAM | $/hr (on-demand) | Est. time (300 ep) | Notes |
|----------|-----|------|-----|-------------------|--------------------|-------|
| g5.2xlarge | 1x A10G | 24 GB | 32 GB | ~$1.21 | ~3 hrs | Single GPU baseline |
| **g5.12xlarge** | **4x A10G** | **4x 24 GB** | **192 GB** | **~$5.67** | **~45 min** | **Recommended for fast training** |
| g5.48xlarge | 8x A10G | 8x 24 GB | 768 GB | ~$16.29 | ~25 min | Max speed |

**VRAM estimate** (5-stage 3D U-Net, 23.6M params, patch 128^3, AMP on):
- bs=4 per GPU: ~6-8 GB peak per GPU
- Multi-GPU: each GPU gets bs=4, effective batch = 4 x num_gpus

**Other settings**:
- At least 100 GB EBS storage (dataset ~15 GB + model checkpoints)
- Open SSH (port 22) in security group

## 2. Connect & Setup Environment

```bash
ssh -i your-key.pem ubuntu@<instance-ip>

# Create a venv (Base AMI has Python + CUDA, no conda)
python3 -m venv ~/vessel-env
source ~/vessel-env/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install SimpleITK nibabel numpy scipy scikit-image
```

## 3. Clone Repo

```bash
cd ~
git clone https://github.com/SamihAmer/Deep-Learning-for-Computer-Vision.git
cd Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2
```

## 4. Get Data

TopCoW 2024 dataset from Zenodo (check exact DOI on grand-challenge.org/topcow):

```bash
mkdir -p ~/data && cd ~/data

# Option A: Download directly on instance (if you have the Zenodo URL)
wget <zenodo-download-url> -O topcow2024.zip
unzip topcow2024.zip -d topcow2024

# Option B: Upload from local machine via scp
#   (run from your local terminal)
#   scp -i your-key.pem topcow2024.zip ubuntu@<instance-ip>:~/data/

# Option C: From S3 (if you've staged it there)
#   aws s3 cp s3://your-bucket/topcow2024.zip ~/data/
#   unzip ~/data/topcow2024.zip -d ~/data/topcow2024
```

Verify directory structure:
```bash
ls ~/data/topcow2024/imagesTr/  # should see topcow_ct_*.nii.gz
ls ~/data/topcow2024/labelsTr/  # should see matching label files
```

## 5. Run Preflight Check

```bash
cd ~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2

# Full check — verifies deps, GPU, data, spacing, forward pass (~1-2 min)
python preflight.py --data_dir ~/data/topcow2024

# Quick version (skip forward pass + full spacing survey)
python preflight.py --data_dir ~/data/topcow2024 --quick
```

All checks should show `[PASS]`. Fix any `[FAIL]` before proceeding.
Review `[WARN]` items — especially spacing variation warnings.

## 6. Train

### Single GPU (g5.2xlarge)

```bash
python train.py --data_dir ~/data/topcow2024 --loss dice_ce
```

### Multi-GPU (g5.12xlarge / g5.48xlarge)

```bash
# 4 GPUs (g5.12xlarge)
torchrun --nproc_per_node=4 train.py --data_dir ~/data/topcow2024 --loss dice_ce

# 8 GPUs (g5.48xlarge)
torchrun --nproc_per_node=8 train.py --data_dir ~/data/topcow2024 --loss dice_ce

# With topology loss
torchrun --nproc_per_node=4 train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice
```

DDP is auto-detected — `torchrun` sets environment variables, `python` runs single-GPU.
LR is automatically scaled by world size. Validation is parallelized across GPUs.

### Run in background (survives SSH disconnect):

```bash
# Using tmux (recommended)
tmux new -s train
torchrun --nproc_per_node=4 train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice
# Ctrl+B, D to detach — reconnect later with: tmux attach -t train

# Or using nohup
nohup torchrun --nproc_per_node=4 train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice \
    > train.log 2>&1 &
tail -f train.log
```

### Monitor training:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail the training log
tail -f runs/*/training_log.json | python -m json.tool
```

## 7. Download Results

From your local machine:

```bash
# Grab the best model + logs
scp -i your-key.pem -r ubuntu@<instance-ip>:~/Deep-Learning-for-Computer-Vision/vessel_seg_v2/CTA\ Seg\ V2/runs/ ./runs/
```

Or push to S3:
```bash
# On the instance
aws s3 sync runs/ s3://your-bucket/vessel-seg-runs/
```

## Notes

- **Spot instances**: g5.12xlarge spot is ~$1.70/hr (70% cheaper). Use with checkpointing — training auto-resumes via `--resume runs/<run>/latest_checkpoint.pth`
- **Multi-GPU**: Uses PyTorch DDP via `torchrun`. Training + validation are both parallelized. LR auto-scales with GPU count.
- **Resume training**: `torchrun --nproc_per_node=4 train.py --data_dir ~/data/topcow2024 --resume runs/<run>/latest_checkpoint.pth`
- **Don't forget to stop the instance when done.**
