# Training on AWS GPU Instance

## 1. Launch Instance

**AMI**: Deep Learning Base AMI (Ubuntu 22.04) — `ami-0fed8ea08e4253cad`
Has CUDA + cuDNN preinstalled, no bloated frameworks.

**Instance type** (pick one):

| Instance | GPU | VRAM | $/hr (on-demand) | $/hr (spot) | Notes |
|----------|-----|------|-------------------|-------------|-------|
| **g5.2xlarge** | A10G | 24 GB | ~$1.21 | ~$0.36 | Recommended. 32 GB system RAM for preloading all volumes |
| g6.xlarge | L4 | 24 GB | ~$0.80 | ~$0.25 | Newer Ada arch, similar perf |
| g4dn.xlarge | T4 | 16 GB | ~$0.53 | ~$0.16 | Budget option, bs=4 only |

**VRAM estimate** (5-stage 3D U-Net, 23.6M params, patch 128^3, AMP on):
- bs=4: ~6-8 GB peak (comfortable on 24 GB, tight on 16 GB)
- bs=8: ~12-14 GB peak (needs 24 GB)

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

```bash
# Baseline (Dice + CE)
python train.py --data_dir ~/data/topcow2024 --loss dice_ce

# With topology loss (recommended)
python train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice

# Or skeleton recall (faster topology loss)
python train.py --data_dir ~/data/topcow2024 --loss dice_ce_skeleton
```

### Run in background (survives SSH disconnect):

```bash
# Using tmux
tmux new -s train
python train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice
# Ctrl+B, D to detach — reconnect later with: tmux attach -t train

# Or using nohup
nohup python train.py --data_dir ~/data/topcow2024 --loss dice_ce_cldice \
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

- **Spot instances**: g5.xlarge spot is ~$0.30/hr (70% cheaper). Use with checkpointing — training auto-resumes via `--resume runs/<run>/latest_checkpoint.pth`
- **Multi-GPU**: Not currently supported (single GPU training). g5.xlarge is sufficient.
- **Expected training time**: ~8-12 hours for 300 epochs on A10G (24GB), faster on V100/A100.
- **Don't forget to stop the instance when done.**
