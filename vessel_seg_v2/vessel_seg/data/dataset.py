"""
Dataset and patch sampler for TopCoW CTA volumes.

Expected directory layout after downloading from Zenodo:
    topcow2024/
        imagesTr/
            topcow_ct_000.nii.gz
            topcow_ct_001.nii.gz
            ...
        labelsTr/
            topcow_ct_000.nii.gz
            ...

Adjust paths in _discover_cases() if your layout differs.
"""

import os
import glob
import random
from typing import Tuple, List, Optional, Dict

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


# ─── Volume I/O ──────────────────────────────────────────────────────────────

def load_nifti(path: str) -> Tuple[np.ndarray, dict]:
    """Load a NIfTI file and return (array, metadata_dict)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: (D, H, W)
    meta = {
        "spacing": img.GetSpacing()[::-1],     # (D, H, W) order
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "path": path,
    }
    return arr.astype(np.float32), meta


def preprocess_ct(volume: np.ndarray, hu_window: Tuple[int, int]) -> np.ndarray:
    """Clip HU values and normalize to [0, 1]."""
    lo, hi = hu_window
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-8)
    return volume


def binarize_labels(label: np.ndarray) -> np.ndarray:
    """Collapse multi-class CoW labels into binary vessel mask.
    Set to passthrough (return label unchanged) for multi-class training.
    """
    return (label > 0).astype(np.float32)


# ─── Patch sampling ──────────────────────────────────────────────────────────

def sample_foreground_center(label: np.ndarray) -> Tuple[int, int, int]:
    """Return a random voxel coordinate where label > 0."""
    coords = np.argwhere(label > 0)
    idx = random.randint(0, len(coords) - 1)
    return tuple(coords[idx])


def sample_random_center(shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return a uniformly random voxel coordinate within volume."""
    return tuple(random.randint(0, s - 1) for s in shape)


def extract_patch(
    volume: np.ndarray,
    label: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a patch centered at `center`, with zero-padding for boundary cases.
    Returns (image_patch, label_patch) each of shape patch_size.
    """
    d, h, w = volume.shape
    pd, ph, pw = patch_size

    # compute start/end with clamping
    starts, ends = [], []
    pad_before, pad_after = [], []
    for c, ps, dim in zip(center, patch_size, volume.shape):
        s = c - ps // 2
        e = s + ps
        pb = max(0, -s)
        pa = max(0, e - dim)
        s = max(0, s)
        e = min(dim, e)
        starts.append(s)
        ends.append(e)
        pad_before.append(pb)
        pad_after.append(pa)

    img_crop = volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    lbl_crop = label[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # pad if needed
    if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
        padding = list(zip(pad_before, pad_after))
        img_crop = np.pad(img_crop, padding, mode="constant", constant_values=0)
        lbl_crop = np.pad(lbl_crop, padding, mode="constant", constant_values=0)

    return img_crop, lbl_crop


# ─── Augmentation ────────────────────────────────────────────────────────────

class PatchAugmentor:
    """Simple 3D augmentations applied on-the-fly to numpy patches."""

    def __init__(self, cfg: dict):
        self.gamma_range = cfg.get("aug_gamma_range", (0.7, 1.5))
        self.mirror = cfg.get("aug_mirror", False)

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Random gamma correction
        gamma = random.uniform(*self.gamma_range)
        image = np.power(image.clip(0, 1), gamma)

        # Random axis flips (only if mirror is enabled; disabled for anatomical labels)
        if self.mirror:
            for ax in range(3):
                if random.random() > 0.5:
                    image = np.flip(image, axis=ax).copy()
                    label = np.flip(label, axis=ax).copy()

        # Random 90-degree rotation in axial plane
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(1, 2)).copy()

        return image, label


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TopCoWPatchDataset(Dataset):
    """
    Yields random patches from preloaded TopCoW CTA volumes.

    Each __getitem__ call samples one patch from a random volume.
    The dataset length is (num_volumes * patches_per_volume) so that
    one epoch roughly covers all volumes with controlled patch count.
    """

    def __init__(
        self,
        case_list: List[Dict[str, str]],
        cfg: dict,
        augment: bool = True,
    ):
        self.cases = case_list
        self.cfg = cfg
        self.patch_size = tuple(cfg["patch_size"])
        self.patches_per_volume = cfg["patches_per_volume"]
        self.foreground_ratio = cfg["foreground_ratio"]
        self.hu_window = tuple(cfg["hu_window"])
        self.augmentor = PatchAugmentor(cfg) if augment else None
        self.binary = cfg["num_classes"] == 2

        # Preload volumes into memory (TopCoW is small enough at ~125 volumes)
        # For larger datasets, switch to lazy loading with caching
        self.volumes = []
        self.labels = []
        print(f"Loading {len(case_list)} volumes into memory...")
        for case in case_list:
            vol, _ = load_nifti(case["image"])
            lbl, _ = load_nifti(case["label"])
            vol = preprocess_ct(vol, self.hu_window)
            if self.binary:
                lbl = binarize_labels(lbl)
            self.volumes.append(vol)
            self.labels.append(lbl)
        print("Done.")

    def __len__(self):
        return len(self.volumes) * self.patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx % len(self.volumes)
        volume = self.volumes[vol_idx]
        label = self.labels[vol_idx]

        # Decide foreground vs random center
        if random.random() < self.foreground_ratio and label.sum() > 0:
            center = sample_foreground_center(label)
        else:
            center = sample_random_center(volume.shape)

        img_patch, lbl_patch = extract_patch(volume, label, center, self.patch_size)

        # Augmentation
        if self.augmentor is not None:
            img_patch, lbl_patch = self.augmentor(img_patch, lbl_patch)

        # To tensor: add channel dim -> (1, D, H, W)
        img_tensor = torch.from_numpy(img_patch[np.newaxis].copy()).float()
        lbl_tensor = torch.from_numpy(lbl_patch.copy()).long()

        return img_tensor, lbl_tensor


# ─── Helpers ─────────────────────────────────────────────────────────────────

def discover_cases(data_dir: str) -> List[Dict[str, str]]:
    """Find all image/label pairs in the TopCoW directory."""
    image_dir = os.path.join(data_dir, "imagesTr")
    label_dir = os.path.join(data_dir, "labelsTr")

    images = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    cases = []
    for img_path in images:
        basename = os.path.basename(img_path)
        lbl_path = os.path.join(label_dir, basename)
        if os.path.exists(lbl_path):
            cases.append({"image": img_path, "label": lbl_path})
        else:
            print(f"Warning: no label found for {basename}, skipping.")
    return cases


def train_val_split(
    cases: List[Dict[str, str]], ratio: float = 0.8, seed: int = 42
) -> Tuple[List, List]:
    """Deterministic random split into train and validation sets."""
    rng = random.Random(seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * ratio)
    return shuffled[:n_train], shuffled[n_train:]
