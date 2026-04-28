"""
expand_checkpoint_to_3class.py — Expand a 2-class vessel checkpoint to 3 classes (bg, artery, vein).

For each 1x1x1 segmentation head (final_head + ds_heads[i]), the new 3-channel
weight tensor is initialized by duplicating the old "vessel" channel:
    new[0] <- old[0]   (background unchanged)
    new[1] <- old[1]   (artery starts as a copy of the vessel detector)
    new[2] <- old[1]   (vein also starts from the vessel detector)

Both foreground channels begin identical and differentiate during fine-tuning.

Usage:
    python expand_checkpoint_to_3class.py \
        --in_checkpoint runs/skeleton_topbrain.../best_model.pth \
        --out_checkpoint runs/skeleton_topbrain_3class_init.pth
"""

import argparse
import re
import torch


# Matches "...ds_heads.N.weight|bias" or "...final_head.weight|bias" (any prefix).
HEAD_KEY_RE = re.compile(r"(^|\.)((ds_heads\.\d+)|(final_head))\.(weight|bias)$")


def expand_head(old: torch.Tensor) -> torch.Tensor:
    """Expand a 2-class head tensor to 3 classes by duplicating the vessel row.

    Accepts weight (2, C_in, 1, 1, 1) or bias (2,); returns the same shape with
    first dim = 3.
    """
    if old.shape[0] != 2:
        raise ValueError(
            f"Expected first dimension == 2 (2-class head), got shape {tuple(old.shape)}"
        )
    bg = old[0:1]
    vessel = old[1:2]
    return torch.cat([bg, vessel, vessel.clone()], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_checkpoint", required=True)
    ap.add_argument("--out_checkpoint", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.in_checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    new_state = {}
    expanded = []
    for key, tensor in state.items():
        if HEAD_KEY_RE.search(key):
            new_state[key] = expand_head(tensor)
            expanded.append((key, tuple(tensor.shape), tuple(new_state[key].shape)))
        else:
            new_state[key] = tensor

    if not expanded:
        raise RuntimeError(
            "No ds_heads.*/final_head.* keys found in checkpoint — nothing to expand. "
            f"Keys present: {list(state.keys())[:8]} ..."
        )

    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = new_state
    else:
        ckpt = new_state

    # Update config if present
    if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
        ckpt["config"] = dict(ckpt["config"])
        ckpt["config"]["num_classes"] = 3

    torch.save(ckpt, args.out_checkpoint)

    print(f"Expanded checkpoint written to {args.out_checkpoint}")
    print("Expanded heads:")
    for k, old_shape, new_shape in expanded:
        print(f"  {k}: {old_shape} -> {new_shape}")


if __name__ == "__main__":
    main()
