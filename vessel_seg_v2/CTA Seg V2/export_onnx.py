"""
export_onnx.py -- Export trained 3D U-Net to ONNX format.

Usage:
    python export_onnx.py --checkpoint runs/dice_ce_20260325_080955/best_model.pth
    python export_onnx.py --checkpoint runs/dice_ce_20260325_080955/best_model.pth --output vessel_seg.onnx
"""

import argparse
import torch
from models.unet3d import UNet3D


class InferenceWrapper(torch.nn.Module):
    """Wraps the U-Net to return only the full-res logits for ONNX export."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        return self.unet(x)[0]


def main():
    parser = argparse.ArgumentParser(description="Export 3D U-Net to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="vessel_seg.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    # Load checkpoint and recover config
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})

    model = UNet3D(
        in_channels=cfg.get("in_channels", 1),
        num_classes=cfg.get("num_classes", 2),
        base_filters=cfg.get("base_filters", 32),
        num_stages=cfg.get("num_stages", 5),
        deep_supervision=cfg.get("deep_supervision", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = InferenceWrapper(model)

    patch_size = tuple(cfg.get("patch_size", (128, 128, 128)))
    dummy = torch.randn(1, 1, *patch_size)

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {2: "D", 3: "H", 4: "W"},
            "logits": {2: "D", 3: "H", 4: "W"},
        },
        opset_version=args.opset,
    )

    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
