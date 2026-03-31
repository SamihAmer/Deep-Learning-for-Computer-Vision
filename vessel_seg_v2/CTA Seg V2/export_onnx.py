"""
export_onnx.py -- Export trained 3D U-Net to ONNX format for SurgicalAR.

Produces a single-channel sigmoid output (1, 1, D, H, W) compatible with
SurgicalAR's MultiClass=false, OutputClassCount=1, LogitsConfidenceThreshold=0.5
configuration.

Usage:
    python export_onnx.py --checkpoint runs/dice_ce_20260325_080955/best_model.pth
    python export_onnx.py --checkpoint runs/dice_ce_20260325_080955/best_model.pth --output vessel_seg.onnx
    python export_onnx.py --checkpoint runs/dice_ce_20260325_080955/best_model.pth --raw-logits
"""

import argparse
import torch
from models.unet3d import UNet3D


class InferenceWrapper(torch.nn.Module):
    """Wraps the U-Net to return only the full-res logits for ONNX export.

    When sigmoid=True (default), converts the 2-channel output to a single
    channel vessel probability via sigmoid(vessel_logit - background_logit).
    This is mathematically equivalent to softmax channel 1 but has no
    ONNX opset compatibility issues.
    """

    def __init__(self, unet, sigmoid=True):
        super().__init__()
        self.unet = unet
        self.sigmoid = sigmoid

    def forward(self, x):
        logits = self.unet(x)[0]  # full-res output: (B, 2, D, H, W)
        if self.sigmoid:
            bg = logits[:, 0:1, :, :, :]   # (B, 1, D, H, W)
            vessel = logits[:, 1:2, :, :, :]  # (B, 1, D, H, W)
            return torch.sigmoid(vessel - bg)  # (B, 1, D, H, W)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Export 3D U-Net to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--raw-logits", action="store_true",
                        help="Export raw 2-channel logits instead of single-channel sigmoid")
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

    use_sigmoid = not args.raw_logits
    wrapper = InferenceWrapper(model, sigmoid=use_sigmoid)

    patch_size = tuple(cfg.get("patch_size", (128, 128, 128)))
    dummy = torch.randn(1, 1, *patch_size)

    if use_sigmoid:
        output_name = "output"
        print(f"Exporting single-channel sigmoid model (SurgicalAR compatible)")
    else:
        output_name = "logits"
        print(f"Exporting raw 2-channel logits model")

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={
            "input": {2: "D", 3: "H", 4: "W"},
            output_name: {2: "D", 3: "H", 4: "W"},
        },
        opset_version=args.opset,
    )

    # Verify output shape
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Output name: '{output_name}', shape: {list(out.shape)}")
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
