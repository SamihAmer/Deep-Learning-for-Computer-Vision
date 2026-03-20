"""
3D U-Net with optional deep supervision.

Architecture follows the nnU-Net design philosophy:
  - Instance normalization + LeakyReLU
  - Residual connections within each stage
  - Strided convolutions for downsampling (no max-pool)
  - Transposed convolutions for upsampling
  - Deep supervision heads at each decoder level
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with residual connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

        # 1x1 projection for residual if channel mismatch
        self.residual = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)) + res)
        return x


class Encoder(nn.Module):
    """Contracting path: ConvBlock + strided conv downsampling at each stage."""

    def __init__(self, in_channels: int, base_filters: int, num_stages: int):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = in_channels
        for i in range(num_stages):
            ch_out = base_filters * (2 ** i)
            self.stages.append(ConvBlock(ch_in, ch_out))
            if i < num_stages - 1:
                self.downsamples.append(
                    nn.Conv3d(ch_out, ch_out, kernel_size=2, stride=2, bias=False)
                )
            ch_in = ch_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of feature maps from each stage (high-res first)."""
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return features


class Decoder(nn.Module):
    """Expanding path with skip connections and optional deep supervision."""

    def __init__(
        self,
        base_filters: int,
        num_stages: int,
        num_classes: int,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.ds_heads = nn.ModuleList()  # deep supervision output heads

        for i in range(num_stages - 2, -1, -1):
            ch_below = base_filters * (2 ** (i + 1))
            ch_skip = base_filters * (2 ** i)

            self.upsamples.append(
                nn.ConvTranspose3d(ch_below, ch_skip, kernel_size=2, stride=2, bias=False)
            )
            self.stages.append(ConvBlock(ch_skip * 2, ch_skip))  # concat doubles channels

            if deep_supervision:
                self.ds_heads.append(nn.Conv3d(ch_skip, num_classes, kernel_size=1))

        # Final output head (highest resolution)
        if not deep_supervision:
            ch_final = base_filters
            self.final_head = nn.Conv3d(ch_final, num_classes, kernel_size=1)

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            encoder_features: list from encoder, index 0 = highest res, -1 = bottleneck

        Returns:
            If deep_supervision: list of logit maps at each decoder level (highest res first)
            Otherwise: list with single output at full resolution
        """
        x = encoder_features[-1]
        outputs = []

        for i, (up, stage) in enumerate(zip(self.upsamples, self.stages)):
            skip = encoder_features[-(i + 2)]
            x = up(x)

            # Handle size mismatches from odd input dimensions
            if x.shape != skip.shape:
                x = _match_size(x, skip)

            x = torch.cat([x, skip], dim=1)
            x = stage(x)

            if self.deep_supervision:
                outputs.append(self.ds_heads[i](x))

        if self.deep_supervision:
            outputs.reverse()  # highest resolution first
            return outputs
        else:
            return [self.final_head(x)]


def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Crop or pad x to match target spatial dimensions."""
    diffs = [t - s for s, t in zip(x.shape[2:], target.shape[2:])]
    if all(d == 0 for d in diffs):
        return x
    # simple center crop/pad
    slices = [slice(None), slice(None)]  # batch, channel
    for d in diffs:
        if d > 0:
            slices.append(slice(0, x.shape[len(slices)]))  # will pad below
        elif d < 0:
            start = (-d) // 2
            slices.append(slice(start, start + target.shape[len(slices)]))
        else:
            slices.append(slice(None))
    x = x[tuple(slices)]
    # pad if needed
    pad = []
    for d in reversed(diffs):
        if d > 0:
            pad.extend([0, d])
        else:
            pad.extend([0, 0])
    if any(p > 0 for p in pad):
        x = nn.functional.pad(x, pad)
    return x


class UNet3D(nn.Module):
    """
    Full 3D U-Net.

    Args:
        in_channels: input channels (1 for CT)
        num_classes: output classes (2 for binary vessel seg)
        base_filters: channels in first encoder stage
        num_stages: total encoder depth
        deep_supervision: return multi-scale outputs for deep supervision loss
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        num_stages: int = 5,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_filters, num_stages)
        self.decoder = Decoder(base_filters, num_stages, num_classes, deep_supervision)
        self.deep_supervision = deep_supervision

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out")
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List of logit tensors. If deep_supervision, index 0 is full-res,
            subsequent indices are progressively lower resolution.
            If not deep_supervision, single-element list.
        """
        features = self.encoder(x)
        return self.decoder(features)


# ─── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet3D(in_channels=1, num_classes=2, base_filters=32, num_stages=5)
    x = torch.randn(1, 1, 128, 128, 128)
    outputs = model(x)
    for i, o in enumerate(outputs):
        print(f"Output {i}: {o.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")
