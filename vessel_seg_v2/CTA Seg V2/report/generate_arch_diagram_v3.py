"""
Generate high-resolution 3D U-Net architecture diagram — LIGHT THEME.
Proper U-shape, white background, dark text, same accent colors.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "figures_v3")
os.makedirs(OUTDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(-1, 17)
ax.set_ylim(-1.5, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("#FAFBFD")
ax.set_facecolor("#FAFBFD")

# Colors — light theme
ENC_COLORS = ["#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF"]
BOTTLENECK_COLOR = "#7C3AED"
DEC_COLORS = ["#F59E0B", "#D97706", "#B45309", "#92400E"]
SKIP_COLOR = "#059669"
DS_COLOR = "#DC2626"
TEXT_DARK = "#1E293B"
TEXT_MID = "#475569"
TEXT_MUTED = "#94A3B8"
CARD_BORDER = "#CBD5E1"

BLOCK_W = 2.2
BLOCK_H_BASE = 0.9

enc_positions = [(1.0, 8.0), (1.0, 6.2), (1.0, 4.4), (1.0, 2.6)]
bottleneck_pos = (7.0, 0.8)
dec_positions = [(13.0, 2.6), (13.0, 4.4), (13.0, 6.2), (13.0, 8.0)]

enc_channels = [32, 64, 128, 256]
bottleneck_ch = 512
dec_channels = [256, 128, 64, 32]


def draw_block(ax, x, y, w, h, color, label, sublabel=None):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor="white",
                           linewidth=2, alpha=0.92, zorder=3)
    ax.add_patch(rect)
    # White text on colored blocks
    ax.text(x + w/2, y + h/2 + 0.08, label, ha="center", va="center",
            fontsize=16, color="white", fontweight="bold", zorder=4, fontfamily="sans-serif")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha="center", va="center",
                fontsize=8, color=(1, 1, 1, 0.7), zorder=4, fontfamily="sans-serif")


def draw_arrow(ax, x1, y1, x2, y2, color=TEXT_MUTED, style="-|>", lw=1.8,
               linestyle="-", connectionstyle=None, zorder=2):
    props = dict(arrowstyle=style, color=color, lw=lw, linestyle=linestyle)
    if connectionstyle:
        props["connectionstyle"] = connectionstyle
    arrow = FancyArrowPatch((x1, y1), (x2, y2), **props, zorder=zorder)
    ax.add_patch(arrow)


# Draw encoder
for i, ((x, y), ch) in enumerate(zip(enc_positions, enc_channels)):
    h = BLOCK_H_BASE + 0.1 * i
    draw_block(ax, x, y, BLOCK_W, h, ENC_COLORS[i], f"{ch}", "Conv3D×2")

    if i < 3:
        next_x, next_y = enc_positions[i + 1]
        next_h = BLOCK_H_BASE + 0.1 * (i + 1)
        draw_arrow(ax, x + BLOCK_W/2, y, x + BLOCK_W/2, next_y + next_h,
                   color="#3B82F6", style="-|>", lw=2.2)
        ax.text(x + BLOCK_W/2 + 0.2, (y + next_y + next_h) / 2, "stride 2³",
                fontsize=7, color=TEXT_MUTED, ha="left", va="center", rotation=90,
                fontfamily="sans-serif")

# Encoder to bottleneck
last_enc_x, last_enc_y = enc_positions[3]
last_enc_h = BLOCK_H_BASE + 0.3
bx, by = bottleneck_pos
draw_arrow(ax, last_enc_x + BLOCK_W/2, last_enc_y,
           bx, by + BLOCK_H_BASE + 0.4,
           color="#3B82F6", style="-|>", lw=2.2,
           connectionstyle="arc3,rad=-0.15")
ax.text(3.5, 1.6, "stride 2³", fontsize=7, color=TEXT_MUTED, ha="center",
        rotation=45, fontfamily="sans-serif")

# Bottleneck
bh = BLOCK_H_BASE + 0.4
draw_block(ax, bx, by, BLOCK_W + 0.6, bh, BOTTLENECK_COLOR, f"{bottleneck_ch}", "Bottleneck")

# Bottleneck to decoder
first_dec_x, first_dec_y = dec_positions[0]
draw_arrow(ax, bx + BLOCK_W + 0.6, by + bh/2,
           first_dec_x, first_dec_y,
           color="#D97706", style="-|>", lw=2.2,
           connectionstyle="arc3,rad=-0.15")
ax.text(12.5, 1.6, "upsample 2³", fontsize=7, color=TEXT_MUTED, ha="center",
        rotation=-45, fontfamily="sans-serif")

# Decoder
for i, ((x, y), ch) in enumerate(zip(dec_positions, dec_channels)):
    h = BLOCK_H_BASE + 0.1 * (3 - i)
    draw_block(ax, x, y, BLOCK_W, h, DEC_COLORS[i], f"{ch}", "Conv3D×2")

    if i < 3:
        next_x, next_y = dec_positions[i + 1]
        next_h = BLOCK_H_BASE + 0.1 * (3 - (i + 1))
        draw_arrow(ax, x + BLOCK_W/2, y + h, x + BLOCK_W/2, next_y,
                   color="#F59E0B", style="-|>", lw=2.2)
        ax.text(x + BLOCK_W/2 + 0.2, (y + h + next_y) / 2, "upsample 2³",
                fontsize=7, color=TEXT_MUTED, ha="left", va="center", rotation=90,
                fontfamily="sans-serif")

# Skip connections
for i in range(4):
    ex, ey = enc_positions[i]
    dx, dy = dec_positions[3 - i]
    eh = BLOCK_H_BASE + 0.1 * i
    dh = eh
    y_mid = ey + eh / 2
    draw_arrow(ax, ex + BLOCK_W + 0.1, y_mid,
               dx - 0.1, dy + dh / 2,
               color=SKIP_COLOR, style="-|>", lw=2.0, linestyle="--")
    mid_x = (ex + BLOCK_W + dx) / 2
    ax.text(mid_x, y_mid + 0.2, "concat", fontsize=7, color=SKIP_COLOR,
            ha="center", va="bottom", fontfamily="sans-serif", fontstyle="italic", alpha=0.8)

# Deep supervision heads
for i, ((x, y), ch) in enumerate(zip(dec_positions, dec_channels)):
    h = BLOCK_H_BASE + 0.1 * (3 - i)
    ds_x = x + BLOCK_W + 0.3
    ds_y = y + h / 2

    ds_rect = FancyBboxPatch((ds_x + 0.5, ds_y - 0.2), 0.5, 0.4,
                              boxstyle="round,pad=0.04",
                              facecolor="#FEE2E2", edgecolor=DS_COLOR,
                              linewidth=1.2, alpha=0.9, zorder=3)
    ax.add_patch(ds_rect)
    ax.text(ds_x + 0.75, ds_y, "2", ha="center", va="center",
            fontsize=8, color=DS_COLOR, fontweight="bold", zorder=4)
    draw_arrow(ax, x + BLOCK_W, ds_y, ds_x + 0.5, ds_y,
               color=DS_COLOR, style="-|>", lw=1.2)
    weights = ["1.0", "0.5", "0.25", "0.125"]
    ax.text(ds_x + 0.75, ds_y - 0.35, f"w={weights[i]}", fontsize=6,
            color=DS_COLOR, ha="center", fontfamily="sans-serif", alpha=0.8)

# Input / Output labels
inp_x, inp_y = enc_positions[0]
inp_h = BLOCK_H_BASE
ax.annotate("", xy=(inp_x + BLOCK_W/2, inp_y + inp_h + 0.1),
            xytext=(inp_x + BLOCK_W/2, inp_y + inp_h + 0.8),
            arrowprops=dict(arrowstyle="-|>", color=TEXT_DARK, lw=1.5))
ax.text(inp_x + BLOCK_W/2, inp_y + inp_h + 1.0, "Input\n1×128³",
        ha="center", va="bottom", fontsize=11, color=TEXT_DARK,
        fontfamily="sans-serif", fontweight="bold")

out_x, out_y = dec_positions[3]
out_h = BLOCK_H_BASE
ax.text(out_x + BLOCK_W/2, out_y + out_h + 1.0, "Output\n2×128³",
        ha="center", va="bottom", fontsize=11, color=TEXT_DARK,
        fontfamily="sans-serif", fontweight="bold")
ax.annotate("", xy=(out_x + BLOCK_W/2, out_y + out_h + 0.1),
            xytext=(out_x + BLOCK_W/2, out_y + out_h + 0.8),
            arrowprops=dict(arrowstyle="-|>", color=TEXT_DARK, lw=1.5))

# Path labels
ax.text(0.3, 5.0, "ENCODER", fontsize=11, color="#3B82F6", fontweight="bold",
        rotation=90, ha="center", va="center", fontfamily="sans-serif", alpha=0.5)
ax.text(15.8, 5.0, "DECODER", fontsize=11, color="#D97706", fontweight="bold",
        rotation=-90, ha="center", va="center", fontfamily="sans-serif", alpha=0.5)

# Legend
legend_y = -0.8
legend_items = [
    ("Conv Block", "2× [Conv3D(3³) → InstanceNorm → LeakyReLU(0.01)] + Residual", "#3B82F6"),
    ("Downsampling", "Strided Conv3D(2³, stride=2) — learnable, no max-pool", "#64748B"),
    ("Skip Connection", "Concatenation of encoder features with decoder", SKIP_COLOR),
    ("Deep Supervision", "1×1 Conv3D → 2-class output at each decoder level", DS_COLOR),
]
for i, (name, desc, color) in enumerate(legend_items):
    lx = 0.5 + i * 4.1
    rect = FancyBboxPatch((lx, legend_y), 0.3, 0.25,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor="none", alpha=0.85)
    ax.add_patch(rect)
    ax.text(lx + 0.45, legend_y + 0.2, name, fontsize=8, color=TEXT_DARK,
            fontweight="bold", va="center", fontfamily="sans-serif")
    ax.text(lx + 0.45, legend_y - 0.05, desc, fontsize=6.5, color=TEXT_MID,
            va="center", fontfamily="sans-serif")

# Title
ax.text(8.0, 9.7, "3D U-Net Architecture  |  24.9M Parameters",
        ha="center", va="center", fontsize=14, color=TEXT_DARK,
        fontweight="bold", fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#BFDBFE"))

plt.tight_layout(pad=0.5)
fig.savefig(os.path.join(OUTDIR, "unet3d_architecture.png"), dpi=250,
            facecolor=fig.get_facecolor(), edgecolor="none")
fig.savefig(os.path.join(OUTDIR, "unet3d_architecture.pdf"),
            facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved unet3d_architecture (light theme)")
