"""
Generate a high-resolution 3D U-Net architecture diagram with proper U-shape.

The diagram shows:
  - Encoder on the left descending
  - Bottleneck at the bottom center
  - Decoder on the right ascending
  - Skip connections as horizontal arrows
  - Deep supervision heads
  - Block details (channels, operations)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "figures_v2")
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(-1, 17)
ax.set_ylim(-1.5, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("#0F1724")
ax.set_facecolor("#0F1724")

# Colors
ENC_COLORS = ["#2563EB", "#1D4ED8", "#1E40AF", "#1E3A8A", "#1E3A8A"]
DEC_COLORS = ["#D97706", "#B45309", "#92400E", "#78350F"]
SKIP_COLOR = "#10B981"
DS_COLOR = "#EF4444"
TEXT_COLOR = "#F8FAFC"
MUTED = "#94A3B8"
ARROW_COLOR = "#64748B"
BG_CARD = "#1E293B"

# Block dimensions - wider blocks, proper spacing for U-shape
BLOCK_W = 2.2
BLOCK_H_BASE = 0.9  # will vary by level for visual effect

# Encoder positions (left side, going DOWN)
# x stays on left, y descends
enc_positions = [
    (1.0, 8.0),    # Level 0: 32 ch
    (1.0, 6.2),    # Level 1: 64 ch
    (1.0, 4.4),    # Level 2: 128 ch
    (1.0, 2.6),    # Level 3: 256 ch
]

# Bottleneck (bottom center)
bottleneck_pos = (7.0, 0.8)

# Decoder positions (right side, going UP)
dec_positions = [
    (13.0, 2.6),   # Level 3: 256 ch
    (13.0, 4.4),   # Level 2: 128 ch
    (13.0, 6.2),   # Level 1: 64 ch
    (13.0, 8.0),   # Level 0: 32 ch
]

enc_channels = [32, 64, 128, 256]
bottleneck_ch = 512
dec_channels = [256, 128, 64, 32]

# =============================================================================
# Helper functions
# =============================================================================
def draw_block(ax, x, y, w, h, color, label, sublabel=None, alpha=0.95):
    """Draw a rounded rectangle block with label."""
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor="#475569",
                           linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(rect)

    # Channel label (big)
    ax.text(x + w/2, y + h/2 + 0.08, label, ha="center", va="center",
            fontsize=16, color=TEXT_COLOR, fontweight="bold", zorder=4,
            fontfamily="sans-serif")

    # Sublabel (small, below)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha="center", va="center",
                fontsize=8, color=MUTED, zorder=4, fontfamily="sans-serif")


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, style="-|>", lw=1.8,
               linestyle="-", connectionstyle=None, zorder=2):
    """Draw an arrow between two points."""
    props = dict(arrowstyle=style, color=color, lw=lw, linestyle=linestyle)
    if connectionstyle:
        props["connectionstyle"] = connectionstyle
    arrow = FancyArrowPatch((x1, y1), (x2, y2), **props, zorder=zorder)
    ax.add_patch(arrow)
    return arrow


# =============================================================================
# Draw encoder blocks
# =============================================================================
for i, ((x, y), ch) in enumerate(zip(enc_positions, enc_channels)):
    h = BLOCK_H_BASE + 0.1 * i  # slightly taller as we go deeper
    draw_block(ax, x, y, BLOCK_W, h, ENC_COLORS[i], f"{ch}", "Conv3D×2")

    # Downsampling arrow (to next level or to bottleneck)
    if i < 3:
        next_x, next_y = enc_positions[i + 1]
        next_h = BLOCK_H_BASE + 0.1 * (i + 1)
        draw_arrow(ax, x + BLOCK_W/2, y, x + BLOCK_W/2, next_y + next_h,
                   color="#3B82F6", style="-|>", lw=2.2)
        # Label
        ax.text(x + BLOCK_W/2 + 0.2, (y + next_y + next_h) / 2, "stride 2³",
                fontsize=7, color=MUTED, ha="left", va="center", rotation=90,
                fontfamily="sans-serif")

# Arrow from last encoder to bottleneck
last_enc_x, last_enc_y = enc_positions[3]
last_enc_h = BLOCK_H_BASE + 0.3
bx, by = bottleneck_pos
draw_arrow(ax, last_enc_x + BLOCK_W/2, last_enc_y,
           bx, by + BLOCK_H_BASE + 0.4,
           color="#3B82F6", style="-|>", lw=2.2,
           connectionstyle="arc3,rad=-0.15")
ax.text(3.5, 1.6, "stride 2³", fontsize=7, color=MUTED, ha="center",
        rotation=45, fontfamily="sans-serif")


# =============================================================================
# Draw bottleneck
# =============================================================================
bh = BLOCK_H_BASE + 0.4
draw_block(ax, bx, by, BLOCK_W + 0.6, bh, "#7C3AED",
           f"{bottleneck_ch}", "Bottleneck")

# Arrow from bottleneck to first decoder
first_dec_x, first_dec_y = dec_positions[0]
first_dec_h = BLOCK_H_BASE + 0.3
draw_arrow(ax, bx + BLOCK_W + 0.6, by + bh/2,
           first_dec_x, first_dec_y,
           color="#D97706", style="-|>", lw=2.2,
           connectionstyle="arc3,rad=-0.15")
ax.text(12.5, 1.6, "upsample 2³", fontsize=7, color=MUTED, ha="center",
        rotation=-45, fontfamily="sans-serif")


# =============================================================================
# Draw decoder blocks
# =============================================================================
for i, ((x, y), ch) in enumerate(zip(dec_positions, dec_channels)):
    h = BLOCK_H_BASE + 0.1 * (3 - i)  # mirror encoder heights
    draw_block(ax, x, y, BLOCK_W, h, DEC_COLORS[i], f"{ch}", "Conv3D×2")

    # Upsampling arrow (to next level)
    if i < 3:
        next_x, next_y = dec_positions[i + 1]
        next_h = BLOCK_H_BASE + 0.1 * (3 - (i + 1))
        draw_arrow(ax, x + BLOCK_W/2, y + h, x + BLOCK_W/2, next_y,
                   color="#F59E0B", style="-|>", lw=2.2)
        ax.text(x + BLOCK_W/2 + 0.2, (y + h + next_y) / 2, "upsample 2³",
                fontsize=7, color=MUTED, ha="left", va="center", rotation=90,
                fontfamily="sans-serif")


# =============================================================================
# Draw skip connections (horizontal arrows with curve)
# =============================================================================
for i in range(4):
    ex, ey = enc_positions[i]
    dx, dy = dec_positions[3 - i]
    eh = BLOCK_H_BASE + 0.1 * i
    dh = eh  # same height

    # Arrow from right side of encoder to left side of decoder
    y_mid = ey + eh / 2
    draw_arrow(ax, ex + BLOCK_W + 0.1, y_mid,
               dx - 0.1, dy + dh / 2,
               color=SKIP_COLOR, style="-|>", lw=2.0, linestyle="--")

    # "concat" label at midpoint
    mid_x = (ex + BLOCK_W + dx) / 2
    ax.text(mid_x, y_mid + 0.2, "concat", fontsize=7, color=SKIP_COLOR,
            ha="center", va="bottom", fontfamily="sans-serif",
            fontstyle="italic", alpha=0.8)


# =============================================================================
# Draw deep supervision heads
# =============================================================================
for i, ((x, y), ch) in enumerate(zip(dec_positions, dec_channels)):
    h = BLOCK_H_BASE + 0.1 * (3 - i)
    ds_x = x + BLOCK_W + 0.3
    ds_y = y + h / 2

    # Small DS output box
    ds_rect = FancyBboxPatch((ds_x + 0.5, ds_y - 0.2), 0.5, 0.4,
                              boxstyle="round,pad=0.04",
                              facecolor="#7F1D1D", edgecolor=DS_COLOR,
                              linewidth=1.0, alpha=0.8, zorder=3)
    ax.add_patch(ds_rect)
    ax.text(ds_x + 0.75, ds_y, "2", ha="center", va="center",
            fontsize=8, color=DS_COLOR, fontweight="bold", zorder=4)

    # Arrow
    draw_arrow(ax, x + BLOCK_W, ds_y, ds_x + 0.5, ds_y,
               color=DS_COLOR, style="-|>", lw=1.2)

    # Weight label
    weights = ["1.0", "0.5", "0.25", "0.125"]
    ax.text(ds_x + 0.75, ds_y - 0.35, f"w={weights[i]}", fontsize=6,
            color=DS_COLOR, ha="center", fontfamily="sans-serif", alpha=0.7)


# =============================================================================
# Input / Output labels
# =============================================================================
# Input arrow
inp_x, inp_y = enc_positions[0]
inp_h = BLOCK_H_BASE
ax.annotate("", xy=(inp_x + BLOCK_W/2, inp_y + inp_h + 0.1),
            xytext=(inp_x + BLOCK_W/2, inp_y + inp_h + 0.8),
            arrowprops=dict(arrowstyle="-|>", color=TEXT_COLOR, lw=1.5))
ax.text(inp_x + BLOCK_W/2, inp_y + inp_h + 1.0, "Input\n1×128³",
        ha="center", va="bottom", fontsize=11, color=TEXT_COLOR,
        fontfamily="sans-serif", fontweight="bold")

# Output arrow
out_x, out_y = dec_positions[3]
out_h = BLOCK_H_BASE
ax.text(out_x + BLOCK_W/2, out_y + out_h + 1.0, "Output\n2×128³",
        ha="center", va="bottom", fontsize=11, color=TEXT_COLOR,
        fontfamily="sans-serif", fontweight="bold")
ax.annotate("", xy=(out_x + BLOCK_W/2, out_y + out_h + 0.1),
            xytext=(out_x + BLOCK_W/2, out_y + out_h + 0.8),
            arrowprops=dict(arrowstyle="-|>", color=TEXT_COLOR, lw=1.5))


# =============================================================================
# Labels for encoder / decoder paths
# =============================================================================
ax.text(0.3, 5.0, "ENCODER", fontsize=11, color="#3B82F6", fontweight="bold",
        rotation=90, ha="center", va="center", fontfamily="sans-serif", alpha=0.6)
ax.text(15.8, 5.0, "DECODER", fontsize=11, color="#F59E0B", fontweight="bold",
        rotation=-90, ha="center", va="center", fontfamily="sans-serif", alpha=0.6)


# =============================================================================
# Legend
# =============================================================================
legend_y = -0.8
legend_items = [
    ("Conv Block", "2× [Conv3D(3³) → InstanceNorm → LeakyReLU(0.01)] + Residual", "#3B82F6"),
    ("Downsampling", "Strided Conv3D(2³, stride=2) — learnable, no max-pool", "#64748B"),
    ("Skip Connection", "Concatenation of encoder features with decoder", SKIP_COLOR),
    ("Deep Supervision", "1×1 Conv3D → 2-class output at each decoder level", DS_COLOR),
]

for i, (name, desc, color) in enumerate(legend_items):
    lx = 0.5 + i * 4.1
    # Color indicator
    rect = FancyBboxPatch((lx, legend_y), 0.3, 0.25,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor="none", alpha=0.8)
    ax.add_patch(rect)
    ax.text(lx + 0.45, legend_y + 0.2, name, fontsize=8, color=TEXT_COLOR,
            fontweight="bold", va="center", fontfamily="sans-serif")
    ax.text(lx + 0.45, legend_y - 0.05, desc, fontsize=6.5, color=MUTED,
            va="center", fontfamily="sans-serif")


# =============================================================================
# Title
# =============================================================================
ax.text(8.0, 9.7, "3D U-Net Architecture  |  24.9M Parameters",
        ha="center", va="center", fontsize=14, color=TEXT_COLOR,
        fontweight="bold", fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD, edgecolor="#334155"))


# =============================================================================
# Save
# =============================================================================
plt.tight_layout(pad=0.5)
fig.savefig(os.path.join(OUTDIR, "unet3d_architecture.png"), dpi=250,
            facecolor=fig.get_facecolor(), edgecolor="none")
fig.savefig(os.path.join(OUTDIR, "unet3d_architecture.pdf"),
            facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved unet3d_architecture.png + .pdf (v2)")
