"""Generate a schematic U-Net architecture diagram for the report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect("equal")
ax.axis("off")

# Encoder blocks (left side, descending)
enc_x = [0.5, 2.0, 3.5, 5.0, 6.5]
enc_y = [5.5, 4.5, 3.5, 2.5, 1.5]
enc_w = [1.2, 1.2, 1.2, 1.2, 1.2]
enc_h = [0.7, 0.7, 0.7, 0.7, 0.7]
enc_ch = [32, 64, 128, 256, 512]
enc_colors = ["#4A90D9", "#357ABD", "#2868A6", "#1B5690", "#0F4479"]

# Decoder blocks (right side, ascending)
dec_x = [8.0, 6.5, 5.0, 3.5]
dec_y = [2.5, 3.5, 4.5, 5.5]
dec_ch = [256, 128, 64, 32]
dec_colors = ["#E67E22", "#D35400", "#BA4A00", "#A04000"]

# Draw encoder
for i in range(5):
    rect = mpatches.FancyBboxPatch((enc_x[i], enc_y[i]), enc_w[i], enc_h[i],
                                     boxstyle="round,pad=0.05",
                                     facecolor=enc_colors[i], edgecolor="black", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(enc_x[i] + enc_w[i]/2, enc_y[i] + enc_h[i]/2,
            f"{enc_ch[i]}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Downsampling arrows
    if i < 4:
        ax.annotate("", xy=(enc_x[i+1], enc_y[i+1] + enc_h[i+1]),
                    xytext=(enc_x[i] + enc_w[i], enc_y[i]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

# Draw decoder
for i in range(4):
    rect = mpatches.FancyBboxPatch((dec_x[i], dec_y[i]), enc_w[0], enc_h[0],
                                     boxstyle="round,pad=0.05",
                                     facecolor=dec_colors[i], edgecolor="black", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(dec_x[i] + enc_w[0]/2, dec_y[i] + enc_h[0]/2,
            f"{dec_ch[i]}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Upsampling arrows
    if i < 3:
        ax.annotate("", xy=(dec_x[i+1], dec_y[i+1]),
                    xytext=(dec_x[i], dec_y[i] + enc_h[0]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

# Bottleneck to decoder
ax.annotate("", xy=(dec_x[0], enc_y[3]),
            xytext=(enc_x[4] + enc_w[4], enc_y[4] + enc_h[4]),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

# Skip connections (dashed)
skip_enc = [(enc_x[0] + enc_w[0], enc_y[0] + enc_h[0]/2),
            (enc_x[1] + enc_w[1], enc_y[1] + enc_h[1]/2),
            (enc_x[2] + enc_w[2], enc_y[2] + enc_h[2]/2),
            (enc_x[3] + enc_w[3], enc_y[3] + enc_h[3]/2)]
skip_dec = [(dec_x[3], dec_y[3] + enc_h[0]/2),
            (dec_x[2], dec_y[2] + enc_h[0]/2),
            (dec_x[1], dec_y[1] + enc_h[0]/2),
            (dec_x[0], dec_y[0] + enc_h[0]/2)]

for (x1, y1), (x2, y2) in zip(skip_enc, skip_dec):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8,
                               linestyle="dashed", connectionstyle="arc3,rad=0.3"))

# Deep supervision outputs (small arrows going right from decoder)
for i in range(4):
    ax.annotate("DS", xy=(dec_x[i] + enc_w[0] + 0.4, dec_y[i] + enc_h[0]/2),
                xytext=(dec_x[i] + enc_w[0] + 0.05, dec_y[i] + enc_h[0]/2),
                fontsize=5, color="red", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.5))

# Labels
ax.text(0.5, 6.3, "Input\n(1ch)", ha="center", fontsize=6, style="italic")
ax.annotate("", xy=(enc_x[0] + enc_w[0]/2, enc_y[0] + enc_h[0]),
            xytext=(0.5, 6.1),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

ax.text(10.0, 5.8, "Output\n(2ch)", ha="center", fontsize=6, style="italic")

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#4A90D9", label="Encoder"),
    mpatches.Patch(facecolor="#E67E22", label="Decoder"),
    plt.Line2D([0], [0], color="green", linestyle="--", label="Skip conn."),
    plt.Line2D([0], [0], color="red", marker=">", markersize=3, label="Deep sup.", linewidth=0.5),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=5.5,
          framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "unet3d_architecture.pdf"))
plt.savefig(os.path.join(OUTDIR, "unet3d_architecture.png"))
plt.close()
print("Saved unet3d_architecture.pdf + .png")
