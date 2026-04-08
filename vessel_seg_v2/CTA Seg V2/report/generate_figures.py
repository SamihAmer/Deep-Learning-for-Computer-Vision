"""
Generate figures for the midterm report.

Reads training_log.json files and produces PDF plots for:
  1. Learning curves (loss + dice over epochs)
  2. Global metric comparison bar chart
  3. Stratified per-vessel bar chart
  4. TopBrain fine-tuning grouped bar chart
  5. Large vs small vessel gap chart

Usage:
    python generate_figures.py [--log_dir /path/to/dice_ce_log_dir]

If topology-aware training logs are available, place them as:
    --cldice_log_dir /path/to/dice_ce_cldice_dir
    --skeleton_log_dir /path/to/dice_ce_skeleton_dir
"""

import json
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# IEEE-friendly styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)


def load_log(path):
    with open(path) as f:
        return json.load(f)


def plot_learning_curves(log, label="Dice+CE", color="C0", outfile="learning_curves.pdf"):
    """Plot training loss, training dice, validation dice, and LR schedule."""
    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    train_dice = [e["train_dice"] for e in log]
    lr = [e["lr"] for e in log]

    # Extract validation points
    val_epochs = [e["epoch"] for e in log if "dice" in e]
    val_dice = [e["dice"] for e in log if "dice" in e]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)

    # Top: Loss + LR
    ax1.plot(epochs, train_loss, color="C3", linewidth=0.8, alpha=0.6, label="Train loss")
    # Smoothed loss
    window = 10
    if len(train_loss) > window:
        smooth = np.convolve(train_loss, np.ones(window)/window, mode="valid")
        ax1.plot(epochs[window-1:], smooth, color="C3", linewidth=1.5, label=f"Loss (smoothed)")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, 1.0)

    ax1r = ax1.twinx()
    ax1r.plot(epochs, lr, color="C4", linewidth=1.0, linestyle="--", alpha=0.7, label="Learning rate")
    ax1r.set_ylabel("Learning rate", color="C4")
    ax1r.tick_params(axis="y", labelcolor="C4")
    ax1r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0e}"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    # Bottom: Dice
    ax2.plot(epochs, train_dice, color="C0", linewidth=0.8, alpha=0.4)
    if len(train_dice) > window:
        smooth_dice = np.convolve(train_dice, np.ones(window)/window, mode="valid")
        ax2.plot(epochs[window-1:], smooth_dice, color="C0", linewidth=1.5, label="Train Dice (smoothed)")
    ax2.plot(val_epochs, val_dice, "s-", color="C1", markersize=4, linewidth=1.2, label="Val Dice")
    ax2.set_ylabel("Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0.2, 1.0)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outfile))
    plt.close()
    print(f"Saved {outfile}")


def plot_global_comparison():
    """Bar chart comparing global metrics across three loss configs."""
    models = ["Dice+CE", "Dice+CE\n+clDice", "Dice+CE\n+Skeleton"]
    dsc =    [0.896, 0.864, 0.845]
    cldice = [0.940, 0.916, 0.838]
    hd95 =   [1.36,  2.40,  2.82]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.2))

    x = np.arange(len(models))
    w = 0.35

    # DSC + clDice
    bars1 = ax1.bar(x - w/2, dsc, w, label="DSC", color="C0", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, cldice, w, label="clDice", color="C1", edgecolor="white", linewidth=0.5)
    ax1.set_ylim(0.7, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7)
    ax1.set_ylabel("Score")
    ax1.legend(loc="lower left", fontsize=7)
    ax1.bar_label(bars1, fmt="%.3f", fontsize=6, padding=1)
    ax1.bar_label(bars2, fmt="%.3f", fontsize=6, padding=1)

    # HD95
    bars3 = ax2.bar(x, hd95, 0.5, color=["C0", "C1", "C2"], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=7)
    ax2.set_ylabel("HD95 (mm)")
    ax2.bar_label(bars3, fmt="%.2f", fontsize=6, padding=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "global_comparison.pdf"))
    plt.close()
    print("Saved global_comparison.pdf")


def plot_stratified_vessels():
    """Horizontal bar chart of per-vessel DSC for baseline model."""
    vessels = [
        "R-ICA", "R-MCA", "BA", "R-PCA", "L-ICA", "L-MCA", "L-PCA",
        "Acom", "L-Pcom", "R-ACA", "L-ACA", "R-Pcom"
    ]
    dsc = [0.858, 0.851, 0.821, 0.815, 0.814, 0.813, 0.812,
           0.740, 0.729, 0.490, 0.462, 0.417]
    colors = ["C0"]*7 + ["C3"]*5  # blue for large, red for communicating

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    y = np.arange(len(vessels))
    bars = ax.barh(y, dsc, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(vessels, fontsize=7)
    ax.set_xlabel("DSC")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.axvline(x=0.775, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="C0", label="Large arteries"),
                       Patch(facecolor="C3", label="Communicating")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "stratified_vessels.pdf"))
    plt.close()
    print("Saved stratified_vessels.pdf")


def plot_topbrain_comparison():
    """Grouped bar chart: 6 vessel groups x 3 models (TopBrain fine-tuned)."""
    groups = ["Large CoW", "Communicating", "Distal\nbranches",
              "Posterior\nfossa", "Small\narteries", "Venous\nsinuses"]
    dice_ce  = [0.834, 0.491, 0.694, 0.570, 0.005, 0.468]
    cldice   = [0.829, 0.490, 0.712, 0.533, 0.000, 0.463]
    skeleton = [0.821, 0.513, 0.752, 0.626, 0.589, 0.613]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(groups))
    w = 0.25

    bars1 = ax.bar(x - w, dice_ce, w, label="Dice+CE", color="C0", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, cldice, w, label="+clDice", color="C1", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + w, skeleton, w, label="+Skeleton", color="C2", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=6.5)
    ax.set_ylabel("DSC")
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "topbrain_comparison.pdf"))
    plt.close()
    print("Saved topbrain_comparison.pdf")


def plot_vessel_gap():
    """Side-by-side bar chart: large vs communicating DSC per model."""
    models = ["Dice+CE", "+clDice", "+Skeleton"]
    large = [0.823, 0.797, 0.798]
    comm =  [0.472, 0.435, 0.466]
    gap =   [0.351, 0.363, 0.332]

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    x = np.arange(len(models))
    w = 0.3

    ax.bar(x - w/2, large, w, label="Large arteries", color="C0", edgecolor="white")
    ax.bar(x + w/2, comm, w, label="Communicating", color="C3", edgecolor="white")

    # Annotate gap
    for i in range(len(models)):
        mid = (large[i] + comm[i]) / 2
        ax.annotate(f"$\\Delta$={gap[i]:.3f}", xy=(i, mid), fontsize=6.5,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("DSC")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "vessel_gap.pdf"))
    plt.close()
    print("Saved vessel_gap.pdf")


def plot_per_case_boxplot():
    """Box plot of per-case DSC for baseline model."""
    per_case_dsc = [
        0.880, 0.883, 0.856, 0.914, 0.860, 0.871, 0.901, 0.883,
        0.666, 0.822, 0.876, 0.935, 0.838, 0.916, 0.853, 0.826,
        0.876, 0.913, 0.855, 0.888, 0.878, 0.929, 0.842, 0.925, 0.899
    ]
    per_case_cldice = [
        0.888, 0.937, 0.926, 0.926, 0.914, 0.916, 0.960, 0.917,
        0.899, 0.759, 0.924, 0.950, 0.931, 0.950, 0.926, 0.931,
        0.973, 0.944, 0.927, 0.960, 0.891, 0.958, 0.880, 0.920, 0.916
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    bp = ax.boxplot([per_case_dsc, per_case_cldice],
                    labels=["DSC", "clDice"],
                    patch_artist=True,
                    widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=4))
    bp["boxes"][0].set_facecolor("C0")
    bp["boxes"][1].set_facecolor("C1")
    for box in bp["boxes"]:
        box.set_alpha(0.6)

    ax.set_ylabel("Score")
    ax.set_title("Dice+CE Baseline: Per-Case Distribution (n=25)")
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "per_case_boxplot.pdf"))
    plt.close()
    print("Saved per_case_boxplot.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "..",
                                             "dice_ce_20260331_222218"),
                        help="Path to dice_ce baseline training log directory")
    parser.add_argument("--cldice_log_dir", type=str, default=None,
                        help="Path to dice_ce_cldice training log directory (optional)")
    parser.add_argument("--skeleton_log_dir", type=str, default=None,
                        help="Path to dice_ce_skeleton training log directory (optional)")
    args = parser.parse_args()

    # 1. Learning curves from baseline
    log_path = os.path.join(args.log_dir, "training_log.json")
    if os.path.exists(log_path):
        log = load_log(log_path)
        plot_learning_curves(log, label="Dice+CE")
    else:
        print(f"WARNING: Baseline training log not found at {log_path}")
        print("  Skipping learning curve plot.")
        print("  To generate, copy training_log.json from EC2 or specify --log_dir")

    # 2-6. Static charts from documented results
    plot_global_comparison()
    plot_stratified_vessels()
    plot_topbrain_comparison()
    plot_vessel_gap()
    plot_per_case_boxplot()

    print(f"\nAll figures saved to: {OUTDIR}/")
    print("\nTo include in LaTeX, ensure the figures/ directory is accessible")
    print("from your .tex file location.")


if __name__ == "__main__":
    main()
