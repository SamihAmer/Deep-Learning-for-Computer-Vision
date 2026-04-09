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

def save_fig(fig, name):
    """Save figure as both PDF (for LaTeX) and PNG (for PPTX)."""
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"))
    fig.savefig(os.path.join(OUTDIR, name + ".png"))
    plt.close(fig)
    print(f"Saved {name}.pdf + .png")


def load_log(path):
    with open(path) as f:
        return json.load(f)


def plot_learning_curves_overlay(logs, labels, colors, outfile="learning_curves.pdf"):
    """Plot overlaid training loss and validation dice for multiple models."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
    window = 15

    for log, label, color in zip(logs, labels, colors):
        epochs = [e["epoch"] for e in log]
        train_loss = [e["train_loss"] for e in log]
        train_dice = [e["train_dice"] for e in log]

        val_epochs = [e["epoch"] for e in log if "dice" in e]
        val_dice = [e["dice"] for e in log if "dice" in e]

        # Top: smoothed training loss
        if len(train_loss) > window:
            smooth = np.convolve(train_loss, np.ones(window)/window, mode="valid")
            ax1.plot(epochs[window-1:], smooth, color=color, linewidth=1.3, label=label)
        else:
            ax1.plot(epochs, train_loss, color=color, linewidth=1.3, label=label)

        # Bottom: smoothed training dice + validation dice markers
        if len(train_dice) > window:
            smooth_dice = np.convolve(train_dice, np.ones(window)/window, mode="valid")
            ax2.plot(epochs[window-1:], smooth_dice, color=color, linewidth=1.0, alpha=0.5)
        ax2.plot(val_epochs, val_dice, "s-", color=color, markersize=3, linewidth=1.2, label=label)

    ax1.set_ylabel("Training Loss")
    ax1.set_ylim(0.05, 0.6)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)

    ax2.set_ylabel("Validation Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0.6, 0.95)
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, outfile.replace(".pdf", ""))


def plot_global_comparison():
    """Bar chart comparing global metrics across three loss configs."""
    models = ["Dice+CE", "Dice+CE\n+clDice", "Dice+CE\n+Skeleton"]
    dsc =    [0.858, 0.864, 0.845]
    cldice = [0.907, 0.916, 0.838]
    hd95 =   [2.48,  2.40,  2.82]

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
    save_fig(fig, "global_comparison")


def plot_stratified_vessels():
    """Horizontal bar chart of per-vessel DSC for baseline model."""
    vessels = [
        "L-ICA", "R-ICA", "L-MCA", "R-MCA", "BA", "R-PCA", "L-PCA",
        "R-ACA", "L-ACA",
        "R-Pcom", "L-Pcom", "Acom"
    ]
    dsc = [0.855, 0.846, 0.819, 0.807, 0.793, 0.802, 0.770,
           0.725, 0.740,
           0.459, 0.410, 0.413]
    colors = ["C0"]*9 + ["C3"]*3  # blue for large, red for communicating

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
    save_fig(fig, "stratified_vessels")


def plot_topbrain_comparison():
    """Grouped bar chart: 6 vessel groups x 3 models (TopBrain fine-tuned)."""
    groups = ["Large CoW", "Communicating", "Distal\nbranches",
              "Posterior\nfossa", "Small\narteries", "Venous\nsinuses"]
    dice_ce  = [0.832, 0.466, 0.687, 0.565, 0.000, 0.416]
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
    save_fig(fig, "topbrain_comparison")


def plot_vessel_gap():
    """Side-by-side bar chart: large vs communicating DSC per model."""
    models = ["Dice+CE", "+clDice", "+Skeleton"]
    large = [0.795, 0.797, 0.798]
    comm =  [0.433, 0.435, 0.466]
    gap =   [0.363, 0.363, 0.332]

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
    save_fig(fig, "vessel_gap")


def plot_per_case_boxplot():
    """Box plot of per-case DSC for baseline model."""
    per_case_dsc = [
        0.826, 0.884, 0.826, 0.911, 0.807, 0.853, 0.878, 0.728,
        0.663, 0.906, 0.886, 0.925, 0.857, 0.924, 0.857, 0.852,
        0.908, 0.911, 0.802, 0.867, 0.834, 0.905, 0.819, 0.911, 0.898
    ]
    per_case_cldice = [
        0.836, 0.909, 0.908, 0.934, 0.837, 0.917, 0.937, 0.856,
        0.865, 0.921, 0.942, 0.952, 0.925, 0.934, 0.915, 0.913,
        0.975, 0.930, 0.868, 0.918, 0.849, 0.937, 0.846, 0.908, 0.943
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
    save_fig(fig, "per_case_boxplot")


def main():
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(runs_dir, "dice_ce_20260409_001230"),
                        help="Path to dice_ce baseline training log directory")
    parser.add_argument("--cldice_log_dir", type=str,
                        default=os.path.join(runs_dir, "dice_ce_cldice_20260404_053853"),
                        help="Path to dice_ce_cldice training log directory")
    parser.add_argument("--skeleton_log_dir", type=str,
                        default=os.path.join(runs_dir, "dice_ce_skeleton_20260402_203733"),
                        help="Path to dice_ce_skeleton training log directory")
    args = parser.parse_args()

    # 1. Overlaid learning curves for all three models
    log_paths = [
        (args.log_dir, "Dice+CE", "C0"),
        (args.cldice_log_dir, "+clDice", "C1"),
        (args.skeleton_log_dir, "+Skeleton", "C2"),
    ]
    logs, labels, colors = [], [], []
    for path, label, color in log_paths:
        lp = os.path.join(path, "training_log.json")
        if os.path.exists(lp):
            logs.append(load_log(lp))
            labels.append(label)
            colors.append(color)
            print(f"Loaded {label}: {lp}")
        else:
            print(f"WARNING: {label} log not found at {lp}, skipping")

    if logs:
        plot_learning_curves_overlay(logs, labels, colors)

    # 2-6. Static charts from documented results
    plot_global_comparison()
    plot_stratified_vessels()
    plot_topbrain_comparison()
    plot_vessel_gap()
    plot_per_case_boxplot()

    print(f"\nAll figures saved to: {OUTDIR}/")


if __name__ == "__main__":
    main()
