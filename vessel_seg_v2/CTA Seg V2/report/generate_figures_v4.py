"""
Generate figures for the final report (Sections VI-VII).

Outputs:
  figures_v4/learning_curves_extended.pdf  -- 7-loss training-loss + val-DSC
  figures_v4/topbrain_extended_groups.pdf  -- 7-loss x 6-vessel-group bar chart

Reads from `runs_final_results/runs/<run>/training_log.json` and
`runs_final_results/runs/<run>/stratified_eval.json` -- the bundle that
came back from the EC2 final ablation.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_RUNS = ROOT / "runs_final_results" / "runs"   # from EC2 bundle (new runs)
LOCAL_RUNS = ROOT / "runs"                            # midterm logs already on disk
OUT = Path(__file__).resolve().parent / "figures_v4"
OUT.mkdir(exist_ok=True)


def find_run_file(run_name: str, filename: str) -> Path | None:
    """Look in the EC2 bundle first, fall back to the local midterm runs.

    For midterm runs, the bundle only contains the re-evaluation outputs
    (eval_summary_extended.txt, stratified_eval.json) — the original
    training_log.json lives in the local `runs/` directory.
    """
    for base in (BUNDLE_RUNS, LOCAL_RUNS):
        candidate = base / run_name / filename
        if candidate.exists():
            return candidate
    return None

# Mapping of (loss_tag, run_dir_name, label, color)
FROM_SCRATCH = [
    ("dice_ce",            "dice_ce_20260409_001230",            "Dice+CE",       "#1f77b4"),
    ("dice_ce_cldice",     "dice_ce_cldice_20260404_053853",     "+clDice",       "#ff7f0e"),
    ("dice_ce_skeleton",   "dice_ce_skeleton_20260402_203733",   "+Skeleton",     "#2ca02c"),
    ("dice_ce_ssim",       "dice_ce_ssim_20260428_191619",       "+SSIM",         "#d62728"),
    ("dice_ce_mse_dt",     "dice_ce_mse_dt_20260428_202437",     "+MSE-DT",       "#9467bd"),
    ("dice_ce_perceptual", "dice_ce_perceptual_20260428_223116", "+Perceptual",   "#8c564b"),
    ("dice_ce_cldice_ssim","dice_ce_cldice_ssim_20260428_233832","+Combo",        "#7f7f7f"),
]

TOPBRAIN = [
    ("dice_ce",            "finetune_topbrain_dice_ce_20260409_011117",
                                                                  "Dice+CE",       "#1f77b4"),
    ("dice_ce_cldice",     "finetune_topbrain_dice_ce_cldice_20260404_064204",
                                                                  "+clDice",       "#ff7f0e"),
    ("dice_ce_skeleton",   "finetune_topbrain_dice_ce_skeleton_20260402_215720",
                                                                  "+Skeleton",     "#2ca02c"),
    ("dice_ce_ssim",       "finetune_topbrain_dice_ce_ssim_20260429_005004",
                                                                  "+SSIM",         "#d62728"),
    ("dice_ce_mse_dt",     "finetune_topbrain_dice_ce_mse_dt_20260429_005937",
                                                                  "+MSE-DT",       "#9467bd"),
    ("dice_ce_perceptual", "finetune_topbrain_dice_ce_perceptual_20260429_011452",
                                                                  "+Perceptual",   "#8c564b"),
    ("dice_ce_cldice_ssim","finetune_topbrain_dice_ce_cldice_ssim_20260429_012422",
                                                                  "+Combo",        "#7f7f7f"),
]


def smooth(y: np.ndarray, k: int = 9) -> np.ndarray:
    """Centered moving average preserving length; reflects edges."""
    if len(y) < k:
        return y
    pad = k // 2
    padded = np.concatenate([[y[0]] * pad, y, [y[-1]] * pad])
    out = np.convolve(padded, np.ones(k) / k, mode="valid")
    return out[: len(y)]


def figure_learning_curves():
    """
    Top: training-loss curves, plotted from epoch 10 onward to skip the
    warmup spike (Perceptual starts at ~13 because the VGG-feature L2 has
    a different scale than CE/Dice; including epoch 0-5 squashes every
    other line into a flat strip near zero).

    Bottom: validation DSC at every checkpointed epoch, with a thin line
    connecting consecutive points per loss. This is more readable than
    the per-epoch noisy training-DSC traces.
    """
    fig, (ax_loss, ax_dsc) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    SKIP = 10  # epochs to skip on the loss panel

    for tag, run, label, color in FROM_SCRATCH:
        log_path = find_run_file(run, "training_log.json")
        if log_path is None:
            print(f"WARN missing training_log for {tag}")
            continue
        with open(log_path) as f:
            history = json.load(f)
        epochs = np.array([h["epoch"] for h in history])
        loss = np.array([h["train_loss"] for h in history])
        val_pts = [(h["epoch"], h["dice"]) for h in history
                   if "dice" in h and h.get("dice") is not None]

        keep = epochs >= SKIP
        ax_loss.plot(epochs[keep], smooth(loss[keep], k=11),
                     color=color, label=label, lw=1.4, alpha=0.92)

        if val_pts:
            vx, vy = zip(*val_pts)
            ax_dsc.plot(vx, vy, "o-", color=color, lw=1.0, ms=3.5,
                        alpha=0.92, label=label)

    ax_loss.set_ylabel("Training loss")
    ax_loss.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    ax_loss.grid(alpha=0.25)
    ax_loss.set_ylim(0, 0.55)

    ax_dsc.set_xlabel("Epoch")
    ax_dsc.set_ylabel("Validation DSC")
    ax_dsc.set_ylim(0.55, 0.92)
    ax_dsc.grid(alpha=0.25)

    plt.tight_layout()
    out_pdf = OUT / "learning_curves_extended.pdf"
    out_png = OUT / "learning_curves_extended.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


# Hard-coded vessel group results from EVALUATION_FINDINGS_FINAL.md §3.5
# (numbers verified against runs_final_results/runs/<run>/stratified_eval.json)
VESSEL_GROUPS = ["Large CoW", "Communicating", "Distal\nbranches",
                 "Posterior\nfossa", "Small\narteries", "Venous\nsinuses"]
TB_GROUP_DSC = {
    "Dice+CE":     [0.832, 0.466, 0.687, 0.565, 0.000, 0.416],
    "+clDice":     [0.829, 0.490, 0.712, 0.533, 0.000, 0.463],
    "+Skeleton":   [0.821, 0.513, 0.752, 0.626, 0.589, 0.613],
    "+SSIM":       [0.834, 0.461, 0.674, 0.541, 0.000, 0.425],
    "+MSE-DT":     [0.823, 0.465, 0.683, 0.526, 0.000, 0.456],
    "+Perceptual": [0.825, 0.469, 0.706, 0.551, 0.011, 0.481],
    "+Combo":      [0.831, 0.489, 0.714, 0.558, 0.000, 0.450],
}
COLORS = {
    "Dice+CE":     "#1f77b4",
    "+clDice":     "#ff7f0e",
    "+Skeleton":   "#2ca02c",
    "+SSIM":       "#d62728",
    "+MSE-DT":     "#9467bd",
    "+Perceptual": "#8c564b",
    "+Combo":      "#7f7f7f",
}


def figure_topbrain_groups():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    n_models = len(TB_GROUP_DSC)
    n_groups = len(VESSEL_GROUPS)
    width = 0.8 / n_models
    x = np.arange(n_groups)

    for i, (label, vals) in enumerate(TB_GROUP_DSC.items()):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=COLORS[label],
               label=label, edgecolor="white", linewidth=0.4)

    # Highlight the small-arteries column
    ax.axvspan(4 - 0.45, 4 + 0.45, alpha=0.06, color="red", zorder=0)
    ax.annotate("only +Skeleton segments\nsmall arteries (0.589 vs 0.000)",
                xy=(4, 0.589), xytext=(4.1, 0.78), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red", lw=0.9),
                color="darkred")

    ax.set_xticks(x)
    ax.set_xticklabels(VESSEL_GROUPS, fontsize=9)
    ax.set_ylabel("DSC")
    ax.set_ylim(0, 0.95)
    ax.set_title("TopBrain fine-tuned: DSC by vessel group, 7 loss configurations",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    out_pdf = OUT / "topbrain_extended_groups.pdf"
    out_png = OUT / "topbrain_extended_groups.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    figure_learning_curves()
    figure_topbrain_groups()
