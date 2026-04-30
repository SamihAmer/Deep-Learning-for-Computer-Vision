"""
Paired Wilcoxon signed-rank tests across the 25 TopCoW validation cases for
each pair of from-scratch models. Driven by reviewer feedback: the marginal
deltas in Table V need real statistics, not hand-waved 'wins'.

Parses per-case DSC and clDice from `eval_summary_extended.txt`, runs paired
Wilcoxon (clDice vs each other model), and prints a summary that drives the
language in the report.
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "runs_final_results" / "runs"
LOCAL = ROOT / "runs"

MODELS = [
    ("Dice+CE",     "dice_ce_20260409_001230"),
    ("clDice",      "dice_ce_cldice_20260404_053853"),
    ("Skeleton",    "dice_ce_skeleton_20260402_203733"),
    ("SSIM",        "dice_ce_ssim_20260428_191619"),
    ("MSE-DT",      "dice_ce_mse_dt_20260428_202437"),
    ("Perceptual",  "dice_ce_perceptual_20260428_223116"),
    ("Combo",       "dice_ce_cldice_ssim_20260428_233832"),
]

# Each per-case line looks like:
#   [1/25] topcow_ct_072_0000.nii.gz | DSC: 0.8329 | clDice: 0.8644 | 13s
RE = re.compile(
    r"\[\s*\d+\s*/\s*\d+\]\s+(?P<case>\S+)\s+\|\s+DSC:\s*(?P<dsc>[0-9.]+)\s+\|\s+clDice:\s*(?P<cld>[0-9.]+)"
)


def find_eval_file(run_name: str) -> Path | None:
    for base in (BUNDLE, LOCAL):
        p = base / run_name / "eval_summary_extended.txt"
        if p.exists():
            return p
    return None


def parse_per_case(run_name: str) -> dict:
    """Returns {case: (dsc, cldice)} from the per-case lines in the eval txt."""
    path = find_eval_file(run_name)
    if path is None:
        raise FileNotFoundError(f"no eval_summary_extended.txt for {run_name}")
    out = {}
    with open(path) as f:
        for line in f:
            m = RE.match(line.strip())
            if m:
                out[m.group("case")] = (float(m.group("dsc")), float(m.group("cld")))
    return out


def aligned_arrays(a: dict, b: dict, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (a_vals, b_vals) for the cases present in both, in matched order."""
    common = sorted(set(a) & set(b))
    av = np.array([a[c][idx] for c in common])
    bv = np.array([b[c][idx] for c in common])
    return av, bv


def main():
    per_case = {label: parse_per_case(run) for label, run in MODELS}
    n_cases = len(next(iter(per_case.values())))
    print(f"Parsed per-case metrics for {len(MODELS)} models on {n_cases} cases.\n")

    # clDice is the reference; test it against every other model on DSC and clDice
    REF = "clDice"
    print(f"=== Paired Wilcoxon signed-rank tests (reference = {REF}, two-sided) ===\n")
    print(f"{'metric':<8}{'comparison':<26}{'mean diff':>11}{'median diff':>13}{'W':>10}{'p-value':>11}{'verdict (α=0.05)':>20}")
    print("-" * 99)

    rows = []
    for metric_label, metric_idx in [("DSC", 0), ("clDice", 1)]:
        for label, _ in MODELS:
            if label == REF:
                continue
            a, b = aligned_arrays(per_case[REF], per_case[label], metric_idx)
            diff = a - b
            mean_d = float(np.mean(diff))
            med_d = float(np.median(diff))
            try:
                stat = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                p = stat.pvalue
                W = stat.statistic
            except ValueError:
                # All zeros -> wilcoxon raises; treat as identical
                p, W = 1.0, 0.0
            verdict = "significant" if p < 0.05 else "n.s. (within noise)"
            print(f"{metric_label:<8}{REF + ' vs ' + label:<26}{mean_d:>+11.4f}{med_d:>+13.4f}{W:>10.1f}{p:>11.4f}   {verdict:<18}")
            rows.append((metric_label, label, mean_d, med_d, W, p, verdict))

    # Summary stats for the reviewer-style framing
    print()
    print("=== Multiple-comparisons correction ===")
    print()
    print("We run 6 pairwise comparisons per metric (clDice vs each other config).")
    print("Holm-Bonferroni step-down at family-wise alpha=0.05 (per metric):")
    print()

    for metric_label in ("DSC", "clDice"):
        metric_rows = [r for r in rows if r[0] == metric_label]
        # Sort by raw p-value ascending
        metric_rows = sorted(metric_rows, key=lambda r: r[5])
        m = len(metric_rows)
        print(f"  --- {metric_label} ---")
        print(f"  {'rank':>4}{'comparison':<25}{'raw p':>10}{'Holm thr':>12}{'survives':>14}")
        for rank, r in enumerate(metric_rows, start=1):
            holm_thr = 0.05 / (m - rank + 1)
            survives = "yes" if r[5] < holm_thr else "no"
            print(f"  {rank:>4}{r[1]:<25}{r[5]:>10.4f}{holm_thr:>12.4f}{survives:>14}")
        print()

    # Bonferroni summary (more conservative)
    bonf_thr = 0.05 / 6
    print(f"  --- Bonferroni (alpha=0.05/6 = {bonf_thr:.4f}) ---")
    bonf_dsc = [r for r in rows if r[0] == "DSC" and r[5] < bonf_thr]
    bonf_cld = [r for r in rows if r[0] == "clDice" and r[5] < bonf_thr]
    print(f"  DSC: clDice significantly better than {len(bonf_dsc)}/6 others under Bonferroni:")
    for r in bonf_dsc:
        print(f"    vs {r[1]}: p={r[5]:.4f}")
    print(f"  clDice metric: significantly better than {len(bonf_cld)}/6 others under Bonferroni:")
    for r in bonf_cld:
        print(f"    vs {r[1]}: p={r[5]:.4f}")


if __name__ == "__main__":
    main()
