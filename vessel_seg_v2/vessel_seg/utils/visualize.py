"""
3D visualization of vessel segmentations for qualitative clinical analysis.

Renders predicted vessel trees as 3D surface meshes for side-by-side
comparison across loss variants. The output is what goes in the
"qualitative analysis" section of the report.

Usage:
    python -m utils.visualize \
        --gt /path/to/gt_label.nii.gz \
        --pred_baseline /path/to/pred_dice_ce.nii.gz \
        --pred_cldice /path/to/pred_cldice.nii.gz \
        --pred_skeleton /path/to/pred_skeleton.nii.gz \
        --output comparison.png

Dependencies:
    pip install matplotlib scikit-image SimpleITK
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple, List
import os


def extract_surface_mesh(
    volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    step_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a triangulated surface mesh from a binary volume using
    marching cubes.

    Args:
        volume: (D, H, W) binary mask
        spacing: voxel spacing in mm
        step_size: downsampling factor (higher = faster but coarser)

    Returns:
        verts: (N, 3) vertex coordinates in mm
        faces: (M, 3) triangle face indices
    """
    from skimage.measure import marching_cubes

    if volume.sum() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    verts, faces, _, _ = marching_cubes(
        volume,
        level=0.5,
        spacing=spacing,
        step_size=step_size,
    )
    return verts, faces


def render_vessel_3d(
    ax: plt.Axes,
    volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    color: str = "#378ADD",
    alpha: float = 0.6,
    title: str = "",
    step_size: int = 2,
):
    """
    Render a binary vessel mask as a 3D surface on a matplotlib axis.

    Args:
        ax: matplotlib 3D axis
        volume: (D, H, W) binary vessel mask
        spacing: voxel spacing for correct aspect ratio
        color: surface color
        alpha: surface transparency
        title: subplot title
        step_size: marching cubes resolution
    """
    verts, faces = extract_surface_mesh(volume, spacing, step_size)

    if len(faces) == 0:
        ax.text(0.5, 0.5, 0.5, "No vessels\ndetected",
                transform=ax.transAxes, ha="center", fontsize=12)
        ax.set_title(title)
        return

    mesh = Poly3DCollection(
        verts[faces],
        alpha=alpha,
        facecolor=color,
        edgecolor="none",
    )
    ax.add_collection3d(mesh)

    # Set axis limits from mesh extents
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.set_zlabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=6)

    # Match viewing angle across subplots
    ax.view_init(elev=25, azim=45)


def render_comparison(
    gt: np.ndarray,
    predictions: dict,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_path: str = "comparison.png",
    case_name: str = "",
    step_size: int = 2,
):
    """
    Render side-by-side 3D vessel trees for qualitative comparison.

    Args:
        gt: (D, H, W) ground truth binary mask
        predictions: dict mapping loss name to (D, H, W) binary prediction
            e.g. {"Dice+CE": pred1, "Dice+CE+clDice": pred2, ...}
        spacing: voxel spacing in mm
        output_path: where to save the figure
        case_name: patient/case identifier for the title
        step_size: marching cubes resolution (2 = good balance of speed/quality)
    """
    n_panels = 1 + len(predictions)  # ground truth + each prediction
    fig = plt.figure(figsize=(5 * n_panels, 5))

    if case_name:
        fig.suptitle(f"Case: {case_name}", fontsize=14, fontweight="bold", y=0.98)

    # Ground truth
    ax = fig.add_subplot(1, n_panels, 1, projection="3d")
    render_vessel_3d(ax, gt, spacing, color="#1D9E75", alpha=0.5,
                     title="Ground truth", step_size=step_size)

    # Each prediction
    colors = ["#378ADD", "#EF9F27", "#E24B4A", "#7F77DD"]
    for i, (name, pred) in enumerate(predictions.items()):
        ax = fig.add_subplot(1, n_panels, i + 2, projection="3d")
        color = colors[i % len(colors)]
        render_vessel_3d(ax, pred, spacing, color=color, alpha=0.5,
                         title=name, step_size=step_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved comparison to {output_path}")


def render_difference_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_path: str = "diff_overlay.png",
    title: str = "",
    step_size: int = 2,
):
    """
    Render a single prediction with color-coded correctness:
      - Green: true positive (correct vessel)
      - Red: false negative (missed vessel)
      - Blue: false positive (hallucinated vessel)

    This is the most clinically informative view: a surgeon can
    immediately see where the segmentation fails.
    """
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    tp = gt_bin & pred_bin         # correct
    fn = gt_bin & (~pred_bin)      # missed
    fp = (~gt_bin) & pred_bin      # hallucinated

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if tp.sum() > 0:
        verts, faces = extract_surface_mesh(tp, spacing, step_size)
        if len(faces) > 0:
            mesh = Poly3DCollection(verts[faces], alpha=0.4,
                                    facecolor="#1D9E75", edgecolor="none")
            ax.add_collection3d(mesh)

    if fn.sum() > 0:
        verts, faces = extract_surface_mesh(fn, spacing, step_size)
        if len(faces) > 0:
            mesh = Poly3DCollection(verts[faces], alpha=0.7,
                                    facecolor="#E24B4A", edgecolor="none")
            ax.add_collection3d(mesh)

    if fp.sum() > 0:
        verts, faces = extract_surface_mesh(fp, spacing, step_size)
        if len(faces) > 0:
            mesh = Poly3DCollection(verts[faces], alpha=0.3,
                                    facecolor="#378ADD", edgecolor="none")
            ax.add_collection3d(mesh)

    # Set limits from the union of all volumes
    union = gt_bin | pred_bin
    if union.sum() > 0:
        coords = np.argwhere(union)
        for i, label in enumerate(["x", "y", "z"]):
            lo = coords[:, i].min() * spacing[i]
            hi = coords[:, i].max() * spacing[i]
            getattr(ax, f"set_{label}lim")(lo, hi)

    ax.view_init(elev=25, azim=45)
    ax.set_title(title or "Segmentation overlay", fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1D9E75", alpha=0.5, label="Correct (TP)"),
        Patch(facecolor="#E24B4A", alpha=0.7, label="Missed (FN)"),
        Patch(facecolor="#378ADD", alpha=0.4, label="Hallucinated (FP)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved overlay to {output_path}")


# ─── CLI entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import SimpleITK as sitk

    parser = argparse.ArgumentParser(description="Render 3D vessel comparisons")
    parser.add_argument("--gt", required=True, help="Ground truth NIfTI")
    parser.add_argument("--pred_baseline", required=True, help="Dice+CE prediction")
    parser.add_argument("--pred_cldice", required=True, help="Dice+CE+clDice prediction")
    parser.add_argument("--pred_skeleton", required=True, help="Dice+CE+SkeletonRecall prediction")
    parser.add_argument("--output", default="comparison.png")
    parser.add_argument("--case_name", default="")
    args = parser.parse_args()

    def load(path):
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img), img.GetSpacing()[::-1]

    gt, spacing = load(args.gt)
    gt = (gt > 0).astype(np.uint8)

    preds = {
        "Dice + CE": (load(args.pred_baseline)[0] > 0).astype(np.uint8),
        "Dice + CE + clDice": (load(args.pred_cldice)[0] > 0).astype(np.uint8),
        "Dice + CE + Skel. Recall": (load(args.pred_skeleton)[0] > 0).astype(np.uint8),
    }

    render_comparison(gt, preds, spacing, args.output, args.case_name)

    # Also render per-model difference overlays
    for name, pred in preds.items():
        safe_name = name.replace(" ", "_").replace("+", "").lower()
        overlay_path = args.output.replace(".png", f"_overlay_{safe_name}.png")
        render_difference_overlay(gt, pred, spacing, overlay_path, title=name)
