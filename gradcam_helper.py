#!/usr/bin/env python3
"""
Regenerate the Grad-CAM++ summary grid for Q1 publication.

This script loads the existing gradcam_<Class>_correct.png images (generated
by pipeline_v2_clean_a100.py, which produce good heatmaps) and arranges them
into a clean publication-quality grid:
  - 4 rows (one per class)
  - 3 image-pairs per row (original | Grad-CAM++ overlay)
  - Row labels on the left, column headers at top
  - No suptitle, no per-image text annotations
  - Minimal gaps, tight layout, 300 DPI

Usage:
    python regenerate_gradcam.py
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
N_PER_CLASS = 3

# Percentage of each sub-image's height to crop from the top to remove
# embedded titles (e.g. "Original\nTrue: Chickenpox", "GradCAM++\nPred: ...")
CROP_TOP_PCT = 0.15


# Directory where the existing gradcam PNGs are stored
GRADCAM_DIR = os.path.join('pipeline_v2_single_gpu', 'gradcam')
OUT_DIR     = os.path.join('pipeline_v2_single_gpu', 'gradcam')

# ── Publication-quality matplotlib style ──────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    0.5,
})


def load_and_split_gradcam_png(png_path):
    """
    Load a gradcam_<Class>_correct.png and split it into individual
    (original, overlay) pairs.

    The pipeline generates these PNGs with a 2-column layout:
      - Column 0: Original image
      - Column 1: Grad-CAM++ overlay
    Each row is one sample. We detect the number of rows by the image height
    and split accordingly.

    Returns: list of (original_np, overlay_np) tuples
    """
    img = Image.open(png_path).convert('RGB')
    w, h = img.size
    img_np = np.array(img)

    # The pipeline generates 3 rows (3 correct predictions per class)
    # Each row has 2 columns: Original | Grad-CAM++
    # We need to figure out how many rows there are.
    # The pipeline uses 3 samples per class, so we expect 3 rows.
    n_rows = 3
    row_h = h // n_rows
    col_w = w // 2

    # Crop top portion of each row to remove embedded titles like
    # "Original\nTrue: Chickenpox" and "GradCAM++\nPred: ... | Conf: ..."
    crop_px = int(row_h * CROP_TOP_PCT)

    pairs = []
    for r in range(n_rows):
        y0 = r * row_h + crop_px  # skip title area at top
        y1 = (r + 1) * row_h if r < n_rows - 1 else h
        orig = img_np[y0:y1, :col_w]
        cam  = img_np[y0:y1, col_w:]
        pairs.append((orig, cam))

    return pairs



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load existing gradcam PNGs ─────────────────────────────────────────────
    all_pairs = {}  # class_name -> list of (original, overlay)
    for cls_name in CLASSES:
        png_path = os.path.join(GRADCAM_DIR, f'gradcam_{cls_name}_correct.png')
        if not os.path.isfile(png_path):
            print(f'WARNING: {png_path} not found, skipping {cls_name}')
            all_pairs[cls_name] = []
            continue

        pairs = load_and_split_gradcam_png(png_path)
        all_pairs[cls_name] = pairs
        print(f'  {cls_name}: loaded {len(pairs)} image pairs from {png_path}')

    # ── Create the figure ──────────────────────────────────────────────────────
    # Layout: 4 rows x (N_PER_CLASS * 2) columns
    #   Each row = one class
    #   Columns alternate: Original | Grad-CAM++ | Original | Grad-CAM++ | ...
    n_rows = len(CLASSES)
    n_cols = N_PER_CLASS * 2

    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 1.1, n_rows * 1.1 + 0.3))

    # Remove all spines and ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Column headers: "Original" and "Grad-CAM++" alternating
    for ci in range(N_PER_CLASS):
        col_orig = ci * 2
        col_cam  = ci * 2 + 1
        axes[0, col_orig].set_title('Original', fontsize=7, pad=3)
        axes[0, col_cam].set_title('Grad-CAM++', fontsize=7, pad=3)

    # Row labels (class names) + image plotting
    for ri, cls_name in enumerate(CLASSES):
        pairs = all_pairs.get(cls_name, [])

        # Plot the available images
        for ci, (orig, cam) in enumerate(pairs[:N_PER_CLASS]):
            col_orig = ci * 2
            col_cam  = ci * 2 + 1

            axes[ri, col_orig].imshow(orig)
            axes[ri, col_cam].imshow(cam)

            # Row label on first column only
            if ci == 0:
                axes[ri, col_orig].set_ylabel(cls_name, fontsize=8, labelpad=4,
                                              rotation=90, fontweight='bold')

        # Hide unused axes (when fewer than N_PER_CLASS images available)
        for ci in range(len(pairs), N_PER_CLASS):
            col_orig = ci * 2
            col_cam  = ci * 2 + 1
            axes[ri, col_orig].set_visible(False)
            axes[ri, col_cam].set_visible(False)

    # Tight layout with minimal gaps
    plt.subplots_adjust(wspace=0.02, hspace=0.05)

    save_path = os.path.join(OUT_DIR, 'gradcam_summary_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f'\nSaved: {save_path}')
    print('Done!')


if __name__ == '__main__':
    main()
