"""GCNet-S Testing Script.

Loads the best model, computes test metrics (Dice, IoU, Acc, Precision, Recall, F1),
saves results to outputs/test_metrics.txt, and generates 50 GT vs Prediction
sample figures (3 panels: Input | GT | Prediction) to outputs/test_samples/.

Usage:
    python -m torch.distributed.run --master_port=7850 --nproc_per_node=1 test.py \
        --model_path outputs/models/best_model.pt --seed=42
"""


import os
import sys
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure mmsegmentation is importable
_MMSEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mmsegmentation')
if _MMSEG_PATH not in sys.path:
    sys.path.insert(0, _MMSEG_PATH)

from mmseg.models.utils import resize

from model import load_model
from dataset import NoduleDataset, denormalize
from metrics import MetricAccumulator
from logger import log_test_metrics


# ─── Paths ───────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join('datasets', '24383_70_15_15_Segmentation')
OUTPUT_DIR = 'outputs'
TEST_METRICS_PATH = os.path.join(OUTPUT_DIR, 'test_metrics.txt')
TEST_SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'test_samples')


def parse_args():
    parser = argparse.ArgumentParser(description='GCNet-S Testing')
    parser.add_argument('--model_path', type=str, default='outputs/models/best_model.pt',
                        help='Path to the best model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Test batch size (default: 32)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size (default: 512)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader workers (default: 4)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of GT vs Prediction sample figures to generate (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


@torch.no_grad()
def evaluate_model(model, loader, device, accumulator):
    """Evaluate model on the test set.

    Args:
        model: The loaded model.
        loader: Test DataLoader.
        device: Device.
        accumulator: MetricAccumulator for test metrics.

    Returns:
        dict: Test metrics.
    """
    model.eval()
    accumulator.reset()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            _, c6_logit = outputs
        else:
            c6_logit = outputs

        c6_resized = resize(c6_logit, size=masks.shape[1:], mode='bilinear', align_corners=False)
        preds = torch.argmax(c6_resized, dim=1)

        accumulator.update(preds, masks, loss=0.0, batch_size=images.size(0))

    return accumulator.compute()


@torch.no_grad()
def generate_sample_figures(model, dataset, device, num_samples: int, save_dir: str):
    """Generate GT vs Prediction sample figures.

    Each figure has 3 panels: Input | GT | Prediction.

    Args:
        model: The loaded model.
        dataset: NoduleDataset for the test split.
        device: Device.
        num_samples: Number of sample figures to generate.
        save_dir: Directory to save the figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select sample indices (evenly spaced or random)
    n = len(dataset)
    if num_samples > n:
        num_samples = n
        print(f"[Warning] Requested {num_samples} samples but test set has {n}. "
              f"Generating {n} samples.", flush=True)

    # Evenly sample indices for diversity
    indices = np.linspace(0, n - 1, num_samples, dtype=int)

    model.eval()

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)  # (1, 1, H, W)

        outputs = model(image_batch)
        if isinstance(outputs, (tuple, list)):
            _, c6_logit = outputs
        else:
            c6_logit = outputs

        c6_resized = resize(c6_logit, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        pred = torch.argmax(c6_resized, dim=1).squeeze(0).cpu()  # (H, W)

        # Denormalize image for visualization
        image_vis = denormalize(image).squeeze(0).cpu().numpy()  # (H, W), [0, 1]
        mask_vis = mask.cpu().numpy()  # (H, W), {0, 1}
        pred_vis = pred.numpy()  # (H, W), {0, 1}

        # Create 3-panel figure: Input | GT | Prediction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_vis, cmap='gray')
        axes[0].set_title('Input', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=14)
        axes[1].axis('off')

        axes[2].imshow(pred_vis, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction', fontsize=14)
        axes[2].axis('off')

        fig.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{i+1:04d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"[Test] Generated {num_samples} sample figures to {save_dir}/", flush=True)


def main():
    args = parse_args()

    # ─── Device Setup ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Setup] Device: {device}", flush=True)

    set_seed(args.seed)

    # ─── Load Model ──────────────────────────────────────────────────────────
    print(f"[Model] Loading from {args.model_path}", flush=True)
    model = load_model(args.model_path, device=device, sync_bn=False)
    print(f"[Model] Model loaded successfully.", flush=True)

    # ─── Test Dataset ───────────────────────────────────────────────────────
    test_dataset = NoduleDataset(DATASET_ROOT, split='test', img_size=args.img_size, augment=False)
    print(f"[Dataset] Test: {len(test_dataset)} samples", flush=True)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # ─── Evaluate ────────────────────────────────────────────────────────────
    print(f"\n[Evaluating] Running on test set...", flush=True)
    accumulator = MetricAccumulator(sync=False)
    test_metrics = evaluate_model(model, test_loader, device, accumulator)

    # ─── Log & Save Test Metrics ─────────────────────────────────────────────
    log_test_metrics(test_metrics, save_path=TEST_METRICS_PATH)
    print(f"[Test] Metrics saved to {TEST_METRICS_PATH}", flush=True)

    # ─── Generate Sample Figures ─────────────────────────────────────────────
    print(f"\n[Generating] {args.num_samples} sample figures...", flush=True)
    generate_sample_figures(model, test_dataset, device, args.num_samples, TEST_SAMPLES_DIR)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}", flush=True)
    print(f"  Testing Complete!", flush=True)
    print(f"  Test Metrics: {TEST_METRICS_PATH}", flush=True)
    print(f"  Sample Figures: {TEST_SAMPLES_DIR}/", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == '__main__':
    main()
