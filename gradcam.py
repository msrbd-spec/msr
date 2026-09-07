#!/usr/bin/env python3
"""
Regenerate the Grad-CAM++ summary grid for Q1 publication.

This script loads the best Fold-1 FL checkpoint, runs Grad-CAM++ on the
test set, and produces a clean 2-column figure:
  - 4 rows (one per class)
  - 3 image-pairs per row (original | Grad-CAM++ overlay)
  - Row labels on the left, column headers at top
  - No suptitle, no per-image text annotations
  - Minimal gaps, tight layout, 300 DPI

Usage:
    python regenerate_gradcam_grid.py
"""

import os
import json
import math
import random
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings('ignore')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Constants (mirrors pipeline_v2_clean_a100.py) ────────────────────────────
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
NUM_CLIENTS = 5
IMAGE_SIZE  = 384
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
MSAF_DIM    = 256
GEM_P       = 3.0
DROP_PATH   = 0.10

DATA_ROOT = 'datasets/final_5_fold_pruned/'
OUT_DIR   = 'pipeline_v2_single_gpu'
GRADCAM_DIR = os.path.join(OUT_DIR, 'gradcam')

# Best fold = Fold_1 (highest FL Run1 macro-F1)
BEST_FOLD = 'Fold_1'
BEST_RUN  = 'FL_Run1_Uniform'

# Checkpoint path (from cv_summary.json or known path)
CKPT_PATH = os.path.join(
    OUT_DIR, BEST_FOLD, BEST_RUN, 'fl',
    f'{BEST_FOLD}_{BEST_RUN}_fl_best.pt'
)

# Number of image-pairs per class
N_PER_CLASS = 3

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

# ── Model definition (copied from pipeline_v2_clean_a100.py) ─────────────────
class GeMPool(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)

    def extra_repr(self):
        return f'p={self.p.data.item():.2f}'


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1, init_alpha=0.01):
        super().__init__()
        t = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
        k = max(t if t % 2 else t + 1, 3)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.conv    = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha   = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.sigmoid(self.conv(self.gap(x).view(b, 1, c))).view(b, c, 1, 1)
        return x + self.alpha * (x * w - x)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * noise.float() / (keep + 1e-8)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7,
                 init_alpha=0.01, drop_path=DROP_PATH):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc1   = nn.Linear(channels, reduced, bias=False)
        self.ch_fc2   = nn.Linear(reduced, channels, bias=False)
        self.ch_sig   = nn.Sigmoid()
        pad           = spatial_kernel // 2
        self.sp_conv  = nn.Conv2d(2, 1, spatial_kernel, padding=pad, bias=False)
        self.sp_sig   = nn.Sigmoid()
        self.alpha    = nn.Parameter(torch.tensor(float(init_alpha)))
        self.drop     = StochasticDepth(drop_path)

    def _ch(self, x):
        b, c, _, _ = x.shape
        mx = self.max_pool(x).view(b, c)
        av = self.avg_pool(x).view(b, c)
        gate = self.ch_sig(
            self.ch_fc2(F.relu(self.ch_fc1(mx), inplace=True)) +
            self.ch_fc2(F.relu(self.ch_fc1(av), inplace=True))
        ).view(b, c, 1, 1)
        return x * gate

    def _sp(self, x):
        sp = torch.cat([x.max(dim=1, keepdim=True)[0],
                        x.mean(dim=1, keepdim=True)], dim=1)
        return x * self.sp_sig(self.sp_conv(sp))

    def forward(self, x):
        x_attn = self._sp(self._ch(x))
        return x + self.alpha * self.drop(x_attn - x)


class CrossScaleAttentionHead(nn.Module):
    def __init__(self, dims, d=MSAF_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.gem = nn.ModuleList([GeMPool() for _ in dims])
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(c, d, bias=False), nn.LayerNorm(d))
            for c in dims
        ])
        self.q_lin = nn.Linear(d, d, bias=False)
        self.k_lin = nn.Linear(d, d, bias=False)
        self.v_lin = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5
        self.norm  = nn.LayerNorm(d)
        self.temp  = nn.Parameter(torch.ones(1))
        self.head  = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d // 2, num_classes),
        )

    def forward(self, feat_list):
        tokens = []
        for i, feat in enumerate(feat_list):
            pooled = self.gem[i](feat)
            tokens.append(self.proj[i](pooled))
        seq = torch.stack(tokens, dim=1)
        q = self.q_lin(seq[:, -1:, :])
        k = self.k_lin(seq)
        v = self.v_lin(seq)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        fused = self.norm(fused + tokens[-1]) * self.temp
        return self.head(fused)


class ConvNeXtV2MSAFv5(nn.Module):
    DIMS = [96, 192, 384, 768]

    def __init__(self, num_classes=NUM_CLASSES, attn_type='msaf', use_aux=False):
        super().__init__()
        self.attn_type = attn_type
        self.use_aux   = use_aux
        dims = self.DIMS

        import timm
        # pretrained=False: we load the trained checkpoint immediately after,
        # so pretrained weights are not needed (and the machine may be offline).
        bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False)
        self._backend = 'timm'
        self.stem   = bb.stem
        self.stage0 = bb.stages[0]
        self.stage1 = bb.stages[1]
        self.stage2 = bb.stages[2]
        self.stage3 = bb.stages[3]

        def _eca(ch):  return ECABlock(ch)
        def _cbam(ch): return CBAMBlock(ch, reduction=max(4, ch // 16), drop_path=DROP_PATH)
        def _none():   return nn.Identity()

        self.attn0, self.attn1 = _eca(dims[0]), _eca(dims[1])
        self.attn2, self.attn3 = _cbam(dims[2]), _cbam(dims[3])

        self.head = CrossScaleAttentionHead(
            dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=num_classes)

    def _stages(self, x):
        x  = self.stem(x)
        s0 = self.attn0(self.stage0(x))
        s1 = self.attn1(self.stage1(s0))
        s2 = self.attn2(self.stage2(s1))
        s3 = self.attn3(self.stage3(s2))
        return s1, s2, s3

    def forward(self, x):
        s1, s2, s3 = self._stages(x)
        return self.head([s1, s2, s3])


# ── Patch timm ConvNeXtStage for GradCAM target-layer indexing ───────────────
def patch_convnext_stage():
    import timm
    _dummy = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False)
    _ConvNeXtStageClass = type(_dummy.stages[0])

    def _stage_getitem(self, index):
        return self.blocks[index]

    _ConvNeXtStageClass.__getitem__ = _stage_getitem
    del _dummy
    print("Patched timm's ConvNeXtStage for indexing.")


# ── Dataset with paths ───────────────────────────────────────────────────────
class ImageFolderWithPaths(datasets.ImageFolder):
    """Returns (tensor, label, filepath) for GradCAM grids."""
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        path = self.samples[idx][0]
        return img, label, path


def denormalize(tensor, mean=MEAN, std=STD):
    """Denormalize a tensor for visualization: (C,H,W) -> (H,W,C) float32 [0,1]."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std  = torch.tensor(std).view(-1, 1, 1)
    return (tensor * std + mean).permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)


def main():
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    # ── Patch timm ───────────────────────────────────────────────────────────
    patch_convnext_stage()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {CKPT_PATH}')
    if not os.path.isfile(CKPT_PATH):
        # Try alternate path patterns
        alt_paths = [
            os.path.join(OUT_DIR, BEST_FOLD, BEST_RUN, 'fl',
                         f'{BEST_FOLD}_{BEST_RUN}_fl_best.pt'),
            os.path.join(OUT_DIR, BEST_FOLD, BEST_RUN,
                         f'{BEST_FOLD}_{BEST_RUN}_fl_best.pt'),
        ]
        for ap in alt_paths:
            if os.path.isfile(ap):
                CKPT_PATH_resolved = ap
                break
        else:
            # Search for it
            import glob
            matches = glob.glob(os.path.join(OUT_DIR, BEST_FOLD, '**', '*fl_best.pt'),
                                recursive=True)
            if matches:
                CKPT_PATH_resolved = matches[0]
                print(f'  Found checkpoint via glob: {CKPT_PATH_resolved}')
            else:
                raise FileNotFoundError(f'Cannot find checkpoint. Tried: {CKPT_PATH}')
    else:
        CKPT_PATH_resolved = CKPT_PATH

    model = ConvNeXtV2MSAFv5(num_classes=NUM_CLASSES, attn_type='msaf', use_aux=False)
    state = torch.load(CKPT_PATH_resolved, map_location=DEVICE)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print('  Model loaded successfully.')

    # ── Load test data ─────────────────────────────────────────────────────────
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    fold_dir = os.path.join(DATA_ROOT, BEST_FOLD, BEST_RUN)
    te_datasets = []
    for c in range(1, NUM_CLIENTS + 1):
        base = os.path.join(fold_dir, f'Client_{c}', 'Test')
        if os.path.isdir(base):
            te_datasets.append(ImageFolderWithPaths(base, transform=eval_transform))

    if not te_datasets:
        raise FileNotFoundError(f'No test datasets found in {fold_dir}')

    te_ds = ConcatDataset(te_datasets)
    te_loader = DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=4,
                           pin_memory=True)
    print(f'  Test set: {len(te_ds)} images')

    # ── Run inference + Grad-CAM++ ─────────────────────────────────────────────
    target_layers = [model.stage3[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=None)

    all_imgs_raw = []   # denormalized images for overlay
    all_labels   = []
    all_preds    = []
    all_confs    = []
    all_overlays = []

    print('Running inference + Grad-CAM++ ...')
    for imgs, labels, paths in te_loader:
        imgs = imgs.to(DEVICE)

        # Step 1: inference with no_grad (fast, no gradient needed)
        with torch.no_grad():
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

        # Step 2: Grad-CAM++ needs gradients — do NOT use torch.no_grad()
        for i in range(imgs.size(0)):
            target = ClassifierOutputTarget(preds[i].item())
            grayscale_cam = cam(input_tensor=imgs[i:i+1], targets=[target])[0]
            img_raw = denormalize(imgs[i].cpu())
            overlay = show_cam_on_image(img_raw, grayscale_cam, use_rgb=True)

            all_imgs_raw.append(img_raw)
            all_labels.append(labels[i].item())
            all_preds.append(preds[i].item())
            all_confs.append(confs[i].item())
            all_overlays.append(overlay)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_confs  = np.array(all_confs)
    print(f'  Processed {len(all_labels)} images')

    # ── Select N_PER_CLASS images per class ────────────────────────────────────
    # Strategy: prefer a mix of correct and incorrect predictions.
    # If not enough incorrect, fill with correct.
    selected = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        correct_idx = np.where((all_labels == cls_idx) & (all_preds == cls_idx))[0]
        wrong_idx   = np.where((all_labels == cls_idx) & (all_preds != cls_idx))[0]

        # Prefer 2 correct + 1 wrong if available
        chosen = []
        n_wrong = min(1, len(wrong_idx))
        n_correct = N_PER_CLASS - n_wrong

        if n_correct > 0 and len(correct_idx) > 0:
            chosen.extend(correct_idx[:n_correct])
        if n_wrong > 0 and len(wrong_idx) > 0:
            chosen.extend(wrong_idx[:n_wrong])

        # Fill remaining slots
        while len(chosen) < N_PER_CLASS:
            pool = np.where(all_labels == cls_idx)[0]
            for p in pool:
                if p not in chosen:
                    chosen.append(p)
                    break
            else:
                break

        selected[cls_name] = chosen[:N_PER_CLASS]
        n_c = sum(1 for c in chosen if all_labels[c] == all_preds[c])
        n_w = len(chosen) - n_c
        print(f'  {cls_name}: {len(chosen)} selected ({n_c} correct, {n_w} wrong)')

    # ── Create the figure ──────────────────────────────────────────────────────
    # Layout: 4 rows x (N_PER_CLASS * 2) columns
    #   Each row = one class
    #   Columns alternate: Original | Grad-CAM++ | Original | Grad-CAM++ | ...
    n_rows = len(CLASSES)
    n_cols = N_PER_CLASS * 2

    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 1.1, n_rows * 1.1 + 0.3))

    # Remove all spines and ticks
    for ax in axes.flat if n_rows > 1 else axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Column headers: "Original" and "Grad-CAM++" alternating
    for ci in range(N_PER_CLASS):
        col_orig = ci * 2
        col_cam  = ci * 2 + 1
        if n_rows > 1:
            axes[0, col_orig].set_title('Original', fontsize=7, pad=3)
            axes[0, col_cam].set_title('Grad-CAM++', fontsize=7, pad=3)
        else:
            axes[col_orig].set_title('Original', fontsize=7, pad=3)
            axes[col_cam].set_title('Grad-CAM++', fontsize=7, pad=3)

    # Row labels (class names)
    for ri, cls_name in enumerate(CLASSES):
        chosen = selected[cls_name]
        for ci, idx in enumerate(chosen):
            col_orig = ci * 2
            col_cam  = ci * 2 + 1

            if n_rows > 1:
                ax_orig = axes[ri, col_orig]
                ax_cam  = axes[ri, col_cam]
            else:
                ax_orig = axes[col_orig]
                ax_cam  = axes[col_cam]

            ax_orig.imshow(all_imgs_raw[idx])
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])

            ax_cam.imshow(all_overlays[idx])
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])

            # Row label on first column only
            if ci == 0:
                ax_orig.set_ylabel(cls_name, fontsize=8, labelpad=4,
                                   rotation=90, fontweight='bold')

    # Tight layout with minimal gaps
    plt.subplots_adjust(wspace=0.02, hspace=0.05)

    save_path = os.path.join(GRADCAM_DIR, 'gradcam_summary_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f'\nSaved: {save_path}')
    print('Done!')


if __name__ == '__main__':
    main()
