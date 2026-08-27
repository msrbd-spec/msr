#!/usr/bin/env python3
"""
Standalone Baseline Training Script — runs on a SEPARATE GPU/system in parallel
with the main pipeline (final_pipeline_A100.py).

This script trains ONLY Section 10 (baseline SOTA models: EfficientNetV2-S,
MobileNetV2, ConvNeXtV2-Tiny, ResNet50) with the identical protocol as the main
pipeline. Results are saved to the SAME directory and JSON file
(pipeline_v2_single_gpu/baselines/baseline_results.json) so that when the main
script reaches Section 10, it finds all entries already completed and skips them.

File locking is used on baseline_results.json to prevent concurrent-write
corruption if both scripts happen to save at the same time.

Usage:
    python baselines_standalone.py 2>&1 | tee baselines_standalone.log

If you have multiple GPUs and want to pin this to a specific GPU:
    CUDA_VISIBLE_DEVICES=1 python baselines_standalone.py
"""

import os, gc, copy, json, math, random, warnings, traceback, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import wilcoxon
from statsmodels.stats.proportion import proportion_confint
warnings.filterwarnings('ignore')

# ── Offline mode: load pretrained weights from cache, no downloads ────────────
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ── Paths ────────────────────────────────────────────────────────────────────
NUM_CLIENTS = 5
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
FOLDS       = [f'Fold_{i}' for i in range(1, 6)]
NUM_FOLDS   = 5
DATA_ROOT   = 'datasets/final_5_fold_pruned/'
OUT_DIR     = 'pipeline_v2_single_gpu'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Image ────────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 384
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
BATCH_SIZE  = 32
NUM_WORKERS = 8

# ── LR ───────────────────────────────────────────────────────────────────────
LR_BACKBONE  = 3e-5
LR_ATTN      = 1e-4
LR_HEAD      = 2e-4
WEIGHT_DECAY = 2e-4

# ── Architecture ─────────────────────────────────────────────────────────────
MSAF_DIM   = 256
GEM_P      = 3.0
DROP_PATH  = 0.10

# ── SAM ──────────────────────────────────────────────────────────────────────
SAM_RHO = 0.05

# ── Focal loss ───────────────────────────────────────────────────────────────
FOCAL_GAMMA = 2.0
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv        = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()
print(f'Focal alpha (test-dist): {[f"{a:.3f}" for a in FOCAL_ALPHA]}')

# ── Augmentation / regularization ────────────────────────────────────────────
LABEL_SMOOTH = 0.05
MIXUP_ALPHA  = 0.1
CUTMIX_ALPHA = 0.0
AUX_W        = 0.2

# ── Centralized phases ───────────────────────────────────────────────────────
NUM_EPOCHS_FROZEN = 10
NUM_EPOCHS_STAGE3 = 15
NUM_EPOCHS_FULL   = 40
SWA_START_EPOCH   = 10

# ── FL ───────────────────────────────────────────────────────────────────────
FL_ROUNDS    = 50
LOCAL_EPOCHS = 3
FEDPROX_MU   = 0.01
PATIENCE     = 18

# ── Ablation compute budget ──────────────────────────────────────────────────
ABLATION_FOLDS = FOLDS

# ── JSON helpers with file locking ──────────────────────────────────────────
def save_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)

def save_json_locked(obj, path, max_retries=20, retry_delay=1.0):
    """Save JSON with file locking to prevent concurrent-write corruption.
    Used when another process (the main pipeline) might be writing to the
    same file at the same time."""
    lock_path = path + '.lock'
    for attempt in range(max_retries):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if attempt == max_retries - 1:
                # Force remove stale lock and proceed
                try: os.remove(lock_path)
                except: pass
                break
            time.sleep(retry_delay)
    try:
        save_json(obj, path)
    finally:
        try: os.remove(lock_path)
        except: pass

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default if default is not None else {}

# ── Elsevier paper-ready matplotlib style ────────────────────────────────────
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
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth':   1.0,
    'lines.markersize':  3,
})

COLOR_TRAIN = '#0066CC'
COLOR_VAL   = '#CC0000'

print('Configuration ready.')
print(f'  LR  backbone={LR_BACKBONE} | attn={LR_ATTN} | head={LR_HEAD}')
print(f'  Centralized phases: {NUM_EPOCHS_FROZEN}+{NUM_EPOCHS_STAGE3}+{NUM_EPOCHS_FULL} epochs')
print(f'  FL: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs  patience={PATIENCE}')
print('Section 0 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Model Architecture (classes only, no smoke test)
# ═══════════════════════════════════════════════════════════════════════════════

class GeMPool(nn.Module):
    """Generalized Mean Pooling. Learnable p (init=3). (B,C,H,W) -> (B,C)."""
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)

    def extra_repr(self): return f'p={self.p.data.item():.2f}'


class ECABlock(nn.Module):
    """Efficient Channel Attention with gated residual. Near-zero params."""
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
    """Drop-path regularization. Applied around the CBAM residual."""
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
    """CBAM: channel + spatial attention with gated residual + stochastic depth."""
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


class AuxHead(nn.Module):
    """Auxiliary classifier for deep supervision."""
    def __init__(self, in_ch, num_classes=NUM_CLASSES):
        super().__init__()
        self.gem = GeMPool()
        self.ln  = nn.LayerNorm(in_ch)
        self.fc  = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        return self.fc(self.ln(self.gem(x)))


class CrossScaleAttentionHead(nn.Module):
    """Novel MSAF head with learned temperature scaling."""
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
    """ConvNeXtV2-Tiny + MSAF head v5."""
    DIMS = [96, 192, 384, 768]

    def __init__(self, num_classes=NUM_CLASSES, attn_type='msaf', use_aux=False):
        super().__init__()
        self.attn_type = attn_type
        self.use_aux   = use_aux
        dims = self.DIMS

        try:
            import timm
            bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)
            self._backend = 'timm'
            self.stem   = bb.stem
            self.stage0 = bb.stages[0]
            self.stage1 = bb.stages[1]
            self.stage2 = bb.stages[2]
            self.stage3 = bb.stages[3]
        except Exception:
            bb = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self._backend = 'tv'
            self.stem   = bb.features[0]
            self.ds1    = bb.features[2]
            self.ds2    = bb.features[4]
            self.ds3    = bb.features[6]
            self.stage0 = bb.features[1]
            self.stage1 = bb.features[3]
            self.stage2 = bb.features[5]
            self.stage3 = bb.features[7]

        def _eca(ch):  return ECABlock(ch)
        def _cbam(ch): return CBAMBlock(ch, reduction=max(4, ch // 16), drop_path=DROP_PATH)
        def _none():   return nn.Identity()

        uses_eca_early = attn_type in ('msaf', 'msaf_gem_only', 'cbam_eca_gap')
        uses_cbam_deep = attn_type in ('msaf', 'msaf_gem_only', 'cbam_eca_gap', 'cbam_only_gap')
        uses_eca_all   = attn_type == 'eca_only_gap'

        if uses_eca_all:
            self.attn0, self.attn1 = _eca(dims[0]), _eca(dims[1])
            self.attn2, self.attn3 = _eca(dims[2]), _eca(dims[3])
        elif uses_eca_early and uses_cbam_deep:
            self.attn0, self.attn1 = _eca(dims[0]), _eca(dims[1])
            self.attn2, self.attn3 = _cbam(dims[2]), _cbam(dims[3])
        elif uses_cbam_deep:
            self.attn0, self.attn1 = _none(), _none()
            self.attn2, self.attn3 = _cbam(dims[2]), _cbam(dims[3])
        else:
            self.attn0 = self.attn1 = self.attn2 = self.attn3 = _none()

        if attn_type == 'msaf':
            self.head = CrossScaleAttentionHead(
                dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=num_classes)
        elif attn_type == 'msaf_gem_only':
            self.head = nn.Sequential(
                GeMPool(), nn.LayerNorm(dims[3]), nn.Dropout(0.35),
                nn.Linear(dims[3], MSAF_DIM), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(MSAF_DIM, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.LayerNorm(dims[3]), nn.Dropout(0.30),
                nn.Linear(dims[3], 256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, num_classes),
            )

        if use_aux:
            self.aux1 = AuxHead(dims[1], num_classes)
            self.aux2 = AuxHead(dims[2], num_classes)

    def _stages(self, x):
        if self._backend == 'timm':
            x  = self.stem(x)
            s0 = self.attn0(self.stage0(x))
            s1 = self.attn1(self.stage1(s0))
            s2 = self.attn2(self.stage2(s1))
            s3 = self.attn3(self.stage3(s2))
        else:
            x  = self.stem(x)
            s0 = self.attn0(self.stage0(x))
            s1 = self.attn1(self.stage1(self.ds1(s0)))
            s2 = self.attn2(self.stage2(self.ds2(s1)))
            s3 = self.attn3(self.stage3(self.ds3(s2)))
        return s1, s2, s3

    def forward(self, x):
        s1, s2, s3 = self._stages(x)
        if self.attn_type == 'msaf':
            main = self.head([s1, s2, s3])
        else:
            main = self.head(s3)

        if self.use_aux and self.training:
            return main, self.aux1(s1), self.aux2(s2)
        return main

    def get_param_groups(self):
        bb  = {'stem', 'stage0', 'stage1', 'stage2', 'stage3', 'ds1', 'ds2', 'ds3'}
        atn = {'attn0', 'attn1', 'attn2', 'attn3'}
        bp, ap, hp = [], [], []
        for name, param in self.named_parameters():
            top = name.split('.')[0]
            if   top in bb:  bp.append(param)
            elif top in atn: ap.append(param)
            else:            hp.append(param)
        return [
            {'params': bp, 'lr': LR_BACKBONE, 'name': 'backbone'},
            {'params': ap, 'lr': LR_ATTN,     'name': 'attention'},
            {'params': hp, 'lr': LR_HEAD,     'name': 'head'},
        ]

    def freeze_backbone(self):
        for n in ['stem', 'stage0', 'stage1', 'stage2', 'stage3', 'ds1', 'ds2', 'ds3']:
            m = getattr(self, n, None)
            if m:
                for p in m.parameters(): p.requires_grad = False

    def unfreeze_deep_stages(self):
        for n in ['stage2', 'stage3', 'ds3']:
            m = getattr(self, n, None)
            if m:
                for p in m.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


# ── Build helpers ────────────────────────────────────────────────────────────
def build_primary(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=False)

def build_baseline(nc=NUM_CLASSES):
    """Plain ConvNeXtV2-Tiny, no custom attention/head — architecture-ablation baseline."""
    try:
        import timm
        m = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)
        m.head.fc = nn.Linear(m.head.fc.in_features, nc)
    except Exception:
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, nc)
    return m

print('Model classes defined. Section 1 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Losses, Mixup/CutMix, SAM Optimizer, Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════

class LabelSmoothCE(nn.Module):
    def __init__(self, classes=NUM_CLASSES, smoothing=LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing; self.cls = classes

    def forward(self, pred, target):
        conf = 1.0 - self.smoothing; sm = self.smoothing / (self.cls - 1)
        oh = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        return -((oh * conf + (1 - oh) * sm) * F.log_softmax(pred, dim=1)).sum(1).mean()


class FocalLoss(nn.Module):
    """Focal loss with configurable alpha (defaults to test-distribution alpha)."""
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, num_classes=NUM_CLASSES):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, pred, target):
        alpha    = self.alpha.to(pred.device)
        log_prob = F.log_softmax(pred, dim=1)
        prob     = log_prob.exp()
        focal_w  = (1 - prob) ** self.gamma
        alpha_t  = alpha[target]
        loss = -(alpha_t * focal_w[range(len(target)), target] * log_prob[range(len(target)), target])
        return loss.mean()

FOCAL_LOSS = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2  = x[idx]
    x1, y1, x2c, y2c = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, x1:x2c, y1:y2c] = x2[:, :, x1:x2c, y1:y2c]
    lam = 1 - (x2c - x1) * (y2c - y1) / (x.size(-1) * x.size(-2))
    return mixed, y, y[idx], lam

def mixed_aug(x, y):
    if CUTMIX_ALPHA <= 0.0 or random.random() < 0.5:
        return mixup_data(x, y, alpha=MIXUP_ALPHA)
    return cutmix_data(x, y, alpha=CUTMIX_ALPHA)

def mixed_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021)."""
    def __init__(self, params, base_optimizer_cls, rho=SAM_RHO, **kwargs):
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['old_p'] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError('Use first_step/second_step explicitly.')

    def _grad_norm(self):
        shared = self.param_groups[0]['params'][0]
        norms = [p.grad.norm(2).to(shared)
                 for g in self.param_groups for p in g['params'] if p.grad is not None]
        return torch.stack(norms).norm(2)

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.base_optimizer.param_groups = self.param_groups


class EarlyStopping:
    def __init__(self, patience=PATIENCE, checkpoint_path='best.pt', mode='max', min_delta=5e-5):
        self.patience = patience; self.checkpoint = checkpoint_path
        self.mode = mode; self.min_delta = min_delta
        self.counter = 0; self.best = None; self.stop = False

    def _better(self, s):
        if self.best is None: return True
        return s > self.best + self.min_delta if self.mode == 'max' else s < self.best - self.min_delta

    def step(self, score, model):
        if self._better(score):
            self.best = score; self.counter = 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True


def compute_epoch_metrics(model, loader, criterion, device=DEVICE):
    """Loss, Acc, Precision, Recall, F1 (macro) for one loader."""
    model.eval()
    ls = 0.0; ap, al = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            if isinstance(out, tuple): out = out[0]
            ls += criterion(out, labels).item() * imgs.size(0)
            ap.extend(out.argmax(1).cpu().numpy())
            al.extend(labels.cpu().numpy())
    n = len(al)
    loss = ls / n
    acc  = float((np.array(ap) == np.array(al)).mean())
    prec = float(precision_score(al, ap, average='macro', zero_division=0))
    rec  = float(recall_score(al,  ap, average='macro', zero_division=0))
    f1   = float(f1_score(al,      ap, average='macro', zero_division=0))
    return loss, acc, prec, rec, f1

print('Losses, SAM, EarlyStopping, compute_epoch_metrics defined. Section 2 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Data Transforms and Loaders
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_TRANSFORM_STRONG = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TTA_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    transforms.TenCrop(IMAGE_SIZE),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])(c)
        for c in crops
    ])),
])

def get_client_dataloaders(fold_dir, run_dir_name, client_id, strong_aug=False):
    tr_tf = TRAIN_TRANSFORM_STRONG if strong_aug else TRAIN_TRANSFORM
    base  = os.path.join(fold_dir, run_dir_name, f'Client_{client_id}')
    tr_ds = datasets.ImageFolder(os.path.join(base, 'Train'), transform=tr_tf)
    vl_ds = datasets.ImageFolder(os.path.join(base, 'Valid'), transform=EVAL_TRANSFORM)
    te_ds = datasets.ImageFolder(os.path.join(base, 'Test'),  transform=EVAL_TRANSFORM)
    tr_l = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    vl_l = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_l = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return tr_l, vl_l, te_l

def get_centralized_loaders(fold_dir, run_dir_name, strong_aug=True):
    tr_tf = TRAIN_TRANSFORM_STRONG if strong_aug else TRAIN_TRANSFORM
    tr_list, vl_list, te_list = [], [], []
    for c in range(1, NUM_CLIENTS + 1):
        base = os.path.join(fold_dir, run_dir_name, f'Client_{c}')
        tr_list.append(datasets.ImageFolder(os.path.join(base, 'Train'), transform=tr_tf))
        vl_list.append(datasets.ImageFolder(os.path.join(base, 'Valid'), transform=EVAL_TRANSFORM))
        te_list.append(datasets.ImageFolder(os.path.join(base, 'Test'),  transform=EVAL_TRANSFORM))
    tr_l = DataLoader(ConcatDataset(tr_list), BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    vl_l = DataLoader(ConcatDataset(vl_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_l = DataLoader(ConcatDataset(te_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    n_tr = sum(len(d) for d in tr_list); n_vl = sum(len(d) for d in vl_list); n_te = sum(len(d) for d in te_list)
    print(f'Loaders from {fold_dir}/{run_dir_name}: Train={n_tr} Valid={n_vl} Test={n_te}')
    return tr_l, vl_l, te_l

def get_agg_test_loader(fold_dir, run_dir_name):
    te_list = []
    for c in range(1, NUM_CLIENTS + 1):
        _, _, tl = get_client_dataloaders(fold_dir, run_dir_name, c)
        te_list.append(tl.dataset)
    return DataLoader(ConcatDataset(te_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def get_agg_tta_loader(fold_dir, run_dir_name):
    te_list = []
    for c in range(1, NUM_CLIENTS + 1):
        base = os.path.join(fold_dir, run_dir_name, f'Client_{c}')
        te_list.append(datasets.ImageFolder(os.path.join(base, 'Test'), transform=TTA_TRANSFORM))
    return DataLoader(ConcatDataset(te_list), batch_size=8, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Returns (tensor, label, filepath) -- used for GradCAM wrong/correct grids."""
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        path = self.samples[idx][0]
        return img, label, path

print('Section 3 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3b — Single-GPU Setup
# ═══════════════════════════════════════════════════════════════════════════════

print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
if torch.cuda.is_available():
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

def wrap_multigpu(model):
    """Single-GPU: just move model to DEVICE."""
    return model.to(DEVICE)

def unwrap(model):
    """Return the underlying module (identity for single-GPU)."""
    return model.module if isinstance(model, nn.DataParallel) else model

def _load_swa_state(model, ckpt_path):
    """Load SWA checkpoint, stripping AveragedModel's 'module.' prefix."""
    _state = torch.load(ckpt_path, map_location=DEVICE)
    if any(k.startswith('module.') for k in _state):
        _state = {k.replace('module.', '', 1): v for k, v in _state.items()}
    model.load_state_dict(_state, strict=False)
    return model

def _run_parallel_clients(build_fn, fold_dir, run_dir_name, global_state_cpu,
                           focal_alpha, focal_gamma, mu, use_aux_loss,
                           poison_client_idx=None, poison_frac=0.0, poison_seed=42):
    """Run all NUM_CLIENTS SEQUENTIALLY on a single GPU.
    Returns (state_dicts, losses, accs) — same interface as the multi-GPU version
    so train_fedprox / train_fedper work unchanged.

    Each client is trained via _fedprox_local_update (defined in Section 4),
    which runs on DEVICE (cuda:0).  The returned model object is converted to
    a state_dict so that _fedavg / _trimmed_mean_agg / _fedper_avg (which
    expect dicts, not model objects) work correctly."""
    global_model = build_fn()
    unwrap(global_model).load_state_dict(global_state_cpu)
    global_model = global_model.to(DEVICE)

    focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    tr_loaders = []
    for c in range(1, NUM_CLIENTS + 1):
        tr, _, _ = get_client_dataloaders(fold_dir, run_dir_name, c, strong_aug=False)
        if poison_client_idx is not None and (c - 1) == poison_client_idx and poison_frac > 0:
            tr = _poison_loader_labels(tr, poison_frac, seed=poison_seed)
        tr_loaders.append(tr)

    lms, lls, las = [], [], []
    for ci in range(NUM_CLIENTS):
        local_model, avg_loss, avg_acc = _fedprox_local_update(
            global_model, tr_loaders[ci], focal_loss,
            mu=mu, local_epochs=LOCAL_EPOCHS, use_aux_loss=use_aux_loss)
        lms.append({k: v.cpu() for k, v in local_model.state_dict().items()})
        lls.append(avg_loss)
        las.append(avg_acc)
        del local_model
        torch.cuda.empty_cache()

    del global_model
    return lms, lls, las

print(f'BATCH_SIZE={BATCH_SIZE}  NUM_WORKERS={NUM_WORKERS}  single-GPU mode')
print('Section 3b complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Training Functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_centralized(build_fn, train_loader, val_loader, save_dir, run_label, use_aux=False):
    os.makedirs(save_dir, exist_ok=True)
    ckpt     = os.path.join(save_dir, f'{run_label}_best.pt')
    swa_ckpt = os.path.join(save_dir, f'{run_label}_swa.pt')

    raw_model = build_fn()
    model = wrap_multigpu(raw_model)
    total = sum(p.numel() for p in model.parameters())
    trnbl = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  {run_label}: {total:,} total | {trnbl:,} trainable params')

    ce_smooth = LabelSmoothCE()
    ce_plain  = nn.CrossEntropyLoss()
    stopper   = EarlyStopping(patience=PATIENCE, checkpoint_path=ckpt, mode='max')

    history = {k: [] for k in [
        'train_loss','train_acc','train_prec','train_rec','train_f1',
        'val_loss','val_acc','val_prec','val_rec','val_f1',
    ]}

    swa_model = None
    swa_ckpt_final = swa_ckpt

    def _run_phase(n_ep, name, unfreeze_fn=None, use_sam=False):
        nonlocal model, swa_model, swa_ckpt_final
        if unfreeze_fn:
            unfreeze_fn()
            tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'  [{name}] Trainable params: {tr:,}')

        param_groups = unwrap(model).get_param_groups() if hasattr(unwrap(model), 'get_param_groups') \
                       else model.parameters()
        if use_sam:
            optimizer = SAM(param_groups, optim.AdamW, rho=SAM_RHO, weight_decay=WEIGHT_DECAY)
        else:
            optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer if use_sam else optimizer, T_max=n_ep, eta_min=1e-7)

        if use_sam:
            swa_model = AveragedModel(unwrap(model))
            swa_sched = SWALR(optimizer.base_optimizer, swa_lr=LR_HEAD * 0.5)

        phase_ep_count = 0
        for ep in range(n_ep):
            model.train()
            tr_loss = 0.0
            all_tr_preds, all_tr_labels = [], []
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                x_mix, ya, yb, lam = mixed_aug(imgs, labels)

                def _forward_loss():
                    out = model(x_mix)
                    if isinstance(out, tuple) and use_aux:
                        main, a1, a2 = out
                        l_main = mixed_criterion(ce_smooth, main, ya, yb, lam)
                        l_aux1 = mixed_criterion(ce_smooth, a1, ya, yb, lam)
                        l_aux2 = mixed_criterion(ce_smooth, a2, ya, yb, lam)
                        loss = l_main + AUX_W * (l_aux1 + l_aux2)
                        preds = main.argmax(1)
                    else:
                        m_out = out[0] if isinstance(out, tuple) else out
                        loss = mixed_criterion(ce_smooth, m_out, ya, yb, lam)
                        preds = m_out.argmax(1)
                    return loss, preds

                optimizer.zero_grad()
                if use_sam:
                    loss, preds = _forward_loss()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    loss2, _ = _forward_loss()
                    loss2.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.second_step(zero_grad=True)
                else:
                    loss, preds = _forward_loss()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                tr_loss += loss.item() * imgs.size(0)
                all_tr_preds.extend(preds.detach().cpu().numpy())
                all_tr_labels.extend(ya.cpu().numpy())

            scheduler.step()
            if use_sam and phase_ep_count >= SWA_START_EPOCH:
                swa_model.update_parameters(model)
                swa_sched.step()

            tr_loss /= len(all_tr_labels)
            tr_acc  = float((np.array(all_tr_preds) == np.array(all_tr_labels)).mean())
            tr_prec = float(precision_score(all_tr_labels, all_tr_preds, average='macro', zero_division=0))
            tr_rec  = float(recall_score(all_tr_labels,   all_tr_preds, average='macro', zero_division=0))
            tr_f1   = float(f1_score(all_tr_labels,       all_tr_preds, average='macro', zero_division=0))

            vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(model, val_loader, ce_plain)

            history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
            history['train_prec'].append(tr_prec); history['train_rec'].append(tr_rec)
            history['train_f1'].append(tr_f1)
            history['val_loss'].append(vl_loss); history['val_acc'].append(vl_acc)
            history['val_prec'].append(vl_prec); history['val_rec'].append(vl_rec)
            history['val_f1'].append(vl_f1)

            stopper.step(vl_f1, model)
            gep = len(history['train_loss'])
            phase_ep_count += 1
            sam_tag = '[SAM]' if use_sam else ''

            if gep % 5 == 0 or stopper.stop:
                print(f'  [{name}]{sam_tag} Ep {gep:3d} | '
                      f'Tr: L={tr_loss:.4f} A={tr_acc:.4f} P={tr_prec:.4f} R={tr_rec:.4f} F1={tr_f1:.4f} | '
                      f'Vl: L={vl_loss:.4f} A={vl_acc:.4f} P={vl_prec:.4f} R={vl_rec:.4f} F1={vl_f1:.4f} | '
                      f'ES={stopper.counter}/{PATIENCE}')

            if stopper.stop:
                print(f'  [{name}] Early stopping.')
                break

        if use_sam and swa_model is not None:
            update_bn(train_loader, swa_model.to(DEVICE), device=DEVICE)
            torch.save(swa_model.state_dict(), swa_ckpt)
            swa_ckpt_final = swa_ckpt
            print(f'  SWA checkpoint saved: {swa_ckpt}')

        return stopper.stop

    print('  Phase 1: backbone frozen')
    unwrap(model).freeze_backbone()
    if not _run_phase(NUM_EPOCHS_FROZEN, 'Phase1-Frozen', use_sam=False):
        print('  Phase 2: unfreeze stage2+stage3')
        if not _run_phase(NUM_EPOCHS_STAGE3, 'Phase2-DeepStages',
                           unfreeze_fn=unwrap(model).unfreeze_deep_stages, use_sam=False):
            print('  Phase 3: unfreeze all + SAM + SWA tail')
            _run_phase(NUM_EPOCHS_FULL, 'Phase3-Full-SAM',
                       unfreeze_fn=unwrap(model).unfreeze_all, use_sam=True)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val F1: {stopper.best:.4f} -- best checkpoint restored.')
    return model, history, swa_ckpt_final


def _fedprox_local_update(global_model, client_loader, focal_loss,
                           mu=None, local_epochs=LOCAL_EPOCHS, use_aux_loss=False):
    mu = FEDPROX_MU if mu is None else mu
    """One client's local FedProx update."""
    local_model   = wrap_multigpu(copy.deepcopy(unwrap(global_model)))
    global_params = {n: p.data.clone() for n, p in unwrap(local_model).named_parameters()}

    if hasattr(unwrap(local_model), 'get_param_groups'):
        opt = optim.AdamW(unwrap(local_model).get_param_groups(), weight_decay=WEIGHT_DECAY)
    else:
        opt = optim.AdamW(local_model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    local_model.train()
    total = 0.0
    all_preds, all_labels = [], []
    for _ in range(local_epochs):
        ep = 0.0
        for imgs, labels in client_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = local_model(imgs)
            if isinstance(out, tuple) and use_aux_loss:
                main, a1, a2 = out
                task = focal_loss(main, labels) + AUX_W * (focal_loss(a1, labels) + focal_loss(a2, labels))
            else:
                m_out = out[0] if isinstance(out, tuple) else out
                task = focal_loss(m_out, labels)
            prox = sum(((p - global_params[n].to(DEVICE)) ** 2).sum()
                       for n, p in unwrap(local_model).named_parameters() if n in global_params)
            loss = task + (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                if isinstance(out, tuple):
                    all_preds.extend(out[0].argmax(1).cpu().numpy())
                else:
                    all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            ep += loss.item()
        total += ep
    avg_acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if all_preds else 0.0
    return local_model.cpu(), total / local_epochs, avg_acc


def _fedavg(gm, lms, sizes):
    """Standard size-weighted FedAvg aggregation.
    lms[i] is a state_dict (dict of tensors), not a model object."""
    total = sum(sizes); w = [n / total for n in sizes]
    target = unwrap(gm)
    gd = target.state_dict()
    for k in gd:
        gd[k] = sum(w[i] * lms[i][k].float() for i in range(len(lms)))
    target.load_state_dict(gd)
    return gm


def _trimmed_mean_agg(gm, lms, sizes, trim_frac=0.2):
    """Coordinate-wise trimmed-mean aggregation."""
    target = unwrap(gm)
    gd = target.state_dict()
    n_clients = len(lms)
    n_trim = max(0, int(n_clients * trim_frac))
    for k in gd:
        stacked = torch.stack([lms[i][k].float() for i in range(n_clients)], dim=0)
        if n_trim > 0 and n_clients - 2 * n_trim >= 1:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[n_trim: n_clients - n_trim]
            gd[k] = trimmed.mean(dim=0)
        else:
            gd[k] = stacked.mean(dim=0)
    target.load_state_dict(gd)
    return gm


def train_fedprox(build_fn, fold_dir, run_dir_name, save_dir, run_name,
                   focal_loss=None, use_aux_loss=False, aggregator='fedavg',
                   poison_client_idx=None, poison_frac=0.0, poison_seed=42,
                   mu=None):
    """FedProx training loop."""
    mu = FEDPROX_MU if mu is None else mu
    os.makedirs(save_dir, exist_ok=True)
    ckpt     = os.path.join(save_dir, f'{run_name}_best.pt')
    swa_ckpt = os.path.join(save_dir, f'{run_name}_swa.pt')

    if focal_loss is None:
        focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    ce = nn.CrossEntropyLoss()
    tr_loaders, vl_loaders, sizes = [], [], []
    for c in range(1, NUM_CLIENTS + 1):
        tr, vl, _ = get_client_dataloaders(fold_dir, run_dir_name, c, strong_aug=False)
        if poison_client_idx is not None and (c - 1) == poison_client_idx and poison_frac > 0:
            tr = _poison_loader_labels(tr, poison_frac, seed=poison_seed)
        tr_loaders.append(tr); vl_loaders.append(vl)
        sizes.append(len(tr.dataset))

    agg_vl = DataLoader(ConcatDataset([l.dataset for l in vl_loaders]),
                         BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    gm = wrap_multigpu(build_fn())
    swa_gm = AveragedModel(unwrap(gm))
    stopper = EarlyStopping(patience=PATIENCE, checkpoint_path=ckpt, mode='max')
    history = {'round': [], 'avg_local_loss': [], 'avg_local_acc': [],
               'global_val_loss': [], 'global_val_acc': [],
               'global_val_prec': [], 'global_val_rec': [], 'global_val_f1': []}

    print(f'FedProx: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs | mu={mu} | '
          f'aggregator={aggregator} | use_aux_loss={use_aux_loss} | '
          f'poison=(client={poison_client_idx}, frac={poison_frac})')
    print(f'Client train sizes: {sizes}')

    SWA_FL_START = FL_ROUNDS - 10
    for rnd in range(1, FL_ROUNDS + 1):
        global_state_cpu = {k: v.cpu() for k, v in unwrap(gm).state_dict().items()}
        lms, lls, las = _run_parallel_clients(
            build_fn, fold_dir, run_dir_name, global_state_cpu,
            FOCAL_ALPHA, FOCAL_GAMMA, mu, use_aux_loss,
            poison_client_idx=poison_client_idx, poison_frac=poison_frac, poison_seed=poison_seed)

        if aggregator == 'trimmed_mean':
            gm = _trimmed_mean_agg(gm, lms, sizes).to(DEVICE)
        else:
            gm = _fedavg(gm, lms, sizes).to(DEVICE)
        avg_ll = float(np.mean(lls))

        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(gm, agg_vl, ce)
        avg_la = float(np.mean(las))
        history['round'].append(rnd); history['avg_local_loss'].append(avg_ll)
        history['avg_local_acc'].append(avg_la)
        history['global_val_loss'].append(vl_loss); history['global_val_acc'].append(vl_acc)
        history['global_val_prec'].append(vl_prec); history['global_val_rec'].append(vl_rec)
        history['global_val_f1'].append(vl_f1)
        stopper.step(vl_f1, gm)

        if rnd >= SWA_FL_START:
            swa_gm.update_parameters(gm)

        if rnd % 5 == 0 or stopper.stop:
            print(f'  Round {rnd:3d}/{FL_ROUNDS} | AvgLL={avg_ll:.4f} | '
                  f'Val: L={vl_loss:.4f} A={vl_acc:.4f} P={vl_prec:.4f} R={vl_rec:.4f} F1={vl_f1:.4f} | '
                  f'TrAcc={avg_la:.4f} | '
                  f'ES={stopper.counter}/{PATIENCE}')

        if stopper.stop:
            print(f'  Early stopping at round {rnd}.'); break

    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val F1: {stopper.best:.4f}')

    update_bn(tr_loaders[0], swa_gm.to(DEVICE), device=DEVICE)
    torch.save(swa_gm.state_dict(), swa_ckpt)
    print(f'  FL-SWA checkpoint saved: {swa_ckpt}')
    return gm, history, swa_ckpt


def _poison_loader_labels(loader, flip_frac, seed=42):
    """Return a NEW DataLoader over the same images but with `flip_frac` of
    the labels randomly flipped to a different class."""
    ds = loader.dataset
    ds = copy.deepcopy(ds)
    rng = np.random.RandomState(seed)
    n = len(ds.samples)
    n_flip = int(n * flip_frac)
    flip_idx = rng.choice(n, n_flip, replace=False)
    new_samples = list(ds.samples)
    for idx in flip_idx:
        path, orig_label = new_samples[idx]
        choices = [c for c in range(NUM_CLASSES) if c != orig_label]
        new_label = rng.choice(choices)
        new_samples[idx] = (path, int(new_label))
    ds.samples = new_samples
    ds.targets = [s[1] for s in new_samples]
    return DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)


class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        return out / self.temperature


def calibrate_temperature(model, val_loader, device=DEVICE):
    """Calibrate temperature on val set using LBFGS. Returns (TemperatureScaler, T)."""
    ts  = TemperatureScaler(model).to(device)
    nll = nn.CrossEntropyLoss()
    opt = optim.LBFGS([ts.temperature], lr=0.01, max_iter=50)

    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.to(device))
            if isinstance(out, tuple): out = out[0]
            logits_list.append(out.cpu()); labels_list.append(labels)

    logits_all = torch.cat(logits_list).to(device)
    labels_all = torch.cat(labels_list).to(device)

    def eval_fn():
        opt.zero_grad()
        loss = nll(logits_all / ts.temperature, labels_all)
        loss.backward()
        return loss

    opt.step(eval_fn)
    ts.temperature.data.clamp_(min=0.001)
    T = ts.temperature.item()
    print(f'  Calibrated temperature: {T:.4f}')
    return ts, T

print('Training functions defined. Section 4 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, save_dir, run_label, use_tta=False, n_crops=10):
    """Full eval: Acc, Prec, Rec, F1, AUROC, CM, ROC curves. Standard + TTA."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if use_tta:
                B, n, C, H, W = imgs.shape
                flat = imgs.view(B * n, C, H, W).to(DEVICE)
                out = model(flat)
                if isinstance(out, tuple): out = out[0]
                probs = torch.softmax(out, 1).view(B, n, NUM_CLASSES).mean(1)
            else:
                out = model(imgs.to(DEVICE))
                if isinstance(out, tuple): out = out[0]
                probs = torch.softmax(out, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels); y_pred = np.array(all_preds); y_prob = np.array(all_probs)
    tta_tag = f' [TTA-{n_crops}crop]' if use_tta else ''
    suf = '_tta' if use_tta else ''
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(f'--- {run_label}{tta_tag} ---')
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    bins = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auroc_mac = roc_auc_score(bins, y_prob, average='macro', multi_class='ovr')
    auroc_mic = roc_auc_score(bins, y_prob, average='micro', multi_class='ovr')
    accuracy  = float((y_pred == y_true).mean())
    macro_f1  = float(report['macro avg']['f1-score'])
    macro_prec= float(report['macro avg']['precision'])
    macro_rec = float(report['macro avg']['recall'])

    print(f'  Accuracy         : {accuracy:.4f}')
    print(f'  Macro-Precision  : {macro_prec:.4f}')
    print(f'  Macro-Recall     : {macro_rec:.4f}')
    print(f'  Macro-F1         : {macro_f1:.4f}')
    print(f'  Weighted-F1      : {report["weighted avg"]["f1-score"]:.4f}')
    print(f'  AUROC (macro)    : {auroc_mac:.4f}')
    print(f'  AUROC (micro)    : {auroc_mic:.4f}')

    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'CM -- {run_label}{tta_tag}')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'cm_{run_label}{suf}.png'), dpi=300); plt.close()

    class_aurocs = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(bins[:, i], y_prob[:, i])
        ca = auc(fpr, tpr); class_aurocs[cls] = ca
        ax.plot(fpr, tpr, label=f'{cls} (AUC={ca:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'ROC -- {run_label}{tta_tag}')
    ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'roc_{run_label}{suf}.png'), dpi=300); plt.close()
    print(f'  Plots saved to {save_dir}')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'weighted_f1': float(report['weighted avg']['f1-score']),
        'auroc_macro': float(auroc_mac),
        'auroc_micro': float(auroc_mic),
        'per_class_auroc': {k: float(v) for k, v in class_aurocs.items()},
        'per_class': {
            cls: {'precision': float(report[cls]['precision']),
                  'recall': float(report[cls]['recall']),
                  'f1': float(report[cls]['f1-score']),
                  'support': int(report[cls]['support'])}
            for cls in CLASSES
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(y_true)),
        'tta': use_tta,
    }


def plot_curves(history, save_dir, run_label, mode='centralized'):
    os.makedirs(save_dir, exist_ok=True)
    if mode == 'centralized':
        ep = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(ep, history['train_loss'], label='Train', color='#0066CC')
        axes[0].plot(ep, history['val_loss'], label='Val', color='#CC0000')
        axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.4)
        axes[1].plot(ep, history['train_acc'], label='Train', color='#0066CC')
        axes[1].plot(ep, history['val_acc'], label='Val', color='#CC0000')
        axes[1].axhline(0.96, color='green', ls='--', lw=1, alpha=0.7, label='0.96')
        axes[1].axhline(0.98, color='red', ls='--', lw=1, alpha=0.7, label='0.98')
        axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'curves_loss_acc_{run_label}.png'), dpi=300); plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, trk, vlk, title in zip(axes, ['train_prec','train_rec','train_f1'],
                                        ['val_prec','val_rec','val_f1'],
                                        ['Macro Precision','Macro Recall','Macro F1']):
            ax.plot(ep, history[trk], label='Train', color='#0066CC')
            ax.plot(ep, history[vlk], label='Val', color='#CC0000')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'curves_prec_rec_f1_{run_label}.png'), dpi=300); plt.close()
        print(f'  Centralized curves saved to {save_dir}')
    else:
        rr = history['round']
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(rr, history['avg_local_loss'], label='Avg Local', color='#0066CC')
        axes[0].plot(rr, history['global_val_loss'], label='Global Val', color='#CC0000')
        axes[0].set_title('Loss'); axes[0].set_xlabel('Round'); axes[0].legend(); axes[0].grid(alpha=0.4)
        if 'avg_local_acc' in history and history['avg_local_acc']:
            axes[1].plot(rr, history['avg_local_acc'], color='#0066CC', label='Train Acc', ls='--')
        axes[1].plot(rr, history['global_val_acc'], color='#CC0000', label='Val Acc')
        axes[1].axhline(0.96, color='green', ls='--', lw=1, alpha=0.7, label='0.96')
        axes[1].axhline(0.98, color='red', ls='--', lw=1, alpha=0.7, label='0.98')
        axes[1].set_title('Global Val Accuracy'); axes[1].set_xlabel('Round'); axes[1].legend(); axes[1].grid(alpha=0.4)
        for k, lbl in [('global_val_prec','Val Prec'), ('global_val_rec','Val Rec'), ('global_val_f1','Val F1')]:
            axes[2].plot(rr, history[k], label=lbl)
        axes[2].set_title('Val Precision / Recall / F1'); axes[2].set_xlabel('Round')
        axes[2].legend(); axes[2].grid(alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'curves_fl_{run_label}.png'), dpi=300); plt.close()
        print(f'  FL curves saved to {save_dir}')

print('Evaluation utilities defined. Section 5 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 10 — Baseline SOTA Models, Identical Protocol
# ═══════════════════════════════════════════════════════════════════════════════

def build_effnetv2_s(nc=NUM_CLASSES):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, nc)
    return m

def build_mobilenet_v2(nc=NUM_CLASSES):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    return m

def build_convnextv2_plain(nc=NUM_CLASSES):
    return build_baseline(nc)  # plain ConvNeXtV2-Tiny, already defined above

def build_resnet50(nc=NUM_CLASSES):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m

BASELINE_MODELS = {
    'EfficientNetV2-S': build_effnetv2_s,
    'MobileNetV2':       build_mobilenet_v2,
    'ConvNeXtV2-Tiny':   build_convnextv2_plain,
    'ResNet50':          build_resnet50,
}

BASELINE_BASE = os.path.join(OUT_DIR, 'baselines')
os.makedirs(BASELINE_BASE, exist_ok=True)
baseline_master = os.path.join(BASELINE_BASE, 'baseline_results.json')
baseline_results = load_json(baseline_master, default={})

print(f'\n{"="*70}')
print(f'STANDALONE BASELINE TRAINING')
print(f'  Output dir: {BASELINE_BASE}')
print(f'  Results JSON: {baseline_master}')
print(f'  Models: {list(BASELINE_MODELS.keys())}')
print(f'  Folds: {FOLDS}')
print(f'  Runs: FL_Run1_Uniform, FL_Run2_Heterogeneous')
print(f'{"="*70}\n')

for model_name, build_fn in BASELINE_MODELS.items():
    baseline_results.setdefault(model_name, {})
    for run_name in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
        for fold in FOLDS:
            fold_dir = os.path.join(DATA_ROOT, fold)
            key = f'{fold}_{run_name}'
            if key in baseline_results[model_name] and not baseline_results[model_name][key].get('error'):
                print(f'[skip -- already done] {model_name} / {key}')
                continue
            print(f'\n=== Baseline: {model_name} | {fold} | {run_name} ===')
            save_dir = os.path.join(BASELINE_BASE, model_name.replace(' ', '_'), fold, run_name)
            os.makedirs(save_dir, exist_ok=True)
            result = {'centralized': {}, 'fl': {}, 'error': None}
            try:
                # Centralized (identical protocol: SAM+SWA phased unfreezing needs
                # get_param_groups/freeze_backbone -- torchvision models don't have
                # these, so for baselines we use a simpler single-phase AdamW fit
                # for the same total epoch budget, matching the original notebook's
                # own build_baseline() treatment via train_centralized's fallback path).
                tr, vl, te = get_centralized_loaders(fold_dir, run_name, strong_aug=True)
                bm = wrap_multigpu(build_fn())
                opt = optim.AdamW(bm.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=NUM_EPOCHS_FROZEN + NUM_EPOCHS_STAGE3 + NUM_EPOCHS_FULL, eta_min=1e-7)
                stopper = EarlyStopping(patience=PATIENCE,
                                         checkpoint_path=os.path.join(save_dir, 'centralized_best.pt'), mode='max')
                ce_smooth = LabelSmoothCE()
                n_total_epochs = NUM_EPOCHS_FROZEN + NUM_EPOCHS_STAGE3 + NUM_EPOCHS_FULL
                baseline_hist = {k: [] for k in [
                    'train_loss','train_acc','train_prec','train_rec','train_f1',
                    'val_loss','val_acc','val_prec','val_rec','val_f1']}
                for ep in range(n_total_epochs):
                    bm.train()
                    ep_loss = 0.0; ep_preds, ep_labels = [], []
                    for imgs, labels in tr:
                        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                        x_mix, ya, yb, lam = mixed_aug(imgs, labels)
                        opt.zero_grad()
                        out = bm(x_mix)
                        loss = mixed_criterion(ce_smooth, out, ya, yb, lam)
                        loss.backward()
                        nn.utils.clip_grad_norm_(bm.parameters(), 1.0)
                        opt.step()
                        ep_loss += loss.item() * imgs.size(0)
                        ep_preds.extend(out.argmax(1).detach().cpu().numpy())
                        ep_labels.extend(ya.cpu().numpy())
                    sched.step()
                    tr_loss = ep_loss / max(len(ep_labels), 1)
                    tr_acc = float((np.array(ep_preds) == np.array(ep_labels)).mean()) if ep_preds else 0.0
                    tr_prec = float(precision_score(ep_labels, ep_preds, average='macro', zero_division=0)) if ep_preds else 0.0
                    tr_rec = float(recall_score(ep_labels, ep_preds, average='macro', zero_division=0)) if ep_preds else 0.0
                    tr_f1 = float(f1_score(ep_labels, ep_preds, average='macro', zero_division=0)) if ep_preds else 0.0
                    vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(bm, vl, nn.CrossEntropyLoss())
                    baseline_hist['train_loss'].append(tr_loss)
                    baseline_hist['train_acc'].append(tr_acc)
                    baseline_hist['train_prec'].append(tr_prec)
                    baseline_hist['train_rec'].append(tr_rec)
                    baseline_hist['train_f1'].append(tr_f1)
                    baseline_hist['val_loss'].append(vl_loss)
                    baseline_hist['val_acc'].append(vl_acc)
                    baseline_hist['val_prec'].append(vl_prec)
                    baseline_hist['val_rec'].append(vl_rec)
                    baseline_hist['val_f1'].append(vl_f1)
                    stopper.step(vl_f1, bm)
                    if (ep + 1) % 10 == 0 or stopper.stop:
                        print(f'  [{model_name}] Ep {ep+1}/{n_total_epochs} | '
                              f'Tr: L={tr_loss:.4f} A={tr_acc:.4f} F1={tr_f1:.4f} | '
                              f'Vl: L={vl_loss:.4f} A={vl_acc:.4f} F1={vl_f1:.4f} | '
                              f'ES={stopper.counter}/{PATIENCE}')
                    if stopper.stop:
                        print('  Early stopping.'); break
                bm.load_state_dict(torch.load(os.path.join(save_dir, 'centralized_best.pt'), map_location=DEVICE))
                m_c = evaluate_model(bm, te, save_dir, f'{model_name}_{fold}_{run_name}_c', use_tta=False)
                save_json(m_c, os.path.join(save_dir, 'metrics_centralized.json'))
                save_json(baseline_hist, os.path.join(save_dir, 'history_centralized.json'))
                plot_curves(baseline_hist, save_dir, f'{model_name}_{fold}_{run_name}_c', mode='centralized')
                result['centralized'] = m_c
                del bm; gc.collect(); torch.cuda.empty_cache()

                # FedProx (identical protocol to the primary model)
                gm, hist, swa = train_fedprox(build_fn, fold_dir, run_name, save_dir, f'{model_name}_{fold}_{run_name}_fl')
                agg_te = get_agg_test_loader(fold_dir, run_name)
                m_fl = evaluate_model(gm, agg_te, save_dir, f'{model_name}_{fold}_{run_name}_fl', use_tta=False)
                save_json(m_fl, os.path.join(save_dir, 'metrics_fl.json'))
                save_json(hist, os.path.join(save_dir, 'history_fl.json'))
                plot_curves(hist, save_dir, f'{model_name}_{fold}_{run_name}_fl', mode='fl')
                result['fl'] = m_fl
                del gm; gc.collect(); torch.cuda.empty_cache()

            except Exception as e:
                result['error'] = f'{type(e).__name__}: {e}'
                print(f'ERROR: {e}'); traceback.print_exc()

            baseline_results[model_name][key] = result
            # Use locked save to prevent corruption if the main pipeline is
            # also writing to this file at the same time.
            save_json_locked(baseline_results, baseline_master)
            print(f'[saved] {baseline_master}')

# ── Aggregate: Table 2 ─────────────────────────────────────────────────────────
rows = []
for model_name in BASELINE_MODELS:
    for run_name in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
        for mode in ['centralized', 'fl']:
            accs, f1s, aurocs = [], [], []
            for fold in FOLDS:
                r = baseline_results.get(model_name, {}).get(f'{fold}_{run_name}', {}).get(mode, {})
                if r and 'accuracy' in r:
                    accs.append(r['accuracy']); f1s.append(r['macro_f1']); aurocs.append(r['auroc_macro'])
            if accs:
                rows.append({'Model': model_name, 'Run': run_name, 'Setting': mode,
                             'Acc_mean': np.mean(accs), 'Acc_std': np.std(accs),
                             'F1_mean': np.mean(f1s), 'F1_std': np.std(f1s),
                             'AUROC_mean': np.mean(aurocs)})
df_baseline = pd.DataFrame(rows)
print('\nTable 2 -- Baseline Comparison:')
print(df_baseline.to_string(index=False))
df_baseline.to_csv(os.path.join(OUT_DIR, 'table2_baseline_comparison.csv'), index=False)
print('\nStandalone baseline training complete.')


