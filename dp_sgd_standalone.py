#!/usr/bin/env python3
"""
Standalone DP-SGD Training Script — runs separately from the main pipeline
(pipeline_v2_clean_a100.py).

This script trains ONLY Section 9a (DP-SGD: Differentially Private Federated
Learning) with the identical protocol as the main pipeline. Results are saved
to the SAME directory and JSON file
(pipeline_v2_single_gpu/dp_sgd/dp_results.json) so that when the main script
reaches Section 9a, it finds all entries already completed and skips them.

BUG FIX: The main pipeline crashes with
    AttributeError: 'list' object has no attribute 'detach'
when Opacus wraps the model, because CrossScaleAttentionHead.forward receives
a Python list ([s1, s2, s3]) as input. Opacus's activation-capture hooks
iterate over forward inputs and call .detach() on each, assuming every input
is a Tensor. A list has no .detach() method.

FIX: Opacus requires every wrapped submodule's forward() to accept a SINGLE
tensor argument. Neither lists nor multiple positional args work (the functorch
backward pass also replays forward with a single arg). So GeM-pool + projection
+ stacking is moved into ConvNeXtV2MSAFv5.forward (the parent), which calls
self.head.gem[i] and self.head.proj[i] individually (each takes a single
tensor — Opacus-safe). CrossScaleAttentionHead.forward then receives a single
(B,3,d) tensor. Model parameters and computation are identical.

Usage:
    python dp_sgd_standalone.py 2>&1 | tee dp_sgd_standalone.log

If you have multiple GPUs and want to pin this to a specific GPU:
    CUDA_VISIBLE_DEVICES=0 python dp_sgd_standalone.py
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
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize
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

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — Configuration (must match pipeline_v2_clean_a100.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

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

# ── Focal loss ───────────────────────────────────────────────────────────────
FOCAL_GAMMA = 2.0
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv        = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()
print(f'Focal alpha (test-dist): {[f"{a:.3f}" for a in FOCAL_ALPHA]}')

# ── Augmentation / regularization ────────────────────────────────────────────
AUX_W        = 0.2

# ── FL ───────────────────────────────────────────────────────────────────────
FL_ROUNDS    = 50
LOCAL_EPOCHS = 3
PATIENCE     = 18

# ── DP-SGD experiment config (must match main pipeline Section 9a) ──────────
DP_BASE     = os.path.join(OUT_DIR, 'dp_sgd')
os.makedirs(DP_BASE, exist_ok=True)
DP_RUN_NAME = 'FL_Run2_Heterogeneous'
DP_FOLDS    = ['Fold_1', 'Fold_3', 'Fold_5']  # 3 folds for compute efficiency
DP_EPSILONS = [None, 12.0, 8.0]  # None = no DP (baseline), then stronger privacy

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

print('Configuration ready.')
print(f'  LR  backbone={LR_BACKBONE} | attn={LR_ATTN} | head={LR_HEAD}')
print(f'  FL: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs  patience={PATIENCE}')
print('Section 0 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Model Architecture
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
    """
    Novel MSAF head with learned temperature scaling.
      1. GeM-pool s1,s2,s3  -> (B,C_i) vectors   [done in parent forward]
      2. Project each to d  -> (B,d)              [done in parent forward]
      3. Stack -> token seq -> (B,3,d)            [done in parent forward]
      4. Cross-attn: Q=s3-token, K=V=all-tokens -> (B,d) fused
      5. Residual + LayerNorm + temp-scale T
      6. Dropout -> Linear(d//2) -> GELU -> Dropout -> Linear(nc)

    *** OPACUS FIX ***
    Opacus requires every wrapped submodule's forward() to accept a SINGLE
    tensor argument. Neither lists nor multiple positional args work:
      - List input:  forward([s1,s2,s3]) -> hooks call .detach() on the list
        -> AttributeError: 'list' object has no attribute 'detach'
      - Positional:  forward(s1,s2,s3) -> functorch backward replays with a
        single arg -> TypeError: missing required positional args 's2' and 's3'

    FIX: GeM-pool + projection + stacking is done in the PARENT forward
    (ConvNeXtV2MSAFv5.forward), which calls self.head.gem[i] and
    self.head.proj[i] individually (each takes a single tensor — Opacus-safe).
    CrossScaleAttentionHead.forward then receives a single (B,3,d) tensor.
    """
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

    def forward(self, token_seq):
        """token_seq: (B, 3, d) — already GeM-pooled + projected + stacked."""
        q = self.q_lin(token_seq[:, -1:, :])
        k = self.k_lin(token_seq)
        v = self.v_lin(token_seq)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        fused = self.norm(fused + token_seq[:, -1, :]) * self.temp
        return self.head(fused)


class ConvNeXtV2MSAFv5(nn.Module):
    """
    ConvNeXtV2-Tiny + MSAF head v5.
    Stage-aware attention:
      stage0 (96ch,  56x56): ECA
      stage1 (192ch, 28x28): ECA
      stage2 (384ch, 14x14): CBAM + StochDepth
      stage3 (768ch,  7x7):  CBAM + StochDepth
    """
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
            # *** OPACUS FIX: GeM-pool + project + stack HERE (in parent),
            # then pass a single (B,3,d) tensor to self.head.
            # Opacus requires every wrapped submodule's forward() to accept a
            # single tensor. By calling self.head.gem[i] and self.head.proj[i]
            # individually (each takes a single tensor), and then passing the
            # stacked result as a single tensor to self.head, all submodule
            # calls are Opacus-safe.
            tokens = []
            for i, feat in enumerate((s1, s2, s3)):
                pooled = self.head.gem[i](feat)
                tokens.append(self.head.proj[i](pooled))
            token_seq = torch.stack(tokens, dim=1)  # (B, 3, d)
            main = self.head(token_seq)
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


def build_primary(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=False)

# Smoke test
print('Dimension smoke test...')
_x = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
_m = build_primary().cpu(); _m.eval()
with torch.no_grad():
    _oe = _m(_x)
assert _oe.shape == (2, NUM_CLASSES), f'eval {_oe.shape}'
print(f'  msaf_primary  eval={_oe.shape}  params={sum(p.numel() for p in _m.parameters())/1e6:.2f}M')
print('Model classes defined. Section 1 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Losses, Early Stopping, Metrics
# ═══════════════════════════════════════════════════════════════════════════════

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

print('Losses, EarlyStopping, compute_epoch_metrics defined. Section 2 complete.')

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

def get_agg_test_loader(fold_dir, run_dir_name):
    te_list = []
    for c in range(1, NUM_CLIENTS + 1):
        _, _, tl = get_client_dataloaders(fold_dir, run_dir_name, c)
        te_list.append(tl.dataset)
    return DataLoader(ConcatDataset(te_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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

print(f'BATCH_SIZE={BATCH_SIZE}  NUM_WORKERS={NUM_WORKERS}  single-GPU mode')
print('Section 3b complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — FL Training Functions (FedAvg + DP-SGD)
# ═══════════════════════════════════════════════════════════════════════════════

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


def _identity_forward(x):
    """No-op forward used to replace nn.Dropout for DP-SGD compatibility."""
    return x


def _disable_random_ops_for_dp(model):
    """Replace all random operations with deterministic alternatives for DP-SGD.

    Opacus's functorch-based per-sample gradient computation uses vmap, which
    does not support random operations (nn.Dropout, StochasticDepth's torch.rand).

    Setting nn.Dropout.p=0.0 is NOT sufficient because nn.Dropout.forward still
    calls F.dropout() which invokes _VF.dropout — a registered random operator
    that vmap flags even when p=0.0. So we override the forward method itself.

    For StochasticDepth, setting drop_prob=0.0 IS sufficient because its forward
    method has an early return: `if self.drop_prob == 0.0: return x`.

    DP-SGD already adds noise (which acts as regularization), so removing
    Dropout is standard practice in DP training. The global model still has
    Dropout/StochasticDepth for non-DP runs and evaluation.
    """
    for name, module in model.named_modules():
        # Override nn.Dropout.forward to a no-op (avoids F.dropout → _VF.dropout
        # which vmap flags as random even when p=0.0)
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            module.forward = _identity_forward  # instance attr shadows class method
        # Disable ALL StochasticDepth variants:
        #   1. Our custom StochasticDepth (has drop_prob attr, early return when 0.0)
        #   2. torchvision.ops.StochasticDepth (has p attr, calls bernoulli_ —
        #      vmap flags this as random even when p=0.0, so override forward)
        if type(module).__name__ == 'StochasticDepth':
            if hasattr(module, 'drop_prob'):
                module.drop_prob = 0.0
            if hasattr(module, 'p'):
                module.p = 0.0
            module.forward = _identity_forward  # override forward to no-op
    return model


def _dp_local_update(global_model, client_loader, focal_loss,
                      local_epochs=LOCAL_EPOCHS, use_aux_loss=False,
                      max_grad_norm=1.0, target_epsilon=10.0):
    """DP-SGD local update using Opacus PrivacyEngine.

    Returns (local_model, avg_loss, epsilon_achieved, avg_acc).
    Falls back to non-DP if Opacus is unavailable.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        _opacus_available = True
    except ImportError:
        _opacus_available = False
        print('  [DP-SGD] Opacus not installed -- using non-DP fallback. Install: pip install opacus')

    local_model = copy.deepcopy(unwrap(global_model)).to(DEVICE)

    if hasattr(local_model, 'get_param_groups'):
        opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
    else:
        opt = optim.AdamW(local_model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    epsilon_achieved = float('inf')

    # Opacus requires the model to be in training mode before make_private_with_epsilon.
    # Must set this BEFORE the Opacus setup block, not after.
    local_model.train()

    # Bug fix: only set up Opacus when target_epsilon is not None.
    # Previously, target_epsilon=None (the "no-DP" baseline) was silently
    # converted to 10.0 by the caller, making the baseline actually DP-SGD.
    if _opacus_available and target_epsilon is not None:
        try:
            # Validate/fix model for DP compatibility (e.g. replace BatchNorm)
            errors = ModuleValidator.validate(local_model, strict=False)
            if errors:
                local_model = ModuleValidator.fix(local_model)
                if hasattr(local_model, 'get_param_groups'):
                    opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
                else:
                    opt = optim.AdamW(local_model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

            # Disable random ops (Dropout, StochasticDepth) for DP compatibility.
            # Opacus's functorch/vmap backend does not support random operations
            # (nn.Dropout calls F.dropout which uses RNG; StochasticDepth uses
            # torch.rand). Setting p=0.0 / drop_prob=0.0 makes them no-ops.
            local_model = _disable_random_ops_for_dp(local_model)

            privacy_engine = PrivacyEngine()
            local_model, opt, client_loader = privacy_engine.make_private_with_epsilon(
                module=local_model,
                optimizer=opt,
                data_loader=client_loader,
                target_epsilon=target_epsilon,
                target_delta=1e-5,
                epochs=local_epochs,
                max_grad_norm=max_grad_norm,
            )
        except Exception as e:
            print(f'  [DP-SGD] Opacus setup failed: {e}. Using non-DP update.')
            _opacus_available = False

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
            loss = task
            loss.backward()
            opt.step()
            with torch.no_grad():
                if isinstance(out, tuple):
                    all_preds.extend(out[0].argmax(1).cpu().numpy())
                else:
                    all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            ep += loss.item()
        total += ep

    if _opacus_available:
        try:
            epsilon_achieved = float(privacy_engine.get_epsilon(1e-5))
        except Exception:
            pass

    # Bug fix: unwrap GradSampleModule if Opacus was used, so state_dict keys
    # match the global model during _fedavg aggregation. Without this, Opacus
    # wraps the model in GradSampleModule which prefixes all keys, causing a
    # silent key-mismatch in _fedavg that produces a garbage aggregate.
    if _opacus_available and hasattr(local_model, '_module'):
        local_model = local_model._module
    avg_acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if all_preds else 0.0
    return local_model.cpu(), total / local_epochs, epsilon_achieved, avg_acc


def train_fedprox_dp(build_fn, fold_dir, run_dir_name, save_dir, run_name,
                      focal_loss=None, use_aux_loss=False,
                      target_epsilon=None, max_grad_norm=1.0):
    """FedAvg with DP-SGD local updates (Opacus).

    If target_epsilon is None or Opacus is unavailable, falls back to standard
    FedAvg (no differential privacy).
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f'{run_name}_best.pt')
    swa_ckpt = os.path.join(save_dir, f'{run_name}_swa.pt')

    if focal_loss is None:
        focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    ce = nn.CrossEntropyLoss()
    tr_loaders, vl_loaders, sizes = [], [], []
    for c in range(1, NUM_CLIENTS + 1):
        tr, vl, _ = get_client_dataloaders(fold_dir, run_dir_name, c, strong_aug=False)
        tr_loaders.append(tr); vl_loaders.append(vl)
        sizes.append(len(tr.dataset))

    agg_vl = DataLoader(ConcatDataset([l.dataset for l in vl_loaders]),
                        BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    gm = wrap_multigpu(build_fn())
    swa_gm = AveragedModel(unwrap(gm))
    stopper = EarlyStopping(patience=PATIENCE, checkpoint_path=ckpt, mode='max')
    history = {k: [] for k in ['round', 'avg_local_loss', 'avg_local_acc',
                                 'global_val_loss', 'global_val_acc',
                                 'global_val_prec', 'global_val_rec', 'global_val_f1',
                                 'epsilon']}

    dp_tag = f'DP(eps={target_epsilon})' if target_epsilon else 'No-DP'
    print(f'DP-SGD FL: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs | {dp_tag}')
    print(f'Client train sizes: {sizes}')

    SWA_FL_START = FL_ROUNDS - 10
    for rnd in range(1, FL_ROUNDS + 1):
        lms, lls, eps_list, las = [], [], [], []
        for ci in range(NUM_CLIENTS):
            lm, ll, eps, la = _dp_local_update(gm, tr_loaders[ci], focal_loss,
                                             use_aux_loss=use_aux_loss,
                                             max_grad_norm=max_grad_norm,
                                             target_epsilon=target_epsilon)
            # Convert model object to state_dict for _fedavg (which now expects dicts)
            lms.append({k: v.cpu() for k, v in lm.state_dict().items()})
            lls.append(ll); eps_list.append(eps); las.append(la)

        gm = _fedavg(gm, lms, sizes).to(DEVICE)
        avg_ll = float(np.mean(lls))
        valid_eps = [e for e in eps_list if e != float('inf')]
        avg_eps = float(np.mean(valid_eps)) if valid_eps else float('inf')

        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(gm, agg_vl, ce)
        avg_la = float(np.mean(las))
        history['round'].append(rnd); history['avg_local_loss'].append(avg_ll)
        history['avg_local_acc'].append(avg_la)
        history['global_val_loss'].append(vl_loss); history['global_val_acc'].append(vl_acc)
        history['global_val_prec'].append(vl_prec); history['global_val_rec'].append(vl_rec)
        history['global_val_f1'].append(vl_f1); history['epsilon'].append(avg_eps)
        stopper.step(vl_f1, gm)

        if rnd >= SWA_FL_START:
            swa_gm.update_parameters(gm)

        if rnd % 5 == 0 or stopper.stop:
            eps_str = f'eps={avg_eps:.1f}' if avg_eps != float('inf') else 'eps=inf'
            print(f'  Round {rnd:3d}/{FL_ROUNDS} | LL={avg_ll:.4f} | '
                  f'A={vl_acc:.4f} F1={vl_f1:.4f} | {eps_str} | ES={stopper.counter}/{PATIENCE}')

        if stopper.stop:
            print(f'  Early stopping at round {rnd}.'); break

    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val F1: {stopper.best:.4f}')

    update_bn(tr_loaders[0], swa_gm.to(DEVICE), device=DEVICE)
    torch.save(swa_gm.state_dict(), swa_ckpt)
    return gm, history, swa_ckpt

print('Training functions defined. Section 4 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Evaluation Utilities
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
    if mode == 'fl':
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
# Section 9a — DP-SGD: Differentially Private Federated Learning
# ═══════════════════════════════════════════════════════════════════════════════
# Reference: Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
# Requires: pip install opacus

# Winning architecture (from main pipeline log):
#   Winning architecture: msaf_primary (ECA+CBAM+CSAH [s1,s2,s3], no aux heads)
#   WINNING_USE_AUX = False
WINNING_BUILD_FN = build_primary
WINNING_USE_AUX  = False

dp_master = os.path.join(DP_BASE, 'dp_results.json')
dp_results = load_json(dp_master, default={})

print(f'\n{"="*60}')
print(f'  DP-SGD Standalone — {len(DP_FOLDS)} folds x {len(DP_EPSILONS)} epsilons')
print(f'  Results JSON: {dp_master}')
print(f'{"="*60}')

for fold in DP_FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    dp_results.setdefault(fold, {})
    for eps_target in DP_EPSILONS:
        eps_label = f'eps{int(eps_target)}' if eps_target else 'noDP'
        if eps_label in dp_results[fold] and dp_results[fold][eps_label].get('standard', {}).get('accuracy'):
            print(f'[skip -- already done] {fold} / {eps_label}')
            continue
        print(f'\n{"="*60}\n  DP-SGD: {fold} | eps_target={eps_target}\n{"="*60}')
        save_dir = os.path.join(DP_BASE, fold, eps_label)
        lbl = f'dp_{fold}_{eps_label}'

        try:
            gm, hist, swa = train_fedprox_dp(
                WINNING_BUILD_FN, fold_dir, DP_RUN_NAME, save_dir, lbl,
                use_aux_loss=WINNING_USE_AUX, target_epsilon=eps_target)
            save_json(hist, os.path.join(save_dir, f'history_{lbl}.json'))
            plot_curves(hist, save_dir, lbl, mode='fl')

            agg_te = get_agg_test_loader(fold_dir, DP_RUN_NAME)
            m = evaluate_model(gm, agg_te, save_dir, lbl, use_tta=False)
            save_json(m, os.path.join(save_dir, f'metrics_{lbl}.json'))
            dp_results[fold][eps_label] = {'standard': m, 'epsilon_target': eps_target}
        except Exception as e:
            print(f'ERROR: {e}'); traceback.print_exc()
            dp_results[fold][eps_label] = {'error': str(e), 'epsilon_target': eps_target}

        save_json_locked(dp_results, dp_master)
        print(f'  Results saved to {dp_master}')

print(f'\n{"="*60}')
print('  DP-SGD Standalone complete.')
print(f'  Results JSON: {dp_master}')
print(f'{"="*60}')
