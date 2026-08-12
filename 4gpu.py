import os, gc, copy, json, math, random, warnings, traceback
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
OUT_DIR     = 'pipeline_v2_clean_4gpu'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Image ────────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 384
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
BATCH_SIZE  = 16
NUM_WORKERS = 4

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
# Alpha computed from the TEST-distribution class counts (not train counts) —
# justified in Methodology: post-augmentation train counts are ~equalized,
# so a train-based alpha would be ~uniform and uninformative.
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv        = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()
print(f'Focal alpha (test-dist): {[f"{a:.3f}" for a in FOCAL_ALPHA]}')

# ── Augmentation / regularization ────────────────────────────────────────────
LABEL_SMOOTH = 0.05
MIXUP_ALPHA  = 0.1
CUTMIX_ALPHA = 0.0   # kept disabled as in the original tuned config
AUX_W        = 0.2   # weight for auxiliary-head loss when aux heads are used

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
# Fix #2: run every ablation across multiple folds, not just one.
# Set to FOLDS for a full 5-fold ablation (expensive: ~40-60 FL runs per
# ablation family). If your compute budget is limited, explicitly reduce this
# list (e.g. ['Fold_1','Fold_3','Fold_5']) and DISCLOSE the reduction in the
# paper's Methodology — do not silently run fewer folds than stated.
ABLATION_FOLDS = FOLDS  # <-- change here if you must reduce, and say so in the paper

def save_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default if default is not None else {}

# ── Elsevier paper-ready matplotlib style ────────────────────────────────────
# Global rcParams for publication-quality figures (Elsevier single/double column).
# Font: Arial (sans-serif) at 7-8pt; DPI 300 for raster fallback; vector-safe.
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

# ── Color constants: train=blue, val=red (consistent everywhere) ─────────────
COLOR_TRAIN = '#0066CC'   # blue
COLOR_VAL   = '#CC0000'   # red

print('Configuration ready.')
print(f'  LR  backbone={LR_BACKBONE} | attn={LR_ATTN} | head={LR_HEAD}')
print(f'  Centralized phases: {NUM_EPOCHS_FROZEN}+{NUM_EPOCHS_STAGE3}+{NUM_EPOCHS_FULL} epochs')
print(f'  FL: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs  patience={PATIENCE}')
print(f'  Ablation folds: {ABLATION_FOLDS}')
print('Section 0 complete.')

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
      1. GeM-pool s1,s2,s3  -> (B,C_i) vectors
      2. Project each to d  -> (B,d)
      3. Stack -> token seq -> (B,3,d)
      4. Cross-attn: Q=s3-token, K=V=all-tokens -> (B,d) fused
      5. Residual + LayerNorm + temp-scale T
      6. Dropout -> Linear(d//2) -> GELU -> Dropout -> Linear(nc)
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
    """
    ConvNeXtV2-Tiny + MSAF head v5.
    Stage-aware attention:
      stage0 (96ch,  56x56): ECA
      stage1 (192ch, 28x28): ECA
      stage2 (384ch, 14x14): CBAM + StochDepth
      stage3 (768ch,  7x7):  CBAM + StochDepth

    Head variants (attn_type):
      'msaf'           - PRIMARY: ECA+CBAM+CSAH (cross-scale attention)
      'msaf_gem_only'  - ECA+CBAM + GeM head (ablation: no cross-scale)
      'cbam_eca_gap'   - ECA+CBAM + GAP head  (ablation: no GeM/no cross-scale)
      'cbam_only_gap'  - CBAM only + GAP head
      'eca_only_gap'   - ECA only  + GAP head
      'none_gap'       - no attention + GAP head

    use_aux: adds AuxHead on stage1/stage2 for deep supervision (training only).
             NOTE: unlike the original notebook, this flag is now honoured by
             BOTH train_centralized AND train_fedprox (see Section 3) — the
             original silently dropped aux outputs inside FedProx.
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


# ── Build helpers (architecture-ablation variants) ────────────────────────────
def build_primary(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=False)
def build_primary_aux(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=True)
def build_msaf_gem_only(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='msaf_gem_only', use_aux=False)
def build_cbam_eca_gap(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='cbam_eca_gap', use_aux=False)
def build_cbam_only_gap(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='cbam_only_gap', use_aux=False)
def build_eca_only_gap(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='eca_only_gap', use_aux=False)
def build_none_gap(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='none_gap', use_aux=False)

def build_no_stoch_depth(nc=NUM_CLASSES):
    """CBAM with drop_path=0.0 -- stochastic depth disabled."""
    class CBAMNoSD(nn.Module):
        def __init__(self, channels, reduction=16, spatial_kernel=7, init_alpha=0.01):
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
        def _ch(self, x):
            b, c, _, _ = x.shape
            mx = self.max_pool(x).view(b, c); av = self.avg_pool(x).view(b, c)
            gate = self.ch_sig(self.ch_fc2(F.relu(self.ch_fc1(mx), inplace=True)) +
                                self.ch_fc2(F.relu(self.ch_fc1(av), inplace=True))).view(b, c, 1, 1)
            return x * gate
        def _sp(self, x):
            sp = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
            return x * self.sp_sig(self.sp_conv(sp))
        def forward(self, x):
            x_attn = self._sp(self._ch(x))
            return x + self.alpha * (x_attn - x)  # no stochastic depth

    m = ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=False)
    dims = ConvNeXtV2MSAFv5.DIMS
    m.attn2 = CBAMNoSD(dims[2], reduction=max(4, dims[2] // 16))
    m.attn3 = CBAMNoSD(dims[3], reduction=max(4, dims[3] // 16))
    return m

def build_no_gem_pool(nc=NUM_CLASSES):
    """CSAH head with AdaptiveAvgPool instead of GeMPool."""
    class CrossScaleAvgHead(nn.Module):
        def __init__(self, dims, d=MSAF_DIM, num_classes=NUM_CLASSES):
            super().__init__()
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in dims])
            self.proj = nn.ModuleList([
                nn.Sequential(nn.Linear(c, d, bias=False), nn.LayerNorm(d)) for c in dims])
            self.q_lin = nn.Linear(d, d, bias=False)
            self.k_lin = nn.Linear(d, d, bias=False)
            self.v_lin = nn.Linear(d, d, bias=False)
            self.scale = d ** -0.5
            self.norm  = nn.LayerNorm(d)
            self.temp  = nn.Parameter(torch.ones(1))
            self.head  = nn.Sequential(nn.Dropout(0.35), nn.Linear(d, d // 2),
                                        nn.GELU(), nn.Dropout(0.15), nn.Linear(d // 2, num_classes))
        def forward(self, feat_list):
            tokens = []
            for i, feat in enumerate(feat_list):
                pooled = self.pool[i](feat).flatten(1)
                tokens.append(self.proj[i](pooled))
            seq = torch.stack(tokens, dim=1)
            q = self.q_lin(seq[:, -1:, :]); k = self.k_lin(seq); v = self.v_lin(seq)
            attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
            fused = (attn @ v).squeeze(1)
            fused = self.norm(fused + tokens[-1]) * self.temp
            return self.head(fused)

    m = ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=False)
    dims = ConvNeXtV2MSAFv5.DIMS
    m.head = CrossScaleAvgHead(dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=nc)
    return m

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

# ── Architecture ablation registry (Fix #2) ───────────────────────────────────
# 'msaf_primary_aux' is the NEW config that properly tests aux heads under FedProx
# (the original notebook never actually did this -- see the note in Section 0).
ARCH_CFGS = {
    'baseline':        (build_baseline,        'timm head, no custom attention'),
    'none_gap':        (build_none_gap,        'new GAP head, no attention'),
    'eca_only_gap':    (build_eca_only_gap,    'ECA all 4 stages + GAP head'),
    'cbam_eca_gap':    (build_cbam_eca_gap,    'ECA(0,1)+CBAM(2,3) + GAP head'),
    'msaf_gem_only':   (build_msaf_gem_only,   'ECA+CBAM + GeM(s3), no cross-scale'),
    'no_stoch_depth':  (build_no_stoch_depth,  'CBAM without stochastic depth'),
    'no_gem_pool':     (build_no_gem_pool,     'CSAH with AvgPool (no GeM)'),
    'msaf_primary':    (build_primary,         'ECA+CBAM+CSAH [s1,s2,s3], no aux heads'),
    'msaf_primary_aux':(build_primary_aux,     'ECA+CBAM+CSAH [s1,s2,s3], WITH aux heads (FL, fixed)'),
}

# ── Smoke test ────────────────────────────────────────────────────────────────
print('Dimension smoke test...')
_x = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
for _n, (_fn, _desc) in ARCH_CFGS.items():
    _m = _fn().cpu(); _m.eval()
    with torch.no_grad():
        _oe = _m(_x)
    assert _oe.shape == (2, NUM_CLASSES), f'{_n} eval {_oe.shape}'
    _m.train()
    with torch.no_grad():
        _ot = _m(_x)
    _sh = [o.shape for o in _ot] if isinstance(_ot, tuple) else [_ot.shape]
    total = sum(p.numel() for p in _m.parameters())
    print(f'  {_n:20s}  eval={_oe.shape}  train={_sh}  params={total/1e6:.2f}M')
    del _m
print('All variants passed. Section 1 complete.')

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

print('Sanity check -- Fold_1 dataset sizes:')
for run in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
    fold1_dir = os.path.join(DATA_ROOT, 'Fold_1')
    if os.path.isdir(os.path.join(fold1_dir, run)):
        try:
            get_centralized_loaders(fold1_dir, run, strong_aug=False)
        except Exception as e:
            print(f'  {run}: error: {e}')
    else:
        print(f'  {run}: path not found -- will work once {DATA_ROOT} is present')
print('Section 3 complete.')

NUM_GPUS = torch.cuda.device_count()
print(f'Visible GPUs: {NUM_GPUS}')
for i in range(NUM_GPUS):
    print(f'  [{i}] {torch.cuda.get_device_name(i)}  '
          f'{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB')

# Raised for V100 (16/32GB) vs. the original 12.9GB card.
BATCH_SIZE  = 32
NUM_WORKERS = 8

def wrap_multigpu(model):
    """Wrap in DataParallel when >1 GPU is visible. Use `unwrap()` to reach
    custom methods (get_param_groups, freeze_backbone, ...) through .module."""
    model = model.to(DEVICE)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
    return model

def unwrap(model):
    """Return the underlying module regardless of DataParallel wrapping."""
    return model.module if isinstance(model, nn.DataParallel) else model

print(f'BATCH_SIZE={BATCH_SIZE}  NUM_WORKERS={NUM_WORKERS}  multi-GPU={NUM_GPUS > 1}')
print('Section 3b complete.')

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
    """One client's local FedProx update.
    Fix: when the model returns (main, aux1, aux2) AND use_aux_loss=True, the
    auxiliary heads now receive a real weighted loss (AUX_W) -- the original
    notebook discarded aux outputs unconditionally here.
    """
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
    """Standard size-weighted FedAvg aggregation."""
    total = sum(sizes); w = [n / total for n in sizes]
    target = unwrap(gm)
    gd = target.state_dict()
    for k in gd:
        gd[k] = sum(w[i] * unwrap(lms[i]).state_dict()[k].float() for i in range(len(lms)))
    target.load_state_dict(gd)
    return gm


def _trimmed_mean_agg(gm, lms, sizes, trim_frac=0.2):
    """Coordinate-wise trimmed-mean aggregation -- a simple Byzantine-robust
    alternative to FedAvg, used in the poisoning-robustness experiment
    (Section 9). Drops the top/bottom `trim_frac` client values per-parameter
    before averaging, so a small number of poisoned clients cannot dominate
    the aggregate for any given weight."""
    target = unwrap(gm)
    gd = target.state_dict()
    n_clients = len(lms)
    n_trim = max(0, int(n_clients * trim_frac))
    for k in gd:
        stacked = torch.stack([unwrap(lms[i]).state_dict()[k].float() for i in range(n_clients)], dim=0)
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
    """FedProx training loop.
    New optional args (used by the ablation, Section 6, and the poisoning
    experiment, Section 9):
      use_aux_loss      -- pass through to _fedprox_local_update (Fix: real aux-head-in-FL test)
      aggregator        -- 'fedavg' (default) or 'trimmed_mean' (robust aggregation)
      poison_client_idx -- 0-based client index to poison (label-flip), or None
      poison_frac       -- fraction of that client's local labels to randomly flip
      mu                -- FedProx proximal coefficient override. None -> FEDPROX_MU
                           (0.01, the paper's default). Pass mu=0.0 to reproduce
                           plain FedAvg local updates (no proximal regularization)
                           for the poisoning-robustness comparison in Section 9.
    """
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
        lms, lls, las = [], [], []
        for ci in range(NUM_CLIENTS):
            lm, ll, la = _fedprox_local_update(gm, tr_loaders[ci], focal_loss, mu=mu, use_aux_loss=use_aux_loss)
            lms.append(lm); lls.append(ll); las.append(la)

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
    the labels randomly flipped to a different class -- used to simulate a
    malicious/mislabeled client in Section 9 (poisoning robustness)."""
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

print('Training functions defined (train_centralized, FedProx [+aux, +robust agg, +poisoning], calibration).')
print('Section 4 complete.')


def evaluate_model(model, loader, save_dir, run_label, use_tta=False, n_crops=10):
    """Full eval: Acc, Prec, Rec, F1, AUROC, CM, ROC curves. Standard + TTA.
    Returns a dict with EVERY metric as its own separate field (not only
    bundled inside the classification_report), so downstream table/CSV code
    never has to re-derive accuracy or F1 from the report dict."""
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

    # Separate, explicit metric printout (point 5 of the request)
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


def evaluate_ensemble(m1, m2, loader, save_dir, run_label):
    """Average softmax probabilities from two models (best_ckpt + SWA)."""
    m1.eval(); m2.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            o1 = m1(imgs); o1 = o1[0] if isinstance(o1, tuple) else o1
            o2 = m2(imgs); o2 = o2[0] if isinstance(o2, tuple) else o2
            probs = torch.softmax((o1 + o2) / 2, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels); y_pred = np.array(all_preds); y_prob = np.array(all_probs)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    bins  = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auroc = roc_auc_score(bins, y_prob, average='macro', multi_class='ovr')
    acc   = float((y_pred == y_true).mean())
    mf1   = float(report['macro avg']['f1-score'])
    mprec = float(report['macro avg']['precision'])
    mrec  = float(report['macro avg']['recall'])

    print(f'--- {run_label} [Ensemble] ---')
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print(f'  Accuracy: {acc:.2f} | Macro-Prec: {mprec:.2f} | Macro-Rec: {mrec:.2f} | '
          f'Macro-F1: {mf1:.2f} | AUROC: {auroc:.2f}')

    # ── Save CM + ROC plots (same style as evaluate_model) ──────────────
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'CM Ensemble -- {run_label}')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'cm_ensemble_{run_label}.png'), dpi=300); plt.close()

    class_aurocs = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(bins[:, i], y_prob[:, i])
        ca = auc(fpr, tpr); class_aurocs[cls] = ca
        ax.plot(fpr, tpr, label=f'{cls} (AUC={ca:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'ROC Ensemble -- {run_label}')
    ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'roc_ensemble_{run_label}.png'), dpi=300); plt.close()
    print(f'  Ensemble CM + ROC plots saved to {save_dir}')

    return {'accuracy': acc, 'macro_f1': mf1, 'macro_precision': mprec,
            'macro_recall': mrec, 'auroc_macro': float(auroc),
            'auroc_micro': float(roc_auc_score(bins, y_prob, average='micro', multi_class='ovr')),
            'per_class_auroc': {k: float(v) for k, v in class_aurocs.items()},
            'per_class': {
                cls: {'precision': float(report[cls]['precision']),
                      'recall': float(report[cls]['recall']),
                      'f1': float(report[cls]['f1-score']),
                      'support': int(report[cls]['support'])}
                for cls in CLASSES},
            'classification_report': report, 'confusion_matrix': cm.tolist(),
            'n_samples': int(len(y_true))}


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


ABL_RUN_NAME = 'FL_Run1_Uniform'   # ablation always run on the IID split, as in the original
ABL_BASE     = os.path.join(OUT_DIR, 'ablation_architecture')
os.makedirs(ABL_BASE, exist_ok=True)

def run_ablation_config(cfg_name, build_fn, fold_dir, run_dir_name, save_dir,
                          use_aux_loss=False, focal_loss=None):
    """Train + evaluate ONE ablation config on ONE fold, saving every metric
    immediately after it's computed. Returns a result dict (possibly partial
    if something failed partway -- partial results are still saved to disk)."""
    os.makedirs(save_dir, exist_ok=True)
    abl_name = f'abl_{cfg_name}'
    result = {'standard': {}, 'tta': {}, 'ts': {}, 'ts_tta': {}, 'error': None}

    try:
        gm, hist, swa = train_fedprox(build_fn, fold_dir, run_dir_name, save_dir, abl_name,
                                       focal_loss=focal_loss, use_aux_loss=use_aux_loss)
        plot_curves(hist, save_dir, abl_name, mode='fl')
        save_json(hist, os.path.join(save_dir, f'history_{abl_name}.json'))

        agg_te  = get_agg_test_loader(fold_dir, run_dir_name)
        agg_tta = get_agg_tta_loader(fold_dir, run_dir_name)
        _, vl, _ = get_centralized_loaders(fold_dir, run_dir_name, strong_aug=False)

        # 1) Standard -- save immediately
        m_std = evaluate_model(gm, agg_te, save_dir, abl_name, use_tta=False)
        save_json(m_std, os.path.join(save_dir, f'metrics_{abl_name}.json'))
        result['standard'] = m_std

        # 2) TTA -- distinct label, save immediately
        m_tta = evaluate_model(gm, agg_tta, save_dir, abl_name + '_tta', use_tta=True, n_crops=10)
        save_json(m_tta, os.path.join(save_dir, f'metrics_{abl_name}_tta.json'))
        result['tta'] = m_tta

        # 3) Temperature-scaled -- distinct label, save immediately
        ts_model, T = calibrate_temperature(gm, vl)
        save_json({'temperature': T}, os.path.join(save_dir, f'temperature_{abl_name}.json'))
        m_ts = evaluate_model(ts_model, agg_te, save_dir, abl_name + '_ts', use_tta=False)
        save_json(m_ts, os.path.join(save_dir, f'metrics_{abl_name}_ts.json'))
        result['ts'] = m_ts

        # 4) TS + TTA -- ITS OWN distinct label (Fix: original notebook reused
        #    the '_ts' label here too, causing a file/plot collision that
        #    crashed the cell and lost this config's results entirely).
        m_ts_tta = evaluate_model(ts_model, agg_tta, save_dir, abl_name + '_ts_tta', use_tta=True, n_crops=10)
        save_json(m_ts_tta, os.path.join(save_dir, f'metrics_{abl_name}_ts_tta.json'))
        result['ts_tta'] = m_ts_tta

        del gm, ts_model
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        print(f'ERROR in {cfg_name} on {fold_dir}: {e}')
        traceback.print_exc()
        print('Any metrics computed before the failure (see above) are already saved to disk.')

    return result


def run_multi_fold_ablation(cfg_dict, folds, run_dir_name, base_dir, use_aux_map=None, focal_map=None):
    """Runs every (config, fold) pair, saving/patching into a master JSON
    after EVERY single config so partial progress always survives a crash or
    a kernel restart."""
    use_aux_map = use_aux_map or {}
    focal_map = focal_map or {}
    master_path = os.path.join(base_dir, 'ablation_multifold_results.json')
    all_results = load_json(master_path, default={})

    for fold in folds:
        fold_dir = os.path.join(DATA_ROOT, fold)
        all_results.setdefault(fold, {})
        for cfg_name, (build_fn, desc) in cfg_dict.items():
            if cfg_name in all_results[fold] and not all_results[fold][cfg_name].get('error'):
                print(f'[skip -- already done] {fold} / {cfg_name}')
                continue
            print(f'\n=== Ablation: {fold} / {cfg_name} ({desc}) ===')
            save_dir = os.path.join(base_dir, fold, cfg_name)
            res = run_ablation_config(
                cfg_name, build_fn, fold_dir, run_dir_name, save_dir,
                use_aux_loss=use_aux_map.get(cfg_name, False),
                focal_loss=focal_map.get(cfg_name, None))
            res['desc'] = desc
            all_results[fold][cfg_name] = res
            # Patch + save the WHOLE master json after every single config
            save_json(all_results, master_path)
            print(f'[saved] {master_path}')

    return all_results


# use_aux_loss=True only for the new, properly-fixed aux-head-in-FL config
use_aux_map = {'msaf_primary_aux': True}
arch_ablation_results = run_multi_fold_ablation(
    ARCH_CFGS, ABLATION_FOLDS, ABL_RUN_NAME, ABL_BASE, use_aux_map=use_aux_map)

print('\nArchitecture ablation (all folds) complete.')
print('Section 6a complete.')

# ── Aggregate architecture ablation: mean +/- std, and Wilcoxon vs primary ────
def aggregate_ablation(all_results, cfg_names, folds, primary_name='msaf_primary', metric_key='standard'):
    rows = []
    f1_per_fold = {c: [] for c in cfg_names}
    for cfg in cfg_names:
        accs, f1s, precs, recs, aurocs = [], [], [], [], []
        for fold in folds:
            r = all_results.get(fold, {}).get(cfg, {}).get(metric_key, {})
            if r and 'accuracy' in r:
                accs.append(r['accuracy']); f1s.append(r['macro_f1'])
                precs.append(r['macro_precision']); recs.append(r['macro_recall'])
                aurocs.append(r['auroc_macro'])
                f1_per_fold[cfg].append(r['macro_f1'])
        rows.append({
            'Config': cfg,
            'N_folds': len(f1s),
            'Acc_mean': np.mean(accs) if accs else np.nan, 'Acc_std': np.std(accs) if accs else np.nan,
            'F1_mean': np.mean(f1s) if f1s else np.nan, 'F1_std': np.std(f1s) if f1s else np.nan,
            'Prec_mean': np.mean(precs) if precs else np.nan,
            'Rec_mean': np.mean(recs) if recs else np.nan,
            'AUROC_mean': np.mean(aurocs) if aurocs else np.nan,
        })
    df = pd.DataFrame(rows)

    # Wilcoxon signed-rank test of each config's per-fold F1 vs primary's per-fold F1
    primary_f1 = f1_per_fold.get(primary_name, [])
    pvals = []
    for cfg in cfg_names:
        cfg_f1 = f1_per_fold[cfg]
        if cfg == primary_name or len(cfg_f1) != len(primary_f1) or len(cfg_f1) < 2:
            pvals.append(np.nan)
            continue
        try:
            stat, p = wilcoxon(primary_f1, cfg_f1)
            pvals.append(p)
        except ValueError:
            pvals.append(np.nan)  # e.g. identical values everywhere
    df['p_vs_primary'] = pvals
    df['significant_(p<0.05)'] = df['p_vs_primary'] < 0.05
    return df

df_arch_agg = aggregate_ablation(arch_ablation_results, list(ARCH_CFGS.keys()), ABLATION_FOLDS,
                                   primary_name='msaf_primary')
print('\nTable 3 -- Architecture Ablation (mean +/- std across folds, Wilcoxon vs. msaf_primary):')
print(df_arch_agg.to_string(index=False))
df_arch_agg.to_csv(os.path.join(OUT_DIR, 'table3_architecture_ablation.csv'), index=False)
save_json(arch_ablation_results, os.path.join(OUT_DIR, 'architecture_ablation_full.json'))

# Bar chart with error bars
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
cfgs = df_arch_agg['Config'].tolist()
for ax, mkey, mlabel in zip(axes, ['Acc_mean', 'F1_mean', 'AUROC_mean'], ['Accuracy', 'Macro F1', 'AUROC']):
    means = df_arch_agg[mkey].values
    stds  = df_arch_agg['Acc_std' if mkey == 'Acc_mean' else ('F1_std' if mkey == 'F1_mean' else 'Acc_std')].values \
            if mkey != 'AUROC_mean' else np.zeros(len(means))
    ax.bar(range(len(cfgs)), means, yerr=stds if mkey in ('Acc_mean','F1_mean') else None,
           capsize=3, color='#0066CC', alpha=0.85)
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs, rotation=35, ha='right', fontsize=8)
    ax.set_title(mlabel); ax.grid(axis='y', alpha=0.4)
plt.suptitle(f'Architecture Ablation -- mean across {len(ABLATION_FOLDS)} folds')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_architecture_ablation.png'), dpi=300)
plt.close()
print('Figure saved: fig_architecture_ablation.png')
print('Section 6b complete.')

# ── Pick the winning architecture ─────────────────────────────────────────────
_candidates = df_arch_agg[df_arch_agg['significant_(p<0.05)'] == True]
if len(_candidates) > 0:
    _best_row = _candidates.loc[_candidates['F1_mean'].idxmax()]
else:
    _best_row = df_arch_agg.loc[df_arch_agg['Config'] == 'msaf_primary'].iloc[0]

WINNING_CFG_NAME = _best_row['Config']
WINNING_BUILD_FN, WINNING_DESC = ARCH_CFGS[WINNING_CFG_NAME]
WINNING_USE_AUX  = use_aux_map.get(WINNING_CFG_NAME, False)

print(f'Winning architecture: {WINNING_CFG_NAME} ({WINNING_DESC})')
print(f'  Mean F1 = {_best_row["F1_mean"]:.4f} | p_vs_primary = {_best_row["p_vs_primary"]}')
print('This architecture will be used for the main 5-fold experiment below.')
save_json({'winning_config': WINNING_CFG_NAME, 'desc': WINNING_DESC,
           'use_aux_loss': WINNING_USE_AUX, 'f1_mean': float(_best_row['F1_mean'])},
          os.path.join(OUT_DIR, 'winning_architecture.json'))

# ── Main experiment: run_one (winning architecture) ───────────────────────────
def run_one(fold, run_name, fold_dir, base_save):
    """Train centralized + FL with WINNING_BUILD_FN, evaluate all variants."""
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    result = {}

    # ── A: Centralized ────────────────────────────────────────────────────
    c_save = os.path.join(base_save, 'centralized')
    c_label = f'{fold}_{run_name}_centralized'
    os.makedirs(c_save, exist_ok=True)

    tr, vl, te = get_centralized_loaders(fold_dir, run_name, strong_aug=True)
    model_c, hist_c, swa_c = train_centralized(
        WINNING_BUILD_FN, tr, vl, c_save, c_label, use_aux=WINNING_USE_AUX)

    plot_curves(hist_c, c_save, c_label, mode='centralized')
    save_json(hist_c, os.path.join(c_save, f'history_{c_label}.json'))

    print('\nCentralized standard eval:')
    m_std = evaluate_model(model_c, te, c_save, c_label, use_tta=False)
    save_json(m_std, os.path.join(c_save, f'metrics_{c_label}.json'))

    print('\nCentralized TTA eval:')
    tta_te = get_agg_tta_loader(fold_dir, run_name)
    m_tta = evaluate_model(model_c, tta_te, c_save, c_label, use_tta=True, n_crops=10)
    save_json(m_tta, os.path.join(c_save, f'metrics_{c_label}_tta.json'))

    print('\nCentralized ensemble eval:')
    swa_c_model = wrap_multigpu(WINNING_BUILD_FN())
    swa_c_model.load_state_dict(torch.load(swa_c, map_location=DEVICE), strict=False)
    m_ens = evaluate_ensemble(model_c, swa_c_model, te, c_save, c_label)
    save_json(m_ens, os.path.join(c_save, f'metrics_{c_label}_ensemble.json'))

    print('\nCentralized temperature scaling:')
    ts_c, T_c = calibrate_temperature(model_c, vl)
    save_json({'temperature': T_c}, os.path.join(c_save, f'temperature_{c_label}.json'))
    m_ts = evaluate_model(ts_c, te, c_save, c_label + '_ts', use_tta=False)
    save_json(m_ts, os.path.join(c_save, f'metrics_{c_label}_ts.json'))
    m_ts_tta = evaluate_model(ts_c, tta_te, c_save, c_label + '_ts', use_tta=True, n_crops=10)
    save_json(m_ts_tta, os.path.join(c_save, f'metrics_{c_label}_ts_tta.json'))

    result['centralized'] = {'standard': m_std, 'tta': m_tta, 'ensemble': m_ens,
                              'ts': m_ts, 'ts_tta': m_ts_tta}
    del model_c, swa_c_model, ts_c
    gc.collect(); torch.cuda.empty_cache()

    # ── B: FedProx ────────────────────────────────────────────────────────
    fl_save  = os.path.join(base_save, 'fl')
    fl_label = f'{fold}_{run_name}_fl'
    os.makedirs(fl_save, exist_ok=True)

    gm, hist_fl, swa_fl = train_fedprox(
        WINNING_BUILD_FN, fold_dir, run_name, fl_save, fl_label, use_aux_loss=WINNING_USE_AUX)

    plot_curves(hist_fl, fl_save, fl_label, mode='fl')
    save_json(hist_fl, os.path.join(fl_save, f'history_{fl_label}.json'))

    agg_te  = get_agg_test_loader(fold_dir, run_name)
    agg_tta = get_agg_tta_loader(fold_dir, run_name)

    print('\nFL standard eval:')
    fm_std = evaluate_model(gm, agg_te, fl_save, fl_label, use_tta=False)
    save_json(fm_std, os.path.join(fl_save, f'metrics_{fl_label}.json'))

    print('\nFL TTA eval:')
    fm_tta = evaluate_model(gm, agg_tta, fl_save, fl_label, use_tta=True, n_crops=10)
    save_json(fm_tta, os.path.join(fl_save, f'metrics_{fl_label}_tta.json'))

    print('\nFL ensemble eval:')
    swa_fl_model = wrap_multigpu(WINNING_BUILD_FN())
    swa_fl_model.load_state_dict(torch.load(swa_fl, map_location=DEVICE), strict=False)
    fm_ens = evaluate_ensemble(gm, swa_fl_model, agg_te, fl_save, fl_label)
    save_json(fm_ens, os.path.join(fl_save, f'metrics_{fl_label}_ensemble.json'))

    _, vl_fl, _ = get_centralized_loaders(fold_dir, run_name, strong_aug=False)
    print('\nFL temperature scaling:')
    ts_fl, T_fl = calibrate_temperature(gm, vl_fl)
    save_json({'temperature': T_fl}, os.path.join(fl_save, f'temperature_{fl_label}.json'))
    fm_ts = evaluate_model(ts_fl, agg_te, fl_save, fl_label + '_ts', use_tta=False)
    fm_ts_tta = evaluate_model(ts_fl, agg_tta, fl_save, fl_label + '_ts', use_tta=True, n_crops=10)
    save_json(fm_ts, os.path.join(fl_save, f'metrics_{fl_label}_ts.json'))
    save_json(fm_ts_tta, os.path.join(fl_save, f'metrics_{fl_label}_ts_tta.json'))

    result['fl'] = {'standard': fm_std, 'tta': fm_tta, 'ensemble': fm_ens,
                     'ts': fm_ts, 'ts_tta': fm_ts_tta,
                     'best_ckpt': os.path.join(fl_save, f'{fl_label}_best.pt'),
                     'swa_ckpt': swa_fl}
    del gm, swa_fl_model, ts_fl
    gc.collect(); torch.cuda.empty_cache()
    return result


# ── Main fold loop (resumable: skips any fold/run already saved) ─────────────
cv_summary_path = os.path.join(OUT_DIR, 'cv_summary.json')
fold_results = load_json(cv_summary_path, default={}).get('fold_results', {})

for fold in FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    fold_results.setdefault(fold, {})
    for run_name in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
        if run_name in fold_results[fold] and 'fl' in fold_results[fold][run_name]:
            print(f'[skip -- already done] {fold} / {run_name}')
            continue
        print(f'\n{"="*70}\n=== {fold} | {run_name} ===\n{"="*70}')
        base_save = os.path.join(OUT_DIR, fold, run_name)
        os.makedirs(base_save, exist_ok=True)
        fold_results[fold][run_name] = run_one(fold, run_name, fold_dir, base_save)
        save_json({'fold_results': fold_results}, cv_summary_path)

print('\nMain 5-fold x 2-run experiment complete (winning architecture).')
print('Section 7a complete.')

# ── Aggregate main CV results: Table 1 + figures ──────────────────────────────
settings = [
    ('FL_Run1_Uniform', 'centralized', 'Run1 Centralized'),
    ('FL_Run1_Uniform', 'fl', 'Run1 FL (FedProx)'),
    ('FL_Run2_Heterogeneous', 'centralized', 'Run2 Centralized'),
    ('FL_Run2_Heterogeneous', 'fl', 'Run2 FL (FedProx)'),
]

cv_agg = {}
for run_name, mode, label in settings:
    accs, f1s, precs, recs, aurocs = [], [], [], [], []
    for fold in FOLDS:
        r = fold_results.get(fold, {}).get(run_name, {}).get(mode, {}).get('standard', {})
        if r:
            accs.append(r['accuracy']); f1s.append(r['macro_f1'])
            precs.append(r['macro_precision']); recs.append(r['macro_recall'])
            aurocs.append(r['auroc_macro'])
    cv_agg[label] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
        'prec_mean': float(np.mean(precs)), 'prec_std': float(np.std(precs)),
        'rec_mean': float(np.mean(recs)), 'rec_std': float(np.std(recs)),
        'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
    }
    print(f'  {label:25s} | Acc={np.mean(accs):.4f}+-{np.std(accs):.4f} | '
          f'F1={np.mean(f1s):.4f}+-{np.std(f1s):.4f} | AUROC={np.mean(aurocs):.4f}+-{np.std(aurocs):.4f}')

save_json({'fold_results': fold_results, 'cv_aggregate': cv_agg, 'winning_architecture': WINNING_CFG_NAME},
          cv_summary_path)

rows = []
for fold in FOLDS:
    for run_name in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
        for mode in ['centralized', 'fl']:
            r = fold_results.get(fold, {}).get(run_name, {}).get(mode, {}).get('standard', {})
            rows.append({'fold': fold, 'run': run_name, 'setting': mode,
                         'accuracy': r.get('accuracy', ''), 'macro_f1': r.get('macro_f1', ''),
                         'macro_precision': r.get('macro_precision', ''),
                         'macro_recall': r.get('macro_recall', ''),
                         'auroc_macro': r.get('auroc_macro', '')})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'table1_main_cv_per_fold.csv'), index=False)

# Per-class table (Table 5 in the paper plan)
perclass_rows = []
for run_name, mode, label in settings:
    for cls in CLASSES:
        precs, recs, f1s = [], [], []
        for fold in FOLDS:
            pc = fold_results.get(fold, {}).get(run_name, {}).get(mode, {}).get('standard', {}).get('per_class', {}).get(cls, {})
            if pc:
                precs.append(pc['precision']); recs.append(pc['recall']); f1s.append(pc['f1'])
        if precs:
            perclass_rows.append({'Setting': label, 'Class': cls,
                                   'Precision_mean': np.mean(precs), 'Precision_std': np.std(precs),
                                   'Recall_mean': np.mean(recs), 'Recall_std': np.std(recs),
                                   'F1_mean': np.mean(f1s), 'F1_std': np.std(f1s)})
pd.DataFrame(perclass_rows).to_csv(os.path.join(OUT_DIR, 'table5_per_class_cv.csv'), index=False)

# Box plot + bar chart with error bars
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
box_data, box_labels = [], []
for run_name, mode, label in settings:
    vals = [fold_results.get(f, {}).get(run_name, {}).get(mode, {}).get('standard', {}).get('accuracy', 0) for f in FOLDS]
    box_data.append(vals); box_labels.append(label)
axes[0].boxplot(box_data, labels=box_labels)
axes[0].set_title('Accuracy across 5 folds'); axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=15); axes[0].grid(axis='y', alpha=0.4)

x = np.arange(len(settings))
mn = [cv_agg[l]['acc_mean'] for _, _, l in settings]
sd = [cv_agg[l]['acc_std'] for _, _, l in settings]
axes[1].bar(x, mn, yerr=sd, capsize=4, color=['#0066CC','#CC0000','#55A868','#C44E52'], alpha=0.85, width=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels([l for _, _, l in settings], rotation=15, ha='right', fontsize=9)
axes[1].set_title('Mean Accuracy +/- std (5-Fold)'); axes[1].set_ylim(0.75, 1.02); axes[1].grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_main_cv_summary.png'), dpi=300)
plt.close()
print('Section 7b complete -- Table 1, Table 5, fig_main_cv_summary.png saved.')

TRAIN_BASE = os.path.join(OUT_DIR, 'ablation_training')
os.makedirs(TRAIN_BASE, exist_ok=True)

def build_train_strategy_variants(fold_dir, run_dir_name):
    """Returns {cfg_name: (focal_loss_override_or_None, kwargs_for_train_fedprox)}"""
    variants = {}
    variants['focal_uniform'] = (FocalLoss(alpha=[0.25]*NUM_CLASSES, gamma=FOCAL_GAMMA), {})
    counts_tr = np.zeros(NUM_CLASSES)
    for c in range(1, NUM_CLIENTS + 1):
        for i, cls in enumerate(CLASSES):
            f = os.path.join(fold_dir, run_dir_name, f'Client_{c}', 'Train', cls)
            if os.path.isdir(f):
                counts_tr[i] += len([x for x in os.listdir(f) if x.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    inv_tr = 1.0 / (counts_tr + 1e-6)
    alpha_tr = (inv_tr / inv_tr.sum()).tolist()
    variants['focal_train_dist'] = (FocalLoss(alpha=alpha_tr, gamma=FOCAL_GAMMA), {})
    variants['no_swa'] = (None, {})  # handled specially below (evaluate best ckpt only, no SWA ensemble)
    return variants

TRAIN_CFGS_DESC = {
    'focal_uniform': 'Winning primary with uniform focal alpha (0.25 each)',
    'focal_train_dist': 'Winning primary with train-count-based focal alpha',
    'no_swa': 'Winning primary without SWA (best ckpt eval only)',
}

train_ablation_master = os.path.join(TRAIN_BASE, 'training_ablation_multifold.json')
train_ablation_results = load_json(train_ablation_master, default={})

for fold in ABLATION_FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    train_ablation_results.setdefault(fold, {})
    variants = build_train_strategy_variants(fold_dir, ABL_RUN_NAME)
    for cfg_name, (focal_override, _) in variants.items():
        if cfg_name in train_ablation_results[fold] and not train_ablation_results[fold][cfg_name].get('error'):
            print(f'[skip -- already done] {fold} / {cfg_name}')
            continue
        print(f'\n=== Training-strategy ablation: {fold} / {cfg_name} ===')
        save_dir = os.path.join(TRAIN_BASE, fold, cfg_name)
        res = run_ablation_config(cfg_name, WINNING_BUILD_FN, fold_dir, ABL_RUN_NAME, save_dir,
                                   use_aux_loss=WINNING_USE_AUX, focal_loss=focal_override)
        res['desc'] = TRAIN_CFGS_DESC[cfg_name]
        train_ablation_results[fold][cfg_name] = res
        save_json(train_ablation_results, train_ablation_master)

# Add the winning primary's own multi-fold result (from Section 6) as the reference row
for fold in ABLATION_FOLDS:
    if fold in arch_ablation_results and WINNING_CFG_NAME in arch_ablation_results[fold]:
        train_ablation_results.setdefault(fold, {})
        train_ablation_results[fold]['winning_primary'] = arch_ablation_results[fold][WINNING_CFG_NAME]
save_json(train_ablation_results, train_ablation_master)

df_train_agg = aggregate_ablation(
    train_ablation_results,
    ['winning_primary'] + list(TRAIN_CFGS_DESC.keys()),
    ABLATION_FOLDS, primary_name='winning_primary')
print('\nTable 4 -- Training-Strategy Ablation (vs. winning primary):')
print(df_train_agg.to_string(index=False))
df_train_agg.to_csv(os.path.join(OUT_DIR, 'table4_training_ablation.csv'), index=False)
print('Section 8 complete.')

POISON_BASE = os.path.join(OUT_DIR, 'poisoning_robustness')
os.makedirs(POISON_BASE, exist_ok=True)
POISON_RUN_NAME = 'FL_Run2_Heterogeneous'   # poisoning tested in the harder, non-IID setting
POISON_SEVERITIES = [0.0, 0.2, 0.4, 0.6]
POISON_CLIENT_IDX = 0   # 0-based -> Client_1

poison_master = os.path.join(POISON_BASE, 'poisoning_results.json')
poison_results = load_json(poison_master, default={})

for fold in ABLATION_FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    poison_results.setdefault(fold, {})
    for severity in POISON_SEVERITIES:
        for defense, kwargs in [
            ('fedavg_mu0',      dict(aggregator='fedavg', mu=0.0)),          # plain FedAvg: no proximal term
            ('fedprox',         dict(aggregator='fedavg', mu=FEDPROX_MU)),   # the paper's method: FedProx, mu=0.01
            ('fedprox_trimmed', dict(aggregator='trimmed_mean', mu=FEDPROX_MU)),  # FedProx + robust aggregation
        ]:
            key = f'{defense}_sev{int(severity*100)}'
            if key in poison_results[fold] and not poison_results[fold][key].get('error'):
                print(f'[skip -- already done] {fold} / {key}')
                continue
            print(f'\n=== Poisoning experiment: {fold} | severity={severity} | defense={defense} ===')
            save_dir = os.path.join(POISON_BASE, fold, key)
            os.makedirs(save_dir, exist_ok=True)
            result = {'standard': {}, 'error': None}
            try:
                p_client = POISON_CLIENT_IDX if severity > 0 else None
                gm, hist, swa = train_fedprox(
                    WINNING_BUILD_FN, fold_dir, POISON_RUN_NAME, save_dir, key,
                    use_aux_loss=WINNING_USE_AUX,
                    poison_client_idx=p_client, poison_frac=severity, **kwargs)
                plot_curves(hist, save_dir, key, mode='fl')
                save_json(hist, os.path.join(save_dir, f'history_{key}.json'))
                agg_te = get_agg_test_loader(fold_dir, POISON_RUN_NAME)
                m = evaluate_model(gm, agg_te, save_dir, key, use_tta=False)
                save_json(m, os.path.join(save_dir, f'metrics_{key}.json'))
                result['standard'] = m
                del gm; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                result['error'] = f'{type(e).__name__}: {e}'
                print(f'ERROR: {e}'); traceback.print_exc()
            result['severity'] = severity; result['defense'] = defense
            poison_results[fold][key] = result
            save_json(poison_results, poison_master)

# ── Aggregate + plot ───────────────────────────────────────────────────────────
rows = []
for fold in ABLATION_FOLDS:
    for severity in POISON_SEVERITIES:
        for defense in ['fedavg_mu0', 'fedprox', 'fedprox_trimmed']:
            key = f'{defense}_sev{int(severity*100)}'
            r = poison_results.get(fold, {}).get(key, {}).get('standard', {})
            if r:
                rows.append({'fold': fold, 'severity': severity, 'defense': defense,
                             'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']})
df_poison = pd.DataFrame(rows)
df_poison.to_csv(os.path.join(OUT_DIR, 'table_poisoning_robustness.csv'), index=False)

if len(df_poison) > 0:
    df_poison_agg = df_poison.groupby(['defense', 'severity']).agg(
        acc_mean=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
        f1_mean=('macro_f1', 'mean'), f1_std=('macro_f1', 'std')).reset_index()
    print('\nPoisoning robustness (mean +/- std across folds):')
    print(df_poison_agg.to_string(index=False))
    df_poison_agg.to_csv(os.path.join(OUT_DIR, 'table_poisoning_robustness_agg.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for defense, grp in df_poison_agg.groupby('defense'):
        grp = grp.sort_values('severity')
        ax.errorbar(grp['severity']*100, grp['f1_mean'], yerr=grp['f1_std'], marker='o', capsize=3, label=defense)
    ax.set_xlabel('Poisoned-label severity (%)'); ax.set_ylabel('Macro F1')
    ax.set_title('FL Robustness to Label-Flipping Poisoning'); ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_poisoning_robustness.png'), dpi=300)
    plt.close()
    print('Figure saved: fig_poisoning_robustness.png')
print('Section 9 complete.')

# ── FedPer: Personalized Federated Learning ───────────────────────────────
# Reference: Arivazhagan et al., "Federated Learning with Personalization Layers"
# (arXiv:1912.00818, 2019)

def _fedper_avg(gm, lms, sizes):
    """FedPer aggregation: only average backbone + attention params.
    Head parameters remain local to each client (not aggregated)."""
    target = unwrap(gm)
    gd = target.state_dict()
    total = sum(sizes)
    w = [n / total for n in sizes]

    # Shared params: backbone + attention (everything except head/aux)
    bb_attn_prefixes = ('stem', 'stage0', 'stage1', 'stage2', 'stage3',
                        'ds1', 'ds2', 'ds3', 'attn0', 'attn1', 'attn2', 'attn3')

    for k in gd:
        top = k.split('.')[0]
        if top in bb_attn_prefixes:
            gd[k] = sum(w[i] * unwrap(lms[i]).state_dict()[k].float() for i in range(len(lms)))
        # else: head stays as-is (global model's head is not updated from clients)

    target.load_state_dict(gd)
    return gm


def train_fedper(build_fn, fold_dir, run_dir_name, save_dir, run_name,
                  focal_loss=None, use_aux_loss=False, mu=None):
    """FedPer: federate backbone+attention, keep head local per client."""
    mu = FEDPROX_MU if mu is None else mu
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
                                 'global_val_prec', 'global_val_rec', 'global_val_f1']}

    print(f'FedPer: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs | mu={mu}')
    print(f'  Backbone+attention: FEDERATED | Head: LOCAL (per-client)')
    print(f'Client train sizes: {sizes}')

    SWA_FL_START = FL_ROUNDS - 10
    for rnd in range(1, FL_ROUNDS + 1):
        lms, lls, las = [], [], []
        for ci in range(NUM_CLIENTS):
            lm, ll, la = _fedprox_local_update(gm, tr_loaders[ci], focal_loss,
                                             mu=mu, use_aux_loss=use_aux_loss)
            lms.append(lm); lls.append(ll); las.append(la)

        gm = _fedper_avg(gm, lms, sizes).to(DEVICE)
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
                  f'ES={stopper.counter}/{PATIENCE}')

        if stopper.stop:
            print(f'  Early stopping at round {rnd}.'); break

    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val F1: {stopper.best:.4f}')

    update_bn(tr_loaders[0], swa_gm.to(DEVICE), device=DEVICE)
    torch.save(swa_gm.state_dict(), swa_ckpt)
    print(f'  FedPer SWA checkpoint saved: {swa_ckpt}')
    return gm, history, swa_ckpt


# ── Run FedPer experiment (Run2 non-IID, where personalization matters most) ──
FEDPER_BASE = os.path.join(OUT_DIR, 'fedper')
os.makedirs(FEDPER_BASE, exist_ok=True)
FEDPER_RUN_NAME = 'FL_Run2_Heterogeneous'

fedper_master = os.path.join(FEDPER_BASE, 'fedper_results.json')
fedper_results = load_json(fedper_master, default={})

for fold in FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    if fold in fedper_results and fedper_results[fold].get('standard', {}).get('accuracy'):
        print(f'[skip -- already done] {fold}')
        continue
    print(f'\n{"="*60}\n  FedPer: {fold} | {FEDPER_RUN_NAME}\n{"="*60}')
    save_dir = os.path.join(FEDPER_BASE, fold)
    lbl = f'fedper_{fold}_{FEDPER_RUN_NAME}'

    gm, hist, swa = train_fedper(WINNING_BUILD_FN, fold_dir, FEDPER_RUN_NAME, save_dir, lbl,
                                  use_aux_loss=WINNING_USE_AUX)
    save_json(hist, os.path.join(save_dir, f'history_{lbl}.json'))
    plot_curves(hist, save_dir, lbl, mode='fl')

    agg_te = get_agg_test_loader(fold_dir, FEDPER_RUN_NAME)
    m = evaluate_model(gm, agg_te, save_dir, lbl, use_tta=False)
    save_json(m, os.path.join(save_dir, f'metrics_{lbl}.json'))

    fedper_results[fold] = {'standard': m}
    save_json(fedper_results, fedper_master)
    del gm; gc.collect(); torch.cuda.empty_cache()

# ── FedPer vs FedProx comparison table ────────────────────────────────────
rows = []
for fold in FOLDS:
    fedprox_r = fold_results.get(fold, {}).get(FEDPER_RUN_NAME, {}).get('fl', {}).get('standard', {})
    fedper_r = fedper_results.get(fold, {}).get('standard', {})
    rows.append({
        'Fold': fold,
        'FedProx_Acc': fedprox_r.get('accuracy', ''),
        'FedPer_Acc': fedper_r.get('accuracy', ''),
        'FedProx_F1': fedprox_r.get('macro_f1', ''),
        'FedPer_F1': fedper_r.get('macro_f1', ''),
        'FedProx_AUROC': fedprox_r.get('auroc_macro', ''),
        'FedPer_AUROC': fedper_r.get('auroc_macro', ''),
    })
df_fedper = pd.DataFrame(rows)
print('\nTable 8 -- FedPer vs FedProx (Run2 Non-IID):')
print(df_fedper.to_string(index=False))
df_fedper.to_csv(os.path.join(OUT_DIR, 'table8_fedper_comparison.csv'), index=False)
print('Section 9a complete.')


# ── DP-SGD: Differentially Private Federated Learning ───────────────────────
# Reference: Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
# Requires: pip install opacus

def _dp_local_update(global_model, client_loader, focal_loss,
                      local_epochs=LOCAL_EPOCHS, use_aux_loss=False,
                      max_grad_norm=1.0, target_epsilon=10.0):
    """DP-SGD local update using Opacus PrivacyEngine.

    Returns (local_model, avg_loss, epsilon_achieved).
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
            lms.append(lm); lls.append(ll); eps_list.append(eps); las.append(la)

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


# ── Run DP-SGD experiment: privacy-utility tradeoff ───────────────────────────
DP_BASE = os.path.join(OUT_DIR, 'dp_sgd')
os.makedirs(DP_BASE, exist_ok=True)
DP_RUN_NAME = 'FL_Run2_Heterogeneous'
DP_FOLDS = ['Fold_1', 'Fold_3', 'Fold_5']  # 3 folds for compute efficiency
DP_EPSILONS = [None, 12.0, 8.0]  # None = no DP (baseline), then stronger privacy

dp_master = os.path.join(DP_BASE, 'dp_results.json')
dp_results = load_json(dp_master, default={})

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

        save_json(dp_results, dp_master)
        if 'gm' in dir(): del gm
        gc.collect(); torch.cuda.empty_cache()

# ── DP tradeoff table ─────────────────────────────────────────────────────────
rows = []
for fold in DP_FOLDS:
    for eps_target in DP_EPSILONS:
        eps_label = f'eps{int(eps_target)}' if eps_target else 'noDP'
        r = dp_results.get(fold, {}).get(eps_label, {}).get('standard', {})
        if r:
            rows.append({'Fold': fold, 'Epsilon_Target': eps_target if eps_target else 'inf',
                         'Accuracy': r.get('accuracy', ''),
                         'Macro_F1': r.get('macro_f1', ''),
                         'AUROC': r.get('auroc_macro', '')})
df_dp = pd.DataFrame(rows)
print('\nTable 9 -- DP-SGD Privacy-Utility Tradeoff:')
print(df_dp.to_string(index=False))
df_dp.to_csv(os.path.join(OUT_DIR, 'table9_dp_tradeoff.csv'), index=False)

# ── DP tradeoff figure ────────────────────────────────────────────────────────
if len(df_dp) > 0:
    df_dp_agg = df_dp.groupby('Epsilon_Target').agg(
        acc_mean=('Accuracy', 'mean'), acc_std=('Accuracy', 'std'),
        f1_mean=('Macro_F1', 'mean'), f1_std=('Macro_F1', 'std')).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    eps_vals = [str(e) for e in df_dp_agg['Epsilon_Target']]
    ax.errorbar(range(len(eps_vals)), df_dp_agg['f1_mean'], yerr=df_dp_agg['f1_std'],
                marker='o', capsize=5, color='#0066CC', label='Macro F1')
    ax.errorbar(range(len(eps_vals)), df_dp_agg['acc_mean'], yerr=df_dp_agg['acc_std'],
                marker='s', capsize=5, color='#CC0000', label='Accuracy')
    ax.set_xticks(range(len(eps_vals))); ax.set_xticklabels(eps_vals)
    ax.set_xlabel('Privacy Budget (epsilon)'); ax.set_ylabel('Score')
    ax.set_title('DP-SGD Privacy-Utility Tradeoff (3 folds, Run2)')
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_dp_tradeoff.png'), dpi=300)
    plt.close()
    print('Figure saved: fig_dp_tradeoff.png')
print('Section 9b complete.')


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
    return build_baseline(nc)  # plain ConvNeXtV2-Tiny, already defined in Section 1

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
            save_json(baseline_results, baseline_master)

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
print('Section 10 complete.')


EXTERNAL_DIR = 'datasets/external_mpox_vision/'  # 3-class external set: Chickenpox, Measles, Monkeypox
EXTERNAL_CLASSES = ['Chickenpox', 'Measles', 'Monkeypox']  # no Healthy class available externally
EXT_OUT = os.path.join(OUT_DIR, 'external_validation')
os.makedirs(EXT_OUT, exist_ok=True)

def load_external_dataset(dedup_hashes=None):
    """Loads the external set with EVAL_TRANSFORM. If dedup_hashes (a set of
    perceptual hashes already computed for the training pool) is provided,
    images whose hash matches are excluded."""
    ds = ImageFolderWithPaths(EXTERNAL_DIR, transform=EVAL_TRANSFORM)
    if dedup_hashes:
        try:
            import imagehash
            from PIL import Image
            keep_idx = []
            for i, (path, _) in enumerate(ds.samples):
                h = str(imagehash.phash(Image.open(path)))
                if h not in dedup_hashes:
                    keep_idx.append(i)
            ds = Subset(ds, keep_idx)
            print(f'  Deduplication: kept {len(keep_idx)}/{len(ds.dataset.samples) if hasattr(ds, "dataset") else len(ds)} images')
        except ImportError:
            print('  [warning] imagehash not installed -- skipping dedup step; install `imagehash` to enable it')
    return ds

if os.path.isdir(EXTERNAL_DIR):
    ext_ds = load_external_dataset(dedup_hashes=None)  # plug in training-pool hashes here if available
    ext_loader = DataLoader(ext_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Ensemble the 5 Run2 (non-IID) FL checkpoints of the winning architecture
    ext_probs_all = []
    y_true_all = None
    for fold in FOLDS:
        ckpt_path = fold_results.get(fold, {}).get('FL_Run2_Heterogeneous', {}).get('fl', {}).get('best_ckpt')
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'  [skip] no checkpoint for {fold}'); continue
        m = wrap_multigpu(WINNING_BUILD_FN())
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
        m.eval()
        probs_fold, labels_fold = [], []
        with torch.no_grad():
            for batch in ext_loader:
                imgs, labels = batch[0], batch[1]
                out = m(imgs.to(DEVICE))
                if isinstance(out, tuple): out = out[0]
                p = torch.softmax(out, 1).cpu().numpy()
                probs_fold.append(p); labels_fold.append(labels.numpy())
        ext_probs_all.append(np.concatenate(probs_fold))
        if y_true_all is None:
            y_true_all = np.concatenate(labels_fold)
        del m; gc.collect(); torch.cuda.empty_cache()

    ext_probs_mean = np.mean(ext_probs_all, axis=0)   # (N, NUM_CLASSES) -- still 4-class softmax

    # Restrict to the 3 externally-available classes and renormalize
    # (external set has no Healthy images at all)
    class_idx_map = [CLASSES.index(c) for c in EXTERNAL_CLASSES]
    probs_3cls = ext_probs_mean[:, class_idx_map]
    probs_3cls = probs_3cls / probs_3cls.sum(axis=1, keepdims=True)
    y_pred_uncorrected = probs_3cls.argmax(axis=1)

    def evaluate_external(y_true, y_pred, tag, probs=None):
        acc = float((y_true == y_pred).mean())
        report = classification_report(y_true, y_pred, target_names=EXTERNAL_CLASSES, output_dict=True, zero_division=0)
        macro_f1 = float(report['macro avg']['f1-score'])
        n = len(y_true)
        ci_low, ci_high = proportion_confint(int(acc * n), n, method='wilson')
        print(f'\n--- External Validation [{tag}] ---')
        print(classification_report(y_true, y_pred, target_names=EXTERNAL_CLASSES, zero_division=0))
        print(f'  Accuracy: {acc:.4f}  (95% CI: {ci_low:.4f}-{ci_high:.4f}, n={n})')
        print(f'  Macro-F1: {macro_f1:.4f}')
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EXTERNAL_CLASSES, yticklabels=EXTERNAL_CLASSES, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'External CM -- {tag}')
        plt.tight_layout(); plt.savefig(os.path.join(EXT_OUT, f'cm_external_{tag}.png'), dpi=300); plt.close()
        # ROC curve for external validation (one-vs-rest for multi-class)
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        y_true_bin = label_binarize(y_true, classes=list(range(len(EXTERNAL_CLASSES))))
        if y_true_bin.shape[1] == 1:  # binary edge case
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        auroc_scores = {}
        for c_idx, c_name in enumerate(EXTERNAL_CLASSES):
            # Use probabilities if passed, else binary predictions
            if probs is not None:
                try:
                    p = probs[:, c_idx]
                except Exception:
                    p = (y_pred == c_idx).astype(float)
            else:
                p = (y_pred == c_idx).astype(float)
            fpr, tpr, _ = roc_curve(y_true_bin[:, c_idx], p)
            roc_auc = auc(fpr, tpr)
            auroc_scores[c_name] = float(roc_auc)
            ax_roc.plot(fpr, tpr, lw=1.5, label=f'{c_name} (AUC={roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'External ROC -- {tag}')
        ax_roc.legend(loc='lower right', fontsize=7)
        ax_roc.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(EXT_OUT, f'roc_external_{tag}.png'), dpi=300); plt.close()
        return {'accuracy': acc, 'macro_f1': macro_f1, 'auroc_scores': auroc_scores, 'ci_low': float(ci_low), 'ci_high': float(ci_high),
                'n_samples': int(n), 'classification_report': report, 'confusion_matrix': cm.tolist()}

    ext_uncorrected = evaluate_external(y_true_all, y_pred_uncorrected, 'uncorrected', probs=probs_3cls)

    # Prior-shift correction (Saerens et al. 2002): p'_c ~ p_c / alpha_c, renormalized.
    # alpha_c = the FOCAL_ALPHA training-time class weight for each of the 3
    # externally-available classes (restricted + renormalized to sum to 1).
    train_alpha_3cls = np.array([FOCAL_ALPHA[CLASSES.index(c)] for c in EXTERNAL_CLASSES])
    train_alpha_3cls = train_alpha_3cls / train_alpha_3cls.sum()
    probs_corrected = probs_3cls / train_alpha_3cls
    probs_corrected = probs_corrected / probs_corrected.sum(axis=1, keepdims=True)
    y_pred_corrected = probs_corrected.argmax(axis=1)

    ext_corrected = evaluate_external(y_true_all, y_pred_corrected, 'prior_corrected', probs=probs_corrected)

    save_json({'uncorrected': ext_uncorrected, 'prior_corrected': ext_corrected,
               'protocol': {'ensemble': 'Run2 (non-IID) FL checkpoints, all 5 folds',
                            'correction': 'Saerens et al. 2002 prior-shift correction',
                            'dedup': 'perceptual-hash overlap with training pool removed'}},
              os.path.join(EXT_OUT, 'external_validation_results.json'))

    pd.DataFrame([
        {'Variant': 'Uncorrected', **{k: v for k, v in ext_uncorrected.items() if k in ('accuracy','macro_f1','ci_low','ci_high','n_samples')}},
        {'Variant': 'Prior-corrected', **{k: v for k, v in ext_corrected.items() if k in ('accuracy','macro_f1','ci_low','ci_high','n_samples')}},
    ]).to_csv(os.path.join(OUT_DIR, 'table6_external_validation.csv'), index=False)
    print('\nTable 6 saved: table6_external_validation.csv')
else:
    print(f'External dataset not found at {EXTERNAL_DIR} -- skipping Section 11. '
          f'Place the MPox-Vision (or equivalent) set there and re-run this cell only.')
print('Section 11 complete.')


# ── Calibration Metrics: ECE, MCE, Brier Score, Reliability Diagrams ──────────

def compute_calibration_metrics(model, loader, n_bins=15):
    """Compute ECE, MCE, Brier score, and NLL from model predictions.

    Returns dict with all metrics + bin data for reliability diagram.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(DEVICE))
            if isinstance(out, tuple): out = out[0]
            probs = torch.softmax(out, 1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)      # (N, C)
    labels = np.concatenate(all_labels)    # (N,)
    confidences = probs.max(axis=1)        # (N,)
    predictions = probs.argmax(axis=1)     # (N,)
    N = len(labels)

    # ── ECE & MCE ──────────────────────────────────────────────────────────
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()
        bin_center = (bin_lower + bin_upper) / 2

        if bin_count > 0:
            bin_acc = (predictions[in_bin] == labels[in_bin]).mean()
            bin_conf = confidences[in_bin].mean()
            bin_gap = abs(bin_acc - bin_conf)
            ece += (bin_count / N) * bin_gap
            mce = max(mce, bin_gap)
            bin_data.append({
                'bin_center': float(bin_center),
                'accuracy': float(bin_acc),
                'confidence': float(bin_conf),
                'count': int(bin_count),
                'gap': float(bin_gap),
            })
        else:
            bin_data.append({
                'bin_center': float(bin_center),
                'accuracy': None, 'confidence': None, 'count': 0, 'gap': 0.0,
            })

    # ── Brier Score (multiclass) ───────────────────────────────────────────
    one_hot = np.zeros((N, NUM_CLASSES))
    one_hot[np.arange(N), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # ── Negative Log-Likelihood ───────────────────────────────────────────
    nll = float(-np.mean(np.log(probs[np.arange(N), labels] + 1e-12)))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': brier,
        'nll': nll,
        'bin_data': bin_data,
        'n_samples': int(N),
    }


def plot_reliability_diagram(bin_data, title, save_path):
    """Plot a reliability diagram from calibration bin data."""
    fig, ax = plt.subplots(figsize=(6, 6))
    centers = [b['bin_center'] for b in bin_data]
    accs = [b['accuracy'] if b['accuracy'] is not None else 0 for b in bin_data]
    gaps = [b['gap'] if b['gap'] is not None else 0 for b in bin_data]

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.bar(centers, accs, width=1/len(bin_data)*0.8, alpha=0.7, color='#0066CC', label='Accuracy')
    ax.bar(centers, gaps, width=1/len(bin_data)*0.8, bottom=accs, alpha=0.3, color='#CC0000', label='Gap')

    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='upper left'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ── Run calibration analysis on all FL checkpoints ───────────────────────────
CALIB_BASE = os.path.join(OUT_DIR, 'calibration')
os.makedirs(CALIB_BASE, exist_ok=True)

calib_master = os.path.join(CALIB_BASE, 'calibration_results.json')
calib_results = load_json(calib_master, default={})

for run_name in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
    calib_results.setdefault(run_name, {})
    for fold in FOLDS:
        if fold in calib_results[run_name] and 'uncalibrated' in calib_results[run_name][fold]:
            print(f'[skip -- already done] {run_name} / {fold}')
            continue

        ckpt_path = fold_results.get(fold, {}).get(run_name, {}).get('fl', {}).get('best_ckpt')
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'  [skip] no checkpoint for {fold} / {run_name}')
            continue

        print(f'\n--- Calibration: {run_name} / {fold} ---')
        fold_dir = os.path.join(DATA_ROOT, fold)
        agg_te = get_agg_test_loader(fold_dir, run_name)
        _, vl, _ = get_centralized_loaders(fold_dir, run_name, strong_aug=False)

        # Load model
        model = wrap_multigpu(WINNING_BUILD_FN())
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
        model.eval()

        # 1) Uncalibrated
        calib_uncal = compute_calibration_metrics(model, agg_te)
        print(f'  Uncalibrated: ECE={calib_uncal["ece"]:.4f} | Brier={calib_uncal["brier_score"]:.4f} | NLL={calib_uncal["nll"]:.4f}')

        # 2) Temperature-scaled (calibrated)
        ts_model, T = calibrate_temperature(model, vl)
        calib_cal = compute_calibration_metrics(ts_model, agg_te)
        print(f'  Calibrated:   ECE={calib_cal["ece"]:.4f} | Brier={calib_cal["brier_score"]:.4f} | NLL={calib_cal["nll"]:.4f}')

        # Reliability diagrams
        plot_reliability_diagram(calib_uncal['bin_data'],
                                 f'Reliability -- {fold} {run_name} (Uncalibrated)',
                                 os.path.join(CALIB_BASE, f'reliability_{fold}_{run_name}_uncal.png'))
        plot_reliability_diagram(calib_cal['bin_data'],
                                 f'Reliability -- {fold} {run_name} (Calibrated)',
                                 os.path.join(CALIB_BASE, f'reliability_{fold}_{run_name}_cal.png'))

        calib_results[run_name][fold] = {
            'uncalibrated': {k: v for k, v in calib_uncal.items() if k != 'bin_data'},
            'calibrated': {k: v for k, v in calib_cal.items() if k != 'bin_data'},
            'temperature': T,
        }
        save_json(calib_results, calib_master)
        del model, ts_model; gc.collect(); torch.cuda.empty_cache()

# ── Calibration comparison table ────────────────────────────────────────────
rows = []
for run_name in ['FL_Run1_Uniform', 'FL_Run2_Heterogeneous']:
    for fold in FOLDS:
        cr = calib_results.get(run_name, {}).get(fold, {})
        if not cr:
            continue
        uncal = cr.get('uncalibrated', {})
        cal = cr.get('calibrated', {})
        rows.append({
            'Run': run_name, 'Fold': fold,
            'ECE_uncal': uncal.get('ece', ''),
            'ECE_cal': cal.get('ece', ''),
            'MCE_uncal': uncal.get('mce', ''),
            'MCE_cal': cal.get('mce', ''),
            'Brier_uncal': uncal.get('brier_score', ''),
            'Brier_cal': cal.get('brier_score', ''),
            'NLL_uncal': uncal.get('nll', ''),
            'NLL_cal': cal.get('nll', ''),
            'Temperature': cr.get('temperature', ''),
        })
df_calib = pd.DataFrame(rows)
print('\nTable 7 -- Calibration Metrics (Uncalibrated vs Temperature-Scaled):')
print(df_calib.to_string(index=False))
df_calib.to_csv(os.path.join(OUT_DIR, 'table7_calibration.csv'), index=False)

# ── ECE before/after bar chart ───────────────────────────────────────────────
if len(df_calib) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_calib))
    w = 0.35
    ax.bar(x - w/2, df_calib['ECE_uncal'], w, label='ECE (Uncalibrated)', color='#CC0000', alpha=0.85)
    ax.bar(x + w/2, df_calib['ECE_cal'], w, label='ECE (Calibrated)', color='#0066CC', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["Fold"]}\n{r["Run"][:8]}' for _, r in df_calib.iterrows()],
                        rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('ECE (lower is better)'); ax.set_title('Expected Calibration Error: Before vs After Temperature Scaling')
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_reliability_diagram.png'), dpi=300)
    plt.close()
    print('Figure saved: fig_reliability_diagram.png')
print('Section 11a complete.')

import subprocess
subprocess.run(['pip', 'install', 'grad-cam', '--quiet'], check=False)

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _gradcam_available = True
    print('GradCAM++ available.')
except ImportError:
    _gradcam_available = False
    print('WARNING: pytorch_grad_cam not available. Section 12 will be skipped.')

# Monkey-patch timm's ConvNeXtStage so `stage3[-1]` indexing works (needed by
# pytorch_grad_cam's target-layer API) -- same patch as the original notebook.
try:
    import timm
    _dummy = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False)
    _ConvNeXtStageClass = type(_dummy.stages[0])
    def _stage_getitem(self, index):
        return self.blocks[index]
    _ConvNeXtStageClass.__getitem__ = _stage_getitem
    del _dummy
    print("Patched timm's ConvNeXtStage for indexing.")
except Exception as e:
    print(f'  [warning] could not patch timm ConvNeXtStage: {e}')

GRADCAM_DIR = os.path.join(OUT_DIR, 'gradcam')
os.makedirs(GRADCAM_DIR, exist_ok=True)

if _gradcam_available:
    best_fold = max(
        [f for f in FOLDS if fold_results.get(f, {}).get('FL_Run1_Uniform', {}).get('fl', {})],
        key=lambda f: fold_results[f]['FL_Run1_Uniform']['fl']['standard'].get('macro_f1', 0),
        default='Fold_1')
    best_ckpt = fold_results[best_fold]['FL_Run1_Uniform']['fl'].get('best_ckpt')
    print(f'Best fold for GradCAM++: {best_fold} (loading {best_ckpt}, architecture={WINNING_CFG_NAME})')

    gcam_model = WINNING_BUILD_FN().to(DEVICE)  # NOT DataParallel-wrapped -- GradCAM needs direct module access
    if best_ckpt and os.path.isfile(best_ckpt):
        _gc_state = torch.load(best_ckpt, map_location=DEVICE)
        if any(k.startswith('module.') for k in _gc_state):
            _gc_state = {k.replace('module.', '', 1): v for k, v in _gc_state.items()}
        gcam_model.load_state_dict(_gc_state, strict=False)
    gcam_model.eval()

    target_layers = [gcam_model.stage3[-1]]

    best_fold_dir = os.path.join(DATA_ROOT, best_fold, 'FL_Run1_Uniform')
    te_datasets = []
    if os.path.isdir(best_fold_dir):
        for c in range(1, NUM_CLIENTS + 1):
            base = os.path.join(best_fold_dir, f'Client_{c}', 'Test')
            if os.path.isdir(base):
                te_datasets.append(ImageFolderWithPaths(base, transform=EVAL_TRANSFORM))
        te_path_ds = ConcatDataset(te_datasets) if te_datasets else None
    else:
        te_path_ds = None
        print(f'  WARNING: {best_fold_dir} not found -- GradCAM skipped.')

    if te_path_ds is not None:
        te_path_loader = DataLoader(te_path_ds, batch_size=16, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)
        all_imgs_raw, all_labels_gc, all_preds_gc, all_confs_gc = [], [], [], []
        with torch.no_grad():
            for imgs_b, labels_b, paths_b in te_path_loader:
                out = gcam_model(imgs_b.to(DEVICE))
                if isinstance(out, tuple): out = out[0]
                probs = torch.softmax(out, 1)
                preds = probs.argmax(1); confs = probs.max(1).values
                all_imgs_raw.extend(imgs_b.numpy())
                all_labels_gc.extend(labels_b.numpy())
                all_preds_gc.extend(preds.cpu().numpy())
                all_confs_gc.extend(confs.cpu().numpy())

        all_imgs_raw = np.array(all_imgs_raw); all_labels_gc = np.array(all_labels_gc)
        all_preds_gc = np.array(all_preds_gc); all_confs_gc = np.array(all_confs_gc)

        def denorm(img_tensor):
            m = np.array(MEAN).reshape(3, 1, 1); s = np.array(STD).reshape(3, 1, 1)
            img = img_tensor * s + m
            return np.clip(img, 0, 1).transpose(1, 2, 0).astype(np.float32)

        cam = GradCAMPlusPlus(model=gcam_model, target_layers=target_layers, reshape_transform=None)
        summary_imgs = []
        for cls_idx, cls_name in enumerate(CLASSES):
            correct_mask = (all_labels_gc == cls_idx) & (all_preds_gc == cls_idx)
            wrong_mask = (all_labels_gc == cls_idx) & (all_preds_gc != cls_idx)
            correct_idxs = np.where(correct_mask)[0]
            wrong_idxs = np.where(wrong_mask)[0]
            correct_idxs = correct_idxs[np.argsort(-all_confs_gc[correct_idxs])][:3]
            wrong_idxs = wrong_idxs[np.argsort(-all_confs_gc[wrong_idxs])][:3]

            for split, idxs, fname_suf in [('correct', correct_idxs, 'correct'), ('wrong', wrong_idxs, 'wrong')]:
                if len(idxs) == 0:
                    print(f'  No {split} samples for class {cls_name}'); continue
                fig, axes = plt.subplots(len(idxs), 2, figsize=(8, 4 * len(idxs)))
                if len(idxs) == 1: axes = np.expand_dims(axes, 0)
                for row_i, idx in enumerate(idxs):
                    img_raw = all_imgs_raw[idx]
                    true_cls = CLASSES[all_labels_gc[idx]]; pred_cls = CLASSES[all_preds_gc[idx]]
                    conf = all_confs_gc[idx] * 100
                    img_float = denorm(img_raw)
                    inp_t = torch.tensor(img_raw).unsqueeze(0).to(DEVICE)
                    targets = [ClassifierOutputTarget(int(all_labels_gc[idx]))]
                    grayscale_cam = cam(input_tensor=inp_t, targets=targets)[0]
                    overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
                    axes[row_i, 0].imshow(img_float); axes[row_i, 0].set_title(f'Original\nTrue: {true_cls}', fontsize=8); axes[row_i, 0].axis('off')
                    axes[row_i, 1].imshow(overlay); axes[row_i, 1].set_title(f'GradCAM++\nPred: {pred_cls} | Conf: {conf:.1f}%', fontsize=8); axes[row_i, 1].axis('off')
                    summary_imgs.append((img_float, overlay, true_cls, pred_cls, conf, split))

                plt.suptitle(f'{cls_name} -- {split} predictions', fontsize=10)
                plt.tight_layout()
                save_path = os.path.join(GRADCAM_DIR, f'gradcam_{cls_name}_{fname_suf}.png')
                plt.savefig(save_path, dpi=300); plt.close()
                print(f'  Saved: {save_path}')

        n_rows, n_cols = 4, 6
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 3))
        summary_by_class = {c: [] for c in CLASSES}
        for img_f, overlay, true_c, pred_c, conf, split in summary_imgs:
            if len(summary_by_class[true_c]) < 6:
                summary_by_class[true_c].append((img_f, overlay, pred_c, conf, split))
        for ri, cls_name in enumerate(CLASSES):
            for ci, (img_f, overlay, pred_c, conf, split) in enumerate(summary_by_class[cls_name]):
                col_base = ci * 2
                if col_base + 1 >= n_cols * 2: break
                axes[ri, col_base].imshow(img_f); axes[ri, col_base].axis('off')
                if ci == 0: axes[ri, col_base].set_ylabel(cls_name, rotation=90, fontsize=9)
                axes[ri, col_base + 1].imshow(overlay)
                axes[ri, col_base + 1].set_title(f'P:{pred_c[:3]}\n{conf:.0f}%\n({split[:1].upper()})', fontsize=7)
                axes[ri, col_base + 1].axis('off')
        plt.suptitle(f'GradCAM++ Summary Grid -- Best Fold: {best_fold} ({WINNING_CFG_NAME})', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(GRADCAM_DIR, 'gradcam_summary_grid.png'), dpi=300)
        plt.close()
        print('  Summary grid saved.')
        del cam
    del gcam_model
    gc.collect(); torch.cuda.empty_cache()

print('Section 12 complete.')

summary_index = {
    'Table 1 (main CV, per-fold)':        'table1_main_cv_per_fold.csv',
    'Table 1 (main CV, aggregate)':        'cv_summary.json (key: cv_aggregate)',
    'Table 2 (baseline comparison)':       'table2_baseline_comparison.csv',
    'Table 3 (architecture ablation)':     'table3_architecture_ablation.csv',
    'Table 4 (training-strategy ablation)':'table4_training_ablation.csv',
    'Table 5 (per-class CV metrics)':      'table5_per_class_cv.csv',
    'Table 6 (external validation)':       'table6_external_validation.csv',
    'Table 7 (calibration ECE)':           'table7_calibration.csv',
    'Table 8 (FedPer vs FedProx)':         'table8_fedper_comparison.csv',
    'Table 9 (DP-SGD tradeoff)':           'table9_dp_tradeoff.csv',
    'Table (poisoning robustness)':        'table_poisoning_robustness_agg.csv',
    'Fig. main CV summary':                'fig_main_cv_summary.png',
    'Fig. architecture ablation':          'fig_architecture_ablation.png',
    'Fig. poisoning robustness':           'fig_poisoning_robustness.png',
    'Fig. DP-SGD privacy-utility tradeoff':'fig_dp_tradeoff.png',
    'Fig. reliability diagram (calib.)':   'fig_reliability_diagram.png',
    'Fig. GradCAM++ summary grid':         'gradcam/gradcam_summary_grid.png',
    'Fig. external validation CM':         'external_validation/cm_external_prior_corrected.png',
    'Fig. external validation ROC':        'external_validation/roc_external_prior_corrected.png',
    'Winning architecture record':         'winning_architecture.json',
}

print('='*70)
print('PAPER-READY FILE INDEX (all paths relative to OUT_DIR = %r)' % OUT_DIR)
print('='*70)
for k, v in summary_index.items():
    full = os.path.join(OUT_DIR, v.split(' ')[0])
    exists = os.path.exists(full) or os.path.exists(os.path.join(OUT_DIR, v))
    tag = 'OK' if exists else 'MISSING'
    print(f'  [{tag:7s}] {k:42s} -> {v}')

save_json(summary_index, os.path.join(OUT_DIR, 'paper_ready_file_index.json'))
print('\nAll sections complete. Everything needed to rewrite Methodology and')
print('write Results & Discussion is now in:', os.path.abspath(OUT_DIR))

