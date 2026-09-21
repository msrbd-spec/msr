## 1. Diagram audit — it has real errors, and most of them are in Section C

The layout and colours are good, but the CSAH block is scientifically wrong in the generated image: the operation order is reversed (attention before GeM/projection), Q/K/V are assigned to the wrong tensors, four scales are tapped instead of three, the residual bypass is missing, τ has drifted into Section D, and the MLP head dimensions don't match your equations. A reviewer who reads your Methodology and then this figure will see a contradiction, so this needs fixing before submission.

Here is a follow-up correction prompt to feed back to the image model:

````markdown
# Correction pass — Figure 2 (PoxCSAF-Net architecture)

Keep the overall layout, panel colours, header ribbons, fonts, the four attached class
photographs, the input panel (A), the backbone block chain of Section B, the
"ConvNeXtV2 Block (simplified)" card, and the four output class chips **exactly as they are**.
Fix only the items listed below. Do not restyle anything else.

## 0. Global
- **Export with a fully transparent background (PNG with alpha).** The current image has an
  opaque white page fill. Remove it. Panels and cards keep their own opaque fills.

## 1. Section B — backbone (small fixes)
1.1 The stem label is wrong. Change `Stem / 7 × 7 Conv / s = 4` to `Stem / 4 × 4 Conv / stride 4`.
1.2 Add to each `ECA` card the tag `gated residual α = 0.01`.
1.3 Add to each `CBAM` card the tags `Channel + Spatial Attn`, `DropPath p_d = 0.10`,
    `gated residual α = 0.01`.
1.4 Each ECA/CBAM card must be connected to its stage by a **curved gray (#888888) loop arrow**
    that leaves the stage output, enters the attention card, and returns to the stage output with
    a small `⊕` marker — showing a gated residual, not a serial block in the chain.
1.5 In the "ConvNeXtV2 Block (simplified)" card, change the chain to
    `DW Conv 7 × 7 → LN → MLP (with GRN) → ⊕` and keep the caption
    `with Layer Scale and Stochastic Depth (DropPath)`.
1.6 Add a thin bracket under the stage chain labelled `LayerNorm + GRN throughout`.

## 2. Section C — Cross-Scale Attention Head: REBUILD THIS PANEL
The current content is incorrect and must be replaced. The correct order of operations is
**GeM pooling → projection → token stack → cross-attention → residual + LayerNorm → τ**,
i.e. pooling and projection come FIRST and attention comes AFTER. Redraw Section C
top-to-bottom as follows:

2.1 **Only three scales are used, not four.** Delete the `F₀` cube entirely — Stage 0 is never
    tapped. The three inputs are, left to right:
    `S₁ — 48 × 48 × 192`, `S₂ — 24 × 24 × 384`, `S₃ — 12 × 12 × 768`.
    Label the row `Multi-scale features (Stages 1–3)`.
    In Section B, draw three tap arrows leaving the outputs of Stage 1, Stage 2 and Stage 3
    (after their attention modules) labelled `S₁`, `S₂`, `S₃`. Stage 0 has no tap arrow.

2.2 Row 1 — three parallel light-green `GeM Pool` blocks (one per scale), tag
    `p = 3.0 (learnable)`, outputs labelled `→ 192-d`, `→ 384-d`, `→ 768-d`.

2.3 Row 2 — three parallel light-purple `Proj + LayerNorm` blocks, each tagged `→ 256-d`,
    outputs labelled `t₁`, `t₂`, `t₃`.

2.4 Row 3 — the three arrows merge into ONE medium-purple block:
    `T = [t₁ ; t₂ ; t₃]` with tag `B × 3 × 256`.
    **Delete the current "Fused Tokens T — N × 768" block; 768 and N are wrong.**

2.5 Row 4 — one gold `Cross-Scale Attention` block. Fix the Q/K/V assignment, which is
    currently wrong in three ways:
    - `Q = t₃` (the deepest token only) — **not** "from F₀, F₁"
    - `K = T` and `V = T` (all three tokens supply both keys and values) — **not** "K from F₂",
      "V from F₃"
    - Formula inside: `f = softmax(QKᵀ / √d) · V`, tag `d = 256`
    Arrows must point **into** the attention block from `T` and from `t₃`, and **out of** it to the
    next row. The current upward-pointing Q/K/V arrows are ambiguous — redraw them as
    clearly directed inputs.

2.6 Row 5 — a single card with three stacked rows:
    `f + t₃` (teal, with a `⊕` marker) → `LayerNorm` (light gray) → `× τ` (red,
    tag `τ = 1.0 (learnable)`). Output labelled `f′` with tag `256-d`.
    **Delete the "Global feature (1 × 768)" label; the CSAH output is 256-d, not 768-d.**

2.7 Add the **missing residual bypass**: a curved teal (#008080) arrow running down the right
    inside edge of Section C from `t₃` (row 2), past the cross-attention block, into the `⊕` of
    row 5. Label it `residual (t₃)`. This connection is currently absent and is essential.

2.8 Move the `τ` tag OUT of Section D. τ belongs inside the CSAH (row 5), applied after
    LayerNorm. It is a feature-scaling parameter, not the calibration temperature — add the
    small italic note `feature scaling, not calibration temperature`.

## 3. Section D — classification head: FIX THE DIMENSIONS
The current head (`LayerNorm → τ → MLP Head 768 → 4 → Dropout p = 0.2 → Softmax`) does not
match the model. Replace the whole chain with, top to bottom:
```
Dropout (0.35)  →  Linear: 256 → 128  →  GELU  →  Dropout (0.15)  →  Linear: 128 → 4
```
wrapped in one light-coral `MLP Head` card, then `ŷ ∈ ℝ⁴`, then `softmax`, then the four
class chips (keep the chips and their photographs unchanged).
- Delete the standalone `LayerNorm` block at the top of Section D (LayerNorm is inside CSAH).
- Delete `Dropout (p = 0.2)` — that value does not exist in this model.
- `768 → 4` is wrong; the head input is 256-d.

## 4. Bottom detail strip
4.1 Card 1 (`ECA Module`): `k = 3` is wrong, the kernel size is adaptive. Replace with
    `k = |(log₂C + b) / γ|_odd , γ = 2, b = 1` and extend the chain to
    `X → GAP → Conv1D (k adaptive) → σ → X ⊙ w → ⊕`, with the line
    `X′ = X + α (X ⊙ w − X),  α = 0.01`.
4.2 Card 2 (`CBAM Module`): add the missing specifics —
    `Channel Attn (shared MLP, r = 16)` → `Spatial Attn (7 × 7 Conv)` →
    `DropPath (p_d = 0.10)` → `⊕`, with the line `X′ = X + α · DropPath(X̂ − X),  α = 0.01`.
4.3 Card 3 (`Cross-Scale Attention Head`): the mini-flow is in the wrong order. Replace with
    `Multi-scale features S₁–S₃ → GeM → Proj + LN → tokens t₁,t₂,t₃ → Cross-Attention
    (Q = t₃; K,V = T) → ⊕ t₃ → LN → × τ`.
4.4 Card 4 (`Training Objective`): remove the bullet `Personalized federated training` —
    personalization (FedPer) is a separate experiment and is not part of the training objective.
    Replace the two bullets with `Class imbalance handling` and
    `Proximal term stabilises client drift`. Keep the focal + FedProx lines and the equation.

## 5. Arrows — general
- Every arrow must be single-headed with a clean arrowhead and must point in the direction of
  computation. No arrow may point backwards along the pipeline except the labelled teal
  residual bypass and the curved gray gated-residual loops.
- The three scale paths in Section C must read as three parallel columns that merge downward
  into one token block; they must not cross each other.
- Keep the four bold black chevrons between A → B → C → D as they are.

## 6. Do not add
No parameter counts, FLOPs, accuracies, dataset names, baseline model names, or any number
not listed above. No new imagery of any kind.
````

---

## 2. The two "centralized" runs — how to handle this for Frontiers

### First, the good news: it is probably not a bug

I looked at `get_centralized_loaders` in your script. It builds the centralized set by concatenating the **per-client `Train` folders of that run directory**:

```python
for c in range(1, NUM_CLIENTS + 1):
    base = os.path.join(fold_dir, run_dir_name, f'Client_{c}')
    tr_list.append(datasets.ImageFolder(os.path.join(base, 'Train'), ...))
```

Your augmentation is applied **offline, per client, to ≈1,500 images per client** (confirmed by the comment at the top of the script: "post-augmentation train counts are ~equalized"). So:

- **Experiment 1 (IID):** each client holds ~20 % of the unique training images, each upsampled ~5×. Pooled set ≈ 7,500 images with a roughly **uniform duplication factor**.
- **Experiment 2 (quantity skew):** Client 1 holds 10 % of unique images upsampled ~15×, Client 3 holds 30 % upsampled ~5×. Pooled set is also ≈ 7,500 images, but the **duplication factor varies ~3× across images** — images assigned to small clients are over-represented.

So the two centralized runs are trained on the **same unique images under two different effective sample-weighting distributions**. They are genuinely two different training distributions, and small differences (0.9134 vs 0.9102 macro-F1, well inside one SD) are exactly what you'd expect. That is a defensible, verifiable, non-embarrassing explanation.

**But you must verify it before writing it.** If the two pooled sets turn out to be byte-identical, the difference is pure GPU non-determinism and the explanation above would be false — which is far worse than the original problem. Run this first:

```python
# verify_centralized_pools.py — self-contained, no deps beyond stdlib
import os, hashlib, collections, json

DATA_ROOT = "/path/to/your/DATA_ROOT"          # <-- edit
FOLDS     = [f"Fold_{i}" for i in range(1, 6)]
RUNS      = ["FL_Run1_Uniform", "FL_Run2_Heterogeneous"]
NUM_CLIENTS = 5

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def pool_hashes(fold_dir, run):
    hashes = []
    for c in range(1, NUM_CLIENTS + 1):
        tr = os.path.join(fold_dir, run, f"Client_{c}", "Train")
        for cls in sorted(os.listdir(tr)):
            d = os.path.join(tr, cls)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    hashes.append(md5(p))
    return hashes

report = {}
for fold in FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    if not os.path.isdir(fold_dir):
        print(f"skip {fold}: not found"); continue
    entry = {}
    counters = {}
    for run in RUNS:
        h = pool_hashes(fold_dir, run)
        cnt = collections.Counter(h)
        counters[run] = cnt
        entry[run] = {
            "total_images":   len(h),
            "unique_images":  len(cnt),
            "dup_factor_min": min(cnt.values()),
            "dup_factor_max": max(cnt.values()),
            "dup_factor_mean": round(sum(cnt.values()) / len(cnt), 3),
        }
    a, b = counters[RUNS[0]], counters[RUNS[1]]
    entry["identical_multiset"] = (a == b)
    entry["same_unique_set"]    = (set(a) == set(b))
    entry["n_unique_only_in_run1"] = len(set(a) - set(b))
    entry["n_unique_only_in_run2"] = len(set(b) - set(a))
    report[fold] = entry
    print(fold, json.dumps(entry, indent=2))

with open("centralized_pool_audit.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved centralized_pool_audit.json")
```

Then branch on the result:

| Audit outcome | What it means | What to do |
|---|---|---|
| `same_unique_set: true`, `identical_multiset: false`, dup-factor range wider in Run 2 | Two different effective training distributions over the same images | **Case A** below — the strong option |
| `identical_multiset: true` | The two centralized runs really were the same experiment twice | **Case B** below |
| `same_unique_set: false` | The splits differ in unique content — a genuine data-pipeline problem | **Case C** below — must be fixed, not explained |

---

### Case A (most likely): keep one centralized row in the main table, report the second transparently

Do **not** delete the second run silently, and do **not** leave both labelled "Run 1 Centralized" / "Run 2 Centralized" with no explanation. The first invites "selective reporting", the second invites "your centralized baseline is not reproducible" — both are major-revision triggers at Frontiers.

The fix is cheap: **one centralized reference row, plus one explicit sentence and one supplementary row.**

**Main table (`tab:main_cv`) becomes three rows:**

```latex
Centralized (pooled) & ... \\
FL (FedProx), Experiment~1: IID uniform & ... \\
FL (FedProx), Experiment~2: Non-IID quantity skew & ... \\
```

Use the Experiment 1 numbers as the centralized row (`0.9218 ± 0.006` accuracy, `0.9134 ± 0.009` macro-F1), because its pooled set has a near-uniform duplication factor and is therefore the cleaner centralized reference.

**Methodology — add to `\subsection{Federated Client Partitioning}`, right after the paragraph about $n_k$:**

```latex
Because class-balanced offline augmentation is applied independently within each
client shard, the union of the client training sets is not identical across the two
partitions: the unique images are the same, but the number of augmented copies per
unique image varies with the size of the client that holds it. Under the
quantity-skew partition, images assigned to small clients are therefore replicated
more often than images assigned to large clients. To avoid conflating this
augmentation-induced reweighting with the effect of federation, a single centralized
reference model is reported. It is trained on the pooled training data of the uniform
partition, in which the duplication factor is approximately constant across images.
```

**Results — add immediately after the main-table paragraph:**

```latex
The centralized reference is reported once, since centralized training does not
depend on client assignment. For completeness, a second centralized model trained on
the pooled shards of the quantity-skew partition, in which the per-image duplication
factor is non-uniform, reached an accuracy of $0.9211\pm0.015$ and a macro-F1 of
$0.9102\pm0.017$; both values lie within one standard deviation of the reported
centralized reference, indicating that the pooled centralized performance is not
sensitive to the augmentation-induced reweighting. Full metrics for this
configuration are given in Supplementary Table~S1.
```

**Limitations — one sentence:**

```latex
Centralized and federated models were trained once per fold with a fixed seed;
run-to-run variation from non-deterministic GPU kernels was not separately
quantified, and reported standard deviations reflect variation across folds rather
than across repeated runs.
```

**Supplementary:** Table S1 with the second centralized configuration's full metrics (accuracy, macro-P/R/F1, AUROC per fold), plus the duplication-factor statistics from the audit script (min / mean / max copies per unique image for each partition). That table is what converts "we dropped a row" into "we characterised a nuisance factor".

**Where things go, at a glance:** mechanism → Methodology §Federated Client Partitioning; the numbers → Results §Main Classification Performance (one sentence); the caveat → Limitations; the full metrics → Supplementary.

You should also update the per-class table (`tab:per_class`, lines ~155 and ~169) the same way: one centralized block instead of two. Since you're dropping the baseline-architecture table anyway, that removes the other eight centralized rows.

---

### Case B: the pools are identical

Then the two runs are two repeats of one experiment under non-deterministic CUDA kernels, and you must say exactly that. Two honest options, in order of preference:

1. **Treat them as two seeds.** Report the centralized row as the mean of the two runs and state: "the centralized model was trained twice per fold; reported values average the two runs, and the between-run difference in macro-F1 (0.0032) provides an estimate of run-to-run variability under non-deterministic GPU kernels." This turns the accident into a genuine reproducibility datapoint, which reviewers like.
2. **Report one, disclose the other** with the same Results sentence as Case A but with the mechanism replaced by "repeated training under non-deterministic kernels".

What you must **not** do in Case B is attribute the difference to the partition. That claim would be falsifiable from your own released code.

---

### Case C: the unique image sets differ between partitions

Then something upstream went wrong in split generation (e.g. the two partitions were built from different shuffles of the fold), and no wording fixes it — the FL comparison between Experiment 1 and Experiment 2 would no longer be on matched data. Rebuild the partitions from a single fixed fold split and rerun. Flag this early; it's the only branch that costs compute.

---

### Two things that matter more than this issue

1. **Focal-loss alpha is computed from the test-set class counts.** Your script:

```python
# Alpha computed from the TEST-distribution class counts (not train counts)
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
```

but `methodology.md` Table 4 says *"Focal-loss class weights: Fold-specific, training data only."* That is a direct contradiction between manuscript and code, and it is test-set information entering the training objective. At a journal that encourages code sharing, a reviewer who opens the notebook will find this in minutes, and it is a far bigger rejection risk than two centralized rows. Either rerun with train-derived (or validation-derived) alpha, or — if a rerun isn't feasible — correct the Methodology to state exactly what was done and justify it as a fixed, fold-independent prior, then quantify the impact with the `focal_train_dist` ablation you already have (`0.924 ± 0.016`, essentially identical), which is strong evidence the choice didn't buy you anything. Do not leave the table saying "training data only".

2. **The `Run 1 / Run 2` labels should become `Experiment 1 / Experiment 2`** throughout `result_and_discussion.md` (lines 16–17, 49, 63, and the figure captions at 100–112) to match the figures you're generating. Directory names in the released code can stay as they are; just note the mapping once in the Methodology.
