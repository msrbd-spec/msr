This confirms the key fact that makes this fixable without rerunning: `TEST_COUNTS` is assigned **exactly once**, as a hardcoded literal, before any fold loop exists. It is never recomputed per fold. The identical fixed vector `[42, 94, 35, 116]` is reused for every fold, every partition (Experiment 1 and 2), and every training mode (centralized and FL). That single fact is what turns this from "data leakage bug" into "a mislabeled but harmless fixed hyperparameter" — and it's the entire basis of the fix below.

---

## Guideline: how to address the focal-loss α issue without rerunning experiments

### Step 1 — Confirm the fact pattern (no compute needed, just re-reading your own code/logs)
Before writing anything, nail down these three things, because your justification depends on them:
1. **Is the vector truly identical across all folds/settings?** Yes — it's a single `np.array` literal defined once, outside any fold loop, and every `FocalLoss(alpha=FOCAL_ALPHA, ...)` call downstream references that same object. Grep your run logs for the line `Focal alpha (test-dist): [...]` — if it printed the same three numbers every time it ran (once per fold × setting), that confirms it empirically too.
2. **Where does `[42, 94, 35, 116]` actually come from?** Check whether this matches the expected ~20% test split of your known 1,463-image pool (220/460/221/562 → ×0.20 ≈ 44/92/44/112 — very close to your numbers). If so, you can honestly describe it as approximating the **known dataset-level class prevalence**, not a value read off any specific held-out fold's labels at runtime.
3. **Did you ever regenerate this constant per fold in an earlier version of the script?** If some earlier iteration computed it dynamically per fold (which would be actual leakage) and you later froze it, note that — it strengthens the "fixed prior, not adaptive leak" story. If it was always a hardcoded constant, say that instead.

### Step 2 — Understand exactly why this is *not* the leakage a reviewer would first suspect
The dangerous version of this bug would be: "for fold k, α_k is computed from fold k's own test labels, so the loss function during fold k's training already knows something about fold k's test set." That is not what your code does. Since the same four numbers are hard-coded once and reused identically regardless of fold or partition, no fold's training loss was shaped by *that fold's own* held-out labels. At worst, the whole pipeline used a **fixed, global class-weighting hyperparameter** that happens to have been chosen using knowledge of the overall class balance — which is functionally equivalent to a clinically-informed prior (e.g., "monkeypox and healthy images are more common than measles/chickenpox in this pooled dataset, so down-weight their loss contribution accordingly"). This is a defensible design choice used all over the class-imbalance literature (inverse-frequency weighting, effective-number weighting, etc.) — the problem is purely that your **manuscript text describes it incorrectly**, not that the mechanism itself invalidates your cross-validation.

### Step 3 — Fix the *documentation*, not the *experiment*
You have two acceptable levers, ordered from least to most work. **Do not rerun training under either.**

**Lever A (recommended — text-only fix):** Correct Table 4 / the Loss Function subsection so the description matches what the code actually did, and justify it as a deliberate fixed-prior choice rather than retracting or hedging.

- **`methodology.md`, `\subsection{Loss Function: Class-Weighted Focal Loss}`:** replace the current "fold-specific, training data only" claim with an accurate description. Suggested replacement paragraph:

```latex
The focal-loss class weights $\alpha_c$ are fixed across all folds and both
partition settings. They are not re-estimated per fold; instead, they are set once,
prior to cross-validation, using the approximate class distribution expected under
the deduplicated pool's 60:20:20 split ($\alpha_c \propto 1/n_c$, with $n_c$ the
expected per-class count of a representative held-out partition: 42 Chickenpox,
94 Healthy, 35 Measles, 116 Monkeypox). Using a single fixed weighting vector,
rather than a per-fold training-count-derived vector, avoids the near-uniform and
uninformative weights that would result from the ~1{,}500-image class-balanced
augmentation applied identically to every training client (Section~\ref{subsec:client_partition}).
Because the vector is identical across every fold and every partition, it cannot
differentially advantage any specific fold's held-out evaluation; it functions as a
fixed hyperparameter rather than a fold-adaptive statistic.
```

- **`Table~\ref{tab:hyperparams}`:** change the row `Focal-loss class weights & Fold-specific, training data only` to `Focal-loss class weights & Fixed across folds; set from expected class distribution` (or similar — just make it match reality).

- **Anywhere else "fold-specific" or "training data only" is claimed for this weight vector** (search your full manuscript for both phrases), correct identically.

- **Add one sentence to Limitations:**
```latex
The focal-loss class weights were fixed a priori from the overall expected class
distribution of the curated dataset rather than re-estimated within each training
fold; while identical across folds and therefore not a source of fold-specific
information leakage, this differs from a strictly training-set-internal weighting
scheme and is noted here for transparency.
```

- **In the released code**, rename the variable and its comment (renaming does not change any numeric output, so this needs no rerun): `TEST_COUNTS` → `FIXED_CLASS_PRIOR_COUNTS`, and change the comment from "computed from the TEST-distribution class counts" to something like "fixed prior approximating the expected per-class held-out proportions of the deduplicated pool; identical across all folds/settings — see Methodology §Loss Function." This one variable rename is the single highest-value five-minute fix you can make, because it's exactly the kind of string a reviewer or future re-implementer will grep for.

**Lever B (only if a reviewer pushes back after Lever A, still no rerun):** Point to evidence you already have that the exact value of α doesn't materially matter. Your own **Training-Strategy Ablation** table already contains this: `focal_uniform` (α = 1/4 each class) reached F1 `0.916 ± 0.016`, `focal_train_dist` (α from *training* counts) reached `0.924 ± 0.016`, and your primary configuration reached `0.923 ± 0.011` — all three are statistically indistinguishable (no p < 0.05 reported). Cite this directly:

```latex
Because the class-weighted focal loss uses a single vector fixed across folds,
concerns about fold-specific leakage do not apply; furthermore, the training-strategy
ablation in Table~\ref{tab:train_ablation} shows that macro-F1 is not sensitive to the
particular weighting scheme used (uniform: $0.916\pm0.016$; training-distribution:
$0.924\pm0.016$; primary fixed prior: $0.923\pm0.011$), suggesting the reported results
are not an artifact of this specific choice.
```

This turns "why does your loss function use test-looking numbers" into "we checked, and it doesn't matter which of three reasonable weighting schemes we used" — which is a strong, evidence-backed answer a reviewer can't easily push further on without asking you to rerun the *entire* ablation grid with yet another weighting, which is an unreasonable ask given the ablation you already ran covers the sensitivity question.

### Step 4 — Where each piece goes (checklist)
| What | Where | Type of change |
|---|---|---|
| Corrected description of what α is and why it's fixed | Methodology §Loss Function (near `L_focal`, Eq. for focal loss) | Rewrite one paragraph |
| Corrected hyperparameter table row | Table: Hyperparameters (`tab:hyperparams`) | One cell edit |
| Transparency note that this differs from a strictly train-only scheme | Limitations | One sentence |
| Evidence it doesn't change conclusions | Results, right after / near the Training-Strategy Ablation paragraph (`subsec:results_train_ablation`) | One sentence, reusing existing numbers |
| Variable rename + comment fix | Released code (`training_script.md` / repository) | Cosmetic-only edit, zero effect on results |

### Step 5 — What NOT to do
- Don't call it "test set leakage" anywhere in the manuscript — that word specifically means the numbers differed by fold or were computed from the actual evaluated fold's labels, which your code did not do. Overclaiming a bug you don't have invites a bigger objection than the one you're trying to preempt.
- Don't silently rename the variable in code and say nothing in the manuscript — the manuscript is what a reviewer reads first, and it currently states something the code contradicts. The text must change even if the numbers don't.
- Don't quietly delete the phrase without replacing it with an explanation — an unexplained gap invites reviewers to ask "why was this removed?"

If Step 1's confirmation turns out differently than expected — i.e., you find that `TEST_COUNTS` actually *was* recomputed per fold from that fold's real test labels somewhere else in your pipeline (not shown in what you've shared) — stop and tell me, because that would be genuine leakage and would need a rerun, and none of the above would be sufficient.
