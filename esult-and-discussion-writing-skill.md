---
name: result-and-discussion-writing-skill
description: Use this skill whenever the user provides experiment code and/or output files (CSV tables, JSON metric dumps, PNG figures) from a monkeypox/medical-image classification study built on a custom CNN and Federated Learning (FL), and asks for the Results and Discussion section (or Results, or Discussion, separately) to be written or revised for a Q1/top-tier journal submission. This is the companion skill to method-writing-skill: use that one for the Methodology section and this one for Results/Discussion. Trigger this whenever the user mentions ablation results, FL robustness/poisoning results, calibration, GradCAM/explainability results, external validation, baseline comparison tables, or asks to interpret/justify/explain experiment output files for the paper.

---

# Results and Discussion Writing Skill (Custom CNN + Federated Learning, Medical Imaging)

This skill governs how to read a large set of experiment output files (CSVs, JSONs, PNGs produced by the training/evaluation code) and turn them into a Results and Discussion section written as LaTeX/Overleaf source, ready to paste into the paper. It shares its format, citation, and style conventions with `method-writing-skill`; read that skill's Sections 0, 3, 5, and 6 as the baseline convention set if it is available, since the two sections must read as one consistent voice in the same paper.

## 0. Format note (read this first)

The deliverable is a `.md` file whose entire body is valid LaTeX/Overleaf source, wrapped in a single fenced ` ```latex ` code block, ready to paste directly into the user's Overleaf project. Do not produce markdown headers, markdown tables, or markdown bold as a substitute for LaTeX. Use `\section{}`/`\subsection{}`, `table`/`tabular`, `figure`, `equation` (if any new metric formulas are introduced), `\cite{}`, `\label{}`, and `\ref{}`/`\eqref{}` exactly as LaTeX commands. Match whatever preamble conventions (citation command, label naming scheme) the paper's existing Methodology draft already uses, if the user has shared or referenced it, so the whole manuscript stays internally consistent.

If a previous Results and Discussion draft already exists and the user is asking for a revision rather than a first draft, follow the same diff-first workflow as `method-writing-skill` Section 0.1: read the old draft in full, diff the new output files against what it claims, and build an explicit change list before writing a single sentence.

## 1. Before writing anything: how to read the output files without exhausting the context window

The experiment produces dozens of files per fold, per run, per configuration. Reading them naively (opening every JSON and PNG in full) will blow the context window long before the section is written. Follow this reading strategy strictly.

### 1.1 Start from the index, not the raw dumps

If `paper_ready_file_index.json` (or equivalent) exists, read it first. It is the map of every table and figure the code produced and where it lives. Use it to build a mental checklist of what needs to be covered, and to confirm a file actually exists before writing a sentence that depends on it.

### 1.2 Aggregate CSVs are the primary source of truth

Every major result already has a small, pre-aggregated CSV (typically named `tableN_*.csv`). These are the numbers that belong in the paper's tables and are cheap to read in full (a handful of rows, mean/std already computed). Always read the CSV before reaching for the underlying JSON. Concretely:

- Table 1 (main cross-validated performance): the per-fold CSV plus the `cv_aggregate` key inside the main CV summary JSON, not the full `fold_results` tree in that same JSON (which nests a complete classification report and confusion matrix per fold, per run, per evaluation mode, and is enormous). Extract only `cv_aggregate` with a targeted read (`jq`, `python -c`, or a `grep`/`sed` snippet), never a full `view` of the file.
- Table 2 (baseline comparison), Table 3 (architecture ablation), Table 4 (training-strategy ablation), Table 5 (per-class metrics), Table 6 (external validation), Table 7 (calibration), Table 8 (FedPer vs FedProx), Table 9 (DP-SGD tradeoff), and the poisoning-robustness aggregate table: read the CSV directly. These are already the numbers to cite in text.
- Only open the larger companion JSON (e.g. `architecture_ablation_full.json`, `poisoning_results.json`, `dp_results.json`, `calibration_results.json`, `baseline_results.json`, `fedper_results.json`) when a specific number is missing from the CSV, a `p_vs_primary`/significance value needs closer inspection, or an `error` field needs checking. Even then, query the specific key with a script rather than viewing the whole file.

### 1.3 Use scripts to extract, never bulk-view large JSON

For any JSON file above roughly a few hundred lines, do not use a full-file view. Instead run something equivalent to:

```bash
python3 -c "
import json
d = json.load(open('path/to/file.json'))
print(json.dumps(d.get('cv_aggregate', d), indent=2)[:4000])
"
```

or a `jq` query for the exact nested key needed (e.g. `.Fold_3.msaf_primary.standard.macro_f1`). Pull only the fields the current sentence needs. This keeps each read small and targeted instead of loading megabytes of nested classification reports and confusion-matrix arrays that will never be quoted.

### 1.4 Figures: read selectively, not exhaustively

The code produces far more PNGs than a Results section can or should individually narrate:

- **Always inspect** the small set of aggregate/summary figures that anchor each subsection: the main CV summary figure, the architecture-ablation bar chart, the poisoning-robustness curve, the DP-SGD tradeoff curve, the calibration/reliability bar chart, the GradCAM++ summary grid, and the external-validation confusion matrix/ROC pair. These are the ones actually discussed in the main text.
- **Sample, don't enumerate**, the large families of near-duplicate per-fold, per-config plots (individual confusion matrices, ROC curves, training curves, per-fold reliability diagrams, per-class GradCAM correct/wrong grids). View one or two representative examples if a qualitative visual claim is needed (e.g., what a GradCAM heatmap pattern looks like, or what a reliability diagram's shape looks like before vs after calibration); derive every quantitative claim from the CSVs instead of trying to "read" a number off a plot.
- Note explicitly in the deliverable which per-fold/per-config figures are being treated as supplementary material rather than main-text figures, and suggest the user route them to a Supplementary Information file rather than inflating the main Results section, which is standard practice for this volume of output in Q1 venues.

### 1.5 Work subsection by subsection

Do not attempt to load every table and figure into context before writing anything. Process one subsection at a time: pull only the CSV(s)/figure(s) that subsection needs, write that subsection in full (results plus justification), and move to the next. This keeps each step's working context small and avoids losing earlier detail to truncation.

### 1.6 Never fabricate a number

If a CSV cell is blank, a JSON key has an `error` field, or a fold/config combination never completed, say so explicitly in the text (e.g., note the reduced fold count for the DP-SGD experiment, which the code itself restricts to three folds for compute efficiency) rather than inventing a plausible value or silently omitting the caveat.

## 2. Persona for the writing pass

Write as the same two people as in `method-writing-skill`: a professional algorithm designer who ran these experiments and understands why each number came out the way it did, and a professional methodology/results writer for Q1 journals who reports findings precisely, ties every number to a stated hypothesis or design choice, and never pads with unearned adjectives. For Results and Discussion specifically, add a third instinct: that of a careful reviewer, someone who, for every strong claim, immediately asks "is this within the noise across folds, or is it a real effect," and answers that question in the text using the standard deviations, confidence intervals, or significance tests the code already computed, not vibes.

## 3. Suggested structure (data-driven, adapt freely)

The subsections below map onto what the code actually produces. Reorder, merge, or split them to fit the journal's conventions and the actual strength of each result; a subsection with a weak or null result should still be reported, just framed honestly rather than dropped silently (a Q1 reviewer will ask about missing pieces implied by the Methodology).

1. **Overview of the experimental protocol** — one short paragraph reminding the reader of the evaluation design before presenting numbers: 5-fold cross-validation, two federated data-distribution settings (an IID/uniform partition and a non-IID/heterogeneous partition), and the evaluation variants computed for each trained model (standard inference, test-time augmentation, best-checkpoint/SWA ensemble, and temperature-scaled calibration). This paragraph orients every table that follows.
2. **Main classification performance: centralized vs. federated, uniform vs. heterogeneous** — Table 1 (per-fold and aggregate), Table 5 (per-class breakdown), the main CV summary figure. Report mean ± std across folds for accuracy, macro-F1, macro-precision, macro-recall, and macro-AUROC in each of the four settings, and discuss the centralized-to-federated gap and the uniform-to-heterogeneous gap separately, since they are different phenomena with different causes.
3. **Comparison against established baseline architectures** — Table 2. Situate the proposed architecture's numbers against EfficientNetV2-S, MobileNetV2, plain ConvNeXtV2-Tiny, and ResNet50 trained under an equivalent protocol, and justify the margin (or lack of one) in terms of architectural design choices, not just "our model wins."
4. **Architecture ablation study** — Table 3, the architecture-ablation bar figure. Walk through the ablation ladder in the order the configs actually build on each other (no attention, ECA only, ECA+CBAM with a plain GAP head, ECA+CBAM with GeM pooling but no cross-scale fusion, then the full cross-scale attention head; separately, the stochastic-depth-removed and GeM-removed variants; separately again, the auxiliary-head variant tested properly under FL). For each step, state what the previous step lacked, what this step adds, and what actually happened to the metric, referencing the Wilcoxon signed-rank `p_vs_primary` column to say plainly which differences are and are not statistically distinguishable from the primary configuration across folds.
5. **Training-strategy ablation** — Table 4. Isolate the effect of the focal-loss alpha choice (test-distribution-informed vs. uniform vs. train-count-informed) and of removing SWA, against the winning primary architecture.
6. **Calibration analysis** — Table 7, the reliability/ECE bar figure, one or two representative reliability diagrams. Report ECE, MCE, Brier score, and NLL before and after temperature scaling, and explain in words what a lower ECE after scaling means for a clinical-adjacent tool (the model's stated confidence becomes trustworthy, not just its argmax).
7. **Explainability via Grad-CAM++** — the GradCAM++ summary grid, one or two representative per-class correct/incorrect panels. Describe, in words grounded in the CBAM/ECA design from the Methodology, what the attention maps localize onto for correctly classified cases per class, and what they diffuse onto or miss for misclassified cases; tie this back to why the architecture's attention modules were included in the first place.
8. **Federated learning robustness to data poisoning** — the poisoning-robustness aggregate table and figure. Explain the label-flipping poisoning protocol (severity levels, which client is poisoned) and compare the three aggregation/optimization defenses (plain FedAvg with no proximal term as the undefended reference, FedProx as the paper's proximal-regularized default, and FedProx combined with coordinate-wise trimmed-mean aggregation as the explicit Byzantine-robust option), reporting how macro-F1 degrades with severity under each and which defense is actually earning its complexity.
9. **Personalized federated learning: FedPer vs. FedProx** — Table 8. Report the comparison specifically on the heterogeneous/non-IID split, since that is where keeping classification heads local is expected to matter, and be honest if the effect is small or mixed across folds.
10. **Privacy-preserving FL: DP-SGD privacy-utility tradeoff** — Table 9, the DP tradeoff figure. Report the accuracy/F1 cost of moving from no privacy guarantee to the tested epsilon budgets, note the reduced fold count used for this experiment and why (compute cost), and frame the tradeoff honestly rather than claiming privacy was achieved "for free."
11. **External validation** — Table 6, the external confusion matrix and ROC figures. Explain the class mismatch between the external set and the training classes (no Healthy class available externally), the ensembling protocol used (the non-IID FL checkpoints across folds), the deduplication step against the training pool, and the prior-shift correction applied, then report uncorrected vs. corrected performance with the Wilson confidence interval, and discuss what the corrected-vs-uncorrected gap implies about class-prior mismatch between the internal and external populations.
12. **Discussion** — a synthesis subsection (not just a restatement) that connects the pieces: why the winning architecture's design choices paid off where they did and not elsewhere, what the poisoning and privacy results together imply about deploying this pipeline across real institutions, and how the calibration and explainability results support (or complicate) clinical usability claims. Relate findings to the cited prior work where genuinely relevant, in one or two sentences per comparison, not a re-run of the literature review.
13. **Limitations and future work** — grounded specifically in what the code shows, not generic boilerplate: the external validation is limited to three of the four classes and a single external source; the DP-SGD and poisoning experiments were run on a reduced fold subset for compute reasons; the baseline architectures were trained with a simplified single-phase schedule rather than the primary model's full phased-unfreezing protocol, which should be disclosed as a fairness caveat on Table 2; and any other honest gap the user's own code comments flag (the code itself contains several such disclosure notes — treat them as required content, not optional detail).

## 4. Statistical rigor and justification requirements (mandatory)

- Every headline number in prose must be accompanied by its spread across folds (mean ± std, or the 95% CI where the code computed one, as in the external validation Wilson interval) the first time it is introduced. Do not report a bare mean without its uncertainty.
- Wherever the code produced a significance test (the Wilcoxon signed-rank `p_vs_primary` column in the ablation tables), use it to qualify claims: a numerically higher mean with a non-significant p-value is described as such, not silently promoted to "outperforms."
- "Justification" means answering *why*, not just *what*: for every table/figure, after stating the numbers, explain the mechanism the Methodology section would predict for that number's direction (e.g., why removing the cross-scale attention head should be expected to cost macro-F1 on the minority classes specifically, given what that head does architecturally) and check whether the data actually match that expectation. If a result contradicts the expected mechanism, say so explicitly and offer a grounded hypothesis rather than glossing over it.
- Distinguish clearly, in every subsection that compares centralized vs. federated or IID vs. non-IID settings, between a performance gap that is a property of federation itself and one that is a property of data heterogeneity, since the two-run design (uniform vs. heterogeneous) exists specifically to let the text make that distinction.
- When a metric is reported in multiple evaluation variants (standard, TTA, ensemble, temperature-scaled), state which variant is the headline number used for cross-table comparisons, and briefly justify that choice once near the top of the section rather than re-justifying it every time it recurs.

## 5. Equation, table, and figure conventions (mandatory, LaTeX-native)

Follow `method-writing-skill` Section 3 exactly: `\label{}` + `\ref{}`/`\eqref{}` for every equation, table, and figure, never a hand-typed number; every table/figure referenced explicitly in the text before or immediately after it appears; self-contained captions. In addition, specific to this section:

- **Figure placeholders**: since the actual PNG files are not being generated by this writing pass, every figure the section discusses gets a complete `figure` environment with a `\includegraphics{}` path pointing at the real output location from the code (e.g. `\includegraphics[width=\linewidth]{pipeline_v2_single_gpu/fig_main_cv_summary.png}`, `\includegraphics[width=\linewidth]{pipeline_v2_single_gpu/gradcam/gradcam_summary_grid.png}`), a full `\caption{}` describing exactly what the figure shows and how it was produced, and a `\label{}` for in-text referencing. Use the exact relative output paths the code writes to (mirroring `OUT_DIR` and its subfolders), so the user can literally drop the PNG into that path in their Overleaf project folder and have the figure resolve with no further edits. If the code's output directory name changes in a future revision, update every placeholder path to match, the same way `method-writing-skill` requires updating stale hyperparameters.
- **Tables**: build the `table`/`tabular` content directly from the CSV columns actually read in Section 1 above, not from memory or approximation; column headers and units should match the CSV's own column names in spirit (renamed to publication-friendly labels, e.g. `Acc_mean`/`Acc_std` becomes "Accuracy (mean ± SD)").
- Every table caption states which folds/runs/configurations it aggregates over and how (mean ± SD across how many folds), since that is exactly the information a Q1 reviewer checks first.

## 6. Citing prior work relevant to Results and Discussion

Beyond the methods already cited in the Methodology section, the Results and Discussion section typically needs to cite work that justifies an *evaluation choice* rather than a *design choice*: the Wilcoxon signed-rank test, the expected calibration error / reliability diagram formulation, the prior-shift correction used for external validation, and any related classification-benchmark papers referenced in the comparison discussion. Follow the same citation-command consistency and no-fabricated-citation rule as `method-writing-skill` Section 5, using a visible `\cite{CITATION_NEEDED_...}` placeholder for anything the user hasn't supplied a reference for yet.

## 7. Writing style

Follow `method-writing-skill` Section 6 in full (no em dashes or double-hyphen dashes, varied sentence openings, the banned AI-tell vocabulary list, no filler adjectives without a number or citation behind them, minimal bullet/bold usage inside prose, no rhetorical questions, no exclamation points, the final read-aloud check). Two additions specific to Results and Discussion:

- Resist the pull toward triumphant framing. A results section reports what happened, including the places where the proposed method did not clearly win; state those plainly and explain them, since a Q1 reviewer trusts a paper more, not less, for reporting an honest null or mixed result inside an ablation.
- When two numbers are close, do not manufacture a narrative difference between them; say they are comparable and let the significance test settle it.

## 8. Consistency and accuracy pass (do this last, always)

1. Cross-check every number quoted in prose against the CSV/JSON cell it came from; if a number changes between drafts because the code changed, update every place that number appears in the text, not just the table.
2. Confirm every table and figure has a `\label{}` that is actually referenced by a `\ref{}` somewhere in the prose, and that no reference points to a label that doesn't exist.
3. Confirm terminology matches the Methodology section exactly (the same run names, e.g. "Run 1 (Uniform)"/"Run 2 (Heterogeneous)", the same architecture-variant names, the same metric names) so the two sections read as one document.
4. Confirm every caveat the code itself flags (reduced fold counts, simplified baseline training protocol, missing external classes, `error` entries in a results JSON) has a corresponding sentence somewhere in Results, Discussion, or Limitations; a disclosed limitation in the code that never makes it into the text is a reproducibility gap waiting to be caught in review.
5. Produce a short changelog/summary for the user (outside the LaTeX code block) listing which tables and figures were covered, which are flagged as supplementary rather than main-text, and any `[VALUE NOT FOUND / please confirm]` or `\cite{CITATION_NEEDED_...}` placeholders left to resolve.

## 9. Output format

- Deliver the Results and Discussion section as a single fenced ` ```latex ` code block, organized under `\section{}`/`\subsection{}` commands following Section 3 above (or the paper's existing section names if already established), with `table`, `figure`, and any needed `equation` environments embedded in the correct places, ready to paste directly into the user's Overleaf `.tex` file.
- Do not include a document preamble unless explicitly asked for a standalone compilable file.
- After the LaTeX code block, add the changelog/summary from Section 8, step 5, as plain text outside the code block.
- Never deliver this as rendered markdown prose describing what the LaTeX would contain; the code block itself is the deliverable.