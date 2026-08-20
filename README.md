## 1. What's in `pipeline_v2_single_gpu/Fold_1`, `Fold_2`, etc.?

These folders contain the **main experiment results** (Section 7a) — the winning architecture (`msaf_primary`) trained on each fold × each FL run. Each `Fold_X` folder has this structure:

```
pipeline_v2_single_gpu/Fold_X/
├── FL_Run1_Uniform/
│   ├── centralized/     ← centralized training (SAM+SWA phased unfreezing)
│   │   ├── *_best.pt, *_swa.pt (checkpoints)
│   │   ├── metrics_*.json (standard, TTA, ensemble, temperature-scaled evals)
│   │   ├── history_*.json (training curves)
│   │   └── curves/CM/ROC plots
│   └── fl/              ← FedProx FL training
│       ├── *_best.pt, *_swa.pt
│       ├── metrics_*.json (standard, TTA, ensemble, TS, TS+TTA)
│       └── history/curves/plots
└── FL_Run2_Heterogeneous/
    └── (same structure)
```

So `Fold_1`–`Fold_5` = the **main 5-fold × 2-run CV experiment** with the winning architecture. This is your **Table 1** (main results) and **Table 5** (per-class).

In contrast:
- `ablation_architecture/` = **Section 6** (architecture ablation — 9 configs × 5 folds, FL only, lighter eval)
- `baselines/` = **Section 10** (4 baseline models × 5 folds × 2 runs, centralized + FL)

---

## 2. Purpose of each JSON/CSV file

| File | Section | Purpose |
|------|---------|---------|
| `ablation_architecture/ablation_multifold_results.json` | 6a | **Raw per-config per-fold results** for the architecture ablation. Written incrementally after each (config, fold) completes. Contains standard/TTA/TS/TS+TTA metrics for all 9 architectures × 5 folds. This is the **working/checkpoint file** — it's what the script reads to skip already-done runs. |
| `winning_architecture.json` | 6b | **Single record** of which architecture won the ablation (here: `msaf_primary`), its description, mean F1, and whether aux loss is used. Used by Section 7a onward to know which `build_fn` to use. |
| `table3_architecture_ablation.csv` | 6b | **Paper-ready summary table** — mean±std across folds for each config, plus Wilcoxon p-values vs. `msaf_primary`. This is Table 3 in your paper. |
| `architecture_ablation_full.json` | 6b | **Complete dump** of all architecture ablation results (same data as `ablation_multifold_results.json` but saved at the end of Section 6b as a final snapshot). Essentially a backup of the full ablation results. |
| `cv_summary.json` | 7b | **Main experiment master file** — contains `fold_results` (all centralized + FL results for every fold/run) and `cv_aggregate` (mean±std across folds for the 4 settings). This is the resumable checkpoint for Section 7a and the source for Table 1. |

### In short:
- `ablation_multifold_results.json` → working file for ablation (incremental, resumable)
- `architecture_ablation_full.json` → final snapshot of ablation (backup)
- `table3_architecture_ablation.csv` → paper table (summary stats + p-values)
- `winning_architecture.json` → decision record (which model to use next)
- `cv_summary.json` → working file + final results for the main experiment (resumable + Table 1 source)
