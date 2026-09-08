Here are concrete ways to shrink the section while keeping all main information, ranked by space saved:

## 1. Move 4 classification-report tables to Supplementary (saves ~250 lines)
These are the biggest space consumers and their key numbers are already discussed in the text:
- **`tab:clf_report_main`** (lines 75-147, ~72 lines) — the aggregate table already has the headline numbers; per-fold P/R/F1/S is supplementary detail
- **`tab:clf_report_ablation`** (lines 357-379, ~22 lines) — the ablation table already has the aggregate F1
- **`tab:clf_report_dp`** (lines 756-777, ~21 lines) — the DP tradeoff table already has the aggregate F1
- **`tab:clf_report_external`** (lines 869-884, ~15 lines) — the external table already has accuracy/F1

**Total: ~130 lines saved.** Replace each with a one-sentence note: "Per-class details are provided in Supplementary Table~S\ref{...}."

## 2. Merge `fig:dp_cm` + `fig:dp_roc` into a single 2×2 figure (saves ~15 lines)
Currently they're two separate figures. Combine into one 4-panel figure with a single caption.

## 3. Merge `fig:external_cm` + `fig:external_roc` into a single 2×2 figure (saves ~10 lines)
Same approach — one figure with CMs on top, ROCs on bottom.

## 4. Tighten the Discussion subsection (saves ~10 lines)
The Discussion (lines 927-948) repeats findings already stated in each subsection. It can be compressed from 3 paragraphs to 2 by merging the first two paragraphs.

## 5. Tighten the Limitations subsection (saves ~5 lines)
The Limitations (lines 952-972) can be converted from prose to a compact bulleted list, which is more space-efficient.

## Summary of savings:
| Change | Lines saved |
|--------|------------|
| 4 clf-report tables → Supplementary | ~130 |
| Merge DP CM+ROC figures | ~15 |
| Merge External CM+ROC figures | ~10 |
| Compress Discussion | ~10 |
| Limitations → bullet list | ~5 |
| **Total** | **~170 lines** |

This would bring the section from ~972 lines to ~800 lines, a ~17% reduction, while keeping all aggregate tables, all figures, and all key findings in the main text.

**Recommendation:** Do #1 first — it's the single biggest win and is standard practice (per-class classification reports are classic Supplementary material). The rest are incremental.

**Please toggle to Act mode** if you'd like me to implement any of these.
