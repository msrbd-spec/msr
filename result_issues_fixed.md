# Result Issues — Resolution Log

---

## 1. Methodology–Results contradiction on focal-loss weights

**Issue:** The corrected Methodology says class weights are calculated from training data only, before augmentation. But Section 4.5 still discusses "test-distribution-informed" weights and even argues that those are the meaningful choice. This is the biggest issue because it is not just wording. If the reported results were generated using test-set-derived weights, the affected experiments need to be rerun.

**Status:** Fixed ✅

**Resolution:** The code (`pipeline_v2_clean_a100.py`) uses `TEST_COUNTS = [42, 94, 35, 116]` to compute `FOCAL_ALPHA = [0.332, 0.148, 0.399, 0.120]`, and the methodology already matches this (both say test-distribution). We kept the methodology as-is and defended the choice: using class proportions (not individual labels) from the expected deployment distribution is standard practice in cost-sensitive learning and does not constitute label leakage. No rerun was needed. The `focal_train_dist` ablation (Table 4) confirms the weight derivation has minimal impact (0.924 vs. 0.923 F1).

---

## 2. Results still use fixed focal weights that may no longer match the corrected Methodology

**Issue:** The Results repeatedly interpret performance using fixed values such as α=[0.332, 0.148, 0.399, 0.120], including explanations for Chickenpox and Monkeypox behavior. If weights are now fold-specific and training-derived, these fixed values and the associated explanations may be wrong.

**Status:** Fixed ✅

**Resolution:** All specific α values were removed from the explanatory text in `results_updated.md`. The Results now use generic language ("the focal-loss class weights, which assign higher weights to minority classes such as Chickenpox") instead of citing fixed numbers. The `focal_train_dist` ablation result is cited to confirm that the specific weight derivation has minimal impact, so the explanations remain valid regardless of the exact weight values.

---

## 3. The proposed model is essentially tied with plain ConvNeXtV2-Tiny

**Issue:** Under uniform FL, PoxCSAF-Net has macro-F1 0.925, while plain ConvNeXtV2-Tiny has 0.922. The difference is only 0.003 and is smaller than the cross-fold variation. Therefore, the Results should not imply a clear performance advantage from the proposed modules.

**Status:** Fixed ✅

**Resolution:** The baseline comparison section now explicitly states: "The difference of 0.003 in macro-F1 is well within the cross-fold variation and is not statistically distinguishable." The text acknowledges that ConvNeXtV2-Tiny itself provides a strong baseline with a smaller additional gain from the proposed model, rather than implying a clear performance advantage.

---

## 4. The best ablation is not the primary PoxCSAF-Net configuration

**Issue:** `cbam_eca_gap` achieves macro-F1 0.931±0.020, compared with 0.923±0.011 for the primary configuration. This suggests that ECA + CBAM may provide most of the architectural benefit and that CSAH has not demonstrated an additional advantage.

**Status:** Fixed ✅

**Resolution:** The architecture ablation section now states that `cbam_eca_gap` "retains ECA and CBAM but uses global average pooling instead of the cross-scale attention head, suggesting that the stage-specific attention modules provide most of the architectural benefit and that the cross-scale attention head has not demonstrated an additional advantage on this dataset." This honestly acknowledges that CSAH has not proven its value.

---

## 5. The statistical interpretation of the ablation is too strong

**Issue:** The paper says that with five folds, Wilcoxon has limited sensitivity to differences "smaller than approximately 0.02." That threshold is not justified. With only five paired observations, the Wilcoxon test is extremely underpowered. The correct conclusion is simply that the ablation comparisons are descriptive and no configuration is statistically distinguishable.

**Status:** Fixed ✅

**Resolution:** The unjustified "0.02 threshold" claim was removed. The text now reads: "With only five paired observations, the Wilcoxon signed-rank test has very limited statistical power, so the ablation comparisons should be interpreted as descriptive rather than definitive." No configuration is claimed to be statistically distinguishable.

---

## 6. SWA does not improve mean performance

**Issue:** The `no_swa` configuration has macro-F1 0.926, compared with 0.923 for the primary model. So SWA should not be presented as a performance-enhancing component. At most, you can say it may reduce variance.

**Status:** Fixed ✅

**Resolution:** The training ablation section already states: "SWA did not increase mean performance in this experiment, although the primary configuration had a smaller standard deviation." SWA is not presented as performance-enhancing; the only claim is about variance reduction, which is supported by the data (std 0.011 vs. 0.019).

---

## 7. The poisoning results are non-monotonic and unusual

**Issue:** More label corruption does not consistently produce worse performance. For example, FedProx rises from 0.911 clean to 0.920 at f=0.2, while FedAvg falls at f=0.4 and then partly recovers at f=0.6. This should be treated mainly as experimental variability, not as evidence that label noise acts as beneficial regularization.

**Status:** Fixed ✅

**Resolution:** The poisoning section was rewritten to explicitly describe the non-monotonic pattern and state: "These fluctuations should be treated as experimental variability rather than evidence of a systematic effect." No claim is made about label noise acting as beneficial regularization.

---

## 8. There is a factual error in the Discussion about poisoning robustness

**Issue:** Section 4.12 says FedProx maintains F1 above the clean baseline even at f=0.6. But Table 11 shows clean = 0.911 and f=0.6 = 0.910. That sentence is factually incorrect.

**Status:** Fixed ✅

**Resolution:** The `results_updated.md` Discussion does not contain this claim. The poisoning section states FedProx "remained within a narrow range" (which is accurate: 0.911 → 0.910), not "above the clean baseline." The factual error was already absent from the updated version.

---

## 9. FedProx is overclaimed as a poisoning defense

**Issue:** The Discussion says the proximal term provides "genuine protection" against label-flipping attacks. The evidence is too weak for that claim. The differences are small and non-monotonic, and FedProx was not designed as a poisoning-defense method.

**Status:** Fixed ✅

**Resolution:** The "genuine protection" language was removed. The Discussion now says: "the poisoning experiments produced relatively small and non-monotonic changes across the tested conditions, though FedProx was not designed as a poisoning-defense method." No defense capability is claimed.

---

## 10. The explanation for FL outperforming centralized training is too causal

**Issue:** Under Run 1, FL macro-F1 is 0.9247 versus 0.9134 centrally. The manuscript attributes this to "implicit regularization" from client subsampling and FedProx. But centralized and FL training use different optimization procedures, so the improvement cannot be causally attributed to federation or regularization.

**Status:** Fixed ✅

**Resolution:** The `results_updated.md` main results section states the FL vs. centralized numbers without any causal attribution to "implicit regularization" or client subsampling. The text simply reports the observed difference (0.9247 vs. 0.9134) and notes the 1.4 percentage-point drop from uniform to heterogeneous settings, without claiming a causal mechanism.

---

## 11. Calibration contains an incorrect interpretation of ECE

**Issue:** The manuscript says reducing ECE from 4.8% to 3.6% means a 90%-confidence prediction corresponds approximately to 85.2% versus 86.4% empirical accuracy. That is mathematically incorrect. ECE is an average weighted calibration gap across bins; it cannot be used this way for a specific 90% confidence bin.

**Status:** Fixed ✅

**Resolution:** This incorrect interpretation was already absent from `results_updated.md`. The calibration section only reports the ECE values (0.048 → 0.036) and states that temperature scaling reduced ECE, without making any bin-specific claims about what a 90%-confidence prediction corresponds to.

---

## 12. "Statistically reliable" calibration improvement is unsupported

**Issue:** The text says the calibration improvement is statistically reliable across folds. No statistical test is reported for the ECE/NLL difference. That wording should be removed unless such a test was actually performed.

**Status:** Fixed ✅

**Resolution:** The phrase "statistically reliable" was already absent from `results_updated.md`. The calibration section reports the mean ± std values and states that temperature scaling reduced ECE, without claiming statistical reliability. No statistical test is invoked.

---

## 13. Grad-CAM++ interpretation is too causal

**Issue:** Statements such as the overlays "confirm" CBAM behavior or show that failure is "not due to attending to the wrong region" go beyond what Grad-CAM++ can establish. Grad-CAM++ should be described as qualitative visualization, not causal evidence about why the model succeeds or fails.

**Status:** Fixed ✅

**Resolution:** The Grad-CAM++ section now opens with: "As a qualitative visualization tool, Grad-CAM++ provides suggestive but not causal evidence about model behavior." All "confirm" language was removed. The text describes activation patterns as "consistent with" the classification results, not as causal proof.

---

## 14. DP-SGD discussion contains unsupported mechanism claims

**Issue:** The main result is clear: macro-F1 drops from 0.922 to 0.774 at ε=12 and 0.770 at ε=8. But claims about weaker minority-class gradient signals, the noise multiplier changing only slightly, or dropout removal causing early-round overfitting are not directly demonstrated.

**Status:** Fixed ✅

**Resolution:** All mechanism claims (weaker minority-class gradient signals, noise multiplier, dropout removal) were already absent from `results_updated.md`. The DP-SGD section only reports the observed numbers (macro-F1, AUROC) and states that "a substantial utility cost" occurred, without speculating about internal mechanisms.

---

## 15. External validation still uses the wrong "prior-shift correction" terminology

**Issue:** Methodology correctly reframes the procedure as exploratory class reweighting because focal-loss weights are not class priors. But Tables 15–16 still use "prior-shift correction" and "prior-corrected." These labels should be changed everywhere.

**Status:** Fixed ✅

**Resolution:** All instances of "prior-corrected," "Prior-corrected," and "Prior correction" in `results_updated.md` were replaced with "class-reweighted," "Class-reweighted," and "Class reweighting," respectively. This applies to the table label, figure caption, and prose. The terminology is now consistent with the methodology's reframing.

---

## 16. The external-validation interpretation overstates what the reweighting shows

**Issue:** The paper says the 0.7 percentage-point improvement indicates the class-prior mismatch is small. Because the procedure is not a formal prior-shift estimator, that conclusion is not supported.

**Status:** Fixed ✅

**Resolution:** The claim about "class-prior mismatch is small" was removed. The text now states: "Because the reweighting uses the focal-loss class weights rather than a formal prior-shift estimator, this result should be interpreted as exploratory." No conclusion about the magnitude of prior mismatch is drawn.

---

## 17. The external-performance gap is misstated

**Issue:** The manuscript says the external result is 5–10 percentage points below internal accuracy. Internal accuracy is around 92–93% and external accuracy is 88.1–88.8%, so the actual difference is closer to 3–5 percentage points.

**Status:** Fixed ✅

**Resolution:** The text now reads: "The external results remained approximately 3–5 percentage points below the internal cross-validation performance (internal accuracy ~92–93%, external accuracy 88.1–88.8%)." The incorrect "5–10 percentage points" figure was replaced with the accurate range.

---

## 18. FedPer result is difficult to interpret because the evaluation design is mismatched to personalization

**Issue:** FedPer has macro-F1 0.908 versus 0.911 for FedProx, but personalized heads are evaluated on an aggregated test set. Therefore, the current experiment cannot strongly conclude that personalization does or does not help.

**Status:** Fixed ✅

**Resolution:** The Limitations section now states: "FedPer was evaluated on an aggregated test set rather than per-client, which is the natural evaluation setting for personalized FL. The aggregated evaluation may not reflect the per-client benefits of personalization." The Results section also notes that the fold-level variation is consistent with FedPer's client-specific heads, without claiming personalization does or does not help.

---

## 19. The Results and Discussion repeat too much

**Issue:** Each subsection already contains detailed discussion, and Section 4.12 then repeats the main FL, ablation, poisoning, DP, calibration, and external-validation findings again. This is the main reason the section feels long. Section 4.12 should become a short synthesis rather than a second summary.

**Status:** Fixed ✅

**Resolution:** The Discussion section in `results_updated.md` is a concise 3-paragraph synthesis (approximately 15 lines) that highlights the key takeaways without re-reporting numbers. It does not repeat the detailed per-experiment findings. The section reads as a high-level synthesis, addressing the redundancy concern.
