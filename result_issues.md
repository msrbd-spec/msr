•  Methodology–Results contradiction on focal-loss weights.
The corrected Methodology says class weights are calculated from training data only, before augmentation. 
But Section 4.5 still discusses “test-distribution-informed” weights and even argues that those are the meaningful choice. 
This is the biggest issue because it is not just wording. If the reported results were generated using test-set-derived weights, the affected experiments need to be rerun. 
•  Results still use fixed focal weights that may no longer match the corrected Methodology.
The Results repeatedly interpret performance using fixed values such as α=[0.332,0.148,0.399,0.120]\alpha=[0.332,0.148,0.399,0.120], including explanations for Chickenpox and Monkeypox behavior. 
If weights are now fold-specific and training-derived, these fixed values and the associated explanations may be wrong. 
•  The proposed model is essentially tied with plain ConvNeXtV2-Tiny.
Under uniform FL, PoxCSAF-Net has macro-F1 0.9250.925, while plain ConvNeXtV2-Tiny has 0.9220.922. The difference is only 0.003 and is smaller than the cross-fold variation. 
Therefore, the Results should not imply a clear performance advantage from the proposed modules. 
•  The best ablation is not the primary PoxCSAF-Net configuration.
cbam_eca_gap achieves macro-F1 0.931±0.0200.931\pm0.020, compared with 0.923±0.0110.923\pm0.011 for the primary configuration. 
This suggests that ECA + CBAM may provide most of the architectural benefit and that CSAH has not demonstrated an additional advantage. 
•  The statistical interpretation of the ablation is too strong.
The paper says that with five folds, Wilcoxon has limited sensitivity to differences “smaller than approximately 0.02.” 
That threshold is not justified. With only five paired observations, the Wilcoxon test is extremely underpowered. The correct conclusion is simply that the ablation comparisons are descriptive and no configuration is statistically distinguishable. 
•  SWA does not improve mean performance.
The no_swa configuration has macro-F1 0.9260.926, compared with 0.9230.923 for the primary model. 
So SWA should not be presented as a performance-enhancing component. At most, you can say it may reduce variance. 
•  The poisoning results are non-monotonic and unusual.
More label corruption does not consistently produce worse performance. For example, FedProx rises from 0.9110.911 clean to 0.9200.920 at f=0.2f=0.2, while FedAvg falls at f=0.4f=0.4 and then partly recovers at f=0.6f=0.6. 
This should be treated mainly as experimental variability, not as evidence that label noise acts as beneficial regularization. 
•  There is a factual error in the Discussion about poisoning robustness.
Section 4.12 says FedProx maintains F1 above the clean baseline even at f=0.6f=0.6. 
But Table 11 shows clean =0.911=0.911 and f=0.6=0.910f=0.6=0.910. 
That sentence is factually incorrect. 
•  FedProx is overclaimed as a poisoning defense.
The Discussion says the proximal term provides “genuine protection” against label-flipping attacks. 
The evidence is too weak for that claim. The differences are small and non-monotonic, and FedProx was not designed as a poisoning-defense method. 
•  The explanation for FL outperforming centralized training is too causal.
Under Run 1, FL macro-F1 is 0.92470.9247 versus 0.91340.9134 centrally. 
The manuscript attributes this to “implicit regularization” from client subsampling and FedProx. 
But centralized and FL training use different optimization procedures, so the improvement cannot be causally attributed to federation or regularization. 
•  Calibration contains an incorrect interpretation of ECE.
The manuscript says reducing ECE from 4.8% to 3.6% means a 90%-confidence prediction corresponds approximately to 85.2% versus 86.4% empirical accuracy. 
That is mathematically incorrect. ECE is an average weighted calibration gap across bins; it cannot be used this way for a specific 90% confidence bin. 
•  “Statistically reliable” calibration improvement is unsupported.
The text says the calibration improvement is statistically reliable across folds. 
I do not see a statistical test reported for the ECE/NLL difference. That wording should be removed unless such a test was actually performed. 
•  Grad-CAM++ interpretation is too causal.
Statements such as the overlays “confirm” CBAM behavior or show that failure is “not due to attending to the wrong region” go beyond what Grad-CAM++ can establish. 
Grad-CAM++ should be described as qualitative visualization, not causal evidence about why the model succeeds or fails. 
•  DP-SGD discussion contains unsupported mechanism claims.
The main result is clear: macro-F1 drops from 0.9220.922 to 0.7740.774 at ϵ=12\epsilon=12 and 0.7700.770 at ϵ=8\epsilon=8. 
But claims about weaker minority-class gradient signals, the noise multiplier changing only slightly, or dropout removal causing early-round overfitting are not directly demonstrated. 
•  External validation still uses the wrong “prior-shift correction” terminology.
Methodology correctly reframes the procedure as exploratory class reweighting because focal-loss weights are not class priors. But Tables 15–16 still use “prior-shift correction” and “prior-corrected.” 
These labels should be changed everywhere. 
•  The external-validation interpretation overstates what the reweighting shows.
The paper says the 0.7 percentage-point improvement indicates the class-prior mismatch is small. 
Because the procedure is not a formal prior-shift estimator, that conclusion is not supported. 
•  The external-performance gap is misstated.
The manuscript says the external result is 5–10 percentage points below internal accuracy. 
Internal accuracy is around 92–93% and external accuracy is 88.1–88.8%, so the actual difference is closer to 3–5 percentage points. 
•  FedPer result is difficult to interpret because the evaluation design is mismatched to personalization.
FedPer has macro-F1 0.9080.908 versus 0.9110.911 for FedProx, but personalized heads are evaluated on an aggregated test set. 
Therefore, the current experiment cannot strongly conclude that personalization does or does not help. 
•  The Results and Discussion repeat too much.
Each subsection already contains detailed discussion, and Section 4.12 then repeats the main FL, ablation, poisoning, DP, calibration, and external-validation findings again. 
This is the main reason the section feels long. Section 4.12 should become a short synthesis rather than a second summary.
