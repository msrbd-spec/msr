
```

---

## Changelog (relative to previous draft)

### Corrected
1. **Augmentation pipeline**: Removed RandomAffine and RandomPerspective (not in code). Fixed CutMix α from 1.0 to 0.0 (disabled). Added Resize(416)→RandomCrop(384) detail. Clarified that centralized training uses strong augmentation (RandAugment, RandomErasing, MixUp) while federated training uses a lighter pipeline (flips + ColorJitter only).
2. **TTA**: Fixed crop size from 224×224/240×240 to 384×384/400×400 (TenCrop at IMAGE_SIZE=384).
3. **Algorithm**: Rewrote Algorithm 2 (FedProx) to reflect the actual FL training loop — no phased unfreezing, no SAM, no MixUp, no label smoothing in federated training. Added Algorithm 1 (centralized phased training) and Algorithm 3 (DP-SGD local update).
4. **SWA in FL**: Corrected from "epoch 10 within Phase 3" to "rounds 41–50" (FL_ROUNDS - 10).
5. **Ensemble**: Corrected from "averaging logit outputs" to "averaging softmax probability outputs" (code computes softmax((o1+o2)/2)).
6. **Focal Loss scope**: Clarified that Focal Loss is used in federated training, while centralized training uses label-smoothed cross-entropy with MixUp.
7. **Temperature scaling**: Added L-BFGS optimizer detail.

### Added (new subsections)
8. **DP-SGD (Section 2.9)**: Opacus PrivacyEngine, make_private_with_epsilon, model adaptations (single-tensor forward, random-op disabling), GradSampleModule unwrapping, 3 folds × 3 epsilons.
9. **FedPer (Section 2.10)**: Federate backbone+attention only, head stays local.
10. **Poisoning Robustness (Section 2.11)**: Label-flipping on Client_1, 3 severities, 3 defenses (FedAvg μ=0, FedProx μ=0.01, FedProx+trimmed_mean).
11. **Calibration (Section 2.12)**: ECE, MCE, Brier, NLL, reliability diagrams, temperature scaling with L-BFGS.
12. **External Validation (Section 2.13)**: 3-class ensemble, prior-shift correction (Saerens et al. 2002), Wilson CIs.
13. **Baseline Models (Section 2.14)**: EfficientNetV2-S, MobileNetV2, ConvNeXtV2-Tiny (plain), ResNet50.
14. **Auxiliary heads (Section 2.5.5)**: Optional deep supervision mechanism.
15. **Trimmed mean aggregation (Section 2.6.3)**: Coordinate-wise trimmed mean for robust aggregation.

### Unchanged (confirmed accurate)
16. Dataset construction details (sources, class counts, split ratio) — confirmed by user as done in a separate script.
17. Client partitioning (IID/Non-IID weights) — confirmed by user.
18. 1,500 images per client via offline augmentation — confirmed by user.
19. Focal alpha values [0.332, 0.148, 0.399, 0.120] — verified against code (TEST_COUNTS = [42, 94, 35, 116]).
20. All architecture details (ECA, CBAM, CSAH, GeM, dimensions) — verified against code.

### Citation placeholders to resolve
All citations use `\cite{CITATION_NEEDED_...}` placeholders. The following keys need BibTeX entries:
- `CITATION_NEEDED_randaugment` — RandAugment (Cubuk et al., 2020)
- `CITATION_NEEDED_mixup` — MixUp (Zhang et al., 2018)
- `CITATION_NEEDED_convnextv2` — ConvNeXtV2 (Woo et al., 2023)
- `CITATION_NEEDED_eca` — ECA-Net (Wang et al., 2020)
- `CITATION_NEEDED_cbam` — CBAM (Woo et al., 2018)
- `CITATION_NEEDED_stochastic_depth` — Stochastic Depth (Huang et al., 2016)
- `CITATION_NEEDED_gem` — GeM Pooling (Radenović et al., 2019)
- `CITATION_NEEDED_fedavg` — FedAvg (McMahan et al., 2017)
- `CITATION_NEEDED_fedprox` — FedProx (Li et al., 2020)
- `CITATION_NEEDED_trimmed_mean` — Trimmed Mean (Yin et al., 2018)
- `CITATION_NEEDED_sam` — SAM (Foret et al., 2021)
- `CITATION_NEEDED_swa` — SWA (Izmailov et al., 2018)
- `CITATION_NEEDED_focal_loss` — Focal Loss (Lin et al., 2017)
- `CITATION_NEEDED_opacus` — Opacus (Yousefpour et al., 2021)
- `CITATION_NEEDED_dpsgd` — DP-SGD (Abadi et al., 2016)
- `CITATION_NEEDED_fedper` — FedPer (Arivazhagan et al., 2019)
- `CITATION_NEEDED_saerens` — Prior-shift correction (Saerens et al., 2002)
- `CITATION_NEEDED_lbfgs` — L-BFGS (Liu & Nocedal, 1989)
- `CITATION_NEEDED_effnetv2` — EfficientNetV2 (Tan & Le, 2021)
- `CITATION_NEEDED_mobilenetv2` — MobileNetV2 (Sandler et al., 2018)
- `CITATION_NEEDED_resnet50` — ResNet (He et al., 2016)
- `CITATION_NEEDED_timm` — timm (Wightman, 2019)



----------------------------------------------------

__Before pasting, also make sure your Overleaf project has these packages in the preamble:__

- `amsmath` (for `equation`, `align` environments)
- `amssymb` (for `\mathbb{}`, `\mathcal{}`)
- `graphicx` (for `\includegraphics`)
- `booktabs` (for `\toprule`, `\midrule`, `\bottomrule`)
- `algorithm` + `algorithmic` (for the `algorithm`/`algorithmic` environments)
- `hyperref` (optional, for `\ref{}` linking)
