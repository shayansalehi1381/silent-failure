# Frozen analysis plan - Phase 5 (sample-level detection of silent failures)

This document is the pre-registration of the Phase 5 analysis (Blueprint Sec 5). It is committed BEFORE any Phase 5
statistic is computed; the commit hash of this file is the freeze point. Everything below was fixed from the Phase 4
descriptive outputs only (no Phase 5 contrast, fusion fit or bootstrap had been run). Deviations from the Blueprint are
listed in Sec 10.

## 1. Models, families and evaluation pools
- Classifiers (frozen, Phases 2-3): PathMNIST standard, Camelyon standard, PathMNIST shortcut v2 (100% cue, 16 px;
  primary family C) and shortcut v1 (95% cue; supplement only); 5 seeds each.
- Families and their pre-specified evaluation pools (rows of `outputs/stage4/monitoring.parquet`):
  - **A** (MedMNIST-C, PathMNIST standard models): clean + all 11 corruptions x 5 severities on the fixed 3,000-image
    id_test sample (56 conditions).
  - **B** (Camelyon standard models): Camelyon `id_test` + `ood_c1` + `ood_c2`.
  - **C** (shortcut v2 models): `aligned` + `removed` + `reversed` on the 3,000-image sample.
  - **X** (supplement, PathMNIST standard models): PathMNIST `id_test` + `nct7k_test`.
- Pooled results are primary; per-condition / per-severity breakdowns are secondary.

## 2. Labels
- **Primary: `sf_090`** = wrong prediction AND calibrated confidence >= 0.90.
- Sensitivity labels: `sf_095` (>= 0.95); `sf_tau` (Blueprint 5.2 rule: smallest confidence threshold with >= 95%
  precision on val_cal; degenerate for 19/20 models, i.e. every error); `sf_099` (same rule with a >= 99% precision target,
  per model, computed on that model's val_cal only; degeneracy is reported).

## 3. Signals and standardisation
- **M0 (uncertainty):** log(1 - MSP) and log(entropy) of the calibrated softmax (logits / T), computed in log space
  (`sfd.signals.uncertainty.log_one_minus_msp`, `log_entropy`). These are strictly monotone in 1 - MSP and entropy, so
  rankings are identical except that saturated confidences (v2 shortcut models, T ~ 0.12) no longer create float ties.
  The tie fraction per model is reported; the float-probability versions are a sensitivity check.
- **M1 = M0 + representation:** Mahalanobis distance to the predicted class (`r_maha_pred`, shared Ledoit-Wolf
  covariance, train-only reference). kNN (`r_knn`, k=10) is a sensitivity variant, never swapped in as primary.
- **M2 = M1 + explanation:** layer4 Grad-CAM JS divergence to the predicted class's reference mean CAM, normalised by
  the transform-induced variability (`e_js_norm_l4`). Raw JS, PCA-embedding distance, layer3 and layer2 CAMs are
  sensitivity variants, never swapped in as primary.
- Every feature is standardised per model with its **val_cal percentile**: the mid-rank ECDF of the source val_cal
  values, extended outside the val_cal range by hyperbolic tails (`normalize.ecdf_percentile_ext`; identical to the
  ECDF inside the range, strictly monotone outside, never 0 or 1), so extreme shifted values keep their order. No
  shifted data are used for standardisation.

## 4. Fusion (held-out-family rotation)
- Rotation: for each held-out family H in {A, B, C}, the fusion is tuned on the other two families and evaluated on H.
  X (supplement) is evaluated with weights tuned on A + B + C.
- Model: logistic regression on the standardised features with **non-negative weights** (monotone; intercept
  unconstrained), small L2 penalty (lambda = 1e-3 on the weights), bounded L-BFGS-B, per seed. The two tuning families
  get equal total weight in the loss (each family's rows share weight 0.5). M0, M1 and M2 are tuned identically.
- Only the tuning families' `sf_090` labels are used. No label, threshold, normalisation or reference object from the
  held-out family is used (asserted in code and by a test).
- Single-signal scores (Fig 3) need no fitting.

## 5. Endpoints and primary contrasts
- Endpoints per held-out family: detection **AUROC** and **AUPRC** (average precision) of silent failures (`sf_090`), and
  **AURC** (area under the risk-coverage curve when accepting cases in increasing order of the fused risk score;
  risk = silent-failure rate among accepted cases; lower is better).
- **Primary contrasts:** M1 - M0 and M2 - M1 for dAUROC, dAUPRC and dAURC, on each held-out family A, B, C
  (3 families x 2 contrasts x 3 metrics = 18 primary estimates), each with a paired 95% CI. No multiplicity adjustment
  on the declared primary contrasts. Verdict text is generated from these numbers: an increment is called "improves"
  only if the 95% CI of dAUROC lies entirely above 0, "worse" if entirely below, otherwise "inconclusive".
- Negative-result-ready summary: does M2 improve over M1 out-of-family (per held-out family), with CIs.

## 6. Inference
- Paired two-level cluster bootstrap, B = 2,000, bootstrap seed 0: (1) resample the 5 seeds with replacement; (2) resample
  cluster units with replacement - PathMNIST near-duplicate clusters (`group` in `splits.parquet`) for A and C,
  Camelyon patients for B; the same cluster draw is applied to every seed and to M0/M1/M2 (paired). Fusion weights are
  NOT refitted inside the bootstrap (they were tuned on other families). Metric per replicate = mean over the drawn
  seeds of the cluster-weighted metric; 95% percentile CIs; two-sided bootstrap p = 2 min(P(d <= 0), P(d >= 0)).
- Seed jackknife (leave one seed out) of every primary delta is reported.
- AUPRC is always reported next to AUROC.

## 7. Secondary analyses (BH-FDR within each secondary family)
- Per-condition breakdowns (B: id_test / ood_c1 / ood_c2; C: aligned / removed / reversed), per corruption type and per
  severity (A) for dAUROC and dAUPRC; each breakdown is one BH family.
- Incremental logistic regression on the held-out family (unconstrained, per seed): fixed tuned M0 score, then
  + Mahalanobis, then + CAM drift; likelihood-ratio and Wald tests. Rows are clustered, so these p-values are
  descriptive (secondary).
- Seed stability of every primary metric.

## 8. Sensitivity analyses (one BH family)
kNN instead of Mahalanobis; layer3 and layer2 CAM instead of layer4; raw instead of normalised JS; PCA-embedding CAM
distance; labels `sf_095`, `sf_tau`, `sf_099`; X supplement; v1 weak-shortcut models in place of v2 for family C;
float-probability 1 - MSP / entropy instead of the log-scale versions.

## 9. Family C inversion diagnosis (descriptive)
Phase 4 showed single-signal AUROC far below 0.5 on C v2 reversed. Pre-declared diagnosis: AUROC pooled vs within each
cue state; characteristics (class, confidence, signal percentiles) of the few correct predictions in the reversed state;
whether the inversion is an artefact of pooling or holds within the reversed condition.

## 10. Deviations from the Blueprint (all fixed before any Phase 5 statistic)
1. The 95%-precision acceptance threshold (Blueprint 5.2) degenerated (val_cal precision >= 95% for 19/20 models), so a
   fixed threshold (0.90) is the primary silent-failure label; the Blueprint rule and a 99% variant are sensitivities.
2. Log-scale uncertainty and tail-extended percentiles were introduced to remove float ties from saturated confidences.
3. A layer2 Grad-CAM sensitivity (16 x 16 grid, same train-only reference procedure, own frozen marker) was added.
4. Family C uses the strong-shortcut v2 models (chosen by the pre-registered val_es pilot); v1 is a supplement.

## 11. Code and data
`sfd/stage5.py` (analysis), `scripts/run_stage5.py`; inputs: `outputs/stage4/monitoring.parquet` and
`outputs/stage4/signals_extra/*.parquet` (git-ignored, checksums of the reference objects committed). Frozen splits and
references are read-only.
