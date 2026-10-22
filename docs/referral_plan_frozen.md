# Frozen referral-policy plan - Phase 7 (accept / review / refer under a fixed budget)

Pre-registration of the Phase 7 analysis (Blueprint Sec 5.6 referral). It is committed BEFORE any referral result is computed;
the commit hash of this file is the freeze point. Frozen splits, references, classifiers, Phase 5 fusion weights and all
earlier outputs are read-only. Only Phase 5/6 facts are used to design this plan (no residual-rate, capture or AURC-at-budget
number has been computed). Lessons carried over: (i) methods are compared only at EQUAL workload (Phase 6 FAR lesson: val_cal
thresholds do not transfer across source splits); (ii) rule-based verdicts are reported together with effect magnitudes
(Phase 5 zero-weight lesson).

## 1. Pools, models, labels
- Deployment pools = the Phase 5 evaluation pools (`sfd.stage5.family_pool`), per classifier seed (5 seeds), pooled over conditions:
  - **A**: PathMNIST standard models; clean + all 11 corruptions x 5 severities on the fixed 3,000-image id_test sample (56 conditions).
  - **B**: Camelyon standard models; `id_test` + `ood_c1` + `ood_c2`.
  - **C**: shortcut v2 models (primary); `aligned` + `removed` + `reversed`. v1 (weak shortcut) is a sensitivity.
  - **X** (supplement): PathMNIST standard models; `id_test` + `nct7k_test`.
- The pool composition (and hence the silent-failure prevalence) differs by family; budgets are shares of the SAME pool the
  policy is evaluated on. Absolute residual rates are therefore comparable across methods within a family, not across families.
- **Primary label `sf_090`** (wrong AND calibrated confidence >= 0.90). Sensitivities: `sf_095`, `sf_tau`.
  The residual ERROR rate (any wrong prediction) is reported next to it.

## 2. Monitors (scores; higher = refer first)
- **M0** = log(1-MSP) + log(entropy) fusion, **M1** = M0 + Mahalanobis, **M2** = M1 + layer4 normalised CAM JS: exactly the Phase 5
  detectors, with the Phase 5 rotation weights (non-negative logistic regression, tuned per seed ONLY on the two tuning families,
  equal family weight, features = val_cal tail-extended percentiles). Weights are re-fitted with the same function and
  ASSERTED equal (1e-6) to `outputs/stage5/fusion_weights.csv`; any difference aborts the run.
- **Random**: referral without information; **Oracle**: ranking by the true label (silent failures first), as a ceiling.
  Both are reference baselines, not detectors.
- Never used for scores, weights or thresholds: any held-out-family label or feature, any val_cal-derived threshold.

## 3. Primary protocol - exact-budget ranking
- For every budget B in **{5, 10, 20}%** and every held-out family, the top B% of the held-out family's deployment pool by score
  (separately per seed) leave automatic acceptance. No threshold is transferred, so the realised workload is exactly B (up to ties:
  a tie group straddling the boundary is flagged fractionally, which makes every method, the random baseline and the oracle
  exactly unbiased at workload B; the share of rows in boundary tie groups is reported).
- **Accept / review / refer:** B is the total share leaving automatic acceptance; **refer** = the top half of the flagged (B/2 of the
  pool, highest scores), **review** = the other half. Workload split reported (B/2 + B/2). NO cost model is used or invented; residual
  metrics depend only on B. The split is used for the descriptive tier table (share of all silent failures falling into the refer tier
  vs the review tier).
- A full budget grid {1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50}% is evaluated for the figure; only {5, 10, 20}% enter the primary contrasts.

## 4. Metrics (per held-out family, budget, method; mean over 5 seeds)
1. **Residual silent-failure rate (PRIMARY)** = share of accepted cases that are silent failures (sf_090).
2. Residual error rate = share of accepted cases that are errors.
3. Fraction of all silent failures captured by the flagged set (recall at workload B).
4. AURC of the risk-coverage curve (risk = silent-failure rate among accepted, accepting lowest score first; identical to the
   Phase 5 definition and asserted equal to it on unit weights), and the risk-coverage curve for the figure.
- Random baseline residual = pool prevalence (unbiased in expectation); oracle residual = (S - min(S, BN)) / (N - BN) with random fill.

## 5. Primary contrasts and inference
- Per held-out family H in {A, B, C} and budget B in {5, 10, 20}%: **M1 - M0** and **M2 - M1** in residual silent-failure rate
  (3 x 3 x 2 = 18 declared primary estimates), plus the reference contrasts **M0 - random, M1 - random, M2 - random** (also unadjusted).
  The X supplement reports the same contrasts and is not a primary.
- Inference: paired two-level cluster bootstrap exactly as Phase 5 (B = 2,000, bootstrap seed 0): (1) resample the 5 seeds with
  replacement, (2) resample cluster units (PathMNIST near-duplicate clusters for A, C, X; Camelyon patients for B) with replacement;
  the same draw is applied to every method (paired). In every replicate the policy is re-applied to the weighted resample (the top B%
  of the resample's cluster-weighted mass is flagged), so workload is B in every replicate. Fusion weights are NOT refitted inside
  the bootstrap. 95% percentile CIs; two-sided bootstrap p = 2 min(P(d <= 0), P(d >= 0)). Seed jackknife (leave-one-seed-out) of every
  primary contrast. No multiplicity adjustment on the declared primaries.
- **Verdict rule (generated from the numbers):** a contrast d (negative = fewer residual silent failures) is "reduces" if the CI lies
  entirely below 0 AND |d| >= 0.001 (0.1 percentage point); "increases" if entirely above 0 AND |d| >= 0.001; "no practical difference"
  if |d| < 0.001; otherwise "inconclusive". Magnitudes (points and relative to the comparator) are always printed with the verdict.
- Negative-result-ready statement: does M1 or M2 reduce residual silent failures at a fixed budget out-of-family, with CIs.

## 6. Secondary protocol - Blueprint-literal threshold transfer (descriptive)
- For each budget B, held-out family H and seed: the score threshold is the family-balanced weighted (1-B) quantile of the fused
  score over the pooled TWO tuning families' rows (flag score >= t, minimal t with tuning flag mass >= B). The threshold is applied to the
  held-out rows. Reported: **realised workload** (share flagged on the held-out family), residual silent-failure rate, capture; every
  realised workload deviating from B by more than +-1 percentage point is FLAGGED. Random at the same workload has residual = prevalence.
  Contrasts between methods at different realised workloads are marked "not workload-matched" (|difference in workload| > 1 point).
  Paired cluster bootstrap with the flagged set fixed by the threshold. No val_cal-derived threshold anywhere.

## 7. Sensitivities and secondary breakdowns (BH-FDR within each family; never used to change the primaries)
- **Sensitivity family (one BH family, M1-M0 and M2-M1 at B in {5, 10, 20}%):** kNN instead of Mahalanobis for M1/M2; layer3 CAM and
  layer2 CAM for M2; v1 weak-shortcut models for family C; labels sf_095 and sf_tau (scores re-fitted with that label, weights tuned on
  tuning families only); seed jackknife; the X supplement (weights tuned on A+B+C).
- **Breakdowns (each its own BH family):** A by corruption and by severity, B by condition, C by condition. The exact-budget policy
  is re-applied inside the condition (the deployment pool is that condition), with the same rotation weights.

## 8. Code and outputs
`sfd/stage7.py` (analysis), `scripts/run_stage7.py`, `tests/test_stage7.py`; inputs: Phase 4/5 tables (`sfd.stage5.load`); outputs in
`outputs/stage7/` (`fig6_data.csv`, `table3_referral.csv`, `tiers.csv`, `threshold_transfer.csv`, `sensitivity.csv`,
`breakdowns.csv`, `seed_jackknife.csv`, `boundary_ties.csv`, `summary_stage7.md`, `figures/fig6_residual_vs_budget.png`).
