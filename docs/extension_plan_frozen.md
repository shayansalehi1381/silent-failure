# Frozen extension plan - Phase 10 (post-hoc extensions E1, E2)

Pre-registration committed BEFORE any extension result is computed (the commit hash of this file is the freeze). Everything here is
post-hoc and labelled **EXTENSION**: it never replaces M0 / M1 / M2 and changes no earlier output, frozen split, reference or plan.
No retraining, no GPU, no image re-extraction.

## E1. Deep-ensemble uncertainty baseline
**Feasibility check (done before this plan, no result computed).** For every model kind (PathMNIST standard, Camelyon standard,
shortcut v2) the per-seed evaluation rows are IDENTICAL and identically ordered across the 5 seeds (same uid / family / condition /
severity / label arrays; hash-verified). Per-seed calibrated probability VECTORS were not saved, only the maximum probability; they are
reconstructed exactly as in Phase 8 from the saved penultimate features and the frozen final linear layer with the model's temperature T
(Phase 8 verified: 99.9th percentile of |conf error| < 1e-3, max 1.4e-3 on the saturated v2 models; the check is repeated and asserted
here). If the reconstruction check fails for any model, the run aborts.

**Target (unchanged):** silent failure (`sf_090`) of the single seed-s classifier, exactly as in Phases 5-8; unit of the seed level = s.
**Ensemble handling of the seed level.** For unit s the ensemble is the leave-seed-s-out ensemble of the other 4 seeds (E4_s): the
monitor is a peer ensemble that does not contain the monitored model, so seed units stay exchangeable and the seed-level bootstrap
(resample units s, then clusters) is applied exactly as before; the ensemble signal for unit s is a function of the other seeds only.
Sensitivity: the 5-member ensemble including s (E5). The variability caused by the choice of ensemble members is therefore only
partly captured (limitation, stated in the notebook).
**Signals (per row; p_k = softmax(z_k / T_k), p-bar = mean_k p_k over members, computed in float64 in probability space):**
- `ens_msp` = log(1 - max p-bar) (stable: log of the summed non-max probabilities, clipped at 1e-300);
- `ens_ent` = log H(p-bar), H = predictive entropy;
- `ens_mi` = log MI, MI = H(p-bar) - mean_k H(p_k) (mutual information / member disagreement; clipped at 1e-12 before the log).
Every signal is standardised by its val_cal tail-extended percentile (Phase 5 function; the ensemble's own val_cal rows, never shifted data).
**Methods per held-out family H in {A, B, C(v2)} (held-out-family rotation, label `sf_090`, tuning families only, exactly as Phase 5):**
M0 and M1 (Phase 5 definitions; weights asserted equal to Phase 5), **E-fuse** = non-negative fusion of {ens_msp, ens_ent} (the ensemble
analogue of M0), **E-msp**, **E-ent**, **E-mi** (single signals, no fitting), **E-M1** = E-fuse + Mahalanobis (seed-s Mahalanobis percentile),
and the E5 sensitivities **E5-fuse**, **E5-M1**. Random and oracle references as in Phase 7.
**Endpoints:** sample level AUROC, AUPRC, AURC; referral = residual silent-failure rate at exact budgets 5 / 10 / 20% (Phase 7 protocol,
policy re-applied in each bootstrap replicate), residual error rate and capture are also saved.
**Contrasts (all EXTENSION, unadjusted, paired two-level cluster bootstrap, B = 2,000, seed 0, leave-one-seed-out jackknife):**
E-fuse - M0 (does an ensemble improve on one model?), M1 - E-fuse (does representation drift add beyond an ensemble?), E-M1 - E-fuse (the
same with the ensemble as base), E-mi - E-msp. Verdict rules as in Phase 8 (residual: CI excludes 0 and |d| >= 0.1 pt; AUROC: |d| >= 0.005).
Reading rule fixed now: "the ensemble baseline changes a conclusion" only if M1 - E-fuse (or E-M1 - E-fuse) has a different verdict class
(improves / worse / inconclusive) from M1 - M0 on the same family, metric and budget.

## E2. In-family (source) vs out-of-family detection, saved tables only
Uses only committed tables (Phase 5 `secondary_breakdowns.csv`, `table2.csv`, `incremental_logit.csv`; Phase 7 `breakdowns.csv`). No computation
other than formatting and arithmetic on saved estimates.
- **Source vs shifted detection:** AUROC / AUPRC of M0, M1, M2 on source-distribution rows (family A `clean`, severity 0; family B `id_test`;
  family C `aligned`, which has no silent failures and is reported as not estimable) versus the shifted rows of the same family (A severities 1-5;
  B `ood_c1`, `ood_c2`; C `removed`, `reversed`), with the saved 95% CIs; and the residual-rate versions from Phase 7 breakdowns.
- **In-sample "in-family" information:** the Phase 5 incremental logistic regression on the held-out family itself (unconstrained, descriptive,
  fitted and evaluated in the same family): likelihood-ratio statistic and CAM / Mahalanobis coefficients, contrasted with the out-of-family monotone
  results. Reading: if CAM drift carries information in-family (significant coefficient) but not out-of-family under the monotone fusion, the
  information exists but does not transfer as a positive-weight alarm (sign matters; see Phase 8 P2).
- Limitation stated in advance: these tables do not contain a fusion tuned and tested inside the same family on disjoint clusters; "in-family"
  here means source-distribution evaluation plus in-sample logistic information, not a within-family train/test split.

## Code and outputs
`sfd/stage10.py`, `scripts/run_stage10.py`, `tests/test_stage10.py` (alignment across seeds, leave-seed-out ensemble excludes the monitored seed,
no held-out label or feature in weights, ensemble of one equals the single model, MI >= 0, reconstruction check). Outputs in `outputs/stage10/`
(small CSV / PNG / MD only; per-row augmented features git-ignored).
