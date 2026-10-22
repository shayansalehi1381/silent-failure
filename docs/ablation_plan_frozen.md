# Frozen ablation plan - Phase 8 (ablations and sensitivity)

Pre-registration of the Phase 8 analysis. It is committed BEFORE any ablation result is computed; the commit hash of this file
is the freeze point. Frozen splits, references, classifiers, Phase 5/6/7 outputs are read-only. Design inputs: Phase 5-7 results
(known) and the user's decisions; no Phase 8 number exists at freeze time.

## 0. Data reuse (no retraining, no GPU, no image re-extraction)
- Pools, labels, clusters, val_cal tail-extended percentile features, rotation: exactly Phase 5/7 (`sfd.stage5.load`, `rotation`,
  `fit_scores` machinery). Held-out families: A, B, C (v2 primary, v1 supplement). X is not re-analysed (supplement of Phases 5/7).
- **Reference-size ablation (Mahalanobis only).** The Phase 4 reference bank (`outputs/stage4/refs/*.npz`: 20,000 PathMNIST /
  40,000 Camelyon source-TRAIN features; labels from the frozen splits through `ref_uid_pos`) is sub-sampled: nested random subsets of
  **500 / 1,000 / 2,000 / full bank** (seeded permutation, seed 0, same for every model/seed of that dataset; class counts reported; a
  class with < 2 rows is flagged). Class means + shared Ledoit-Wolf precision are re-fitted from the stored (fp16) bank features; the
  scores of every row are recomputed from the stored per-row features (`outputs/stage6/rows/*.npz`) and standardised with the val_cal
  rows of that model. The "full bank" refit is validated against the stored Phase 4 reference (rank correlation reported); the stored
  reference stays the primary M1. "Full train set" beyond the saved bank would need a re-extraction and is NOT done; this is stated in the
  results. kNN reference size is not ablated (cost).
- **Temperature ablation.** Logits are reconstructed from the stored penultimate features with the frozen final linear layer (checked
  against the stored calibrated confidence: max abs difference asserted < 1e-3). "Before temperature scaling" = M0 features computed
  from the T = 1 softmax (log(1-MSP) and log-entropy at T = 1, val_cal percentiles at T = 1); the silent-failure LABEL stays the
  Phase 5 `sf_090` (calibrated confidence >= 0.90), so the target is identical in both arms; Mahalanobis is unaffected by T.

## 1. Endpoints
- **Primary endpoint:** residual silent-failure rate (`sf_090`) among accepted cases at exact budgets B in {5, 10, 20}% on each
  held-out family (Phase 7 exact-budget protocol: top B% by score of the held-out pool, weighted ties, policy re-applied in every
  bootstrap replicate). **Secondary:** AUROC, AUPRC (Phase 5 definitions), AURC, residual error rate, silent-failure capture.
- Paired two-level cluster bootstrap exactly as Phases 5/7 (B = 2,000, seed 0, seeds x clusters; one draw shared by all methods);
  fusion weights/gates are NOT refitted in the bootstrap. 95% percentile CIs; p = 2 min(P(d <= 0), P(d >= 0)); seed jackknife
  (leave-one-seed-out min/max) for every contrast. Verdict rule of Phase 7 (CI excludes 0 AND |d| >= 0.001 on the residual rate; for
  AUROC a CI excluding 0 and |d| >= 0.005).
- Variants are fitted per seed ONLY on the two tuning families (rotation), never on the held-out family.

## 2. Detectors and the two fusion constraints
Standardised features are the val_cal tail-extended percentiles (`x_*`). **NN** = non-negative logistic fusion (Phase 5, primary);
**SF** = sign-free logistic fusion (same loss, L2 1e-3 on the weights, intercept and weights unconstrained, L-BFGS). M0 = {log(1-MSP),
log-entropy}, M1 = M0 + Mahalanobis, M2 = M1 + layer4 normalised CAM JS. "Single signal" = the percentile feature itself (no fit).

## 3. Ablation groups (each yields one summary table with 95% CIs, a plain-language paragraph, and figures where stated)
| id | ablation | levels | contrasts (variant - reference) |
|---|---|---|---|
| G1 | signal groups | M0 (uncertainty only), M1 (+Mahalanobis), M0+CAM (no Mahalanobis), M2; single signals: log(1-MSP), log-entropy, Mahalanobis, kNN, layer4 normalised CAM | M1-M0, M0+CAM-M0, M2-M1, each single signal - log(1-MSP) |
| G2 | fusion constraint | NN vs SF for M0, M1, M2 | SF-NN per detector; SF M1 - SF M0; SF M2 - SF M1 |
| G3 | CAM layer / form | M1 + CAM {l4 norm (primary), l3 norm, l2 norm, l4 raw JS, l4 PCA distance}, NN and SF | CAM variant - M1 (same constraint); layer/form variant - l4 norm |
| G4 | Mahalanobis vs kNN | M1 and M2 with kNN, NN and SF | kNN - Mahalanobis |
| G5 | reference size | M1 with Mahalanobis fitted on 500 / 1,000 / 2,000 / full bank, NN and SF | size - full bank; M1(size) - M0 |
| G6 | temperature scaling | M0 and M1 from T=1 (before) vs calibrated T (after), NN | before - after |
| G7 | family A severity and corruption | policy applied inside each severity 0 (clean), 1-5 and each of the 11 corruptions + clean; M0, M1, M2 (NN), SF M1 | M1-M0, M2-M1, SF M1 - M1 per level |
| G8 | family C cue state | aligned / removed / reversed, v2 (primary) and v1 (supplement); M0, M1, M2 (NN), SF M1, SF M2 | M1-M0, M2-M1, SF M1 - M0 per state |
| G9 | label variants | `sf_090` (primary), `sf_095`, `sf_tau`, `sf_099`; M0, M1, M2 (NN) and SF M1 | M1-M0, M2-M1, SF M1 - M0 per label |
| G10 | seed jackknife | every contrast above | leave-one-seed-out min/max reported in each table |

**Declared primary contrasts (unadjusted, residual silent-failure rate, held-out A/B/C x B in {5,10,20}% = 27 estimates):**
P1 = SF M1 - SF M0 (does representation drift help when the weight sign is free?), P2 = SF M2 - SF M1 (does CAM drift help when the
sign is free?), P3 = SF M1 - NN M1 (does the sign constraint matter?). All other contrasts are SECONDARY: BH-FDR is applied only
inside each ablation group (one BH family per group over its residual-rate contrasts at the three budgets); AUROC/AUPRC/AURC contrasts
are descriptive.

## 4. Exploratory, post-hoc targeted fusion (clearly labelled; never replaces M0/M1/M2 in any main table)
Motivation (Phase 7): a silent failure needs calibrated confidence >= 0.90, whereas uncertainty-first ranking spends small budgets on
low-confidence cases. Variants, exact-budget protocol, weights/gates fitted ONLY on the two tuning families:
- **(b) sign-free ungated:** SF M0, SF M1, SF M2 (the sign of the uncertainty weight is free).
- **(a) confidence-gated ranking:** the gate is the fixed rule calibrated confidence >= 0.90 (the label's own threshold, not tuned and
  not a held-out quantity; it uses the case's own confidence, available at deployment). Cases inside the gate are ranked first; scores
  inside the gate: **G-M0** (NN M0 re-fitted on the gated tuning rows), **G-M1** (NN M1 refitted on gated tuning rows), **G-SF M1**
  and **G-SF M2** (sign-free fusion refitted on gated tuning rows). The flagged set fills the exact budget B from the gated pool; if the
  gated pool is smaller than B, the remainder is filled by the plain Phase 7 M0 rank among non-gated cases; the share of the flagged set
  filled that way, and the gated share of each pool, are reported.
- Reported in a separate table headed "exploratory, post-hoc" with oracle and random references, contrasts vs plain M0, plain M1 and
  random, per held-out family and budget (95% CIs, unadjusted, descriptive; the verdict rule is applied but not interpreted as
  confirmatory). Question answered: does a targeted (gated) policy reduce residual silent failures vs M0/M1 out-of-family.

## 5. Figures and tables (paper)
`fig8a_incremental_deltas.png` (incremental delta of residual rate and AUROC by ablation), `fig8b_severity_A.png` (family A by
severity), `fig8c_shortcut_states.png` (C: aligned / removed / reversed, v2 and v1), `fig8d_exploratory_gated.png`; summary tables
`outputs/stage8/tables/ablation_G*.csv`, `exploratory_targeted.csv`; `summary_stage8.md` with generated one-paragraph conclusions.

## 6. Code and outputs
`sfd/stage8.py`, `scripts/run_stage8.py` (resumable per group), `tests/test_stage8.py` (no held-out label/feature in weights or gates;
exact budget; oracle >= every method; gated-pool logic incl. fill; sign-free fit; NN fit equals Phase 5; paired bootstrap;
reference-refit and logit-reconstruction checks). Per-row augmented features are git-ignored; small CSV/PNG/MD are committed.
