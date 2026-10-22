# Frozen batch-monitoring plan - Phase 6 (window level)

Pre-registration of the Phase 6 analysis (Blueprint Sec 4 batch column, 5.1, 5.4 batch testing, 5.5). It is committed BEFORE
any batch (window) statistic is computed; the commit hash of this file is the freeze point. Phase 5 results, the frozen splits,
references and classifiers are read-only. Only per-image arrays (features, CAMs) are extracted before the freeze is used.

## 1. Models, pools and windows
- Models: the 20 frozen classifiers (PathMNIST standard, Camelyon standard, shortcut v2, shortcut v1; seeds 0-4); every
  result is per classifier seed and aggregated over seeds. Shortcut v2 is the primary family C model, v1 a supplement.
- **Window** = `n` rows sampled WITHOUT replacement inside the window, from a seeded RNG keyed by (model seed, cell, window
  size); windows of different cells/seeds are independent draws, so results are deterministic and re-runnable.
  Primary size **n = 100**; sensitivities n = 50 and n = 200 (primary cells only).
- A window with shifted fraction `f` contains round(f*n) rows from the family's shifted pool and the rest from its ID pool;
  f = 0 are source-only windows. Fractions {0, 0.1, 0.25, 0.5, 0.75, 1.0}.
- Pools (per model kind):
  - **A** (PathMNIST standard): ID pool = `id_test` rows NOT in the fixed 3,000-image Stage-3 sample (7,247); shifted pool =
    the corrupted versions of the sample, grouped by severity 1-5 (pooled over the 11 corruptions; per-corruption cells are
    secondary, at fraction 1.0).
  - **B** (Camelyon standard): ID pool = Camelyon `id_test`; shifted pools = `ood_c1` and `ood_c2` separately.
  - **C** (shortcut v2; v1 = supplement): the 3,000-image sample split once into two disjoint seeded halves; ID pool =
    `aligned` rows of half 1, shifted pool = `reversed` (primary) or `removed` (secondary) rows of half 2.
  - **X** (supplement, PathMNIST standard): ID pool = `id_test` (all 10,247), shifted = `nct7k_test`.
- Source-only windows for thresholds and silent-window bounds: 3,000 windows per model and size from `val_cal` (with the
  model's own training-cue recipe for shortcut models). Windows for the ACTUAL false-alarm rate: 2,000 source-only windows per
  model and size from the held-out ID pool (A, X: all `id_test`; B: Camelyon `id_test`; C: all 3,000 `aligned` rows).
- **Windows per cell and seed: 200** (>= 200 as required); cells: primary = A severity 1-5, B ood_c1/ood_c2, C reversed,
  X, each at 5 fractions (+ a source-only cell per family); secondary = C removed, per-corruption (11 x 5 severities, f = 1.0).
  Compute check: a speed pilot is run first; if the projected CPU time exceeds 60 min, per-corruption cells are reduced to
  severities {1, 3, 5} (pre-declared fallback; primary cells are never reduced). The choice is reported.

## 2. Batch statistics (higher = more shifted)
Per-image inputs are the Phase 4 objects (frozen): calibrated probabilities, 512-d penultimate features, Grad-CAMs.
1. **Uncertainty:** `u_mean` = window mean of log(1 - MSP) (stable log scale, from logits / T; PRIMARY uncertainty statistic);
   `u_p90` = its 90th percentile; `u_ks` = KS statistic between the window's log(1 - MSP) values and the `val_cal` values
   (KS is invariant to the monotone log transform, so it is the confidence-distribution shift statistic).
2. **Representation:** `r_mmd` = unbiased RBF-kernel MMD^2 between the window's features and a FIXED 2,000-row train
   reference (first 2,000 rows of the Phase 4 train reference sample), bandwidth sigma = median pairwise L2 distance of a
   1,000-row `val_cal` subsample (val_cal only). Sensitivities: `r_maha_mean` (mean per-image Mahalanobis), `r_knn_mean` (mean per-image kNN distance).
3. **Explanation:** `e_cam_l4` (PRIMARY) = JS divergence between the window's MEAN unit-mass Grad-CAM (layer4) and the
   class-conditional source reference mixture sum_c pi_c * meanCAM_c (pi_c = share of window rows predicted as class c;
   meanCAM_c = Phase 4 train-only class reference). Sensitivities: `e_cam_l3`, `e_cam_l2` (same with layer3 / layer2
   references), `e_img_mean` (mean per-image transform-normalised layer4 CAM JS from Phase 4).
4. **Combined monitor:** non-negative logistic regression (L-BFGS-B, L2 1e-3, intercept free, equal family weight) on the
   standardised statistics {u_mean, u_p90, u_ks, r_mmd, e_cam_l4}; standardisation = z-score with the mean/sd of the model's
   `val_cal` source windows; tuned per seed on the windows of the TWO tuning families (label = window contains shifted rows,
   i.e. f > 0, vs f = 0), evaluated on the held-out third (same A/B/C rotation as Phase 5; X evaluated with weights tuned on
   A+B+C). The alarm threshold of the combined score is the 95th percentile of its value over the held-out model's
   `val_cal` source windows.

## 3. Alert thresholds and false alarms
- Threshold of every statistic and every size = **95th percentile over the 3,000 `val_cal` source-only windows of that model**
  (5% nominal false alarm); no shifted data are ever used. A window alarms when its statistic exceeds the threshold.
- **Actual FAR** is measured separately on the 2,000 held-out `id_test` source windows (see Sec 1) per statistic, family and
  size; tolerance 3-7%: any statistic outside is flagged in the tables and the text.

## 4. Evaluation-only ground truth: silent windows (Blueprint 5.5)
- From the model's `val_cal` source windows: `err_ub` = 95th percentile of the window error rate; `[c_lo, c_hi]` = 2.5-97.5%
  interval of the window mean calibrated confidence.
- A window is **silently failed** when its true error rate > `err_ub` AND its mean calibrated confidence is inside
  `[c_lo, c_hi]`. Labels are used for evaluation only, never for thresholds or fusion.
- **PRIMARY endpoint: detection power at 5% nominal FAR on silently-failed windows** (share of silent windows with alarm);
  secondary: power on all shifted windows (f > 0). Reported by statistic and by shifted fraction / severity, together with
  the share of shifted windows that are silently failed. By construction mean-confidence/uncertainty statistics cannot
  detect silent windows well (the silent definition requires the mean confidence to look normal). Cells with fewer than 20
  silent windows (summed over seeds) are reported but excluded from the contrasts ("insufficient").

## 5. Detection delay
- Streams: 20 source-only windows, then onset: abrupt at fractions {0.25, 0.5, 1.0} or one gradual ramp (fractions
  0.1, 0.25, 0.5, 0.75, 1.0, six windows each); post-onset horizon 30 windows. Stream windows are independent draws (with
  replacement across streams) from the pre-computed window pool of the corresponding cell. 100 streams per (cell, onset,
  seed). Cells: A severity {1, 3, 5}, B ood_c1, ood_c2, C reversed, X.
- Delay = number of post-onset windows observed up to and including the first alarm; right-censored at 30 if none.
  Reported: weighted Kaplan-Meier median delay with the censored share; bootstrap 95% CI; pre-onset alarm rate.

## 6. Inference
- Bootstrap B = 2,000, paired across statistics (every replicate evaluates all statistics): (1) resample the 5 classifier
  seeds with replacement; (2) resample cluster units of the family's rows (Camelyon patients; PathMNIST near-duplicate
  clusters) with replacement; a window's weight = mean cluster multiplicity of its rows, times its seed's multiplicity.
  Power = weighted share of alarming windows. 95% percentile CIs; two-sided bootstrap p = 2 min(P(d<=0), P(d>=0)).
- **Declared primary contrasts** (power on silent windows, pooled over the held-out family's shifted cells with >= 20 silent
  windows; no multiplicity adjustment): (a) combined monitor vs `u_mean`; (b) `r_mmd` vs `u_mean`; (c) `e_cam_l4` vs `r_mmd`.
  Verdict from the 95% CI of the power difference: "improves" if entirely > 0, "worse" if entirely < 0, else "inconclusive".
- BH-FDR only inside the prespecified secondary families: window size (50/200), CAM layer (l3/l2), per-corruption, per-severity
  (each its own family). Seed spread (per-seed power) is reported for every primary quantity.
- Sensitivities: window sizes 50 and 200; v1 weak shortcut for family C; kNN-based window statistic; CAM layers l3 / l2;
  mean-Mahalanobis; mean per-image CAM JS; the X supplement.

## 7. Code and data
`sfd/stage6.py`, `scripts/run_stage6.py`; per-image arrays and window statistics are git-ignored (`outputs/stage6/rows/`,
`outputs/stage6/windows/`); small summary CSVs, figures and the threshold/silent-bound tables are committed.
Tests: thresholds built only from `val_cal` windows; actual FAR on held-out windows near 5%; paired bootstrap; deterministic
window sampling; MMD ~ 0 for identical distributions; KS equals scipy.
