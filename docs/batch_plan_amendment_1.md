# Amendment 1 to the frozen batch plan (post-freeze, outcome-informed - labelled as such)

Frozen plan: `docs/batch_plan_frozen.md` (commit 931fb36). It is NOT edited. All pre-registered analyses (val_cal thresholds)
are run and reported unchanged as the primary results.

## Why this amendment exists
The pre-registered false-alarm check (Sec 3) FAILED: with thresholds set on `val_cal` source windows, the actual false-alarm
rate on held-out `id_test` source windows is far outside 3-7% for most statistics (e.g. mean log-uncertainty 0.37 on
PathMNIST). Diagnosis (done before this amendment, no power or contrast result was inspected): fresh windows drawn from
`val_cal` itself alarm at 4.6-6.2% (the pipeline is unbiased), so `val_cal` and `id_test` are not exchangeable at the window
level: id_test has slightly heavier uncertainty tails, and window means (sampling sd shrinks by sqrt(n)) turn that small
row-level difference into a ~1 sd shift of the window statistic. At unequal actual false-alarm rates, the power of different
statistics is not comparable, so the declared contrasts are confounded by false alarms.

## Amendment A1: FAR-matched thresholds from a disjoint, exchangeable source half
Everything is unchanged (cells, fractions, statistics, silent definition, combined monitor, bootstrap, contrasts, delays)
except where thresholds, silent bounds and fusion standardisation come from:
- Each model's ID pool (as defined in plan Sec 1) is split ROW-wise into two disjoint halves by a fixed hash of the row uid
  (`sha256(uid) mod 2`): the **calibration half** (hash 0) and the **evaluation half** (hash 1). Row-level (not cluster-level)
  splitting keeps the halves exchangeable; a cluster-level split would put the giant near-duplicate clusters into one half.
- Thresholds (95th percentile of each statistic), silent-window bounds (`err_ub`, `[c_lo, c_hi]`) and the fusion
  standardisation / combined-monitor threshold are computed from 3,000 source-only windows drawn from the calibration half.
- The ID component of every mixture window and the 2,000 false-alarm windows are drawn from the evaluation half only.
  For family C the aligned rows (all 3,000) are split by the same hash; the shifted pool (half 2 of the sample) is unchanged.
- The reference quantities of the statistics themselves (KS reference, MMD bandwidth, class CAM references) remain the
  val_cal/train objects of the frozen plan.
A1 results are reported in a separate set of tables and called "FAR-matched (post-freeze amendment A1)". They are a
sensitivity analysis, never relabelled as the pre-registered primary. The actual false-alarm rate is re-measured on the
evaluation half and flagged against 3-7% in the same way; the declared contrasts are repeated with the same bootstrap.

## What stays pre-registered
The primary pre-registered results (val_cal thresholds), including their failed FAR check, remain the primary endpoint and
are reported first. A1 is added only to make the power comparison fair; where the two disagree, both are reported.
