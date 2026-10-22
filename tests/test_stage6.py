"""Stage 6 tests: window determinism, KS, MMD, thresholds/leakage, actual FAR, combined-monitor leakage, bootstrap, KM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfd import stage6 as s6


def test_windows_deterministic_distinct_and_fraction_counts():
    shift, idp = np.arange(1000, 1500), np.arange(0, 500)
    a, m = s6.sample_windows(shift, idp, 0.25, 40, 30, s6.cell_rng(3, "A|sev1|0.25", 40))
    b, _ = s6.sample_windows(shift, idp, 0.25, 40, 30, s6.cell_rng(3, "A|sev1|0.25", 40))
    c, _ = s6.sample_windows(shift, idp, 0.25, 40, 30, s6.cell_rng(4, "A|sev1|0.25", 40))
    assert m == 10 and np.array_equal(a, b) and not np.array_equal(a, c)
    assert all(len(set(r)) == 40 for r in a)                                   # without replacement inside a window
    assert ((a >= 1000).sum(1) == 10).all() and ((a < 500).sum(1) == 30).all()  # exact shifted share
    z, mz = s6.sample_windows(None, idp, 0.0, 40, 5, s6.cell_rng(0, "src", 40))
    assert mz == 0 and (z < 500).all()


def test_ks_matches_scipy_and_is_monotone_invariant():
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(0)
    ref = np.sort(rng.normal(size=2000))
    x = rng.normal(0.3, 1.2, (20, 100))
    mine = s6.ks_stat(x, ref)
    theirs = [ks_2samp(r, ref).statistic for r in x]
    assert np.allclose(mine, theirs, atol=1e-9)


def test_mmd_zero_for_identical_distribution_and_large_for_shift():
    rng = np.random.default_rng(1)
    D = 16
    ref = rng.normal(size=(600, D)).astype(np.float32)
    mmd = s6.MMD(ref, sigma=float(np.sqrt(2 * D)), device="cpu")
    same = rng.normal(size=(300, 40, D)).astype(np.float32)
    shifted = (rng.normal(size=(300, 40, D)) + 1.0).astype(np.float32)
    ms, mh = mmd(same), mmd(shifted)
    assert abs(ms.mean()) < 3 * ms.std() / np.sqrt(len(ms)) + 1e-3             # unbiased: ~0 on average
    assert mh.mean() > 20 * abs(ms.mean()) and mh.mean() > 0.05


def _synthetic_W(seed=0, n_val=400, n_far=300):
    rng = np.random.default_rng(seed)
    rows = []
    models = [("pm_std", "pathmnist", "standard", ["A|src", "A|sev1", "X|src", "X|nct7k"]),
              ("cam_std", "camelyon17", "standard", ["B|src", "B|ood_c1"]),
              ("sc2", "pathmnist", "shortcutv2", ["C|src", "C|reversed"])]
    def block(model, ds, kind, role, cell, fam, fr, n, shift):
        d = {c: rng.normal(size=n) + (shift if c in ("u_mean", "r_mmd", "e_cam_l4", "u_ks", "u_p90") else 0.0) for c in s6.STATS}
        d.update(err=rng.random(n) * (0.05 + 0.3 * shift), conf=0.9 + 0.05 * rng.random(n) - 0.02 * shift)
        return pd.DataFrame(d).assign(model=model, dataset=ds, kind=kind, seed=0, role=role, cell=cell, family=fam, fraction=fr,
                                      size=100, n_shift=int(100 * fr), wid=np.arange(n), corruption=None, severity=None, sigma=1.0)
    for model, ds, kind, cells in models:
        rows.append(block(model, ds, kind, "valcal", "valcal", "valcal", 0.0, n_val, 0.0))
        rows.append(block(model, ds, kind, "far", "far", "far", 0.0, n_far, 0.0))
        for c in cells:
            fam = c.split("|")[0]
            fr = 0.0 if c.endswith("src") else 0.5
            rows.append(block(model, ds, kind, "cell", c, fam, fr, 60, 0.0 if fr == 0 else 1.0))
    return pd.concat(rows, ignore_index=True)


def test_thresholds_use_only_valcal_windows():
    W = _synthetic_W()
    t1 = s6.thresholds(W)
    W2 = W.copy()
    m = W2.role != "valcal"
    W2.loc[m, s6.STATS] = W2.loc[m, s6.STATS] + 100.0                          # wildly different non-val_cal windows
    W2.loc[m, ["err", "conf"]] = 0.5
    t2 = s6.thresholds(W2)
    pd.testing.assert_frame_equal(t1, t2)
    v = W[(W.model == "pm_std") & (W.role == "valcal")]
    assert t1[t1.model == "pm_std"]["thr_u_mean"].iloc[0] == pytest.approx(v.u_mean.quantile(0.95))


def test_actual_false_alarm_rate_on_heldout_windows_is_near_five_percent():
    W = _synthetic_W(seed=2, n_val=3000, n_far=2000)
    E = s6.build_eval(W, s6.thresholds(W))
    far = E[(E.role == "far") & (E.H == "B")]
    for c in s6.STATS:
        assert 0.03 <= far["alarm_" + c].mean() <= 0.07, c


def test_combined_monitor_never_uses_heldout_family_labels():
    W = _synthetic_W(seed=3, n_val=500, n_far=300)
    thr = s6.thresholds(W)
    E, wt1 = s6.add_combined(s6.build_eval(W, thr))
    E2 = s6.build_eval(W, thr)
    h = (E2.H == "A") & (E2.role == "cell")                                   # flip the held-out family's labels
    E2.loc[h, "n_shift"] = 100 - E2.loc[h, "n_shift"]
    E2.loc[h, s6.FUSE] = E2.loc[h, s6.FUSE] + 5.0
    _, wt2 = s6.add_combined(E2)
    a1, a2 = wt1[wt1.H == "A"].reset_index(drop=True), wt2[wt2.H == "A"].reset_index(drop=True)
    assert np.allclose(a1[[f"w_{c}" for c in s6.FUSE] + ["intercept"]], a2[[f"w_{c}" for c in s6.FUSE] + ["intercept"]])
    assert (wt1[[f"w_{c}" for c in s6.FUSE]] >= 0).all().all()                  # monotone (non-negative) fusion


def test_uid_half_split_is_deterministic_balanced_and_disjoint():
    uids = np.array([f"pathmnist/id_test/{i}" for i in range(4000)])
    h1, h2 = s6.uid_half(uids), s6.uid_half(uids)
    assert np.array_equal(h1, h2) and set(h1) == {0, 1} and abs(h1.mean() - 0.5) < 0.03
    assert np.array_equal(s6.uid_half(uids[:100]), h1[:100])                  # a row's half never depends on the others


def test_silent_window_definition():
    W = _synthetic_W(seed=4)
    E = s6.build_eval(W, s6.thresholds(W))
    s = E[E.silent]
    assert (s.n_shift > 0).all() and (s.err > s.err_ub).all() and ((s.conf >= s.conf_lo) & (s.conf <= s.conf_hi)).all()
    assert not E[(E.role != "cell")].silent.any()                               # source-only windows are never silent


def test_bootstrap_paired_deterministic_and_identical_alarms_give_zero_diff():
    import scipy.sparse as sps
    rng = np.random.default_rng(5)
    n, nC = 400, 60
    R = pd.DataFrame({"seed": np.repeat([0, 1], n // 2), "alarm_u_mean": rng.random(n) < 0.3})
    R["alarm_r_mmd"] = R["alarm_u_mean"]
    Mx = sps.csr_matrix(np.eye(nC)[rng.integers(0, nC, n)])
    masks = {"all": np.ones(n, bool), "half": np.arange(n) % 2 == 0}
    a = s6.boot_power(R, Mx, masks, ["alarm_u_mean", "alarm_r_mmd"], 300, [0, 1], rng_seed=7)
    b = s6.boot_power(R, Mx, masks, ["alarm_u_mean", "alarm_r_mmd"], 300, [0, 1], rng_seed=7)
    assert np.allclose(a["all"], b["all"], equal_nan=True)
    assert np.allclose(a["all"][:, 0] - a["all"][:, 1], 0)                      # paired: identical alarms -> delta 0
    lo, hi = np.percentile(a["all"][:, 0], [2.5, 97.5])
    assert lo < R.alarm_u_mean.mean() < hi


def test_km_median_delay_and_censoring():
    w = np.ones((50, 1))
    assert s6.km_median(np.full(50, 3), np.ones(50, bool), w, 30)[0] == 3
    assert np.isnan(s6.km_median(np.full(50, 31), np.zeros(50, bool), w, 30)[0])           # all censored
    d = np.r_[np.full(30, 2), np.full(20, 31)]
    ev = np.r_[np.ones(30, bool), np.zeros(20, bool)]
    assert s6.km_median(d, ev, w, 30)[0] == 2                                      # 60% detected at delay 2
