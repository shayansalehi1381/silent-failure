"""Stage 7 tests: exact budget, leakage (weights / thresholds), oracle ceiling, unbiased random baseline, paired bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfd import stage5 as s5
from sfd import stage7 as s7


def _data(n=2000, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.08
    err = y | (rng.random(n) < 0.10)
    score = rng.normal(size=n) + 1.5 * y
    if ties:
        score = np.round(score, 1)
    return score, y, err


def test_exact_budget_flagged_count_is_ceil_b_n_up_to_ties():
    score, y, err = _data()
    n = len(score)
    for b in (0.05, 0.10, 0.20):
        f = s7.flag_fractions(score, np.ones(n), b)
        assert set(np.unique(f)) <= {0.0, 1.0}                                   # continuous scores: no fractional flags
        assert int(f.sum()) == int(np.ceil(b * n))
        assert score[f == 1].min() >= score[f == 0].max()                        # the top-b by score are flagged
    st, _ = s7.Policy(score, y, err).stats(np.ones(n), [0.05, 0.10, 0.20])
    k = np.ceil(np.array([0.05, 0.10, 0.20]) * n)
    top = np.argsort(-score)
    manual = [(y.sum() - y[top[: int(kk)]].sum()) / (n - kk) for kk in k]
    assert np.allclose(st[:, 0], manual)


def test_ties_flagged_fractionally_so_workload_is_exact_and_unbiased():
    score, y, err = _data(ties=True)
    w = np.random.default_rng(3).integers(0, 4, len(score)).astype(float)       # bootstrap-like weights (zeros included)
    for b in (0.05, 0.10, 0.20):
        f = s7.flag_fractions(score, w, b)
        assert np.isclose((w * f).sum(), b * w.sum())
    # a constant score carries no information: residual rate = prevalence, for ANY weights
    st, _ = s7.Policy(s7.random_score(len(score)), y, err).stats(w, [0.05, 0.10, 0.20])
    assert np.allclose(st[:, 0], (w * y).sum() / w.sum())
    assert np.allclose(st[:, 2], [0.05, 0.10, 0.20])                              # captures exactly B of the silent failures


def test_random_referral_is_unbiased_monte_carlo():
    _, y, err = _data(seed=2)
    rng = np.random.default_rng(0)
    res = []
    for _ in range(400):
        sc = rng.random(len(y))                                                   # a uniformly random ranking
        st, _ = s7.Policy(sc, y, err).stats(np.ones(len(y)), [0.10])
        res.append(st[0, 0])
    se = np.std(res) / np.sqrt(len(res))
    assert abs(np.mean(res) - y.mean()) < 4 * se + 1e-4


def test_oracle_is_a_ceiling_for_every_method_and_weights():
    rng = np.random.default_rng(4)
    for seed in range(5):
        score, y, err = _data(seed=seed, ties=bool(seed % 2))
        w = rng.integers(0, 3, len(y)).astype(float)
        bud = np.array([0.01, 0.05, 0.10, 0.20, 0.5])
        o, _ = s7.Policy(s7.oracle_score(y), y, err).stats(w, bud)
        for sc in (score, -score, rng.random(len(y)), s7.random_score(len(y))):
            m, _ = s7.Policy(sc, y, err).stats(w, bud)
            assert (o[:, 0] <= m[:, 0] + 1e-12).all() and (o[:, 2] >= m[:, 2] - 1e-12).all()
        P, W = (w * y).sum(), w.sum()
        assert np.allclose(o[:, 0], (P - np.minimum(bud * W, P)) / (W - bud * W))   # closed form


def test_aurc_matches_phase5_definition():
    score, y, err = _data(seed=5, ties=True)
    a = s5.Ranked(score, y).metrics(np.ones(len(y)))[2]
    b = s7.Policy(score, y, err).stats(np.ones(len(y)), [0.1])[1]
    assert a == pytest.approx(b)


def _block(rng, pool, n):
    y = rng.random(n) < 0.1
    d = pd.DataFrame({"pool": pool, "seed": 0, "uid": [f"{pool}/{i}" for i in range(n)], "family": pool, "condition": "c",
                      "severity": 0})
    d["sf_090"] = y
    d["correct"] = ~(y | (rng.random(n) < 0.1))
    d["cluster"] = [f"g{i % 50}" for i in range(n)]
    d["x_u_log1msp"] = np.clip(rng.random(n) * 0.5 + 0.4 * y, 0, 1)
    d["x_u_logent"] = np.clip(rng.random(n) * 0.5 + 0.4 * y, 0, 1)
    return d


def test_no_heldout_label_in_fusion_weights_or_thresholds(monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.concat([_block(rng, "A", 600), _block(rng, "B", 600), _block(rng, "C", 400)], ignore_index=True)
    det = {"M0": ["u_log1msp", "u_logent"]}
    monkeypatch.setattr(s5, "rotation", lambda d, H, c_kind="shortcutv2": (d[d.pool != H].copy(), d[d.pool == H].copy()))
    sc1 = s7.fit_scores(df, "C", det, "sf_090", [0])
    df2 = df.copy()
    m = df2.pool == "C"
    df2.loc[m, "sf_090"] = ~df2.loc[m, "sf_090"]                                   # flip every held-out label ...
    df2.loc[m, "x_u_log1msp"] = 1.0 - df2.loc[m, "x_u_log1msp"]                    # ... and distort its features
    sc2 = s7.fit_scores(df2, "C", det, "sf_090", [0])
    pd.testing.assert_frame_equal(pd.DataFrame(sc1["weights"]), pd.DataFrame(sc2["weights"]))   # weights unaffected
    assert np.allclose(sc1["tscore"]["M0"][0], sc2["tscore"]["M0"][0])
    t1 = [s7.weighted_threshold(sc1["tscore"]["M0"][0], sc1["tw"][0], b) for b in (0.05, 0.1, 0.2)]
    t2 = [s7.weighted_threshold(sc2["tscore"]["M0"][0], sc2["tw"][0], b) for b in (0.05, 0.1, 0.2)]
    assert t1 == t2                                                                 # thresholds unaffected
    assert (pd.DataFrame(sc1["weights"])[["w_u_log1msp", "w_u_logent"]] >= 0).all().all()   # monotone fusion


def test_weighted_threshold_flags_at_least_b_and_close():
    rng = np.random.default_rng(1)
    s = rng.normal(size=5000)
    w = rng.random(5000)
    for b in (0.05, 0.1, 0.2):
        t = s7.weighted_threshold(s, w, b)
        share = w[s >= t].sum() / w.sum()
        assert b - 1e-9 <= share <= b + 0.002


def test_paired_bootstrap_deterministic_and_identical_methods_give_zero_difference():
    score, y, err = _data(n=600, seed=7)
    seeds = [0, 1]
    te = {s: pd.DataFrame({"seed": s, "cluster": [f"g{i % 40}" for i in range(600)], "correct": ~err, "sf_090": y,
                           "condition": "c", "severity": 0}) for s in seeds}
    sc = {"test": te, "score": {"M0": {s: score for s in seeds}, "M1": {s: score for s in seeds}}, "label": "sf_090"}
    pol = s7.make_policies(sc, ["M0", "M1"], seeds)
    a = s7.paired_bootstrap(pol, sc, seeds, [0.05, 0.1], 200, rng_seed=3)
    b = s7.paired_bootstrap(pol, sc, seeds, [0.05, 0.1], 200, rng_seed=3)
    assert np.allclose(a["boot"]["M0"], b["boot"]["M0"], equal_nan=True)
    assert np.allclose(a["boot"]["M0"] - a["boot"]["M1"], 0)                         # paired: identical scores -> delta 0
    lo, hi = s5.ci(a["boot"]["M0"][:, 0, 0])
    assert lo < a["point"]["M0"][0][0, 0] < hi
    assert (a["boot"]["oracle"][:, :, 0] <= a["boot"]["M0"][:, :, 0] + 1e-12).all()   # oracle ceiling in every replicate
    rows = pd.DataFrame(s7.result_rows(a, "C", {"analysis": "t", "label": "sf_090"}, budgets_keep=[0.05, 0.1]))
    r = rows[(rows.quantity == "M1-M0") & (rows.metric == "dresid_sf")]
    assert (r.verdict == "no practical difference").all()


def test_verdict_rule_uses_ci_and_practical_margin():
    assert s7.verdict(-0.01, -0.02, -0.005) == "reduces"
    assert s7.verdict(0.01, 0.005, 0.02) == "increases"
    assert s7.verdict(-0.01, -0.03, 0.01) == "inconclusive"
    assert s7.verdict(-2e-6, -3e-6, -1e-6) == "no practical difference"             # zero-weight fusion noise
