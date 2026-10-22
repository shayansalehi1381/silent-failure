"""Stage 8 tests: leakage of weights/gates, exact budget, oracle ceiling, gated-pool logic, sign-free fit, paired bootstrap, helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfd import stage5 as s5
from sfd import stage7 as s7
from sfd import stage8 as s8
from sfd.signals import representation as rp


def test_nonneg_fit_equals_phase5_and_signfree_can_be_negative():
    rng = np.random.default_rng(0)
    X = rng.random((800, 3))
    y = rng.random(800) < 0.3 + 0.3 * X[:, 0] - 0.25 * X[:, 1]
    w = rng.random(800)
    assert np.allclose(s8.fit_logreg(X, y, w, nonneg=True), s5.fit_nonneg_logreg(X, y, w))
    sf = s8.fit_logreg(X, y, w, nonneg=False)
    assert sf[2] < 0 < sf[1]                                                          # sign-free recovers the negative weight
    assert (s8.fit_logreg(X, y, w, nonneg=True)[1:] >= 0).all()


def test_gated_scores_order_and_fill():
    rng = np.random.default_rng(1)
    n = 1000
    gate = rng.random(n) < 0.04                                                       # gated pool (4%) smaller than a 10% budget
    sg, m0 = rng.normal(size=n), rng.normal(size=n)
    sc = s8.gated_scores(sg, m0, gate)
    assert sc[gate].min() > sc[~gate].max()                                           # every gated case outranks every non-gated case
    g = np.flatnonzero(gate)
    assert np.array_equal(g[np.argsort(-sc[gate])], g[np.argsort(-sg[gate])])         # gated cases ordered by the gated score
    ng = np.flatnonzero(~gate)
    assert np.array_equal(ng[np.argsort(-sc[~gate])], ng[np.argsort(-m0[~gate])])     # non-gated ordered by plain M0 (fill rule)
    f = s7.flag_fractions(sc, np.ones(n), 0.10)
    assert int(f.sum()) == int(np.ceil(0.10 * n)) and f[gate].all()                   # budget filled: all gated + best non-gated
    top_non = np.flatnonzero(f.astype(bool) & ~gate)
    assert len(top_non) == int(f.sum()) - gate.sum()
    assert m0[top_non].min() >= np.sort(m0[~gate])[::-1][len(top_non)] - 1e-12        # fill = highest-M0 non-gated cases
    f2 = s7.flag_fractions(sc, np.ones(n), 0.02)                                       # budget smaller than the gated pool
    assert f2[gate].sum() == int(f2.sum()) and (sg[f2.astype(bool)].min() >= np.sort(sg[gate])[::-1][int(f2.sum()) - 1] - 1e-12)


def _block(rng, pool, n):
    y = rng.random(n) < 0.15
    d = pd.DataFrame({"pool": pool, "seed": 0, "uid": [f"{pool}/{i}" for i in range(n)], "family": pool, "condition": "c", "severity": 0})
    d["sf_090"] = y
    d["correct"] = ~(y | (rng.random(n) < 0.1))
    d["confidence"] = np.where(y, 0.9 + 0.1 * rng.random(n), rng.random(n))
    d["cluster"] = [f"g{i % 50}" for i in range(n)]
    for c in ("u_log1msp", "u_logent", "r_maha_pred", "e_js_norm_l4"):
        d["x_" + c] = np.clip(rng.random(n) * 0.5 + 0.3 * y * (1 if c != "u_log1msp" else -1) + 0.2, 0, 1)
    return d


def test_no_heldout_label_or_feature_in_weights_gates_or_scores_of_tuning(monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.concat([_block(rng, "A", 700), _block(rng, "B", 700), _block(rng, "C", 400)], ignore_index=True)
    monkeypatch.setattr(s5, "rotation", lambda d, H, c_kind="shortcutv2": (d[d.pool != H].copy(), d[d.pool == H].copy()))
    names = ["M0", "sf_M1", "G_M0", "G_sfM1"]
    c1 = s8.make_ctx(df, "C", "sf_090", [0])
    sc1, w1 = s8.score_methods(c1, names, [0])
    df2 = df.copy()
    m = df2.pool == "C"
    df2.loc[m, "sf_090"] = ~df2.loc[m, "sf_090"]                                      # flip held-out labels, distort features and confidence
    df2.loc[m, "x_r_maha_pred"] = 1 - df2.loc[m, "x_r_maha_pred"]
    df2.loc[m, "confidence"] = 1 - df2.loc[m, "confidence"]
    c2 = s8.make_ctx(df2, "C", "sf_090", [0])
    sc2, w2 = s8.score_methods(c2, names, [0])
    pd.testing.assert_frame_equal(pd.DataFrame(w1), pd.DataFrame(w2))                  # weights and gated-pool fits never see held-out data
    assert (pd.DataFrame(w1).query("model == 'M0'")[["w_u_log1msp", "w_u_logent"]] >= 0).all().all()
    assert not np.allclose(sc1["sf_M1"][0], sc2["sf_M1"][0])                           # scores of the held-out rows do change (sanity)


def test_evaluate_exact_budget_oracle_ceiling_and_paired_zero_difference():
    rng = np.random.default_rng(2)
    n, seeds = 500, [0, 1]
    y = rng.random(n) < 0.12
    err = y | (rng.random(n) < 0.1)
    te = {s: pd.DataFrame({"seed": s, "cluster": [f"g{i % 30}" for i in range(n)], "correct": ~err, "sf_090": y,
                           "condition": "c", "severity": 0}) for s in seeds}
    sa, sb = rng.normal(size=n) + y, rng.normal(size=n)
    scores = {"A": {s: sa for s in seeds}, "A2": {s: sa for s in seeds}, "B": {s: sb for s in seeds}}
    res = s8.evaluate(te, scores, "sf_090", seeds, 120, rng_seed=1)
    assert (res["bl"]["oracle"][:, :, 0] <= res["bl"]["A"][:, :, 0] + 1e-12).all()
    assert (res["bl"]["oracle"][:, :, 0] <= res["bl"]["B"][:, :, 0] + 1e-12).all()
    assert np.allclose(res["bl"]["A"] - res["bl"]["A2"], 0) and np.allclose(res["ba"]["A"] - res["ba"]["A2"], 0)
    k = np.ceil(np.array(s8.BUDGETS) * n)
    top = np.argsort(-sa)
    manual = [(y.sum() - y[top[: int(kk)]].sum()) / (n - kk) for kk in k]
    assert np.allclose(res["point"]["A"][0][:, 0], manual)                              # exact budget on the point estimate
    assert np.allclose(res["point"]["random"][0][:, 0], y.mean())                       # random baseline unbiased
    rows = pd.DataFrame(s8.make_rows(res, {"heldout": "X"}, [("G2", "A", "A2", True), ("G2", "A", "B", False)]))
    r = rows[(rows.quantity == "A-A2") & (rows.metric == "dresid_sf")]
    assert (r.verdict == "no practical difference").all() and r.primary.all()
    assert rows[(rows.quantity == "A-B") & (rows.metric == "dauroc")].estimate.iloc[0] > 0
    assert rows.groupby(["quantity", "metric"]).jackknife_min.count().min() >= 0


def test_maha_helper_matches_phase4_implementation_and_logit_reconstruction():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(300, 16)).astype(np.float32)
    lab = rng.integers(0, 4, 300)
    g = rp.fit_class_gaussian(f, lab, 4)
    pred = rng.integers(0, 4, 300)
    ref, _ = rp.mahalanobis_scores(f, pred, {"means": g["means"], "precision": g["precision"]})
    assert np.allclose(s8._maha_pred(f, pred, g["means"], g["precision"]), ref, rtol=1e-3, atol=1e-3)
    W, b = rng.normal(size=(4, 16)).astype(np.float32), rng.normal(size=4).astype(np.float32)
    lo = s8.reconstruct_logits(f, W, b)
    z = lo / 1.3
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert np.allclose(s8._softmax_conf(lo, 1.3), p.max(1), atol=1e-6)
    p2 = s8.verdict_auc(0.001, -0.01, 0.01), s8.verdict_auc(0.05, 0.02, 0.08), s8.verdict_auc(-0.05, -0.08, -0.02)
    assert p2 == ("no practical difference", "improves", "worse")
