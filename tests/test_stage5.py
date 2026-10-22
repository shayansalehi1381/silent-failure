"""Stage 5 tests: weighted metrics vs sklearn, non-negative fusion, paired bootstrap, pooled-label logic, leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfd import stage5 as s5
from sfd.signals import uncertainty as un


def test_ranked_metrics_match_sklearn_and_handle_ties():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.2
    s = np.round(rng.normal(size=2000) + y, 1)                          # rounding creates many ties
    auroc, ap, aurc = s5.Ranked(s, y).metrics(np.ones(2000))
    assert auroc == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    assert ap == pytest.approx(average_precision_score(y, s), abs=1e-9)
    # AURC: perfect ranking < random ranking
    good = s5.Ranked(y.astype(float), y).metrics(np.ones(2000))[2]
    rnd = s5.Ranked(rng.random(2000), y).metrics(np.ones(2000))[2]
    assert good < rnd


def test_integer_weights_equal_row_duplication():
    rng = np.random.default_rng(1)
    y, s = rng.random(300) < 0.3, rng.normal(size=300)
    w = rng.integers(0, 4, 300)
    a = s5.Ranked(s, y).metrics(w)
    b = s5.Ranked(np.repeat(s, w), np.repeat(y, w)).metrics(np.ones(w.sum()))
    assert np.allclose(a, b)


def test_nonneg_fusion_is_monotone_and_learns_signal():
    rng = np.random.default_rng(2)
    X = rng.random((4000, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=4000)) > 0.6                  # col 0 informative, col 2 anti-informative
    X[:, 2] = 1 - X[:, 0] + 0.05 * rng.normal(size=4000)
    th = s5.fit_nonneg_logreg(X, y, np.ones(4000))
    assert (th[1:] >= 0).all() and th[1] > 1 and th[3] == pytest.approx(0, abs=1e-6)


def test_paired_bootstrap_is_paired_and_deterministic():
    rng = np.random.default_rng(3)
    test = pd.DataFrame({"seed": np.repeat([0, 1], 500), "cluster": np.tile(np.arange(100).repeat(5), 2)})
    y = {s: rng.random(500) < 0.3 for s in (0, 1)}
    base = {s: rng.normal(size=500) + y[s] for s in (0, 1)}
    scores = {"M0": base, "M1": {s: base[s].copy() for s in (0, 1)}}         # identical detectors
    setup = s5.boot_setup(test, [0, 1])
    r1 = s5.paired_bootstrap(scores, y, setup, [0, 1], 50, rng_seed=0)
    r2 = s5.paired_bootstrap(scores, y, setup, [0, 1], 50, rng_seed=0)
    assert np.allclose(r1["boot"]["M0"], r2["boot"]["M0"], equal_nan=True)  # deterministic
    assert np.allclose(r1["boot"]["M1"] - r1["boot"]["M0"], 0)              # paired: identical detectors -> delta 0


def test_bh_monotone_and_bounded():
    q = s5.bh(np.array([0.001, 0.04, 0.03, 0.5, np.nan]))
    assert np.isnan(q[-1]) and (q[:-1] <= 1).all() and q[0] <= q[1]


def _toy_df():
    rows = []
    for fam, ds, kind, cond in (("A", "pathmnist", "standard", "clean"), ("ID", "camelyon17", "standard", "id_test"),
                                ("B", "camelyon17", "standard", "ood_c1"), ("C", "pathmnist", "shortcutv2", "reversed"),
                                ("C", "pathmnist", "shortcut", "reversed"), ("X", "pathmnist", "standard", "nct7k_test"),
                                ("ID", "pathmnist", "standard", "id_test"), ("VALCAL", "pathmnist", "standard", "val_cal")):
        for i in range(20):
            rows.append({"family": fam, "dataset": ds, "model_kind": kind, "condition": cond, "seed": 0, "uid": f"{fam}{cond}{i}"})
    return pd.DataFrame(rows)


def test_pools_are_the_prespecified_ones():
    df = _toy_df()
    assert set(s5.family_pool(df, "A").condition) == {"clean"}
    assert set(s5.family_pool(df, "B").condition) == {"id_test", "ood_c1"} and set(s5.family_pool(df, "B").dataset) == {"camelyon17"}
    assert set(s5.family_pool(df, "C").model_kind) == {"shortcutv2"}
    assert set(s5.family_pool(df, "C", "shortcut").model_kind) == {"shortcut"}
    x = s5.family_pool(df, "X")
    assert set(x.condition) == {"id_test", "nct7k_test"} and set(x.dataset) == {"pathmnist"}
    assert "VALCAL" not in set(pd.concat([s5.family_pool(df, f) for f in "ABCX"]).family)


@pytest.mark.parametrize("H", ["A", "B", "C"])
def test_rotation_never_tunes_on_the_heldout_family(H):
    df = _toy_df()
    tune, test = s5.rotation(df, H)
    assert H not in set(tune["pool"]) and set(test["pool"]) == {H}
    assert not set(tune["uid"]) & set(test["uid"])
    assert set(tune["pool"]) == {"A", "B", "C"} - {H}


def test_log_scale_uncertainty_is_monotone_and_tie_free_when_saturated():
    lo = np.array([[300.0, 0, 0], [250.0, 0, 0], [200.0, 0, 0], [3.0, 0, 0]])
    a = un.log_one_minus_msp(lo, 0.12)
    p = np.exp(lo / 0.12 - (lo / 0.12).max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert (1 - p.max(1))[:3].tolist() == [0.0, 0.0, 0.0]               # float probabilities tie at exactly 1
    assert len(set(a.tolist())) == 4 and np.all(np.diff(a) > 0)         # log scale keeps the order
    e = un.log_entropy(lo, 0.12)
    assert np.all(np.isfinite(e)) and np.all(np.diff(e) > 0)
