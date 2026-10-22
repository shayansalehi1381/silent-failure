"""Stage 10 tests: row alignment, ensemble signals, leave-seed-out construction, leakage of weights, MI >= 0."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfd import stage5 as s5
from sfd import stage8 as s8
from sfd import stage10 as s10
from tests.test_stage8 import _block


def _probs(rng, n=400, C=4, k=5):
    return [s10.softmax64(rng.normal(size=(n, C)) * 3, 1.0 + 0.1 * i) for i in range(k)]


def test_alignment_check_detects_misaligned_rows():
    a = {k: np.arange(5) for k in ("fam", "cond", "sev", "uid", "label")}
    b = {k: v.copy() for k, v in a.items()}
    s10.check_alignment([a, b])
    b["uid"] = b["uid"][::-1].copy()
    with pytest.raises(AssertionError):
        s10.check_alignment([a, b])


def test_one_member_ensemble_equals_single_model_and_mi_nonnegative():
    rng = np.random.default_rng(0)
    P = _probs(rng)
    one = s10.ens_signals([P[0]])
    msp = P[0].max(1)
    assert np.allclose(one["msp"], np.log(np.clip(1 - msp, 1e-300, 1)), atol=1e-9)
    assert np.all(np.isfinite(one["msp"])) and np.allclose(one["mi"], np.log(1e-12))          # no disagreement -> MI clipped to 0
    many = s10.ens_signals(P)
    assert np.exp(many["mi"]).min() >= 1e-12 - 1e-15
    assert (many["mi"] > one["mi"].mean()).mean() > 0.99                                       # disagreement is visible


def test_ensemble_signal_is_stable_for_saturated_probabilities():
    z = np.array([[60.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    p = s10.softmax64(z, 0.1)
    s = s10.ens_signals([p, p])
    assert np.all(np.isfinite(s["msp"])) and s["msp"][0] < -100                                # 1 - MSP ~ 1e-260 stays resolvable


def test_leave_seed_out_membership_and_registry():
    R = s10.registry()
    assert R["E_fuse"]["feats"] == ["ens4_msp", "ens4_ent"] and R["E5_fuse"]["feats"] == ["ens5_msp", "ens5_ent"]
    assert R["E_mi"] == {"kind": "single", "feat": "ens4_mi"}
    assert "M0" in R and "G_M1" in R                                                           # earlier methods untouched


def test_no_heldout_data_in_ensemble_fusion_weights(monkeypatch):
    rng = np.random.default_rng(1)
    df = pd.concat([_block(rng, "A", 600), _block(rng, "B", 600), _block(rng, "C", 400)], ignore_index=True)
    for c in ("ens4_msp", "ens4_ent", "ens4_mi"):
        df["x_" + c] = np.clip(rng.random(len(df)) * 0.5 + 0.3 * df.sf_090, 0, 1)
    monkeypatch.setattr(s5, "rotation", lambda d, H, c_kind="shortcutv2": (d[d.pool != H].copy(), d[d.pool == H].copy()))
    reg = s10.registry()
    c1 = s8.make_ctx(df, "C", "sf_090", [0])
    sc1, w1 = s8.score_methods(c1, ["M0", "E_fuse", "E_M1"], [0], reg=reg)
    df2 = df.copy()
    m = df2.pool == "C"
    df2.loc[m, "sf_090"] = ~df2.loc[m, "sf_090"]
    df2.loc[m, "x_ens4_msp"] = 1 - df2.loc[m, "x_ens4_msp"]
    c2 = s8.make_ctx(df2, "C", "sf_090", [0])
    sc2, w2 = s8.score_methods(c2, ["M0", "E_fuse", "E_M1"], [0], reg=reg)
    pd.testing.assert_frame_equal(pd.DataFrame(w1), pd.DataFrame(w2))
    assert (pd.DataFrame(w1)[["w_ens4_msp", "w_ens4_ent"]].dropna() >= 0).all().all()
