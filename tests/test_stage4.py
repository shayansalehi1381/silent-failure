"""Stage 4 tests: signal shapes, normalisation, determinism, no-leakage of reference inputs, tau, CAM utilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sfd.signals import explanation as ex
from sfd.signals import normalize as nz
from sfd.signals import representation as rp
from sfd.signals import uncertainty as un


def test_uncertainty_shapes_and_direction():
    p = np.array([[0.98, 0.01, 0.01], [0.34, 0.33, 0.33]])
    assert (un.one_minus_msp(p)[1] > un.one_minus_msp(p)[0]) and (un.entropy(p)[1] > un.entropy(p)[0])
    lo = np.array([[8.0, 0, 0], [0.1, 0, 0]])
    assert un.energy(lo, 1.0)[1] > un.energy(lo, 1.0)[0]


def test_mahalanobis_and_knn_detect_outliers():
    rng = np.random.default_rng(0)
    f = np.concatenate([rng.normal(0, 1, (300, 8)), rng.normal(4, 1, (300, 8))]).astype(np.float32)
    lab = np.repeat([0, 1], 300)
    ref = rp.fit_class_gaussian(f, lab, 2)
    q = np.array([[0.0] * 8, [20.0] * 8], np.float32)
    dp, dm = rp.mahalanobis_scores(q, np.array([0, 0]), ref)
    assert dp.shape == (2,) and dp[1] > dp[0] and (dm <= dp + 1e-9).all()
    k = rp.knn_distance(q, f, k=10, device="cpu")
    assert k.shape == (2,) and np.isfinite(k).all()


def test_normalisation_uses_reference_only():
    ref = np.arange(1, 101, dtype=float)
    assert nz.ecdf_percentile(ref, np.array([0.0]))[0] == 0.0 and nz.ecdf_percentile(ref, np.array([1000.0]))[0] == 1.0
    assert abs(nz.ecdf_percentile(ref, np.array([50.0]))[0] - 0.495) < 1e-9
    z = nz.robust_z(ref, np.array([np.median(ref)]))
    assert abs(z[0]) < 1e-12


def test_js_divergence_properties():
    p = np.full((3, 64), 1 / 64)
    q = np.zeros((3, 64)); q[:, 0] = 1
    assert np.allclose(ex.js_divergence(p, p), 0, atol=1e-9)
    d = ex.js_divergence(p, q)
    assert (d > 0).all() and (d <= 1 + 1e-9).all() and np.allclose(d, ex.js_divergence(q, p))


@pytest.mark.parametrize("t", ["hflip", "vflip", "rot90", "rot180", "rot270"])
def test_invert_cam_is_exact_for_flips_and_rot90(t):
    cam = np.random.default_rng(1).random((4, ex.GRID * ex.GRID)).astype(np.float32)
    cam /= cam.sum(1, keepdims=True)
    fwd = ex.apply_transform(torch.from_numpy(cam).view(-1, 1, ex.GRID, ex.GRID), t).flatten(1).numpy()
    assert np.allclose(ex.invert_cam(fwd, t), cam, atol=1e-6)


def test_gradcam_shapes_unit_mass_and_determinism():
    from sfd.train import Net
    torch.manual_seed(0)
    m = Net(9, pretrained=False).eval()
    x = torch.randn(4, 3, 128, 128)
    a, b = ex.gradcam_batch(m, x), ex.gradcam_batch(m, x)
    assert a["logits"].shape == (4, 9) and a["features"].shape == (4, 512) and a["features"].dtype == np.float16
    for ln in ex.LAYERS:
        assert a[ln].shape == (4, ex.GRID * ex.GRID) and np.allclose(a[ln].sum(1), 1, atol=1e-5) and (a[ln] >= 0).all()
        assert np.array_equal(a[ln], b[ln])                               # deterministic


def test_frozen_tau_regular_and_degenerate():
    from sfd.stage4 import frozen_tau
    conf = np.linspace(0.3, 1.0, 1000)
    correct = conf > 0.6                                                  # low-confidence region mostly wrong
    t = frozen_tau(conf, correct, 0.95)
    assert t["degenerate"] is None and t["val_cal_precision_at_tau"] >= 0.95 and 0.3 < t["tau"] < 1.0
    d = frozen_tau(conf, np.ones(1000, bool), 0.95)
    assert d["degenerate"] and d["tau"] == pytest.approx(conf.min()) and d["accept_frac"] == 1.0


def test_cue_mass_box_weights():
    from sfd.stage4 import cue_mass
    cam = np.zeros((2, ex.GRID * ex.GRID)); cam[0, 0] = 1; cam[1, -1] = 1
    m = cue_mass(cam, offset=0, cue_size=16, img_px=128)                  # box == first cell exactly
    assert m[0] == pytest.approx(1.0) and m[1] == pytest.approx(0.0)


def test_reference_inputs_never_include_shifted_or_eval_files():
    from sfd import stage4 as s4
    from sfd.config import ROOT, load_config, resolve
    cfg = load_config(ROOT / "configs" / "full.yaml")
    p = s4.P(cfg, resolve(cfg, "out_dir"))
    if not (p["stage2"] / "markers" / "pathmnist_seed0.done").exists():
        pytest.skip("stage-2 models not present")
    for spec in s4.model_specs(cfg, p):
        files = [str(f).replace("\\", "/") for f in s4.reference_inputs(cfg, p, spec)]
        assert files[0].endswith(f"{spec['dataset']}_train.npy")
        for f in files[:2]:
            assert not any(b.replace("\\", "/") in f for b in s4.FORBIDDEN), f
    bad = dict(s4.model_specs(cfg, p)[0], ckpt=p["stage3"] / "preds" / "x.npz")
    with pytest.raises(AssertionError):
        s4.reference_inputs(cfg, p, bad)
