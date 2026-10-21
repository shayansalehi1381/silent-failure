"""Stage 2 tests: temperature scaling, calibration metrics, augmentation, resume, saved array shapes."""
from __future__ import annotations

import copy
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from sfd import calibration as cal
from sfd.config import ROOT, load_config


def _overconfident(n=20000, k=5, scale=3.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, k))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    return z * scale, y, z


def test_temperature_recovers_scale_and_nll_improves_monotonically():
    logits, y, _ = _overconfident()
    T = cal.fit_temperature(logits, y)
    assert 2.6 < T < 3.4
    assert cal.nll(logits, y, T=T) < cal.nll(logits, y, T=1.0)
    grid = np.linspace(1.0, T, 8)
    vals = [cal.nll(logits, y, T=t) for t in grid]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))        # NLL decreases towards the optimum
    grid2 = np.linspace(T, 2 * T, 8)
    vals2 = [cal.nll(logits, y, T=t) for t in grid2]
    assert all(a <= b + 1e-9 for a, b in zip(vals2, vals2[1:]))      # and increases beyond it (convex in 1/T)


def test_temperature_with_uniform_weights_equals_unweighted_and_prior_weights_mean_one():
    logits, y, _ = _overconfident(4000)
    assert cal.fit_temperature(logits, y, np.ones(len(y))) == pytest.approx(cal.fit_temperature(logits, y), rel=1e-4)
    rng = np.random.default_rng(1)
    fit, target = rng.integers(0, 2, 5000), (rng.random(3000) < 0.3).astype(int)
    w = cal.prior_weights(fit, target, 2)
    assert w.mean() == pytest.approx(1.0, abs=1e-9)
    assert w[fit == 1].mean() < w[fit == 0].mean()                   # target has fewer positives


def test_calibration_metrics_on_perfectly_calibrated_and_perfect_predictions():
    logits, y, z = _overconfident(30000, scale=1.0)
    p = cal.probs(z)
    assert cal.ece(p, y) < 0.02
    onehot = np.eye(4)[np.arange(100) % 4] * 50.0
    lab = np.arange(100) % 4
    assert cal.brier(cal.probs(onehot), lab) < 1e-9 and cal.ece(cal.probs(onehot), lab) < 1e-6
    c, a, n = cal.reliability_curve(p, y)
    assert n.sum() == len(y)


def test_augment_is_per_sample_permutation_without_colour_changes():
    from sfd.train import augment
    torch.manual_seed(0)
    x = torch.rand(64, 3, 16, 16, device="cuda")
    out = augment(x.clone())
    assert out.shape == x.shape
    for i in range(64):                                              # same pixel multiset => only flips/rotations
        assert torch.equal(out[i].flatten().sort().values, x[i].flatten().sort().values)
    assert sum(not torch.equal(out[i], x[i]) for i in range(64)) > 32   # most samples actually transformed


def _tiny_cfg(tmp):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    cfg["train"].update(epochs=2, batch_size=32)
    return cfg


def _tiny_data(n, classes=3, size=32, seed=0):
    from sfd.data.classify_data import SplitData
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    imgs = (rng.random((n, size, size, 3)) * 60 + y[:, None, None, None] * 60).astype(np.uint8)
    return SplitData("t", torch.from_numpy(imgs).cuda(), y.astype(np.int64), np.array([f"u{i}" for i in range(n)]),
                     np.full(n, -1))


def test_train_one_resumes_from_last_checkpoint(tmp_path):
    from sfd.train import GpuGuard, train_one
    cfg, dev = _tiny_cfg(tmp_path), torch.device("cuda")
    tr, va = _tiny_data(256), _tiny_data(64, seed=1)
    guard = GpuGuard(log=lambda *_: None)
    msgs = []
    kw = dict(tr=tr, va=va, n_classes=3, seed=0, ckpt_dir=tmp_path / "ck", log_path=tmp_path / "log.jsonl",
              guard=guard, dev=dev, log=msgs.append, pretrained=False)
    cfg["train"]["epochs"] = 1
    r1 = train_one(cfg, **kw)
    assert r1["epochs_run"] == 1 and (tmp_path / "ck" / "last.pt").exists() and (tmp_path / "ck" / "best.pt").exists()
    cfg["train"]["epochs"] = 2                                       # a later run continues instead of restarting
    r2 = train_one(cfg, **kw)
    assert any("resumed from epoch 1" in m for m in msgs)
    assert r2["epochs_run"] == 2 and len(r2["history"]) == 2
    n_lines = len((tmp_path / "log.jsonl").read_text().splitlines())
    assert n_lines == 2
    r3 = train_one(cfg, **kw)                                        # finished model: nothing more to train
    assert r3["epochs_run"] == 2 and len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_row_reader_matches_np_load(tmp_path):
    from numpy.lib.format import open_memmap
    from sfd.data.classify_data import RowReader
    a = np.random.default_rng(0).integers(0, 255, (40, 8, 8, 3), dtype=np.uint8)
    mm = open_memmap(tmp_path / "r.npy", mode="w+", dtype=np.uint8, shape=a.shape)
    mm[:] = a
    mm.flush()
    r = RowReader(tmp_path / "r.npy")
    idx = np.array([39, 0, 7, 7, 20])
    assert len(r) == 40 and r.nbytes == a.nbytes and np.array_equal(r[idx], a[idx])


def test_train_one_memmap_path_matches_shapes(tmp_path):
    """When the train set does not fit in VRAM the batches are gathered from an SSD memmap."""
    from numpy.lib.format import open_memmap
    from sfd.train import GpuGuard, train_one
    cfg = _tiny_cfg(tmp_path)
    cfg["train"]["epochs"] = 1
    tr = _tiny_data(128)
    mm = open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.uint8, shape=tuple(tr.images.shape))
    mm[:] = tr.images.cpu().numpy()
    mm.flush()
    tr.images = np.load(tmp_path / "x.npy", mmap_mode="r")
    va = _tiny_data(64, seed=1)
    r = train_one(cfg, tr, va, 3, 0, tmp_path / "ck", tmp_path / "l.jsonl", GpuGuard(log=lambda *_: None),
                  torch.device("cuda"), log=lambda *_: None, pretrained=False)
    assert r["epochs_run"] == 1 and np.isfinite(r["best_val_loss"])


@pytest.mark.skipif(not (ROOT / "data" / "pathmnist_128.npz").exists()
                    or not (ROOT / "outputs" / "smoke" / "stage1" / "splits.parquet").exists(),
                    reason="needs PathMNIST file and the smoke stage-1 outputs")
def test_stage2_saved_arrays_marker_resume_and_frozen_splits_untouched(tmp_path):
    from sfd import stage2
    cfg = _tiny_cfg(tmp_path)
    cfg["stage2"] = copy.deepcopy(cfg["stage2"])
    cfg["stage2"].update(datasets=["pathmnist"], cache_dir=str(tmp_path / "cache"))
    cfg["train"]["epochs"] = 1
    (tmp_path / "stage1").mkdir()
    shutil.copy(ROOT / "outputs" / "smoke" / "stage1" / "splits.parquet", tmp_path / "stage1" / "splits.parquet")
    msgs = []
    stage2.run(cfg, tmp_path, log=msgs.append)
    sp = pd.read_parquet(tmp_path / "stage1" / "splits.parquet")
    assert (tmp_path / "stage2" / "markers" / "pathmnist_seed0.done").exists()
    for s in stage2.eval_splits("pathmnist"):
        z = np.load(tmp_path / "stage2" / "preds" / f"pathmnist_seed0_{s}.npz")
        frame = sp[(sp.dataset == "pathmnist") & (sp.split == s)]
        n = len(frame)
        assert z["logits"].shape == (n, 9) and z["logits"].dtype == np.float32
        assert z["features"].shape == (n, 512) and z["features"].dtype == np.float16
        assert z["probs_cal"].shape == (n, 9) and np.allclose(z["probs_cal"].sum(1), 1, atol=1e-4)
        assert np.array_equal(z["uid"], frame["uid"].to_numpy(dtype=str))          # same uid order as splits.parquet
        assert np.array_equal(z["labels"], frame["label"].to_numpy()) and len(z["patient"]) == n
    for name in ("metrics_per_seed_pathmnist.csv", "calibration_pathmnist.csv", "sanity_pathmnist.json"):
        assert (tmp_path / "stage2" / name).exists()
    msgs.clear()
    stage2.run(cfg, tmp_path, log=msgs.append)                                     # resume: nothing is retrained
    assert any("done earlier -> skipped" in m for m in msgs) and not any("training" in m for m in msgs)
