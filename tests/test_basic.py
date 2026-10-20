"""Fast Phase 0 tests: configs, seeding, PathMNIST loader shapes, smoke pipeline."""
from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from sfd.config import ROOT, load_config, resolve, set_seed

SMOKE = ROOT / "configs" / "smoke.yaml"
FULL = ROOT / "configs" / "full.yaml"


@pytest.mark.parametrize("path", [SMOKE, FULL])
def test_config_loads(path):
    cfg = load_config(path)
    for key in ("seed", "paths", "data", "model", "train", "stages"):
        assert key in cfg
    assert cfg["data"]["pathmnist_size"] == 128 and cfg["data"]["camelyon_size"] == 96
    assert cfg["model"]["pretrained"] is False
    assert resolve(cfg, "data_dir").is_absolute()


def test_full_config_decisions():
    t = load_config(FULL)["train"]
    assert t["seeds"] == [0, 1, 2, 3, 4]
    assert t["epochs"] == 12
    assert t["early_stopping"] == {"metric": "val_loss", "patience": 3}
    assert (t["batch_size"], t["grad_accum"], t["amp"], t["channels_last"]) == (128, 1, True, True)


def test_set_seed_reproducible():
    def draw():
        return random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()

    set_seed(123)
    a = draw()
    set_seed(123)
    assert draw() == a
    set_seed(124)
    assert draw() != a


def test_load_pathmnist_shapes():
    data_file = ROOT / "data" / "pathmnist_128.npz"
    if not data_file.exists():
        pytest.skip("PathMNIST 128px file not downloaded")
    from sfd.data.pathmnist import load_pathmnist

    # val/test only: keeps RAM small (the 90k train split alone is ~4.4 GB)
    data = load_pathmnist(ROOT / "data", 128, splits=("val", "test"), download=False)
    for split, expected_n in (("val", 10004), ("test", 7180)):
        x, y = data[split]
        assert x.shape == (expected_n, 128, 128, 3) and x.dtype == np.uint8
        assert y.shape == (expected_n,) and y.dtype == np.int64
        assert set(np.unique(y)) == set(range(9))


def _run_all(*extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"),
                           "--config", str(SMOKE), *extra],
                          capture_output=True, text=True, cwd=ROOT)


def test_run_all_smoke_and_resume():
    r = _run_all()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "run_all finished" in r.stdout
    r = _run_all("--resume")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped (--resume)" in r.stdout
