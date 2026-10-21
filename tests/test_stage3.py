"""Stage 3 tests: shortcut cue, corruption determinism and resize, row I/O, sample, saved array shapes."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sfd import shortcut as sc


def test_palette_distinct_and_derangement():
    pal = sc.make_palette(9, 16)
    assert pal.shape == (9, 16, 16, 3) and pal.dtype == np.uint8
    flat = pal.reshape(9, -1)
    assert len({row.tobytes() for row in flat}) == 9                        # 9 distinct cues
    for seed in range(20):
        p = sc.derangement(9, seed)
        assert sorted(p.tolist()) == list(range(9)) and not (p == np.arange(9)).any()
    assert (sc.derangement(9, 3) == sc.derangement(9, 3)).all()


def test_cue_assignment_rate_and_wrong_cues_never_match():
    lab = np.random.default_rng(0).integers(0, 9, 20000)
    cue = sc.cue_assignment(lab, 0.95, 9, seed=1)
    assert abs((cue == lab).mean() - 0.95) < 0.01
    assert ((cue >= 0) & (cue < 9)).all()
    assert (cue == sc.cue_assignment(lab, 0.95, 9, seed=1)).all()          # deterministic


def test_states_aligned_removed_reversed():
    lab = np.arange(9)
    perm = sc.derangement(9, 0)
    assert (sc.state_cues(lab, "aligned", perm) == lab).all()
    assert (sc.state_cues(lab, "removed", perm) == -1).all()
    rev = sc.state_cues(lab, "reversed", perm)
    assert (rev != lab).all()


def test_uint8_cue_position_and_removed_is_untouched():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (5, 128, 128, 3), dtype=np.uint8)
    pal = sc.make_palette(9, 16)
    out = sc.apply_cue_uint8(imgs, np.array([0, 3, -1, 8, 2]), pal, offset=4)
    assert np.array_equal(out[2], imgs[2])                                   # removed = no cue
    assert np.array_equal(out[1, 4:20, 4:20], pal[3])
    mask = np.ones((128, 128), bool)
    mask[4:20, 4:20] = False
    assert np.array_equal(out[1][mask], imgs[1][mask])                       # nothing else changed


def test_tensor_cue_after_augmentation_stays_at_fixed_position():
    from sfd.train import augment
    pal = sc.make_palette(9, 16)
    pal_t = torch.from_numpy(pal).permute(0, 3, 1, 2).float().div(255).cuda()
    x = torch.rand(32, 3, 64, 64, device="cuda")
    ids = torch.arange(32, device="cuda") % 9
    x = sc.apply_cue_tensor(augment(x), ids, pal_t, 4)                       # cue pasted AFTER flips/rot90
    for i in range(32):
        assert torch.allclose(x[i, :, 4:20, 4:20], pal_t[ids[i]])


def test_corruption_is_deterministic_and_resize_shapes():
    pytest.importorskip("medmnistc")
    from sfd.corrupt import corrupt_one, corruptors, image_seed, resize128
    img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    cor = corruptors()
    for name in ("bubble", "stain_deposit", "jpeg_compression", "defocus_blur"):
        s = image_seed(0, name, 3, "pathmnist/train/1")
        a, b = corrupt_one(cor[name], img, 3, s), corrupt_one(cor[name], img, 3, s)
        assert np.array_equal(a, b) and a.shape == (224, 224, 3)
    for m in ("pil_bicubic", "pil_bilinear", "pil_lanczos", "pil_box", "cv2_area"):
        r = resize128(img, m)
        assert r.shape == (128, 128, 3) and r.dtype == np.uint8
    s = image_seed(0, "motion_blur", 5, "u")                                # needs ImageMagick + Wand
    assert np.array_equal(corrupt_one(cor["motion_blur"], img, 5, s), corrupt_one(cor["motion_blur"], img, 5, s))


def test_rowio_roundtrip(tmp_path):
    from numpy.lib.format import open_memmap
    from sfd.corrupt import RowIO
    mm = open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.uint8, shape=(10, 4, 4, 3))
    del mm
    io_ = RowIO(tmp_path / "x.npy")
    a = np.random.default_rng(0).integers(0, 255, (3, 4, 4, 3), dtype=np.uint8)
    io_.write(5, a)
    assert np.array_equal(io_.read(5, 8), a) and np.array_equal(np.load(tmp_path / "x.npy")[5:8], a)


def test_save_preds_shapes_and_fields(tmp_path):
    from sfd.stage3 import save_preds
    lo = np.random.default_rng(0).normal(size=(50, 9)).astype(np.float32)
    ft = np.random.default_rng(1).normal(size=(50, 512))
    lab = np.arange(50) % 9
    save_preds(tmp_path / "p.npz", lo, ft, lab, [f"u{i}" for i in range(50)], np.full(50, -1), 1.2)
    z = np.load(tmp_path / "p.npz")
    assert z["logits"].shape == (50, 9) and z["features"].dtype == np.float16 and z["features"].shape == (50, 512)
    assert np.allclose(z["probs_cal"].sum(1), 1, atol=1e-5)
    assert (z["pred"] == z["probs_cal"].argmax(1)).all() and (z["correct"] == (z["pred"] == lab)).all()
    assert np.allclose(z["confidence"], z["probs_cal"].max(1))


def test_train_one_with_post_augment_hook(tmp_path):
    import copy
    from sfd.config import ROOT, load_config
    from sfd.data.classify_data import SplitData
    from sfd.train import GpuGuard, train_one
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    cfg["train"].update(epochs=1, batch_size=32)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 128)
    tr = SplitData("t", torch.from_numpy(rng.integers(0, 255, (128, 32, 32, 3), dtype=np.uint8)).cuda(), y,
                   np.array([str(i) for i in range(128)]), np.full(128, -1))
    va = SplitData("v", rng.integers(0, 255, (64, 32, 32, 3), dtype=np.uint8), y[:64], np.array([str(i) for i in range(64)]),
                   np.full(64, -1))
    calls = []

    def hook(x, idx):
        calls.append(int(len(idx)))
        return x
    r = train_one(cfg, tr, va, 9, 0, tmp_path / "ck", tmp_path / "l.jsonl", GpuGuard(log=lambda *_: None),
                  torch.device("cuda"), log=lambda *_: None, pretrained=False, post_augment=hook)
    assert r["epochs_run"] == 1 and calls == [32, 32, 32, 32]


def test_run_tables_end_to_end_on_fake_predictions(tmp_path):
    import copy
    import json
    import pandas as pd
    from numpy.lib.format import open_memmap
    from sfd import stage3 as s3
    from sfd.config import ROOT, load_config
    if not (ROOT / "outputs" / "stage2" / "preds" / "camelyon17_seed4_ood_c2.npz").exists():
        pytest.skip("stage-2 predictions not present")
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "full.yaml"))
    n = 30
    smp = pd.DataFrame({"uid": [f"pathmnist/train/{i}" for i in range(n)], "label": np.arange(n) % 9})
    cache = tmp_path / "cache"
    (cache / "A").mkdir(parents=True)
    for cond in ("clean_s0", "bubble_s5"):
        mm = open_memmap(cache / "A" / f"{cond}.npy", mode="w+", dtype=np.uint8, shape=(n, 128, 128, 3))
        del mm
    p = {"out": tmp_path / "stage3", "stage2": ROOT / "outputs" / "stage2", "cache": cache}
    rng = np.random.default_rng(0)
    for seed in cfg["train"]["seeds"]:
        for fam, cond, sev in (("A", "clean", 0), ("A", "bubble", 1), ("A", "bubble", 5), ("C", "aligned", 0),
                               ("C", "removed", 0), ("C", "reversed", 0)):
            kind = "shortcut" if fam == "C" else "standard"
            s3.save_preds(p["out"] / "preds" / f"{kind}_seed{seed}_{fam}_{cond}_s{sev}.npz",
                          rng.normal(size=(n, 9)).astype(np.float32), rng.normal(size=(n, 512)), smp["label"].to_numpy(),
                          smp["uid"], np.full(n, -1), 1.0)
    (p["out"] / "corruption_report.json").write_text(json.dumps({"determinism_mismatches": []}))
    san = s3.run_tables(cfg, p, None, smp, log=lambda *_: None)
    assert not san["nan_found"] and not san["uid_mismatch_files"]
    long = pd.read_parquet(p["out"] / "predictions_long.parquet")
    need = {"family", "condition", "severity", "model_kind", "dataset", "seed", "uid", "label", "pred", "confidence",
            "correct", "silent_failure_provisional"}
    assert need <= set(long.columns) and set(long.family) == {"A", "B", "C", "X", "ID"}
    for f in ("family_summary.csv", "condition_summary.csv", "shortcut_states.csv", "summary_stage3.md",
              "figures/behaviour_family_A.png", "figures/behaviour_families_BCX.png", "figures/cue_palette.png",
              "figures/examples.png", "figures/supplement_per_corruption.png"):
        assert (p["out"] / f).exists(), f

