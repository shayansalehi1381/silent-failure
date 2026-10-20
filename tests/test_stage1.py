"""Stage 1 tests: splitting, near-duplicate search, audits, Camelyon planning, frozen-split integrity."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from sfd.config import ROOT, load_config
from sfd.data import audit, camelyon, dedup
from sfd.data.remote import SparseFile
from sfd.data.splits import group_stratified_split, stratified_sample

FRAC = {"train": 0.8, "val_es": 0.05, "val_cal": 0.05, "id_test": 0.1}


# ---------------------------------------------------------------- splitting ------------------
def _toy(n_groups=400, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 8, n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    labels = rng.integers(0, 4, len(groups))
    return groups, labels


def test_group_split_keeps_groups_whole_and_is_stratified():
    groups, labels = _toy()
    sp = group_stratified_split(groups, labels, FRAC, seed=1)
    df = pd.DataFrame({"g": groups, "y": labels, "s": sp})
    assert (df.groupby("g")["s"].nunique() == 1).all()
    got = df["s"].value_counts(normalize=True)
    for s, f in FRAC.items():
        assert abs(got[s] - f) < 0.02
    # every class appears in every split
    assert (df.groupby("s")["y"].nunique() == 4).all()


def test_group_split_deterministic_and_seed_sensitive():
    groups, labels = _toy()
    a = group_stratified_split(groups, labels, FRAC, seed=3)
    assert (a == group_stratified_split(groups, labels, FRAC, seed=3)).all()
    assert (a != group_stratified_split(groups, labels, FRAC, seed=4)).any()


def test_group_split_min_groups():
    groups = np.repeat(np.arange(10), 100)
    sp = group_stratified_split(groups, np.zeros(1000, int), FRAC, seed=0, min_groups=1)
    assert set(sp) == set(FRAC)


def test_stratified_sample_exact_size_deterministic():
    df = pd.DataFrame({"hospital": np.repeat([0, 3], 500), "label": np.tile([0, 1], 500)})
    a = stratified_sample(df, 100, ["hospital", "label"], seed=1)
    assert len(a) == 100 and a.index.equals(stratified_sample(df, 100, ["hospital", "label"], seed=1).index)
    assert a.groupby(["hospital", "label"]).size().between(24, 26).all()
    assert len(stratified_sample(df, 5000, ["hospital"], seed=0)) == len(df)


def test_balanced_sample_balances_labels_and_caps_by_availability():
    from sfd.data.splits import balanced_sample
    df = pd.DataFrame({"g": np.repeat([0, 1], 1000), "y": np.r_[np.zeros(900), np.ones(100), np.zeros(500), np.ones(500)]})
    s = balanced_sample(df, 400, "g", "y", seed=0)
    assert len(s) == 400 and s.index.is_unique
    # group 1 (balanced availability) gets ~200 per label share; group 0 has only 100 positives -> capped
    assert s[s.g == 1].y.mean() == pytest.approx(0.5, abs=0.02)
    assert (s[s.g == 0].y == 1).sum() <= 100
    assert s.index.equals(balanced_sample(df, 400, "g", "y", seed=0).index)


# ---------------------------------------------------------------- near-duplicates ------------
def test_dedup_finds_planted_duplicates_and_clusters_them():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (120, 32, 32, 3), dtype=np.uint8)
    imgs[60] = imgs[5]                                                      # exact copy
    imgs[61] = np.clip(imgs[5].astype(int) + rng.integers(-2, 3, imgs[5].shape), 0, 255)   # near copy
    imgs[90] = imgs[20]
    th = dedup.thumbnails(imgs, 8)
    i, j, r, sat = dedup.radius_pairs(th, None, max_rms=0.02, chunk=50, device="cpu")
    assert not sat
    cl = dedup.cluster_labels(len(imgs), i, j, r, 0.02)
    assert cl[5] == cl[60] == cl[61] and cl[20] == cl[90] and cl[5] != cl[20]
    assert len(np.unique(cl)) == 120 - 3
    h = dedup.exact_hashes(imgs)
    assert h[5] == h[60] and h[5] != h[61]


def test_cross_join_indexes_both_sets():
    a = np.zeros((3, 12), np.float32)
    b = np.zeros((5, 12), np.float32)
    b[2] = 0.001
    b[:2] = 1.0
    b[3:] = 0.5
    i, j, r, _ = dedup.radius_pairs(a, b, max_rms=0.01, device="cpu")
    assert sorted(set(j.tolist())) == [2] and sorted(set(i.tolist())) == [0, 1, 2]


def test_sensitivity_table_monotone():
    rng = np.random.default_rng(1)
    th = rng.random((200, 48)).astype(np.float32)
    th[100:110] = th[0:10] + 0.002
    i, j, r, s = dedup.radius_pairs(th, None, max_rms=0.3, device="cpu")
    rows = dedup.sensitivity_table(200, i, j, r, [0.0, 0.01, 0.1, 0.3], np.zeros(200, int), s)
    pairs = [x["n_pairs"] for x in rows]
    assert pairs == sorted(pairs) and rows[1]["n_dup_clusters"] == 10


# ---------------------------------------------------------------- audits ---------------------
def _df():
    return pd.DataFrame({"uid": list("abcdef"), "split": ["train", "train", "val_es", "val_es", "id_test", "excluded"],
                         "group": [1, 1, 2, 2, 3, 1], "label": [0, 1, 0, 1, 0, 1]})


def test_audit_accepts_clean_and_ignores_excluded():
    audit.assert_unique_uid(_df())
    audit.assert_group_disjoint(_df(), "group")        # group 1 also in 'excluded' row: ignored


def test_audit_detects_planted_leaks():
    d = _df()
    d.loc[1, "group"] = 2                              # cluster 2 now in train and val_es
    with pytest.raises(AssertionError, match="cross splits"):
        audit.assert_group_disjoint(d, "group")
    d = _df()
    d.loc[1, "uid"] = "a"
    with pytest.raises(AssertionError, match="duplicated uid"):
        audit.assert_unique_uid(d)
    with pytest.raises(AssertionError):
        audit.assert_splits_nonempty_all_classes(_df(), 2, ["train", "val_es", "id_test"])


def test_pair_leak_counter():
    sp = np.array(["train", "val_es", "train", "excluded"], dtype=object)
    i, j, r = np.array([0, 0, 2]), np.array([1, 2, 3]), np.array([0.0, 0.0, 0.0])
    assert audit.assert_no_pair_across(i, j, r, 0.01, sp, sp) == 1     # (0,1) crosses; (0,2) same; (2,3) excluded


def test_manifest_hash_order_independent():
    d = _df()
    assert audit.manifest_hash(d, ["uid", "split"]) == audit.manifest_hash(d.sample(frac=1, random_state=0), ["uid", "split"])


# ---------------------------------------------------------------- camelyon planning ----------
def _cam_cfg():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg = copy.deepcopy(cfg)
    return cfg


def test_camelyon_plan_group_aware_and_deterministic():
    cfg = _cam_cfg()
    meta = camelyon.synthetic_metadata(seed=0)
    camelyon.validate_metadata(meta)
    a = camelyon.plan_subset(meta, cfg, 0)
    b = camelyon.plan_subset(meta, cfg, 0)
    assert audit.manifest_hash(a, ["uid", "split", "group"]) == audit.manifest_hash(b, ["uid", "split", "group"])
    audit.assert_unique_uid(a)
    audit.assert_group_disjoint(a, "group")                                   # no slide crosses splits
    src = a[a["split"].isin(camelyon.SPLIT_NAMES)]
    assert set(src["hospital"]) <= {0, 3, 4}
    assert set(a.loc[a["split"] == "ood_c1", "hospital"]) == {1} and set(a.loc[a["split"] == "ood_c2", "hospital"]) == {2}
    for s, n in cfg["camelyon"]["sizes"].items():
        assert (a["split"] == s).sum() == n
    assert (a.groupby("split")["label"].nunique() == 2).all()
    assert (a.groupby(src["split"])["hospital"].nunique() == 3).all()          # each source split sees all 3 hospitals


def test_spatial_overlap_report():
    meta = pd.DataFrame({"slide": [1, 1, 1, 2], "hospital": 0, "x_coord": [0, 50, 500, 0], "y_coord": [0, 50, 500, 0]})
    r = camelyon.spatial_overlap_report(meta, 96).set_index("slide")
    assert r.loc[1, "n_overlapping_pairs"] == 1 and abs(r.loc[1, "frac_patches_overlapping"] - 2 / 3) < 1e-3
    assert r.loc[2, "n_overlapping_pairs"] == 0


def test_tissue_flags():
    white = np.full((1, 8, 8, 3), 250, np.uint8)
    pink = np.zeros((1, 8, 8, 3), np.uint8)
    pink[..., 0], pink[..., 1], pink[..., 2] = 220, 120, 180
    dark = np.full((1, 8, 8, 3), 10, np.uint8)
    f = camelyon.tissue_flags(np.concatenate([white, pink, dark]), 0.1, 40)
    assert f["flag_nontissue"].tolist() == [True, False, True] and f["flag_dark"].tolist() == [False, False, True]


def test_sparse_file_merges_and_refuses_unfetched_bytes():
    f = SparseFile(100, [(10, b"abcdef"), (14, b"efgh"), (50, b"zz")])
    f.seek(10)
    assert f.read(8) == b"abcdefgh"
    f.seek(50)
    assert f.read(2) == b"zz"
    f.seek(20)
    with pytest.raises(IOError):
        f.read(1)


# ---------------------------------------------------------------- integration ----------------
def test_stage1_smoke_end_to_end(tmp_path):
    if not (ROOT / "data" / "pathmnist_128.npz").exists():
        pytest.skip("PathMNIST 128px file not downloaded")
    import json

    from sfd.data import stage1
    cfg = _cam_cfg()
    rep = stage1.run(cfg, tmp_path, log=lambda *_: None)
    assert rep["n_failed"] == 0 and rep["repro"]["identical_on_rebuild"]
    sp = pd.read_parquet(tmp_path / "stage1" / "splits.parquet")
    audit.assert_unique_uid(sp)
    for ds, col in (("pathmnist", "group"), ("camelyon17", "group")):
        audit.assert_group_disjoint(sp[sp["dataset"] == ds], col)
    assert (tmp_path / "stage1" / "table1.md").exists() and (tmp_path / "stage1" / "near_duplicate_pairs.png").exists()
    json.loads((tmp_path / "stage1" / "stage1_report.json").read_text())


@pytest.mark.skipif(not (ROOT / "outputs" / "stage1" / "splits.parquet").exists(), reason="frozen splits not built")
def test_frozen_splits_integrity():
    out = ROOT / "outputs" / "stage1"
    sp = pd.read_parquet(out / "splits.parquet")
    cfg = load_config(ROOT / "configs" / "full.yaml")
    audit.assert_unique_uid(sp)
    pm, cam = sp[sp["dataset"] == "pathmnist"], sp[sp["dataset"] == "camelyon17"]
    audit.assert_group_disjoint(pm, "group")
    audit.assert_group_disjoint(cam, "group")
    audit.assert_splits_nonempty_all_classes(pm, 9, ["train", "val_es", "val_cal", "id_test", "nct7k_test"])
    audit.assert_splits_nonempty_all_classes(cam, 2, ["train", "val_es", "val_cal", "id_test", "ood_c1", "ood_c2"])
    audit.assert_fractions(pm[pm["split"].isin(camelyon.SPLIT_NAMES)], cfg["pathmnist"]["split_fractions"],
                           cfg["pathmnist"]["split_tolerance"])
    # the checksum file matches the frozen files
    for line in (out / "checksums.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = ROOT / rel
        if p.exists() and p.stat().st_size < 50 * 2**20:
            assert audit.sha256_file(p) == digest, f"checksum mismatch: {rel}"
