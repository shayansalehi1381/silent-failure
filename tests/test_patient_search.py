"""Phase 1b tests: analytic sampler model == real sampler; search is seeded; frozen plan untouched."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfd.config import ROOT, load_config
from sfd.data import camelyon
from sfd.data.patient_search import SPLITS, Scorer, patient_table, search, split_counts, top_distinct
from sfd.data.splits import balanced_sample


def test_split_counts_matches_balanced_sample():
    rng = np.random.default_rng(0)
    for trial in range(30):
        df = pd.DataFrame({"hospital": rng.choice([0, 3, 4], 3000, p=[0.2, 0.3, 0.5]),
                           "label": (rng.random(3000) < rng.choice([0.02, 0.3, 0.7])).astype(int)})
        avail = np.zeros((3, 2), int)
        for i, h in enumerate([0, 3, 4]):
            for l in (0, 1):
                avail[i, l] = ((df.hospital == h) & (df.label == l)).sum()
        n = int(rng.integers(200, 2500))
        s = balanced_sample(df, n, "hospital", "label", seed=trial)
        size, tum, per_h = split_counts(avail, n)
        assert (size, tum) == (len(s), int(s.label.sum()))
        assert per_h.tolist() == [int((s.hospital == h).sum()) for h in (0, 3, 4)]


def _cfg_meta():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    return cfg, camelyon.synthetic_metadata(seed=0)


def test_assignment_plan_matches_scorer_and_is_patient_disjoint():
    cfg, meta = _cfg_meta()
    pt = patient_table(meta, cfg["camelyon"]["source_hospitals"])
    sizes = {k: cfg["camelyon"]["sizes"][k] for k in SPLITS}
    sc = Scorer(pt, sizes, 0.42, min_patients=1, require_all_hospitals=False)
    asg = np.arange(len(pt)) % 4
    st = sc.stats(asg)
    plan = camelyon.plan_subset(meta, cfg, 0, assignment=dict(zip(pt["patient"], [SPLITS[k] for k in asg])))
    for s in SPLITS:
        p = plan[plan["split"] == s]
        assert len(p) == st[s]["size"] and int(p["label"].sum()) == st[s]["tumour"]
    assert (plan.groupby("patient")["split"].nunique() == 1).all()


def test_search_is_seeded_and_respects_constraints():
    cfg, meta = _cfg_meta()
    pt = patient_table(meta, cfg["camelyon"]["source_hospitals"])
    sizes = {k: cfg["camelyon"]["sizes"][k] for k in SPLITS}
    sc = Scorer(pt, sizes, 0.42, min_patients=2, require_all_hospitals=True)
    a = top_distinct(search(sc, 1, restarts=2, iters=300), 3)
    b = top_distinct(search(sc, 1, restarts=2, iters=300), 3)
    assert all((x[1] == y[1]).all() and x[0] == y[0] for x, y in zip(a, b))
    c = top_distinct(search(sc, 2, restarts=2, iters=300), 3)
    assert any((x[1] != y[1]).any() for x, y in zip(a, c))


def test_extraction_reuses_cached_patches_and_fetches_only_missing(tmp_path, monkeypatch):
    cfg, meta = _cfg_meta()
    cfg["camelyon"]["source"] = "hf_mirror"
    man = camelyon.plan_subset(meta, cfg, 0).iloc[:60].copy()
    man["shard"], man["row_in_shard"] = "data/fake-0.parquet", np.arange(60)
    man["uid"] = "u" + man["image_id"].astype(str)
    size = cfg["data"]["camelyon_size"]
    truth = np.random.default_rng(0).integers(0, 255, (60, size, size, 3), dtype=np.uint8)
    cache = tmp_path / "parts"
    cache.mkdir()
    # a stale part cache that holds the first 40 patches (in a different order)
    order = np.arange(40)[::-1]
    np.savez(cache / "fake-0.npz", images=truth[order], uid=man["uid"].to_numpy(dtype=str)[order])
    requested = []

    def fake_extract(client, grp, imgs, **kw):
        requested.extend(grp["uid"].tolist())
        for idx in grp.index:
            imgs[idx] = truth[idx]

    monkeypatch.setattr(camelyon, "_extract_hf_shard", fake_extract)
    monkeypatch.setattr(camelyon.remote, "RangeClient", lambda url: None)
    out = camelyon.extract_images(man.reset_index(drop=True), cfg, tmp_path / "subset.npz", cache, log=lambda *_: None)
    assert sorted(requested) == sorted(man["uid"].iloc[40:])                  # only the 20 missing patches
    assert np.array_equal(np.load(tmp_path / "subset.npz")["images"], truth)   # cached + fetched are all correct
    assert len(np.load(cache / "fake-0.npz")["uid"]) == 60 and len(out) == 60


def test_assignment_file_in_config_equals_explicit_assignment(tmp_path):
    import json
    cfg, meta = _cfg_meta()
    pt = patient_table(meta, cfg["camelyon"]["source_hospitals"])
    asg = {int(p): SPLITS[i % 4] for i, p in enumerate(pt["patient"])}
    f = tmp_path / "asg.json"
    f.write_text(json.dumps(asg))
    cfg["camelyon"]["assignment_file"] = str(f)
    a = camelyon.plan_subset(meta, cfg, 0)
    b = camelyon.plan_subset(meta, cfg, 0, assignment=asg)
    assert a["uid"].tolist() == b["uid"].tolist() and a["split"].tolist() == b["split"].tolist()


def test_frozen_rank1_assignment_file_is_consistent():
    import json
    f = ROOT / "configs" / "camelyon_assignment_phase1b_rank1.json"
    asg = json.loads(f.read_text())
    assert len(asg) == 26 and pd.Series(asg).value_counts().to_dict() == {"train": 10, "val_cal": 6, "id_test": 6, "val_es": 4}


def test_prefer_ids_keeps_counts_and_reuses_cached_rows():
    cfg, meta = _cfg_meta()
    base = camelyon.plan_subset(meta, cfg, 0)
    pref = set(base["image_id"])
    again = camelyon.plan_subset(meta, cfg, 0, prefer_ids=pref)
    assert again.groupby("split").size().to_dict() == base.groupby("split").size().to_dict()
    assert set(again["image_id"]) == pref                       # same assignment -> exactly the cached rows


@pytest.mark.skipif(not (ROOT / "outputs" / "stage1" / "subset_manifest.parquet").exists(), reason="frozen manifest missing")
def test_default_plan_still_reproduces_frozen_manifest():
    cfg = load_config(ROOT / "configs" / "full.yaml")
    meta = pd.read_parquet(ROOT / "outputs" / "stage1" / "cache" / "camelyon_meta.parquet")
    frozen = pd.read_parquet(ROOT / "outputs" / "stage1" / "subset_manifest.parquet")
    plan = camelyon.plan_subset(meta, cfg, cfg["seed"])
    assert set(plan["uid"]) == set(frozen["uid"])
    assert (plan.set_index("uid")["split"].sort_index() == frozen.set_index("uid")["split"].sort_index()).all()
