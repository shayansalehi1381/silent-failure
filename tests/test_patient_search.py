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
