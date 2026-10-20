"""Phase 1b: search patient-disjoint Camelyon assignments that equalise tumour prior across val_es/val_cal/id_test.

Uses only cached metadata (outputs/stage1/cache/camelyon_meta.parquet) and the frozen manifest (read-only).
Writes outputs/stage1b/{candidates.csv, best_candidate_detail.csv, best_assignment.json, summary.md}.
Never modifies the frozen splits. Downloads nothing (one tiny HF API listing for shard sizes, optional).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfd.config import ROOT, load_config  # noqa: E402
from sfd.data import camelyon  # noqa: E402
from sfd.data.audit import to_markdown  # noqa: E402`nfrom sfd.data.audit import to_markdown  # noqa: E402
from sfd.data.patient_search import SPLITS, Scorer, patient_table, search, top_distinct  # noqa: E402

ROW_GROUP_ROWS = 100          # verified on the mirror: 100 rows per row group (last one partial)


def shard_group_bytes(meta: pd.DataFrame, repo: str) -> dict[str, float]:
    """Average bytes per row group of each shard (shard size / number of row groups). Falls back to
    the dataset-wide average if the HF file listing cannot be fetched."""
    rows = meta.groupby("shard").size()
    groups = np.ceil(rows / ROW_GROUP_ROWS)
    try:
        sizes = {d["file"]: d["size"] for d in camelyon.list_remote_shards(repo)}
        return {s: sizes[s] / groups[s] for s in rows.index}
    except Exception:
        avg = 9.92 * 2**30 / groups.sum()
        return {s: avg for s in rows.index}


def download_estimate(plan: pd.DataFrame, cached_ids: set, gbytes: dict) -> dict:
    need = plan[~plan["image_id"].isin(cached_ids)]
    g = need.assign(rg=need["row_in_shard"] // ROW_GROUP_ROWS).drop_duplicates(["shard", "rg"])
    gib = sum(gbytes[s] for s in g["shard"]) / 2**30
    return {"patches_total": len(plan), "patches_cached": int(plan["image_id"].isin(cached_ids).sum()),
            "patches_new": len(need), "row_groups_to_fetch": len(g), "est_new_download_gib": round(gib, 2)}


def detail(plan: pd.DataFrame) -> pd.DataFrame:
    p = plan[plan["split"].isin(SPLITS)]
    d = p.groupby(["split", "hospital"]).agg(patients=("patient", "nunique"), slides=("slide", "nunique"),
                                             patches=("image_id", "size"), tumour=("label", "mean")).round(3)
    return d.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--out", default="outputs/stage1b")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    c = cfg["camelyon"]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(ROOT / "outputs/stage1/cache/camelyon_meta.parquet")
    frozen = pd.read_parquet(ROOT / "outputs/stage1/subset_manifest.parquet")
    ood = frozen[frozen["split"].isin(["ood_c1", "ood_c2"])]["label"].mean()
    pt = patient_table(meta, c["source_hospitals"])
    assert len(pt) == 26, f"expected 26 source patients, got {len(pt)}"
    sizes = {k: c["sizes"][k] for k in SPLITS}

    rows, found = [], {}
    for mode, strict, share in (("strict_min_hospital_share_10pct", True, 0.10), ("strict_all_hospitals", True, 0.0),
                                ("relaxed_hospitals", False, 0.0)):
        sc = Scorer(pt, sizes, ood, min_patients=3, require_all_hospitals=strict, min_share=share)
        t0 = time.time()
        cands = top_distinct(search(sc, a.seed, a.restarts, a.iters), 5)
        print(f"[{mode}] search {time.time() - t0:.0f}s")
        for rank, (j, asg) in enumerate(cands, 1):
            _, st = sc.score(asg)
            f = [st[s]["frac"] for s in SPLITS[1:]]
            rows.append({"mode": mode, "rank": rank, "score": round(j, 4), "gap_pts": round(100 * (max(f) - min(f)), 1),
                         "max_dev_from_ood_pts": round(100 * max(abs(x - ood) for x in f), 1),
                         **{f"{s}_tumour": round(st[s]["frac"], 3) for s in SPLITS},
                         **{f"{s}_n": st[s]["size"] for s in SPLITS},
                         **{f"{s}_patients": st[s]["patients"] for s in SPLITS},
                         **{f"{s}_hospitals": st[s]["hospitals"] for s in SPLITS},
                         **{f"{s}_min_hosp_share": round(st[s]["min_share"], 3) for s in SPLITS[1:]},
                         "assignment": json.dumps({int(p): SPLITS[k] for p, k in zip(pt["patient"], asg)})})
            found[(mode, rank)] = (asg, st)
    cand = pd.DataFrame(rows)
    cand.to_csv(out / "candidates.csv", index=False)

    # the frozen assignment, for comparison (read-only)
    fz = frozen[frozen["split"].isin(SPLITS)]
    fz_frac = fz.groupby("split")["label"].mean().round(3).to_dict()

    # prefer the most constrained mode that still gets the gap under 5 points
    pick_mode = next((m for m in ("strict_min_hospital_share_10pct", "strict_all_hospitals", "relaxed_hospitals")
                      if cand[cand["mode"] == m]["gap_pts"].min() <= 5), cand.sort_values("score").iloc[0]["mode"])
    best_row = cand[cand["mode"] == pick_mode].sort_values("score").iloc[0]
    asg_map = {int(k): v for k, v in json.loads(best_row["assignment"]).items()}
    (out / "best_assignment.json").write_text(json.dumps(asg_map, indent=1), encoding="utf-8")

    # verify with the REAL planner, and compare with the analytic stats
    plan = camelyon.plan_subset(meta, cfg, cfg["seed"], assignment=asg_map)
    chk = plan[plan["split"].isin(SPLITS)].groupby("split")["label"].agg(["size", "mean"])
    for s in SPLITS:
        assert chk.loc[s, "size"] == best_row[f"{s}_n"] and abs(chk.loc[s, "mean"] - best_row[f"{s}_tumour"]) < 1e-3
    assert (plan.groupby("patient")["split"].nunique() == 1).all()
    d = detail(plan)
    d.to_csv(out / "best_candidate_detail.csv", index=False)
    overall_h = plan[plan["split"].isin(SPLITS)].groupby("hospital")["label"].mean().round(3).to_dict()

    gbytes = shard_group_bytes(meta, c["repo"])
    cached = set(frozen["image_id"])
    est_plain = download_estimate(plan, cached, gbytes)
    plan_pref = camelyon.plan_subset(meta, cfg, cfg["seed"], assignment=asg_map, prefer_ids=cached)
    est_pref = download_estimate(plan_pref, cached, gbytes)
    pf = plan_pref[plan_pref["split"].isin(SPLITS)].groupby("split")["label"].agg(["size", "mean"]).round(3)

    md = [f"# Phase 1b - patient assignment search (seed {a.seed}, {a.restarts} restarts x {a.iters} iterations)\n",
          f"OOD tumour fraction (frozen): {ood:.3f}. Frozen source-split tumour fractions: {fz_frac} "
          f"(gap {100 * (max(fz_frac[s] for s in SPLITS[1:]) - min(fz_frac[s] for s in SPLITS[1:])):.1f} pts).\n",
          "## Top candidates\n", to_markdown(cand.drop(columns=["assignment"])), "\n",
          f"## Best candidate ({pick_mode}, rank 1)\n", to_markdown(d), "\n",
          f"Tumour fraction per hospital over the four source splits: {overall_h}\n",
          "## Download needed for the best candidate\n",
          f"- Same sampler (new random patches): {json.dumps(est_plain)}",
          f"- Cache-first sampler (reuses already-downloaded patches, identical counts): {json.dumps(est_pref)}; "
          f"split sizes/tumour: {pf.to_dict()}",
          "- Estimates use shard size / number of row groups (100 rows each); OOD splits are unchanged (fully cached).\n"]
    (out / "summary.md").write_text("\n".join(str(x) for x in md), encoding="utf-8")
    print("\n".join(str(x) for x in md))


if __name__ == "__main__":
    main()


