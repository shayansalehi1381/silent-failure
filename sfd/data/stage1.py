"""Stage 1: data, frozen splits, leakage audit.

Steps (heavy ones cache their result keyed by the relevant config, so re-runs are cheap and resumable):
  1 PathMNIST fingerprints (thumbnails + exact hashes)       -> cache/pathmnist_fp.npz
  2 PathMNIST near-duplicate search (pool self-join, nct7k->pool cross-join)  -> cache/pathmnist_pairs.npz
  3 Camelyon metadata (+ validation against WILDS v1.0 facts) -> cache/camelyon_meta.parquet
  4 split builders (PathMNIST cluster-level, Camelyon slide-level) + reproducibility check
  5 Camelyon patch extraction (only if camelyon.fetch_images) + flags + near-duplicate audit
  6 audits (asserts), splits.parquet, subset_manifest.parquet, dedup_report.csv, Table 1, figure, checksums
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfd.config import ROOT, resolve
from sfd.data import audit, camelyon, dedup, remote
from sfd.data.splits import group_stratified_split
from sfd.sysinfo import process_mem_gb, ram_gb

POOL_SPLITS = ["train", "val_es", "val_cal", "id_test"]
PM_CLASSES = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth_muscle",
              "normal_mucosa", "stroma", "tumor_epithelium"]
CAM_CLASSES = ["non_tumor", "tumor"]


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Steps:
    """Runtime / RAM / transfer log of the stage."""

    def __init__(self, log=print):
        self.rows, self.log = [], log

    def run(self, name, fn):
        free0, t0, b0 = ram_gb().get("available"), time.time(), remote.DOWNLOADED_BYTES
        self.log(f"[stage1] {name} (free RAM {free0} GB)")
        out = fn()
        row = {"step": name, "seconds": round(time.time() - t0, 1), "free_ram_gb_before": free0,
               "free_ram_gb_after": ram_gb().get("available"), "process_peak_gb": process_mem_gb().get("peak"),
               "downloaded_mib": round((remote.DOWNLOADED_BYTES - b0) / 2**20, 1)}
        self.rows.append(row)
        self.log(f"[stage1] {name} done in {row['seconds']} s, process peak {row['process_peak_gb']} GB")
        return out


# ------------------------------------------------------------------ PathMNIST ---------------
def pathmnist_fingerprints(cfg: dict, cache: Path) -> dict:
    pm = cfg["pathmnist"]
    key = _key([pm["pool_splits"], pm["max_per_source_split"], pm["max_test"], pm["thumb_size"], cfg["seed"],
                cfg["data"]["pathmnist_size"]])
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files}
    path = resolve(cfg, "data_dir") / f"pathmnist_{cfg['data']['pathmnist_size']}.npz"
    z = np.load(path)
    rng = np.random.default_rng(cfg["seed"])
    parts = {"thumbs": [], "hash": [], "label": [], "src_split": [], "src_index": [], "role": []}
    for k, role, limit in [(s, "pool", pm["max_per_source_split"]) for s in pm["pool_splits"]] + \
                          [("test", "test", pm["max_test"])]:
        labels = z[f"{k}_labels"].reshape(-1).astype(np.int64)
        idx = np.arange(len(labels))
        if limit and limit < len(idx):
            idx = np.sort(rng.choice(len(idx), int(limit), replace=False))
        imgs = z[f"{k}_images"]                       # one split in RAM at a time
        sub = imgs[idx] if len(idx) < len(imgs) else imgs
        parts["thumbs"].append(dedup.thumbnails(sub, pm["thumb_size"]))
        parts["hash"].append(dedup.exact_hashes(sub))
        parts["label"].append(labels[idx])
        parts["src_split"].append(np.full(len(idx), k))
        parts["src_index"].append(idx)
        parts["role"].append(np.full(len(idx), role))
        del imgs, sub
    out = {k: np.concatenate(v) for k, v in parts.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=key, **out)
    return out


def pathmnist_pairs(cfg: dict, fp: dict, cache: Path) -> dict:
    pm = cfg["pathmnist"]
    d = pm["dedup"]
    key = _key([d, pm["thumb_size"], pm["pool_splits"], pm["max_per_source_split"], pm["max_test"], cfg["seed"]])
    if cache.exists():
        z = np.load(cache)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files}
    mx = max(d["sensitivity"])
    pool, test = fp["role"] == "pool", fp["role"] == "test"
    kw = dict(max_rms=mx, chunk=d["chunk"], device=d["device"], cap=d["pair_cap"])
    pi, pj, pr, psat = dedup.radius_pairs(fp["thumbs"][pool], None, **kw)
    ci, cj, cr, csat = dedup.radius_pairs(fp["thumbs"][test], fp["thumbs"][pool], **kw)
    ti, tj, tr, tsat = dedup.radius_pairs(fp["thumbs"][test], None, **kw)
    out = dict(key=key, pool_i=pi, pool_j=pj, pool_r=pr, pool_sat=psat, cross_i=ci, cross_j=cj, cross_r=cr,
               cross_sat=csat, test_i=ti, test_j=tj, test_r=tr, test_sat=tsat)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def build_pathmnist_splits(cfg: dict, fp: dict, pairs: dict) -> pd.DataFrame:
    """Cluster-level stratified split of the pool + nct7k_test (cross-set near-duplicates excluded)."""
    pm, thr = cfg["pathmnist"], cfg["pathmnist"]["dedup"]["rms_threshold"]  # within-pool clustering
    thr_x = pm["dedup"]["cross_exclusion_rms"]                                 # nct7k_test -> pool exclusion
    pool, test = fp["role"] == "pool", fp["role"] == "test"
    n_pool, n_test = int(pool.sum()), int(test.sum())
    cl = dedup.cluster_labels(n_pool, pairs["pool_i"], pairs["pool_j"], pairs["pool_r"], thr)
    split = group_stratified_split(cl, fp["label"][pool], pm["split_fractions"], cfg["seed"])
    tcl = dedup.cluster_labels(n_test, pairs["test_i"], pairs["test_j"], pairs["test_r"], thr) + cl.max() + 1
    has_pool_nb = np.zeros(n_test, bool)
    k = pairs["cross_r"] <= thr_x
    has_pool_nb[pairs["cross_i"][k]] = True
    rows = []
    for mask, grp, spl, reason in [(pool, cl, split, ""),
                                   (test, tcl, np.where(has_pool_nb, audit.EXCLUDED, "nct7k_test"),
                                    None)]:
        d = pd.DataFrame({"src_split": fp["src_split"][mask], "src_index": fp["src_index"][mask],
                          "label": fp["label"][mask], "group": grp, "split": spl})
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["reason"] = np.where(df["split"] == audit.EXCLUDED, f"near_duplicate_of_pool(rms<={thr_x})", "")
    df["uid"] = "pathmnist/" + df["src_split"].astype(str) + "/" + df["src_index"].astype(str)
    df["dataset"], df["hospital"] = "pathmnist", -1
    return df


# ------------------------------------------------------------------ orchestration -----------
def _dedup_rows(dataset, scope, n, i, j, r, thr_list, labels, sat) -> list[dict]:
    rows = dedup.sensitivity_table(n, i, j, r, thr_list, labels, sat)
    return [{"dataset": dataset, "scope": scope, **row} for row in rows]


def _cross_rows(dataset, scope, n_a, i, r, thr_list, sat) -> list[dict]:
    return [{"dataset": dataset, "scope": scope, "rms_threshold": t, "n_pairs": int((r <= t).sum()),
             "n_images_with_neighbor": int(len(np.unique(i[r <= t]))), "n_images": n_a,
             "pairs_saturated": bool(sat)} for t in thr_list]


def run(cfg: dict, out_dir: Path, log=print) -> dict:
    out = out_dir / "stage1"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    st = Steps(log)
    pm, cam = cfg["pathmnist"], cfg["camelyon"]
    data_dir = resolve(cfg, "data_dir")
    log(f"[stage1] free RAM at start: {ram_gb()}")

    fp = st.run("pathmnist fingerprints", lambda: pathmnist_fingerprints(cfg, cache / "pathmnist_fp.npz"))
    pairs = st.run("pathmnist near-duplicate search", lambda: pathmnist_pairs(cfg, fp, cache / "pathmnist_pairs.npz"))
    pm_df = st.run("pathmnist split", lambda: build_pathmnist_splits(cfg, fp, pairs))

    # ---- Camelyon metadata and plan
    def meta_step():
        if cam["source"] == "hf_mirror":
            return camelyon.fetch_remote_metadata(cam["repo"], cache / "camelyon_meta.parquet", log)
        if cam["source"] == "local_archive":
            return camelyon.load_local_metadata(Path(cam["archive_path"]))
        return camelyon.synthetic_metadata(seed=cfg["seed"])

    meta = st.run("camelyon metadata", meta_step)
    meta_report = camelyon.validate_metadata(meta)
    cam_plan = st.run("camelyon plan", lambda: camelyon.plan_subset(meta, cfg, cfg["seed"]))

    # ---- reproducibility: the split builders are deterministic given the seed
    h1 = audit.manifest_hash(pm_df, ["uid", "split", "group"])
    h2 = audit.manifest_hash(build_pathmnist_splits(cfg, fp, pairs), ["uid", "split", "group"])
    c1 = audit.manifest_hash(cam_plan, ["uid", "split", "group"])
    c2 = audit.manifest_hash(camelyon.plan_subset(meta, cfg, cfg["seed"]), ["uid", "split", "group"])
    assert h1 == h2 and c1 == c2, "split builders are not deterministic"
    repro = {"pathmnist_hash": h1, "camelyon_hash": c1, "identical_on_rebuild": True}

    # ---- Camelyon images
    cam_df, cam_imgs_ok = cam_plan.copy(), False
    for c in ("tissue_frac", "mean_intensity"):
        cam_df[c] = np.nan
    cam_df["flag_nontissue"], cam_df["flag_dark"] = False, False
    # real subset lives in data/ (set in full.yaml); runs without `subset_npz` (smoke, synthetic) stay in out/cache
    cam_npz = (ROOT / cam["subset_npz"]) if cam.get("subset_npz") else cache / "camelyon_subset_96.npz"
    if cam["source"] == "synthetic":
        assert not str(cam_npz).startswith(str(data_dir)), "synthetic images must never be written to data/"
    dd_rows: list[dict] = []
    cam_pairs = None
    if cam["fetch_images"]:
        cam_df = st.run("camelyon patch extraction", lambda: camelyon.extract_images(
            cam_plan, cfg, cam_npz, cache / "camelyon_parts", log))
        cam_imgs_ok = True
        imgs = np.load(cam_npz)["images"]
        th = dedup.thumbnails(imgs, cam["thumb_size"])
        del imgs
        d = cam["dedup"]
        ci, cj, cr, csat = dedup.radius_pairs(th, None, max_rms=max(d["sensitivity"]), chunk=d["chunk"],
                                              device=d["device"], cap=d["pair_cap"])
        cam_pairs = (ci, cj, cr, csat)
        dd_rows += _dedup_rows("camelyon17", "subset_self_join", len(th), ci, cj, cr, d["sensitivity"],
                               cam_df["label"].values, csat)

    # ---- assemble splits.parquet (small columns only)
    pm_out = pm_df.assign(flag_nontissue=False, flag_dark=False)[
        ["uid", "dataset", "split", "group", "label", "hospital", "src_index", "reason", "flag_nontissue", "flag_dark"]]
    cam_out = cam_df.assign(src_index=cam_df["image_id"], reason="")[
        ["uid", "dataset", "split", "group", "label", "hospital", "src_index", "reason", "flag_nontissue", "flag_dark"]]
    splits = pd.concat([pm_out, cam_out], ignore_index=True)
    splits["src_index"] = splits["src_index"].astype("int64")

    # ---- audits
    checks: list[dict] = []

    def check(name, fn):
        try:
            val = fn()
            checks.append({"check": name, "passed": True, "value": val})
        except AssertionError as e:
            checks.append({"check": name, "passed": False, "value": str(e)})

    pmx = splits[splits["dataset"] == "pathmnist"]
    cax = splits[splits["dataset"] == "camelyon17"]
    pool_df = pmx[pmx["split"].isin(POOL_SPLITS)]
    thr = pm["dedup"]["rms_threshold"]
    thr_x = pm["dedup"]["cross_exclusion_rms"]
    check("uid unique (all datasets)", lambda: audit.assert_unique_uid(splits))
    check("pathmnist: no near-duplicate cluster crosses splits", lambda: audit.assert_group_disjoint(pmx, "group"))
    check("pathmnist: split fractions within tolerance",
          lambda: audit.assert_fractions(pool_df, pm["split_fractions"], pm["split_tolerance"]))
    check("pathmnist: every split has all 9 classes",
          lambda: audit.assert_splits_nonempty_all_classes(pmx, 9, POOL_SPLITS + ["nct7k_test"]))
    # no near-duplicate pair (rms<=thr) between any two different non-excluded splits
    pool_idx = np.where(fp["role"] == "pool")[0]
    pool_split = pm_df["split"].values[:len(pool_idx)].astype(str)
    test_split = pm_df["split"].values[len(pool_idx):].astype(str)
    check("pathmnist: 0 near-duplicate pairs across pool splits",
          lambda: _zero(audit.assert_no_pair_across(pairs["pool_i"], pairs["pool_j"], pairs["pool_r"], thr,
                                                    pool_split, pool_split)))
    check("pathmnist: 0 near-duplicate pairs between nct7k_test and the pool (after exclusion)",
          lambda: _zero(audit.assert_no_pair_across(pairs["cross_i"], pairs["cross_j"], pairs["cross_r"], thr_x,
                                                    test_split, pool_split)))
    src_h, ood = set(cam["source_hospitals"]), cam["ood_hospitals"]
    check("camelyon: no slide crosses splits", lambda: audit.assert_group_disjoint(cax, "group"))
    check("camelyon: source splits use source hospitals only",
          lambda: _all(cax[cax["split"].isin(POOL_SPLITS)]["hospital"].isin(src_h)))
    check("camelyon: OOD splits are exactly their held-out hospital",
          lambda: _all(all((cax[cax["split"] == k]["hospital"] == h).all() for k, h in ood.items())))
    check("camelyon: OOD hospitals absent from source splits",
          lambda: _all(~cax[cax["split"].isin(POOL_SPLITS)]["hospital"].isin(list(ood.values()))))
    check("camelyon: every split has both classes",
          lambda: audit.assert_splits_nonempty_all_classes(cax, 2, POOL_SPLITS + list(ood)))
    check("camelyon: realised split sizes vs requested",
          lambda: {s: [int((cax["split"] == s).sum()), cam["sizes"][s]] for s in cam["sizes"]})
    if cam_pairs is not None:
        cs = cam_df["split"].values.astype(str)
        check("camelyon: near-duplicate pairs (rms<=thr) across different splits [reported, not asserted]",
              lambda: audit.assert_no_pair_across(cam_pairs[0], cam_pairs[1], cam_pairs[2],
                                                  cam["dedup"]["rms_threshold"], cs, cs))
        checks[-1]["passed"] = True       # informational: cross-slide near-duplicates are reported only
    failed = [c for c in checks if not c["passed"]]

    # ---- reports
    for t_ in pm["dedup"]["sensitivity"]:
        pass
    dd_rows = (_dedup_rows("pathmnist", "pool_self_join", int((fp["role"] == "pool").sum()), pairs["pool_i"],
                           pairs["pool_j"], pairs["pool_r"], pm["dedup"]["sensitivity"], fp["label"][fp["role"] == "pool"],
                           bool(pairs["pool_sat"])) +
               _cross_rows("pathmnist", "nct7k_test_to_pool", int((fp["role"] == "test").sum()), pairs["cross_i"],
                           pairs["cross_r"], pm["dedup"]["sensitivity"], bool(pairs["cross_sat"])) + dd_rows)
    pd.DataFrame(dd_rows).to_csv(out / "dedup_report.csv", index=False)
    ex = fp["hash"]
    exact = {"pool_images_in_exact_duplicate_groups": int(pd.Series(ex[fp["role"] == "pool"]).duplicated(keep=False).sum()),
             "test_images_exactly_in_pool": int(np.isin(ex[fp["role"] == "test"], ex[fp["role"] == "pool"]).sum())}

    overlap = camelyon.spatial_overlap_report(meta, cam["patch_px"])
    overlap.to_csv(out / "camelyon_spatial_overlap.csv", index=False)
    sel_overlap = camelyon.spatial_overlap_report(meta[meta["image_id"].isin(cam_plan["image_id"])], cam["patch_px"])
    sel_overlap.to_csv(out / "camelyon_spatial_overlap_subset.csv", index=False)

    splits.to_parquet(out / "splits.parquet", index=False)
    man_cols = [c for c in cam_df.columns if c not in ("dataset",)]
    cam_df[man_cols].to_parquet(out / "subset_manifest.parquet", index=False)

    info = _table1_info(cam)
    t1 = audit.table1(splits[splits["split"] != audit.EXCLUDED], info)
    t1.to_csv(out / "table1.csv", index=False)
    (out / "table1.md").write_text(audit.to_markdown(t1), encoding="utf-8")

    pair_fig = _pair_figure(cfg, fp, pairs, thr, thr_x, out / "near_duplicate_pairs.png", log)

    sums = audit.write_checksums(
        [data_dir / f"pathmnist_{cfg['data']['pathmnist_size']}.npz"] + ([cam_npz] if cam_imgs_ok else []) +
        [out / "splits.parquet",
         out / "subset_manifest.parquet", out / "dedup_report.csv", out / "table1.csv"],
        out / "checksums.sha256", ROOT)

    audit_json = {"checks": checks, "n_failed": len(failed), "exact_duplicates": exact,
                  "repro": repro, "camelyon_metadata": meta_report,
                  "camelyon_images_extracted": cam_imgs_ok,
                  "camelyon_overlap_all": _overlap_summary(overlap),
                  "camelyon_overlap_subset": _overlap_summary(sel_overlap),
                  "pathmnist_excluded_from_nct7k": int((pm_df["split"] == audit.EXCLUDED).sum()),
                  "pathmnist_thresholds": {"within_pool_clustering_rms": thr, "cross_exclusion_rms": thr_x},
                  "pathmnist_nct7k_exclusions_by_threshold": _exclusions_by_threshold(
                      fp, pairs, pm["dedup"]["sensitivity"]),
                  "pathmnist_excluded_by_class": {PM_CLASSES[int(k)]: int(v) for k, v in
                                                  pm_df.loc[pm_df["split"] == audit.EXCLUDED, "label"]
                                                  .value_counts().sort_index().items()},
                  "pathmnist_nct7k_by_class_before_exclusion": {PM_CLASSES[int(k)]: int(v) for k, v in pm_df[
                      pm_df["split"].isin(["nct7k_test", audit.EXCLUDED])]["label"].value_counts().sort_index().items()},
                  "pair_figure_pairs": pair_fig,
                  "nontissue_flagged": int(cam_df["flag_nontissue"].sum()),
                  "dark_flagged": int(cam_df["flag_dark"].sum())}
    camelyon.save_json(audit_json, out / "leakage_audit.json")
    camelyon.save_json({"steps": st.rows, "downloaded_bytes": remote.DOWNLOADED_BYTES, "ram": ram_gb(),
                        "peak_process_gb": process_mem_gb().get("peak"), "checksums": sums},
                       out / "stage1_report.json")
    for c in checks:
        log(f"[audit] {'PASS' if c['passed'] else 'FAIL'}  {c['check']}  {c['value'] if c['passed'] else c['value']}")
    if failed:
        raise AssertionError(f"{len(failed)} leakage-audit check(s) failed: " + "; ".join(c["check"] for c in failed))
    return audit_json


def _exclusions_by_threshold(fp: dict, pairs: dict, thresholds) -> dict:
    """For each cross-exclusion threshold: how many nct7k_test images would be excluded, per class."""
    lab = fp["label"][fp["role"] == "test"]
    out = {}
    for t in thresholds:
        hit = np.unique(pairs["cross_i"][pairs["cross_r"] <= t])
        out[str(t)] = {"n_excluded": int(len(hit)),
                       "by_class": {PM_CLASSES[int(k)]: int(v) for k, v in
                                    pd.Series(lab[hit]).value_counts().sort_index().items()}}
    return out


def _zero(n: int) -> int:
    assert n == 0, f"{n} offending pairs"
    return n


def _all(x) -> bool:
    assert bool(np.all(x)), "condition violated"
    return True


def _overlap_summary(o: pd.DataFrame) -> dict:
    return {"n_slides": int(len(o)), "mean_frac_patches_overlapping": round(float(o["frac_patches_overlapping"].mean()), 4),
            "max_frac_patches_overlapping": round(float(o["frac_patches_overlapping"].max()), 4)}


def _table1_info(cam: dict) -> dict:
    src = {"default": "source (ID)"}
    return {
        "pathmnist": {
            "class_names": PM_CLASSES,
            "domains": {"default": "NCT-CRC-HE-100K (Macenko-normalised H&E, colorectal)",
                        "nct7k_test": "CRC-VAL-HE-7K (different clinical centre)"},
            "families": {"default": "source / in-distribution", "nct7k_test": "X (supplement: real centre shift)"},
            "roles": {"train": "fit classifier", "val_es": "early stopping", "val_cal": "temperature scaling, "
                      "acceptance threshold, monitoring references", "id_test": "in-distribution test",
                      "nct7k_test": "shifted evaluation only"},
            "access": "MedMNIST v2 (CC BY 4.0), 128x128; underlying data NCT-CRC-HE (CC BY 4.0)"},
        "camelyon17": {
            "class_names": CAM_CLASSES,
            "domains": {"default": f"source hospitals {cam['source_hospitals']}",
                        **{k: f"held-out hospital {h}" for k, h in cam["ood_hospitals"].items()}},
            "families": {"default": "source / in-distribution", **{k: "B (held-out hospital)" for k in cam["ood_hospitals"]}},
            "roles": {"train": "fit classifier", "val_es": "early stopping", "val_cal": "temperature scaling, "
                      "acceptance threshold, monitoring references", "id_test": "in-distribution test",
                      **{k: "shifted evaluation only" for k in cam["ood_hospitals"]}},
            "access": "Camelyon17-WILDS v1.0 (CC0), 96x96; via unofficial HF mirror wltjr1007/Camelyon17-WILDS"}}


def _pair_figure(cfg, fp, pairs, thr, thr_x, out_png: Path, log, n_max: int = 8) -> int:
    """Up to n_max near-duplicate pairs: excluded nct7k vs pool first, then within-pool pairs spread over rms."""
    pool_idx = np.where(fp["role"] == "pool")[0]
    test_idx = np.where(fp["role"] == "test")[0]
    cand = []                                   # (rms, global_i, global_j, kind)
    k = np.where(pairs["cross_r"] <= max(thr, thr_x))[0]      # spread over rms; label what the policy does
    if len(k):
        order = k[np.argsort(pairs["cross_r"][k])]
        for q in order[np.unique(np.linspace(0, len(order) - 1, n_max // 2).astype(int))]:
            r = float(pairs["cross_r"][q])
            cand.append((r, test_idx[pairs["cross_i"][q]], pool_idx[pairs["cross_j"][q]],
                         f"nct7k~pool {'EXCLUDED' if r <= thr_x else 'kept'}"))
    k = np.where(pairs["pool_r"] <= thr)[0]
    if len(k):
        order = k[np.argsort(pairs["pool_r"][k])]
        pick = order[np.unique(np.linspace(0, len(order) - 1, n_max - len(cand)).astype(int))] if len(cand) < n_max else []
        for q in pick:
            cand.append((float(pairs["pool_r"][q]), pool_idx[pairs["pool_i"][q]], pool_idx[pairs["pool_j"][q]], "pool~pool"))
    cand = cand[:n_max]
    need: dict[str, set] = {}
    for _, a, b, _ in cand:
        for g in (a, b):
            need.setdefault(str(fp["src_split"][g]), set()).add(int(fp["src_index"][g]))
    path = resolve(cfg, "data_dir") / f"pathmnist_{cfg['data']['pathmnist_size']}.npz"
    z, got = np.load(path), {}
    for s, idxs in need.items():
        arr = z[f"{s}_images"]
        for i in idxs:
            got[(s, i)] = arr[i].copy()
        del arr
    pairs_img = []
    for r, a, b, kind in cand:
        ia, ib = got[(str(fp["src_split"][a]), int(fp["src_index"][a]))], got[(str(fp["src_split"][b]), int(fp["src_index"][b]))]
        pairs_img.append((ia, ib, f"{kind}  rms={r:.4f}\n{fp['src_split'][a]}[{fp['src_index'][a]}] | "
                                  f"{fp['src_split'][b]}[{fp['src_index'][b]}]"))
    audit.pair_grid(pairs_img, out_png)
    return len(pairs_img)

