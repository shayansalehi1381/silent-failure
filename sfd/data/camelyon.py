"""Camelyon17-WILDS v1.0 (96x96) metadata, group-aware subset planning and patch extraction.

Sources (cfg camelyon.source):
  hf_mirror     - unofficial HF parquet mirror `wltjr1007/Camelyon17-WILDS` (the official WILDS host has
                  an expired TLS certificate; verification is never disabled). Metadata columns are read
                  with HTTP range requests; images are streamed shard by shard and never stored whole.
  local_archive - the original WILDS archive (`metadata.csv` + `patches/`) at cfg camelyon.archive_path.
  synthetic     - random tiles with WILDS-like metadata; ONLY to exercise the code path in smoke tests.

Hospital = WILDS `center` integer (0..4). WILDS v1.0: hospitals 0,3,4 = source (train + id_val),
hospital 1 = val (OOD), hospital 2 = test (OOD).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sfd.data import remote
from sfd.data.splits import balanced_sample, group_stratified_split

META_COLS = ["image_id", "label", "center", "patient", "node", "x_coord", "y_coord", "slide"]
SPLIT_NAMES = ["train", "val_es", "val_cal", "id_test"]


# ------------------------------------------------------------------ metadata ---------------
def list_remote_shards(repo: str) -> list[dict]:
    import requests
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}", params={"blobs": "true"}, timeout=60)
    r.raise_for_status()
    out = [{"file": s["rfilename"], "size": s.get("size")} for s in r.json()["siblings"]
           if s["rfilename"].endswith(".parquet")]
    return sorted(out, key=lambda d: d["file"])


def fetch_remote_metadata(repo: str, cache: Path, log=print) -> pd.DataFrame:
    """Metadata columns of every shard (range requests, no image bytes). Cached as parquet."""
    import pyarrow.parquet as pq
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    b0 = remote.DOWNLOADED_BYTES
    for sh in list_remote_shards(repo):
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{sh['file']}"
        client = remote.RangeClient(url)
        tail_start = max(0, client.size - (1 << 20))
        tail = client.fetch(tail_start, client.size)                       # contains the footer
        md = pq.ParquetFile(remote.SparseFile(client.size, [(tail_start, tail)])).metadata
        # the small metadata column chunks of a row group sit right after the big image chunk
        rngs = []
        for g in range(md.num_row_groups):
            rg, lo, hi = md.row_group(g), None, 0
            for ci in range(rg.num_columns):
                c = rg.column(ci)
                if c.path_in_schema == "image.bytes":
                    continue
                offs = [o for o in (c.dictionary_page_offset, c.data_page_offset) if o]
                s = min(offs)
                lo = s if lo is None else min(lo, s)
                hi = max(hi, s + c.total_compressed_size)
            rngs.append((lo, hi))
        pieces = list(zip([r[0] for r in rngs], client.fetch_many(rngs, workers=24)))
        pf = pq.ParquetFile(remote.SparseFile(client.size, pieces + [(tail_start, tail)]))
        t = pf.read(columns=META_COLS).to_pandas()
        t["shard"] = sh["file"]
        t["row_in_shard"] = np.arange(len(t), dtype=np.int64)
        t["mirror_split"] = sh["file"].split("/")[-1].split("-")[0]
        frames.append(t)
        log(f"  metadata {sh['file']}: {len(t)} rows, {pf.num_row_groups} row groups, "
            f"{remote.DOWNLOADED_BYTES / 2**20:.1f} MiB transferred so far")
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.rename(columns={"center": "hospital", "label": "label"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(cache, index=False)
    log(f"  metadata bytes transferred: {(remote.DOWNLOADED_BYTES - b0) / 2**20:.1f} MiB")
    return meta


def load_local_metadata(archive: Path) -> pd.DataFrame:
    """Original WILDS archive: metadata.csv columns patient,node,x_coord,y_coord,tumor,slide,center,split."""
    m = pd.read_csv(Path(archive) / "metadata.csv", index_col=0)
    meta = pd.DataFrame({
        "image_id": np.arange(len(m), dtype=np.int64), "label": m["tumor"].astype(int).values,
        "hospital": m["center"].astype(int).values, "patient": m["patient"].values,
        "node": m["node"].values, "x_coord": m["x_coord"].values, "y_coord": m["y_coord"].values,
        "slide": m["slide"].values})
    meta["shard"], meta["row_in_shard"] = "local", np.arange(len(meta))
    meta["mirror_split"] = m["split"].map({0: "train", 1: "validation", 2: "test"}).values
    return meta


def synthetic_metadata(n_per_slide: int = 200, seed: int = 0) -> pd.DataFrame:
    """WILDS-like layout (5 hospitals x 6 slides), for smoke tests only."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(5):
        for k in range(6):
            slide = h * 6 + k
            x = rng.integers(0, 4000, n_per_slide)
            y = rng.integers(0, 4000, n_per_slide)
            for i in range(n_per_slide):
                rows.append((len(rows), int(rng.random() < 0.5), h, slide // 2, slide % 5, x[i], y[i], slide))
    meta = pd.DataFrame(rows, columns=META_COLS).rename(columns={"center": "hospital"})
    meta = meta.rename(columns={"center": "hospital"})
    meta["shard"], meta["row_in_shard"], meta["mirror_split"] = "synthetic", np.arange(len(meta)), "train"
    return meta


def validate_metadata(meta: pd.DataFrame, expect: dict | None = None) -> dict:
    """Consistency checks against WILDS v1.0 facts; returns a report (raises on structural problems)."""
    assert meta["image_id"].is_unique, "image_id not unique"
    assert set(meta["label"].unique()) <= {0, 1}
    # slide must belong to exactly one hospital
    assert (meta.groupby("slide")["hospital"].nunique() == 1).all(), "a slide spans several hospitals"
    rep = {"n_rows": len(meta), "n_slides": int(meta["slide"].nunique()),
           "per_hospital": meta.groupby("hospital").size().to_dict(),
           "slides_per_hospital": meta.groupby("hospital")["slide"].nunique().to_dict(),
           "tumor_frac_per_hospital": meta.groupby("hospital")["label"].mean().round(4).to_dict()}
    if expect:
        for k, v in expect.items():
            assert rep.get(k) == v, f"metadata mismatch for {k}: {rep.get(k)} != {v}"
    return rep


# ------------------------------------------------------------------ planning --------------
def plan_subset(meta: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Slide-disjoint split of source hospitals + patch sampling; OOD hospitals sampled per patch.

    Returns the manifest rows (one per selected patch) with columns split, group (=slide), hospital...
    """
    c = cfg["camelyon"]
    sizes, src_h = c["sizes"], list(c["source_hospitals"])
    rng_seed = int(seed)
    parts = []

    # 1) source hospitals: assign whole slides to splits (per hospital => each split sees each hospital)
    src = meta[meta["hospital"].isin(src_h)]
    tot = sum(sizes[s] for s in SPLIT_NAMES)
    fractions = {s: sizes[s] / tot for s in SPLIT_NAMES}
    # rows = (slide, label) with the patch count as weight, so the greedy assignment balances each label
    slide_tbl = src.groupby(["hospital", "slide", "label"]).size().rename("n").reset_index()
    slide_split = {}
    for h in src_h:
        st = slide_tbl[slide_tbl["hospital"] == h].reset_index(drop=True)
        assigned = group_stratified_split(st["slide"].values, st["label"].values, fractions, rng_seed + h,
                                          weights=st["n"].values, min_groups=1)
        slide_split.update(dict(zip(st["slide"], assigned)))
    src = src.assign(split=src["slide"].map(slide_split))
    # 2) subsample patches inside each split: hospitals proportional to availability, labels balanced
    for s in SPLIT_NAMES:
        pool = src[src["split"] == s]
        parts.append(balanced_sample(pool, sizes[s], "hospital", "label", rng_seed + 100 + SPLIT_NAMES.index(s)))
    # 3) OOD hospitals: slides proportional to availability, labels balanced
    for name, h in c["ood_hospitals"].items():
        pool = meta[meta["hospital"] == h].assign(split=name)
        parts.append(balanced_sample(pool, sizes[name], "slide", "label", rng_seed + 200 + int(h)))
    man = pd.concat(parts, ignore_index=True)
    man["uid"] = "camelyon17/" + man["image_id"].astype(str)
    man["group"] = man["slide"].astype(int)
    man["dataset"] = "camelyon17"
    return man.sort_values("image_id", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ images -----------------
def tissue_flags(imgs: np.ndarray, min_tissue_frac: float, dark_mean: float, chunk: int = 4096) -> pd.DataFrame:
    """Flag (never drop) non-tissue and dark/artefact patches. imgs: uint8 NxHxWx3."""
    tf, mi = [], []
    for a in range(0, len(imgs), chunk):
        x = imgs[a:a + chunk].astype(np.int16)
        mx, mn = x.max(-1), x.min(-1)
        sat = (mx - mn) / np.maximum(mx, 1)
        tf.append(((sat > 0.15) & (mx < 240)).mean((1, 2)))
        mi.append(x.mean((1, 2, 3)))
    tissue, inten = np.concatenate(tf), np.concatenate(mi)
    return pd.DataFrame({"tissue_frac": tissue.round(4), "mean_intensity": inten.round(2),
                         "flag_nontissue": tissue < min_tissue_frac, "flag_dark": inten < dark_mean})


def _decode(b: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def _extract_hf_shard(client, grp: pd.DataFrame, imgs: np.ndarray, wave: int = 16, workers: int = 8) -> None:
    """Fill imgs[grp.index] from one remote shard: only row groups containing selected rows are fetched."""
    import pyarrow.parquet as pq
    tail_start = max(0, client.size - (1 << 20))
    tail = client.fetch(tail_start, client.size)
    md = pq.ParquetFile(remote.SparseFile(client.size, [(tail_start, tail)])).metadata
    starts = np.cumsum([0] + [md.row_group(g).num_rows for g in range(md.num_row_groups)])
    todo = []
    for g in range(md.num_row_groups):
        sel = grp[(grp["row_in_shard"] >= starts[g]) & (grp["row_in_shard"] < starts[g + 1])]
        if sel.empty:
            continue
        rg, lo, hi = md.row_group(g), None, 0
        for ci in range(rg.num_columns):
            c = rg.column(ci)
            if c.path_in_schema.startswith("image"):
                o = min(x for x in (c.dictionary_page_offset, c.data_page_offset) if x)
                lo = o if lo is None else min(lo, o)
                hi = max(hi, o + c.total_compressed_size)
        todo.append((g, (lo, hi), sel))
    for w in range(0, len(todo), wave):
        batch = todo[w:w + wave]
        pieces = client.fetch_many([t[1] for t in batch], workers=workers)
        for (g, (lo, _), sel), data in zip(batch, pieces):
            pf = pq.ParquetFile(remote.SparseFile(client.size, [(tail_start, tail), (lo, data)]))
            col = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
            for idx, rr in zip(sel.index, sel["row_in_shard"] - starts[g]):
                imgs[idx] = _decode(col[int(rr)]["bytes"])


def extract_images(man: pd.DataFrame, cfg: dict, out_npz: Path, cache_dir: Path, log=print) -> pd.DataFrame:
    """Return the manifest with image-derived flags and write `images`+`uid` to out_npz.

    hf_mirror: streams only the row groups that contain selected patches; one small cache file per
    shard (selected patches only) makes the step resumable.
    """
    c = cfg["camelyon"]
    man = man.reset_index(drop=True)
    size = cfg["data"]["camelyon_size"]
    imgs = np.zeros((len(man), size, size, 3), dtype=np.uint8)
    if c["source"] == "synthetic":
        rng = np.random.default_rng(0)
        for i, (lab, hos) in enumerate(zip(man["label"], man["hospital"])):
            base = rng.integers(100, 230, 3) - 12 * int(hos)
            imgs[i] = np.clip(base + rng.normal(0, 25 if lab else 12, (size, size, 3)), 0, 255)
    elif c["source"] == "local_archive":
        from PIL import Image
        root = Path(c["archive_path"]) / "patches"
        for i, r in enumerate(man.itertuples()):
            nm = f"patient_{r.patient:03d}_node_{r.node}"
            imgs[i] = np.asarray(Image.open(root / nm / f"patch_{nm}_x_{r.x_coord}_y_{r.y_coord}.png").convert("RGB"))
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for shard, grp in man.groupby("shard", sort=True):
            part = cache_dir / (Path(shard).stem + ".npz")
            if part.exists():
                z = np.load(part)
                if np.array_equal(z["uid"], grp["uid"].values.astype(str)):
                    imgs[grp.index.to_numpy()] = z["images"]
                    log(f"  {shard}: loaded from cache")
                    continue
            url = f"https://huggingface.co/datasets/{c['repo']}/resolve/main/{shard}"
            _extract_hf_shard(remote.RangeClient(url), grp, imgs)
            np.savez(part, images=imgs[grp.index.to_numpy()], uid=grp["uid"].values.astype(str))
            log(f"  {shard}: {len(grp)} patches (transferred {remote.DOWNLOADED_BYTES / 2**30:.2f} GiB)")
    fl = tissue_flags(imgs, c["nontissue_min_tissue_frac"], c["dark_mean_intensity"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, images=imgs, uid=man["uid"].values.astype(str))
    return pd.concat([man, fl], axis=1)


# ------------------------------------------------------------------ spatial overlap --------
def spatial_overlap_report(meta: pd.DataFrame, patch: int = 96) -> pd.DataFrame:
    """Per slide: share of patches with another patch of the same slide overlapping it (|dx|,|dy| < patch)."""
    from scipy.spatial import cKDTree
    rows = []
    for slide, g in meta.groupby("slide"):
        xy = g[["x_coord", "y_coord"]].to_numpy(float)
        pairs = cKDTree(xy).query_pairs(r=patch - 1e-6, p=np.inf, output_type="ndarray")
        hit = np.zeros(len(g), bool)
        if len(pairs):
            hit[pairs[:, 0]] = True
            hit[pairs[:, 1]] = True
        rows.append({"slide": int(slide), "hospital": int(g["hospital"].iloc[0]), "n_patches": len(g),
                     "n_overlapping_pairs": int(len(pairs)), "frac_patches_overlapping": round(hit.mean(), 4)})
    return pd.DataFrame(rows)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
