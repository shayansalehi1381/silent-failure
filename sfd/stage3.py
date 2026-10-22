"""Stage 3: shift families A (MedMNIST-C), C (shortcut cue), plus B and X reused from stage 2.

Steps (each resumable through markers in outputs/stage3/markers/):
  sample    one stratified 3,000-image sample of PathMNIST id_test (uid list, committed)
  shortcut  5 shortcut models (cue aligned with p=0.95) + predictions in the aligned/removed/reversed states [GPU]
  corrupt   clean 224 sample (streamed), 224->128 correspondence check, 10 corruptions x 5 severities [CPU]
  inferA    the 5 standard PathMNIST models on clean + all corrupted conditions [GPU]
  tables    unified long table (A, B, C, X + ID references), summaries, figures, sanity checks
The frozen splits are only read. Prediction arrays are git-ignored; summaries and figures are committed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfd import calibration as cal
from sfd import shortcut as sc
from sfd.config import ROOT, resolve
from sfd.data import classify_data as cd
from sfd.data.audit import sha256_file, to_markdown
from sfd.data.splits import stratified_sample
from sfd.sysinfo import ram_gb

PM_CLASSES = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth_muscle", "normal_mucosa",
              "stroma", "tumor_epithelium"]


# --------------------------------------------------------------------------- helpers ---------
def P(cfg: dict, out_dir: Path) -> dict:
    s2c = Path(cfg["stage2"]["cache_dir"])
    return {"out": out_dir / "stage3", "stage1": out_dir / "stage1" / "splits.parquet", "stage2": out_dir / "stage2",
            "cache": ROOT / cfg["stage3"]["cache_dir"], "s2cache": s2c if s2c.is_absolute() else ROOT / s2c,
            "npz224": ROOT / cfg["stage3"]["data_224"]}


def marker(out: Path, name: str) -> Path:
    (out / "markers").mkdir(parents=True, exist_ok=True)
    return out / "markers" / f"{name}.done"


def save_preds(path: Path, lo, ft, labels, uid, patient, T: float) -> None:
    p = cal.probs(lo, T)
    pred = p.argmax(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, logits=lo.astype(np.float32), probs_cal=p.astype(np.float32), features=ft.astype(np.float16),
             labels=labels, uid=np.asarray(uid, dtype=str), patient=patient, T=np.float32(T), pred=pred,
             confidence=p.max(1).astype(np.float32), correct=pred == labels)


def ram_check(min_gb: float = 2.0) -> None:
    from sfd.train import GuardStop
    free = ram_gb().get("available")
    if free is not None and free < min_gb:
        raise GuardStop(f"free RAM {free} GB < {min_gb} GB")


# --------------------------------------------------------------------------- sample ----------
def make_sample(cfg: dict, out: Path, df: pd.DataFrame, log=print) -> pd.DataFrame:
    f = out / "sample_id_test_3000.csv"
    if f.exists():
        return pd.read_csv(f)
    s = cfg["stage3"]["sample"]
    fr = cd.split_frame(df, "pathmnist", s["split"]).assign(pos=lambda d: np.arange(len(d)))
    pick = stratified_sample(fr, s["n"], ["label"], s["seed"]).sort_values("pos")
    parts = pick["uid"].str.split("/")
    smp = pd.DataFrame({"uid": pick["uid"].values, "label": pick["label"].values, "pos": pick["pos"].values,
                        "src_split": parts.str[1].values, "src_index": parts.str[2].astype(int).values})
    out.mkdir(parents=True, exist_ok=True)
    smp.to_csv(f, index=False)
    log(f"[stage3] sample: {len(smp)} id_test images, per class {np.bincount(smp.label, minlength=9).tolist()}")
    return smp


def sample_images128(p: dict, df: pd.DataFrame, smp: pd.DataFrame) -> np.ndarray:
    """Stored 128 px id_test images of the sample (from the stage-2 cache, explicit row reads)."""
    rr = cd.RowReader(cd.split_file(p["s2cache"], "pathmnist", "id_test"))
    return rr[smp["pos"].to_numpy()]


# --------------------------------------------------------------------------- shortcut --------
def run_shortcut(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> None:
    import torch
    from sfd.train import GpuGuard, predict, train_one
    out, c = p["out"], cfg["stage3"]["shortcut"]
    seeds = [s for s in cfg["train"]["seeds"] if not marker(out, f"shortcut_seed{s}").exists()]
    if not seeds:
        return
    g = cfg["stage2"]["guard"]
    guard = GpuGuard(g["slowdown_margin_c"], g["min_free_ram_gb"], g["pause_s"], g["max_pause_s"], log)
    dev = torch.device("cuda")
    pal, off = sc.make_palette(9, c["cue_px"]), c["offset_px"]
    perm = sc.derangement(9, c["derangement_seed"])
    frames = {s: cd.split_frame(df, "pathmnist", s) for s in ("train", "val_es", "val_cal")}
    tr = cd.load_split(p["s2cache"], "pathmnist", "train", frames["train"])
    tr.images = cd.RowReader(cd.split_file(p["s2cache"], "pathmnist", "train"))
    cues_tr = sc.cue_assignment(tr.labels, c["p_aligned"], 9, c["cue_seed"])
    va = cd.load_split(p["s2cache"], "pathmnist", "val_es", frames["val_es"])
    va.images = sc.apply_cue_uint8(np.array(va.images), sc.cue_assignment(va.labels, c["p_aligned"], 9,
                                                                           c["cue_seed"] + 1), pal, off)
    vc = cd.load_split(p["s2cache"], "pathmnist", "val_cal", frames["val_cal"])
    vc.images = sc.apply_cue_uint8(np.array(vc.images), sc.cue_assignment(vc.labels, c["p_aligned"], 9,
                                                                           c["cue_seed"] + 2), pal, off)
    test = sample_images128(p, df, smp)
    lab = smp["label"].to_numpy(np.int64)
    pal_t = torch.from_numpy(pal).permute(0, 3, 1, 2).float().div(255).to(dev)
    cues_t = torch.from_numpy(cues_tr).to(dev)
    (out / "shortcut_design.json").write_text(json.dumps({
        "cue_px": c["cue_px"], "corner": "top-left", "offset_px": off, "p_aligned": c["p_aligned"],
        "reversed_mapping": {PM_CLASSES[i]: PM_CLASSES[int(perm[i])] for i in range(9)},
        "train_cue_aligned_frac": float((cues_tr == tr.labels).mean())}, indent=1))
    log(f"[stage3] shortcut: train cue aligned fraction {(cues_tr == tr.labels).mean():.4f}; reversal {perm.tolist()}")
    for seed in seeds:
        t0 = time.time()
        res = train_one(cfg, tr, va, 9, seed, out / "ckpt" / f"shortcut_seed{seed}", out / "logs" / f"shortcut_seed{seed}.jsonl",
                        guard, dev, log, post_augment=lambda x, idx: sc.apply_cue_tensor(x, cues_t[idx], pal_t, off))
        model = res["model"]
        lo, _ = predict(model, vc.images, dev)
        T = cal.fit_temperature(lo, vc.labels)
        for state in ("aligned", "removed", "reversed"):
            imgs = sc.apply_cue_uint8(test, sc.state_cues(lab, state, perm), pal, off)
            lo, ft = predict(model, imgs, dev)
            save_preds(out / "preds" / f"shortcut_seed{seed}_C_{state}_s0.npz", lo, ft, lab, smp["uid"], np.full(len(lab), -1), T)
        marker(out, f"shortcut_seed{seed}").write_text(json.dumps({"seed": seed, "T": T, "best_epoch": res["best_epoch"],
                                                                     "epochs_run": res["epochs_run"],
                                                                     "seconds": round(time.time() - t0, 1)}))
        log(f"[stage3] shortcut seed {seed}: done in {time.time() - t0:.0f} s, best epoch {res['best_epoch']}, T={T:.3f}")
        del model, res
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- corruptions -----
def extract_clean224(p: dict, smp: pd.DataFrame, log=print) -> Path:
    from numpy.lib.format import open_memmap
    from sfd.data.pathmnist import iter_npz_chunks
    f, ok = p["cache"] / "clean224.npy", p["cache"] / "clean224.ok.json"
    uh = hashlib.sha256("\n".join(smp["uid"]).encode()).hexdigest()[:16]
    if f.exists() and ok.exists() and json.loads(ok.read_text())["uid_hash"] == uh:
        return f
    p["cache"].mkdir(parents=True, exist_ok=True)
    mm = open_memmap(f, mode="w+", dtype=np.uint8, shape=(len(smp), 224, 224, 3))
    del mm
    from sfd.corrupt import RowIO
    io_ = RowIO(f)
    for key, g in smp.reset_index(drop=True).groupby("src_split"):
        idx = g["src_index"].to_numpy()
        order = np.argsort(idx)
        idx, pos = idx[order], g.index.to_numpy()[order]
        for start, chunk in iter_npz_chunks(p["npz224"], f"{key}_images", 128):   # 224-px stream, ~19 MB per chunk
            a, b = np.searchsorted(idx, [start, start + len(chunk)])
            for j in range(a, b):
                io_.write(int(pos[j]), chunk[idx[j] - start][None])
            ram_check()
        log(f"[stage3] clean224: {len(idx)} images from '{key}'")
    ok.write_text(json.dumps({"uid_hash": uh, "n": len(smp)}))
    return f


def correspondence(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, clean224: Path, log=print) -> str:
    from sfd.corrupt import RowIO, resize128
    f = p["out"] / "resize_correspondence.json"
    if f.exists():
        return json.loads(f.read_text())["chosen"]
    n = 300
    ref = sample_images128(p, df, smp.iloc[:n]).astype(np.float64)
    src = RowIO(clean224).read(0, n)
    rms = {}
    for m in cfg["stage3"]["corruptions"]["resize_candidates"]:
        r = np.stack([resize128(im, m) for im in src]).astype(np.float64)
        rms[m] = float(np.sqrt(((r - ref) ** 2).mean()))
    chosen = min(rms, key=rms.get)
    rep = {"n_images": n, "rms_levels_0_255": {k: round(v, 3) for k, v in rms.items()}, "chosen": chosen,
           "max_rms_levels": cfg["stage3"]["corruptions"]["max_rms_levels"]}
    p["out"].mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rep, indent=1))
    log(f"[stage3] 224->128 correspondence (RMS, 0-255 levels): {rep['rms_levels_0_255']} -> {chosen}")
    if rms[chosen] > rep["max_rms_levels"]:
        raise RuntimeError(f"224->128 resize does not match the stored 128 images (best RMS {rms[chosen]:.2f}); stopping")
    return chosen


def run_corruptions(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> None:
    from multiprocessing import get_context
    from numpy.lib.format import open_memmap
    from sfd.corrupt import RowIO, corrupt_one, corruptors, image_seed, resize128, task
    out, cc = p["out"], cfg["stage3"]["corruptions"]
    if marker(out, "corrupt").exists():
        return
    clean224 = extract_clean224(p, smp, log)
    method = correspondence(cfg, p, df, smp, clean224, log)
    adir = p["cache"] / "A"
    adir.mkdir(parents=True, exist_ok=True)
    uids = smp["uid"].tolist()
    n = len(uids)
    # severity 0 (clean) goes through the same resize as every corrupted condition
    clean = adir / "clean_s0.npy"
    if not (adir / "clean_s0.done").exists():
        mm = open_memmap(clean, mode="w+", dtype=np.uint8, shape=(n, 128, 128, 3))
        del mm
        src, dst = RowIO(clean224), RowIO(clean)
        for a in range(0, n, 200):
            dst.write(a, np.stack([resize128(im, method) for im in src.read(a, min(n, a + 200))]))
        (adir / "clean_s0.done").write_text(sha256_file(clean))
    names = [k for k in corruptors() if k not in cc["pending"]]
    todo = [(c, s) for c in names for s in cc["severities"] if not (adir / f"{c}_s{s}.done").exists()]
    log(f"[stage3] corruptions: {len(names)} types x {len(cc['severities'])} severities; {len(todo)} conditions to build "
        f"({cc['workers']} workers); pending (not run): {cc['pending']}")
    ctx = get_context("spawn")
    with ctx.Pool(cc["workers"]) as pool:
        for c, s in todo:
            # yield to a concurrent GPU job: wait (do not fail) while free RAM is below the CPU-job margin
            while (ram_gb().get("available") or 99) < cc.get("cpu_job_min_free_ram_gb", 3.0):
                log(f"[stage3] free RAM {ram_gb().get('available')} GB < {cc.get('cpu_job_min_free_ram_gb', 3.0)} GB: "
                    "corruption job waits 60 s")
                time.sleep(60)
            t0 = time.time()
            f = adir / f"{c}_s{s}.npy"
            mm = open_memmap(f, mode="w+", dtype=np.uint8, shape=(n, 128, 128, 3))
            del mm
            jobs = [(str(clean224), str(f), c, s, a, min(n, a + cc["chunk"]), uids[a:min(n, a + cc["chunk"])],
                     cc["seed"], method) for a in range(0, n, cc["chunk"])]
            for _ in pool.imap_unordered(task, jobs):
                pass
            (adir / f"{c}_s{s}.done").write_text(sha256_file(f))
            log(f"[stage3] corruption {c} s{s}: {time.time() - t0:.0f} s, free RAM {ram_gb().get('available')} GB")
    # determinism: rebuild the first k images of every condition serially and compare bytes
    k, cor, mism = cc["determinism_check_n"], corruptors(), []
    src = RowIO(clean224).read(0, k)
    for c in names:
        for s in cc["severities"]:
            redo = np.stack([resize128(corrupt_one(cor[c], src[i], s, image_seed(cc["seed"], c, s, uids[i])), method)
                             for i in range(k)])
            if not np.array_equal(redo, RowIO(adir / f"{c}_s{s}.npy").read(0, k)):
                mism.append(f"{c}_s{s}")
    rep = {"conditions": ["clean_s0"] + [f"{c}_s{s}" for c in names for s in cc["severities"]],
           "pending": cc["pending"], "resize": method, "determinism_checked_images_per_condition": k,
           "determinism_mismatches": mism,
           "sha256": {f.stem: (adir / f"{f.stem}.done").read_text().strip() for f in sorted(adir.glob("*.npy"))}}
    (out / "corruption_report.json").write_text(json.dumps(rep, indent=1))
    if mism:
        raise RuntimeError(f"corruption not deterministic for {mism}")
    marker(out, "corrupt").write_text(json.dumps({"n_conditions": len(rep["conditions"])}))
    log(f"[stage3] corruptions done; determinism check on {k} images x {len(rep['conditions']) - 1} conditions: OK")


# --------------------------------------------------------------------------- inference A -----
def run_infer_a(cfg: dict, p: dict, smp: pd.DataFrame, log=print) -> None:
    import torch
    from sfd.corrupt import RowIO
    from sfd.train import Net, predict
    out = p["out"]
    dev = torch.device("cuda")
    adir = p["cache"] / "A"
    conds = json.loads((out / "corruption_report.json").read_text())["conditions"]
    lab = smp["label"].to_numpy(np.int64)
    for seed in cfg["train"]["seeds"]:
        mk = marker(out, f"inferA_seed{seed}")
        if mk.exists():
            continue
        ram_check()
        T = json.loads((p["stage2"] / "markers" / f"pathmnist_seed{seed}.done").read_text())["T"]
        model = Net(9, pretrained=False).to(dev).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(p["stage2"] / "ckpt" / f"pathmnist_seed{seed}" / "best.pt", map_location=dev))
        t0 = time.time()
        for cond in conds:
            name, sev = cond.rsplit("_s", 1)
            lo, ft = predict(model, RowIO(adir / f"{cond}.npy").read(0, len(smp)), dev)
            save_preds(out / "preds" / f"standard_seed{seed}_A_{name}_s{sev}.npz", lo, ft, lab, smp["uid"],
                       np.full(len(lab), -1), T)
        mk.write_text(json.dumps({"seed": seed, "T": T, "n_conditions": len(conds), "seconds": round(time.time() - t0, 1)}))
        log(f"[stage3] inferA seed {seed}: {len(conds)} conditions in {time.time() - t0:.0f} s")
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- tables ----------
def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE from confidence and correctness (equal-width bins), same definition as calibration.ece."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def _collect(cfg: dict, p: dict) -> list[dict]:
    """Every prediction file of every family, with its family/condition/severity/model labels."""
    items = []
    import re
    pat = re.compile(r"^(?P<kind>[a-z0-9]+)_seed(?P<seed>\d+)_(?P<fam>[A-Z]+)_(?P<cond>.+)_s(?P<sev>\d+)$")
    for f in sorted((p["out"] / "preds").glob("*.npz")):
        m = pat.match(f.stem)
        assert m, f"unexpected prediction file name {f.name}"
        items.append({"path": f, "family": m["fam"], "condition": m["cond"], "severity": int(m["sev"]),
                      "model_kind": m["kind"], "dataset": "pathmnist", "seed": int(m["seed"])})
    for seed in cfg["train"]["seeds"]:                                   # stage-2 predictions (not recomputed)
        for ds, split, fam, cond in (("pathmnist", "id_test", "ID", "id_test"), ("pathmnist", "nct7k_test", "X", "nct7k_test"),
                                     ("camelyon17", "id_test", "ID", "id_test"), ("camelyon17", "ood_c1", "B", "ood_c1"),
                                     ("camelyon17", "ood_c2", "B", "ood_c2")):
            items.append({"path": p["stage2"] / "preds" / f"{ds}_seed{seed}_{split}.npz", "family": fam,
                          "condition": cond, "severity": 0, "model_kind": "standard", "dataset": ds, "seed": seed})
    return items


def run_tables(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, thr = p["out"], cfg["stage3"]["silent_failure_conf"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, long_parts = [], []
    sanity = {"nan_found": False, "uid_mismatch_files": [], "n_files": 0}
    uid_ref = smp["uid"].to_numpy(dtype=str)
    for it in _collect(cfg, p):
        z = np.load(it["path"])
        pr = z["probs_cal"].astype(np.float64)
        lab = z["labels"]
        conf, pred = pr.max(1), pr.argmax(1)
        corr = pred == lab
        sanity["n_files"] += 1
        if not (np.isfinite(z["logits"]).all() and np.isfinite(pr).all() and np.isfinite(z["features"].astype(np.float32)).all()):
            sanity["nan_found"] = True
        if it["family"] in ("A", "C") and not np.array_equal(z["uid"].astype(str), uid_ref):
            sanity["uid_mismatch_files"].append(it["path"].name)
        sf = (~corr) & (conf >= thr)
        meta = {k: it[k] for k in ("family", "condition", "severity", "model_kind", "dataset", "seed")}
        rows.append({**meta, "n": len(lab), "acc": corr.mean(), "mean_conf": conf.mean(), "ece": _ece(conf, corr),
                     "sf_rate": sf.mean(), "sf_among_errors": sf.sum() / max(1, (~corr).sum())})
        long_parts.append(pd.DataFrame({**meta, "uid": z["uid"].astype(str), "patient": z["patient"], "label": lab,
                                        "pred": pred, "confidence": conf.astype(np.float32), "correct": corr,
                                        "silent_failure_provisional": sf}))
    long = pd.concat(long_parts, ignore_index=True)
    long.to_parquet(out / "predictions_long.parquet", index=False)              # git-ignored (full table)
    per = pd.DataFrame(rows)
    per.to_csv(out / "metrics_per_seed.csv", index=False)
    keys = ["family", "condition", "severity", "model_kind", "dataset"]
    agg = per.groupby(keys).agg(n_seeds=("seed", "nunique"), n=("n", "first"),
                                acc_mean=("acc", "mean"), acc_std=("acc", "std"), conf_mean=("mean_conf", "mean"),
                                conf_std=("mean_conf", "std"), ece_mean=("ece", "mean"), ece_std=("ece", "std"),
                                sf_rate_mean=("sf_rate", "mean"), sf_rate_std=("sf_rate", "std"),
                                sf_among_errors_mean=("sf_among_errors", "mean")).reset_index()
    agg.round(4).to_csv(out / "condition_summary.csv", index=False)               # committed (small)

    # per-family summary: A pooled per severity (mean over corruptions first, then over seeds)
    a = per[per.family == "A"].copy()
    a_sev = a.groupby(["severity", "seed"])[["acc", "mean_conf", "ece", "sf_rate"]].mean().reset_index()
    fam_rows = []
    for sev, g in a_sev.groupby("severity"):
        fam_rows.append({"family": "A", "condition": "all corruptions" if sev else "clean", "severity": sev,
                         "dataset": "pathmnist", "model_kind": "standard", **_ms(g)})
    for (fam, cond, ds, kind), g in per[per.family != "A"].groupby(["family", "condition", "dataset", "model_kind"]):
        fam_rows.append({"family": fam, "condition": cond, "severity": 0, "dataset": ds, "model_kind": kind, **_ms(g)})
    fam = pd.DataFrame(fam_rows).sort_values(["dataset", "family", "severity"])
    fam.to_csv(out / "family_summary.csv", index=False)
    corr_tab = agg[agg.family == "A"].pivot_table(index="condition", columns="severity", values="acc_mean").round(4)
    corr_tab.to_csv(out / "per_corruption_accuracy.csv")
    sf_tab = agg[agg.family == "A"].pivot_table(index="condition", columns="severity", values="sf_rate_mean").round(4)
    sf_tab.to_csv(out / "per_corruption_sf_rate.csv")
    shortcut = fam[fam.family == "C"][["model_kind", "condition", "acc", "acc_std", "conf", "ece", "sf_rate", "sf_rate_std"]]
    shortcut.to_csv(out / "shortcut_states.csv", index=False)

    # sanity: monotone accuracy decline with severity, shortcut ordering, determinism, seed spread
    mono = {}
    for cond, g in agg[agg.family == "A"].groupby("condition"):
        if cond == "clean":
            continue
        accs = [float(agg[(agg.family == "A") & (agg.condition == "clean")]["acc_mean"].iloc[0])] + \
               g.sort_values("severity")["acc_mean"].tolist()
        rises = [i for i in range(1, len(accs)) if accs[i] > accs[i - 1] + 0.005]
        mono[cond] = {"acc_by_severity_0_5": [round(x, 4) for x in accs], "rises_gt_0.5pt_at_severity": rises,
                      "drop_0_to_5_pts": round(100 * (accs[0] - accs[-1]), 2)}
    sc_acc = {r.condition: r.acc for r in shortcut[shortcut.model_kind == "shortcut"].itertuples()} if len(shortcut) else {}
    sc_acc2 = {r.condition: r.acc for r in shortcut[shortcut.model_kind == "shortcutv2"].itertuples()}
    rep = json.loads((out / "corruption_report.json").read_text()) if (out / "corruption_report.json").exists() else {}
    sanity.update({"monotonicity": mono, "non_monotone_corruptions": [c for c, v in mono.items() if v["rises_gt_0.5pt_at_severity"]],
                   "shortcut_order_aligned_gt_removed_gt_reversed": bool(sc_acc and sc_acc.get("aligned", 0) > sc_acc.get("removed", 0)
                                                                         > sc_acc.get("reversed", 1)),
                   "shortcutv2_order_aligned_gt_removed_gt_reversed": bool(sc_acc2 and sc_acc2.get("aligned", 0) > sc_acc2.get("removed", 0) > sc_acc2.get("reversed", 1)),
                   "same_sample_everywhere": not sanity["uid_mismatch_files"], "n_sample": len(smp),
                   "determinism_mismatches": rep.get("determinism_mismatches"),
                   "max_seed_std_acc": float(agg["acc_std"].max())})
    (out / "sanity_stage3.json").write_text(json.dumps(sanity, indent=1))

    _figures(cfg, p, agg, a_sev, fam, smp, plt)
    md = ["# Stage 3 - behaviour under shift (mean Â± std over seeds; provisional silent failure = wrong AND calibrated "
          f"confidence >= {thr})\n", "## Per family\n", to_markdown(_fmt(fam)),
          "\n## Shortcut states (family C = reversed)\n", to_markdown(_fmt(fam[fam.family == "C"])),
          "\n## Accuracy per corruption and severity (mean over seeds)\n", to_markdown(corr_tab.reset_index()),
          "\n## Silent-failure rate per corruption and severity\n", to_markdown(sf_tab.reset_index())]
    (out / "summary_stage3.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[stage3] tables: {sanity['n_files']} prediction files, {len(long):,} rows; nan={sanity['nan_found']}, "
        f"uid mismatches={len(sanity['uid_mismatch_files'])}, non-monotone={sanity['non_monotone_corruptions']}")
    return sanity


def _ms(g: pd.DataFrame) -> dict:
    out = {}
    for k, col in (("acc", "acc"), ("conf", "mean_conf"), ("ece", "ece"), ("sf_rate", "sf_rate")):
        out[k], out[k + "_std"] = float(g[col].mean()), float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
    out["n_seeds"] = int(g["seed"].nunique())
    return out


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    for k in ("acc", "conf", "ece", "sf_rate"):
        t[k] = [f"{m:.4f} Â± {s:.4f}" for m, s in zip(t[k], t[k + "_std"])]
    return t[[c for c in t.columns if not c.endswith("_std")]]


def _figures(cfg, p, agg, a_sev, fam, smp, plt) -> None:
    out = p["out"] / "figures"
    metrics = [("acc", "accuracy"), ("mean_conf", "mean confidence"), ("ece", "ECE"), ("sf_rate", "silent-failure rate")]
    # 1) family A: mean over corruptions vs severity (Â± std over seeds)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for k, (col, name) in enumerate(metrics):
        g = a_sev.groupby("severity")[col].agg(["mean", "std"])
        ax[k].errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3)
        ax[k].set_xlabel("severity (0 = clean)"); ax[k].set_title(f"A (PathMNIST): {name}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "behaviour_family_A.png", dpi=130); plt.close(fig)
    # 2) families B, C, X and their ID references
    sel = fam[fam.family != "A"].copy()
    sel["label"] = sel["dataset"].str[:4] + "/" + sel["family"] + ":" + sel["condition"] + "/" + sel["model_kind"].str[:5]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for k, (col, name) in enumerate([("acc", "accuracy"), ("conf", "mean confidence"), ("ece", "ECE"), ("sf_rate", "silent-failure rate")]):
        ax[k].barh(sel["label"], sel[col], xerr=sel[col + "_std"], capsize=2)
        ax[k].set_title(name, fontsize=9); ax[k].tick_params(labelsize=7)
        if k:
            ax[k].set_yticklabels([])
    fig.tight_layout(); fig.savefig(out / "behaviour_families_BCX.png", dpi=130); plt.close(fig)
    # 3) supplement: per-corruption severity curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    a = agg[agg.family == "A"]
    clean = a[a.condition == "clean"].iloc[0]
    for cond, g in a[a.condition != "clean"].groupby("condition"):
        g = g.sort_values("severity")
        for k, col in enumerate(("acc_mean", "sf_rate_mean")):
            ax[k].plot([0] + g["severity"].tolist(), [clean[col]] + g[col].tolist(), marker=".", label=cond)
    ax[0].set_title("accuracy per corruption"); ax[1].set_title("silent-failure rate per corruption")
    for x in ax:
        x.set_xlabel("severity")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "supplement_per_corruption.png", dpi=130); plt.close(fig)
    # 4) cue palette and 5) example images
    c = cfg["stage3"]["shortcut"]
    pal = sc.make_palette(9, c["cue_px"])
    fig, ax = plt.subplots(1, 9, figsize=(11, 1.8))
    for k in range(9):
        ax[k].imshow(pal[k], interpolation="nearest"); ax[k].set_title(PM_CLASSES[k], fontsize=7); ax[k].axis("off")
    fig.tight_layout(); fig.savefig(out / "cue_palette.png", dpi=130); plt.close(fig)
    v2f = p["out"] / "shortcut_v2_design.json"
    v2 = json.loads(v2f.read_text()) if v2f.exists() else None
    if v2:
        pal2 = sc.palette_for(v2)
        fig, ax = plt.subplots(1, 9, figsize=(11, 1.8))
        for k in range(9):
            ax[k].imshow(pal2[k], interpolation="nearest"); ax[k].set_title(PM_CLASSES[k], fontsize=7); ax[k].axis("off")
        fig.suptitle(f"family C v2 cues ({v2['name']}: {v2['cue_px']} px, frame {v2.get('frame_px', 0)} px, p={v2['p_aligned']})", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "cue_palette_v2.png", dpi=130); plt.close(fig)
    from sfd.corrupt import RowIO
    adir = p["cache"] / "A"
    i = 0
    lab = int(smp["label"].iloc[i])
    perm = sc.derangement(9, c["derangement_seed"])
    clean_img = RowIO(adir / "clean_s0.npy").read(i, i + 1)
    panels = [("clean", clean_img[0])]
    for name in ("bubble", "stain_deposit", "defocus_blur", "jpeg_compression", "contrast_down"):
        if (adir / f"{name}_s5.npy").exists():
            panels.append((f"{name} s5", RowIO(adir / f"{name}_s5.npy").read(i, i + 1)[0]))
    for state in ("aligned", "removed", "reversed"):
        panels.append((f"cue {state}", sc.apply_cue_uint8(clean_img, sc.state_cues(np.array([lab]), state, perm), pal,
                                                         c["offset_px"])[0]))
    if v2:
        for state in ("aligned", "removed", "reversed"):
            panels.append((f"v2 cue {state}", sc.apply_cue_uint8(clean_img, sc.state_cues(np.array([lab]), state, perm),
                                                                 sc.palette_for(v2), c["offset_px"])[0]))
    fig, ax = plt.subplots(1, len(panels), figsize=(2 * len(panels), 2.3))
    for x, (t, im) in zip(ax, panels):
        x.imshow(im); x.set_title(t, fontsize=7); x.axis("off")
    fig.suptitle(f"example: {smp['uid'].iloc[i]} (class {PM_CLASSES[lab]})", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "examples.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- shortcut v2 -----
def _train_shortcut(cfg: dict, p: dict, df: pd.DataFrame, setting: dict, seed: int, run_name: str, epochs: int,
                    log=print):
    """Train one shortcut model for `setting` ({p_aligned, style, cue_px, frame_px}); return (model, info, ctx)."""
    import copy
    import torch
    from sfd.train import GpuGuard, train_one
    out, c = p["out"], cfg["stage3"]["shortcut"]
    g = cfg["stage2"]["guard"]
    guard = GpuGuard(g["slowdown_margin_c"], g["min_free_ram_gb"], g["pause_s"], g["max_pause_s"], log)
    dev = torch.device("cuda")
    pal, off = sc.palette_for(setting), c["offset_px"]
    frames = {s: cd.split_frame(df, "pathmnist", s) for s in ("train", "val_es", "val_cal")}
    tr = cd.load_split(p["s2cache"], "pathmnist", "train", frames["train"])
    tr.images = cd.RowReader(cd.split_file(p["s2cache"], "pathmnist", "train"))
    cues_tr = sc.cue_assignment(tr.labels, setting["p_aligned"], 9, c["cue_seed"])
    raw = {}
    sets = {}
    for k, s in enumerate(("val_es", "val_cal"), 1):
        d = cd.load_split(p["s2cache"], "pathmnist", s, frames[s])
        raw[s] = np.array(d.images)
        d.images = sc.apply_cue_uint8(raw[s], sc.cue_assignment(d.labels, setting["p_aligned"], 9, c["cue_seed"] + k), pal, off)
        sets[s] = d
    pal_t = torch.from_numpy(pal).permute(0, 3, 1, 2).float().div(255).to(dev)
    cues_t = torch.from_numpy(cues_tr).to(dev)
    cfg2 = copy.deepcopy(cfg)
    cfg2["train"]["epochs"] = epochs
    res = train_one(cfg2, tr, sets["val_es"], 9, seed, out / "ckpt" / run_name, out / "logs" / f"{run_name}.jsonl", guard,
                    dev, log, post_augment=lambda x, idx: sc.apply_cue_tensor(x, cues_t[idx], pal_t, off))
    return res, {"pal": pal, "off": off, "sets": sets, "raw": raw, "dev": dev}


def run_shortcut_pilot(cfg: dict, p: dict, df: pd.DataFrame, log=print) -> dict:
    """Pre-registered ladder (seed 0, short training): stop at the first setting whose accuracy on val_es with the cue
    REVERSED is <= the target. Only val_es is used for this choice (never id_test or any shifted data)."""
    import torch
    from sfd.train import predict
    out, v2 = p["out"], cfg["stage3"]["shortcut_v2"]
    f = out / "shortcut_v2_pilot.json"
    rep = json.loads(f.read_text()) if f.exists() else {"ladder": [], "chosen": None}
    if rep["chosen"]:
        return rep
    perm = sc.derangement(9, cfg["stage3"]["shortcut"]["derangement_seed"])
    done = {r["name"] for r in rep["ladder"]}
    for st in v2["ladder"]:
        if st["name"] in done:
            r = next(r for r in rep["ladder"] if r["name"] == st["name"])
        else:
            t0 = time.time()
            res, ctx = _train_shortcut(cfg, p, df, st, 0, f"pilot_{st['name']}_seed0", v2["pilot_epochs"], log)
            lab = ctx["sets"]["val_es"].labels
            accs = {}
            for state in ("aligned", "removed", "reversed"):
                imgs = sc.apply_cue_uint8(ctx["raw"]["val_es"], sc.state_cues(lab, state, perm), ctx["pal"], ctx["off"])
                lo, _ = predict(res["model"], imgs, ctx["dev"])
                accs[state] = float((lo.argmax(1) == lab).mean())
            r = {**st, "val_es_acc": accs, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                 "seconds": round(time.time() - t0, 1)}
            rep["ladder"].append(r)
            f.write_text(json.dumps(rep, indent=1))
            log(f"[stage3] pilot {st['name']}: val_es acc aligned {accs['aligned']:.4f} removed {accs['removed']:.4f} "
                f"reversed {accs['reversed']:.4f} ({r['seconds']:.0f} s)")
            del res
            torch.cuda.empty_cache()
        if r["val_es_acc"]["reversed"] <= v2["target_reversed_acc"]:
            rep["chosen"], rep["reason"] = st["name"], f"first setting with val_es reversed acc <= {v2['target_reversed_acc']}"
            break
    if not rep["chosen"]:
        rep["chosen"], rep["reason"] = v2["ladder"][-1]["name"], "no setting reached the target; last rung taken (pre-registered)"
    f.write_text(json.dumps(rep, indent=1))
    log(f"[stage3] pilot chosen: {rep['chosen']} ({rep['reason']})")
    return rep


def run_shortcut_v2(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> None:
    import torch
    from sfd.train import predict
    out, v2 = p["out"], cfg["stage3"]["shortcut_v2"]
    rep = run_shortcut_pilot(cfg, p, df, log)
    st = next(s for s in v2["ladder"] if s["name"] == rep["chosen"])
    perm = sc.derangement(9, cfg["stage3"]["shortcut"]["derangement_seed"])
    lab = smp["label"].to_numpy(np.int64)
    test = sample_images128(p, df, smp)
    (out / "shortcut_v2_design.json").write_text(json.dumps({**st, "offset_px": cfg["stage3"]["shortcut"]["offset_px"],
                                                             "reversed_mapping": {PM_CLASSES[i]: PM_CLASSES[int(perm[i])] for i in range(9)}}, indent=1))
    for seed in cfg["train"]["seeds"]:
        mk = marker(out, f"shortcutv2_seed{seed}")
        if mk.exists():
            continue
        t0 = time.time()
        res, ctx = _train_shortcut(cfg, p, df, st, seed, f"shortcutv2_seed{seed}", cfg["train"]["epochs"], log)
        model = res["model"]
        vc = ctx["sets"]["val_cal"]
        lo, _ = predict(model, vc.images, ctx["dev"])
        T = cal.fit_temperature(lo, vc.labels)
        for state in ("aligned", "removed", "reversed"):
            imgs = sc.apply_cue_uint8(test, sc.state_cues(lab, state, perm), ctx["pal"], ctx["off"])
            lo, ft = predict(model, imgs, ctx["dev"])
            save_preds(out / "preds" / f"shortcutv2_seed{seed}_C_{state}_s0.npz", lo, ft, lab, smp["uid"], np.full(len(lab), -1), T)
        mk.write_text(json.dumps({"seed": seed, "setting": st["name"], "T": T, "best_epoch": res["best_epoch"],
                                  "epochs_run": res["epochs_run"], "seconds": round(time.time() - t0, 1)}))
        log(f"[stage3] shortcutv2 seed {seed}: done in {time.time() - t0:.0f} s, best epoch {res['best_epoch']}, T={T:.3f}")
        del model, res
        torch.cuda.empty_cache()


