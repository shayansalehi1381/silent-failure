"""Stage 4: monitoring references and signals (Blueprint Sec 4), frozen acceptance threshold (Sec 5.2).

Steps (markers in outputs/stage4/markers/):
  refs     per model (dataset x model_kind x seed): reference objects fitted ONLY on source TRAINING images
           (features, class Gaussians, kNN bank, Grad-CAM class references, transform variability). Then the
           'refs_frozen' marker with checksums is written BEFORE any shifted-family file is opened.
  pilot    Grad-CAM speed pilot -> full sets or one fixed stratified 1,000-per-condition subset (decision saved)
  signals  every Stage-3 row (A, C) plus B, X, ID references and val_cal: uncertainty, representation and
           explanation signals; val_cal percentiles and robust z; frozen tau; silent-failure labels
  tables   tidy monitoring table, validity checks, dose-response, descriptive per-signal AUROC, figures
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
from sfd.config import ROOT
from sfd.data import classify_data as cd
from sfd.data.audit import sha256_file, to_markdown
from sfd.data.splits import stratified_sample
from sfd.signals import explanation as ex
from sfd.signals import normalize as nz
from sfd.signals import representation as rp
from sfd.signals import uncertainty as un
from sfd.sysinfo import ram_gb

SIGNALS = ["u_1msp", "u_entropy", "u_energy", "r_maha_pred", "r_maha_min", "r_knn",
           "e_js_norm_l4", "e_js_raw_l4", "e_pca_l4", "e_js_norm_l3", "e_js_raw_l3", "e_pca_l3"]
PRIMARY = {"uncertainty": "u_1msp", "representation": "r_maha_pred", "explanation": "e_js_norm_l4"}
FORBIDDEN = ("stage3_cache", "nct7k", "ood_c", "id_test", "val_cal", "val_es", "/preds/", "\\preds\\", "pathmnist_224")


# --------------------------------------------------------------------------- setup -----------
def P(cfg: dict, out_dir: Path) -> dict:
    s2c = Path(cfg["stage2"]["cache_dir"])
    return {"out": out_dir / "stage4", "stage1": out_dir / "stage1" / "splits.parquet", "stage2": out_dir / "stage2",
            "stage3": out_dir / "stage3", "s2cache": s2c if s2c.is_absolute() else ROOT / s2c,
            "s3cache": ROOT / cfg["stage3"]["cache_dir"]}


def marker(out: Path, name: str) -> Path:
    (out / "markers").mkdir(parents=True, exist_ok=True)
    return out / "markers" / f"{name}.done"


def model_specs(cfg: dict, p: dict) -> list[dict]:
    """All frozen classifiers: standard (PathMNIST, Camelyon), shortcut v1 (95%/16px) and v2 (chosen pilot setting)."""
    specs = []
    for seed in cfg["train"]["seeds"]:
        for ds in ("pathmnist", "camelyon17"):
            mk = json.loads((p["stage2"] / "markers" / f"{ds}_seed{seed}.done").read_text())
            specs.append({"name": f"{ds}_standard_seed{seed}", "dataset": ds, "kind": "standard", "seed": seed,
                          "ckpt": p["stage2"] / "ckpt" / f"{ds}_seed{seed}" / "best.pt", "T": mk["T"],
                          "n_classes": cd.N_CLASSES[ds], "cue": None})
        v1 = {"p_aligned": cfg["stage3"]["shortcut"]["p_aligned"], "style": "v1", "cue_px": cfg["stage3"]["shortcut"]["cue_px"],
              "frame_px": 0}
        for kind, run, setting in (("shortcut", f"shortcut_seed{seed}", v1), ("shortcutv2", f"shortcutv2_seed{seed}", None)):
            mkf = p["stage3"] / "markers" / f"{run}.done"
            if not mkf.exists():
                continue
            if setting is None:
                chosen = json.loads((p["stage3"] / "shortcut_v2_pilot.json").read_text())["chosen"]
                setting = next(s for s in cfg["stage3"]["shortcut_v2"]["ladder"] if s["name"] == chosen)
            specs.append({"name": f"pathmnist_{kind}_seed{seed}", "dataset": "pathmnist", "kind": kind, "seed": seed,
                          "ckpt": p["stage3"] / "ckpt" / run / "best.pt", "T": json.loads(mkf.read_text())["T"],
                          "n_classes": 9, "cue": setting})
    return specs


def reference_inputs(cfg: dict, p: dict, spec: dict) -> list[Path]:
    """The ONLY files the reference fitting may read: the source-train image cache, the frozen checkpoint and the
    frozen split table (for train labels). Asserted: no shifted-family, test, calibration or prediction file."""
    files = [cd.split_file(p["s2cache"], spec["dataset"], "train"), Path(spec["ckpt"]), p["stage1"]]
    for f in files[:2]:
        s = str(f).replace("\\", "/")
        assert not any(b.replace("\\", "/") in s for b in FORBIDDEN), f"reference input not allowed: {f}"
    assert files[0].name.endswith("_train.npy")
    return files

def load_model(spec: dict, dev):
    import torch
    from sfd.train import Net
    m = Net(spec["n_classes"], pretrained=False).to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(spec["ckpt"], map_location=dev))
    m.eval()
    return m


def _batches(images, cue_ids, palette, offset, dev, bs: int = 256):
    """Yield normalised NCHW batches (cue pasted on the uint8 images first when cue_ids is given)."""
    import torch
    from sfd.train import MEAN, STD
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    for a in range(0, len(images), bs):
        u8 = np.asarray(images[a:a + bs])
        if cue_ids is not None:
            u8 = sc.apply_cue_uint8(u8, cue_ids[a:a + bs], palette, offset)
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(dev).permute(0, 3, 1, 2).float().div_(255)
        yield ((x - mean) / std).contiguous(memory_format=torch.channels_last), a


def run_gradcam(model, images, dev, cue_ids=None, palette=None, offset=0, bs: int = 256) -> dict:
    parts = {k: [] for k in ("logits", "features", *ex.LAYERS)}
    for x, _ in _batches(images, cue_ids, palette, offset, dev, bs):
        r = ex.gradcam_batch(model, x)
        for k in parts:
            parts[k].append(r[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


# --------------------------------------------------------------------------- references ------
def _ref_index(cfg: dict, frame: pd.DataFrame, ds: str) -> np.ndarray:
    n = cfg["stage4"]["ref_n"][ds]
    fr = frame.assign(pos=np.arange(len(frame)))
    if n is None or n >= len(fr):
        return fr["pos"].to_numpy()
    return np.sort(stratified_sample(fr, n, ["label"], cfg["stage4"]["seed"])["pos"].to_numpy())


def _transform_stats(model, imgs_u8, labels, dev, cue_ids, palette, offset, n_classes: int) -> dict:
    """Per class and layer: median and MAD of JS(CAM of the transformed image mapped back, CAM of the original),
    pooled over all label-preserving transforms; plus the mean JS per transform (reported)."""
    import torch
    from sfd.train import MEAN, STD
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    pal_t = torch.from_numpy(palette).permute(0, 3, 1, 2).float().div(255).to(dev) if cue_ids is not None else None
    per = {ln: np.zeros((len(imgs_u8), len(ex.TRANSFORMS))) for ln in ex.LAYERS}
    for a in range(0, len(imgs_u8), 128):
        u8 = torch.from_numpy(np.ascontiguousarray(imgs_u8[a:a + 128])).to(dev).permute(0, 3, 1, 2).float().div_(255)
        cid = torch.from_numpy(cue_ids[a:a + 128]).to(dev) if cue_ids is not None else None

        def run(x):
            if cid is not None:                       # the cue stays at its fixed position, as in training
                x = sc.apply_cue_tensor(x.clone(), cid, pal_t, offset)
            return ex.gradcam_batch(model, ((x - mean) / std).contiguous(memory_format=torch.channels_last))
        base = run(u8)
        for ti, t in enumerate(ex.TRANSFORMS):
            r = run(ex.apply_transform(u8, t))
            for ln in ex.LAYERS:
                per[ln][a:a + len(u8), ti] = ex.js_divergence(ex.invert_cam(r[ln], t), base[ln])
    stats = {}
    for ln in ex.LAYERS:
        med = np.array([np.median(per[ln][labels == c]) for c in range(n_classes)])
        mad = np.array([np.median(np.abs(per[ln][labels == c] - med[c])) for c in range(n_classes)])
        stats[ln] = {"median": med, "mad": mad, "per_transform_mean": per[ln].mean(0)}
    return stats

def fit_references(cfg: dict, p: dict, df: pd.DataFrame, log=print) -> None:
    import torch
    out = p["out"]
    if marker(out, "refs_frozen").exists():
        return
    dev = torch.device("cuda")
    s4 = cfg["stage4"]
    (out / "refs").mkdir(parents=True, exist_ok=True)
    for spec in model_specs(cfg, p):
        f = out / "refs" / f"{spec['name']}.npz"
        if f.exists():
            continue
        inputs = reference_inputs(cfg, p, spec)                      # asserted: train cache + checkpoint only
        t0 = time.time()
        frame = cd.split_frame(df, spec["dataset"], "train")
        idx = _ref_index(cfg, frame, spec["dataset"])
        rr = cd.RowReader(inputs[0])
        labels = frame["label"].to_numpy(np.int64)
        cue_all = pal = None
        off = cfg["stage3"]["shortcut"]["offset_px"]
        if spec["cue"]:
            cue_all = sc.cue_assignment(labels, spec["cue"]["p_aligned"], 9, cfg["stage3"]["shortcut"]["cue_seed"])
            pal = sc.palette_for(spec["cue"])
        model = load_model(spec, dev)
        parts = {k: [] for k in ("logits", "features", *ex.LAYERS)}
        for a in range(0, len(idx), 2048):
            ii = idx[a:a + 2048]
            r = run_gradcam(model, rr[ii], dev, None if cue_all is None else cue_all[ii], pal, off)
            for k in parts:
                parts[k].append(r[k])
            if (ram_gb().get("available") or 99) < 2.0:
                raise RuntimeError("free RAM < 2 GB during reference fitting")
        R = {k: np.concatenate(v) for k, v in parts.items()}
        lab, pred = labels[idx], R["logits"].argmax(1)
        rep_ref = rp.fit_class_gaussian(R["features"].astype(np.float32), lab, spec["n_classes"])
        cam_refs = {ln: ex.fit_cam_reference(R[ln], lab, pred, spec["n_classes"]) for ln in ex.LAYERS}
        sub = np.sort(stratified_sample(pd.DataFrame({"label": lab}), s4["transform_n"], ["label"], s4["seed"]).index.to_numpy())
        tstats = _transform_stats(model, rr[idx[sub]], lab[sub], dev, None if cue_all is None else cue_all[idx[sub]], pal,
                                  off, spec["n_classes"])
        arrays = {"ref_uid_pos": idx, "knn_bank": R["features"].astype(np.float16), "means": rep_ref["means"],
                  "precision": rep_ref["precision"], "lw_shrinkage": rep_ref["shrinkage"], "ref_train_acc": (pred == lab).mean()}
        for ln in ex.LAYERS:
            for k, v in cam_refs[ln].items():
                arrays[f"{ln}_{k}"] = v
            arrays[f"{ln}_t_median"], arrays[f"{ln}_t_mad"] = tstats[ln]["median"], tstats[ln]["mad"]
            arrays[f"{ln}_t_per_transform"] = tstats[ln]["per_transform_mean"]
        np.savez(f, **arrays)
        log(f"[stage4] refs {spec['name']}: {len(idx)} train images, train acc {(pred == lab).mean():.4f}, "
            f"LW shrinkage {rep_ref['shrinkage']:.3f}, PCA k l4={cam_refs['layer4']['pca_k']} ({time.time() - t0:.0f} s)")
        del model
        torch.cuda.empty_cache()
    sums = {f.name: sha256_file(f) for f in sorted((out / "refs").glob("*.npz"))}
    pd.DataFrame({"file": list(sums), "sha256": list(sums.values())}).to_csv(out / "refs_checksums.csv", index=False)
    marker(out, "refs_frozen").write_text(json.dumps({"n_models": len(sums), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                      "sha256": sums}, indent=1))
    log(f"[stage4] references FROZEN: {len(sums)} models (refs_frozen marker written before any shifted file is read)")


def load_ref(p: dict, spec: dict) -> dict:
    z = np.load(p["out"] / "refs" / f"{spec['name']}.npz")
    r = {"rep": {"means": z["means"], "precision": z["precision"]}, "bank": z["knn_bank"]}
    for ln in ex.LAYERS:
        r[ln] = {"mean_cam": z[f"{ln}_mean_cam"], "pca_mu": z[f"{ln}_pca_mu"], "pca_components": z[f"{ln}_pca_components"],
                 "pca_class_means": z[f"{ln}_pca_class_means"], "pca_precision": z[f"{ln}_pca_precision"]}
        r[ln + "_t"] = {"median": z[f"{ln}_t_median"], "mad": z[f"{ln}_t_mad"]}
    return r


# --------------------------------------------------------------------------- signals ---------
def compute_signals(R: dict, T: float, ref: dict) -> pd.DataFrame:
    """All raw signals for one set (R = output of run_gradcam)."""
    lo = R["logits"]
    pr = cal.probs(lo, T)
    pred = pr.argmax(1)
    feats = R["features"].astype(np.float32)
    mp, mm = rp.mahalanobis_scores(feats, pred, ref["rep"])
    d = {"pred": pred, "confidence": pr.max(1), "u_1msp": un.one_minus_msp(pr), "u_entropy": un.entropy(pr),
         "u_energy": un.energy(lo, T), "r_maha_pred": mp, "r_maha_min": mm, "r_knn": rp.knn_distance(feats, ref["bank"], 10)}
    for ln, tag in (("layer4", "l4"), ("layer3", "l3")):
        s = ex.explanation_scores(R[ln], pred, ref[ln], ref[ln + "_t"])
        d[f"e_js_raw_{tag}"], d[f"e_js_norm_{tag}"], d[f"e_pca_{tag}"] = s["js_raw"], s["js_norm"], s["pca_maha"]
    return pd.DataFrame(d)


def frozen_tau(conf: np.ndarray, correct: np.ndarray, target: float = 0.95) -> dict:
    """Smallest confidence threshold whose accepted set (conf >= t) on val_cal has precision >= target."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order].astype(float)
    prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)                # precision of the top-k accepted
    valid = np.where(prec >= target)[0]
    if len(valid) == 0:
        return {"tau": 1.0, "degenerate": "no threshold reaches the target precision", "accept_frac": 0.0}
    k = valid.max()
    tau = float(c[k])
    degenerate = None
    if tau <= c.min() + 1e-12:
        degenerate = f"accepts every val_cal prediction (val_cal precision {ok.mean():.4f} >= {target})"
    return {"tau": tau, "degenerate": degenerate, "accept_frac": float((conf >= tau).mean()),
            "val_cal_precision_at_tau": float(ok[conf[order] >= tau].mean()), "val_cal_acc": float(ok.mean())}


def _sets(cfg: dict, p: dict, df: pd.DataFrame, spec: dict, smp: pd.DataFrame, subset: dict | None) -> list[dict]:
    """(family, condition, severity, images, labels, uid, patient, cue_ids) for one model, val_cal first."""
    ds, kind = spec["dataset"], spec["kind"]
    off = cfg["stage3"]["shortcut"]["offset_px"]
    out = []

    def split_set(split, fam, cond):
        fr = cd.split_frame(df, ds, split)
        return {"family": fam, "condition": cond, "severity": 0, "images": cd.RowReader(cd.split_file(p["s2cache"], ds, split)),
                "pos": np.arange(len(fr)), "labels": fr["label"].to_numpy(np.int64), "uid": fr["uid"].to_numpy(str),
                "patient": fr["patient"].to_numpy(np.int64), "cue": None}
    if kind == "standard":
        out.append(split_set("val_cal", "VALCAL", "val_cal"))
        out.append(split_set("id_test", "ID", "id_test"))
        if ds == "pathmnist":
            out.append(split_set("nct7k_test", "X", "nct7k_test"))
            for cond in json.loads((p["stage3"] / "corruption_report.json").read_text())["conditions"]:
                name, sev = cond.rsplit("_s", 1)
                out.append({"family": "A", "condition": name, "severity": int(sev),
                            "images": cd.RowReader(p["s3cache"] / "A" / f"{cond}.npy"), "pos": np.arange(len(smp)),
                            "labels": smp["label"].to_numpy(np.int64), "uid": smp["uid"].to_numpy(str),
                            "patient": np.full(len(smp), -1), "cue": None})
        else:
            out.append(split_set("ood_c1", "B", "ood_c1"))
            out.append(split_set("ood_c2", "B", "ood_c2"))
    else:
        st = spec["cue"]
        v = split_set("val_cal", "VALCAL", "val_cal")
        v["cue"] = sc.cue_assignment(v["labels"], st["p_aligned"], 9, cfg["stage3"]["shortcut"]["cue_seed"] + 2)
        out.append(v)
        perm = sc.derangement(9, cfg["stage3"]["shortcut"]["derangement_seed"])
        lab = smp["label"].to_numpy(np.int64)
        rr = cd.RowReader(cd.split_file(p["s2cache"], "pathmnist", "id_test"))
        for state in ("aligned", "removed", "reversed"):
            out.append({"family": "C", "condition": state, "severity": 0, "images": rr, "pos": smp["pos"].to_numpy(),
                        "labels": lab, "uid": smp["uid"].to_numpy(str), "patient": np.full(len(lab), -1),
                        "cue": sc.state_cues(lab, state, perm)})
    if subset:                                                       # identical items for every signal and model
        for s in out:
            if s["family"] in ("VALCAL",):
                continue
            keep = subset.get("AC" if s["family"] in ("A", "C") else f"{ds}|{s['condition']}")
            if keep is not None:
                m = np.isin(s["uid"], keep)
                s.update(pos=s["pos"][m], labels=s["labels"][m], uid=s["uid"][m], patient=s["patient"][m],
                         cue=None if s["cue"] is None else s["cue"][m])
    for s in out:
        s["cue_off"] = off
    return out


def _read(s: dict) -> np.ndarray:
    rr, pos = s["images"], s["pos"]
    if isinstance(rr, cd.RowReader):
        return rr[pos]
    return rr.read(0, len(pos))


def run_signals(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> None:
    import torch
    out = p["out"]
    assert marker(out, "refs_frozen").exists(), "references must be frozen before any shifted-family file is read"
    dev = torch.device("cuda")
    dec = json.loads((out / "compute_decision.json").read_text())
    subset = dec.get("subset_uids")
    (out / "signals").mkdir(parents=True, exist_ok=True)
    taus = []
    for spec in model_specs(cfg, p):
        fout = out / "signals" / f"{spec['name']}.parquet"
        mk = marker(out, f"signals_{spec['name']}")
        if mk.exists():
            taus.append(json.loads(mk.read_text())["tau"])
            continue
        t0 = time.time()
        ref = load_ref(p, spec)
        model = load_model(spec, dev)
        pal = sc.palette_for(spec["cue"]) if spec["cue"] else None
        frames, valcal = [], None
        for s in _sets(cfg, p, df, spec, smp, subset):
            if (ram_gb().get("available") or 99) < 2.0:
                raise RuntimeError("free RAM < 2 GB during signal computation")
            R = run_gradcam(model, _read(s), dev, s["cue"], pal, s["cue_off"])
            sig = compute_signals(R, spec["T"], ref)
            meta = {"family": s["family"], "condition": s["condition"], "severity": s["severity"],
                    "model_kind": spec["kind"], "dataset": spec["dataset"], "seed": spec["seed"]}
            sig.insert(0, "uid", s["uid"])
            for k, v in reversed(list(meta.items())):
                sig.insert(0, k, v)
            sig["patient"], sig["label"] = s["patient"], s["labels"]
            sig["correct"] = sig["pred"] == sig["label"]
            if spec["kind"] != "standard" and s["family"] == "C":
                P_ = pal.shape[1]
                for ln, tag in (("layer4", "l4"), ("layer3", "l3")):
                    sig[f"cue_mass_{tag}"] = cue_mass(R[ln], s["cue_off"], P_, 128)
            if s["family"] == "VALCAL":
                valcal = sig
            frames.append(sig)
        allsig = pd.concat(frames, ignore_index=True)
        for col in SIGNALS:                                          # normalisation: source val_cal ONLY
            allsig["pct_" + col] = nz.ecdf_percentile(valcal[col].to_numpy(), allsig[col].to_numpy())
            allsig["rz_" + col] = nz.robust_z(valcal[col].to_numpy(), allsig[col].to_numpy())
        tau = frozen_tau(valcal["confidence"].to_numpy(), valcal["correct"].to_numpy(), cfg["stage4"]["tau_precision"])
        tau.update({"model": spec["name"], "dataset": spec["dataset"], "kind": spec["kind"], "seed": spec["seed"]})
        for name, thr in (("sf_tau", tau["tau"]), ("sf_090", 0.90), ("sf_095", 0.95)):
            allsig[name] = (~allsig["correct"]) & (allsig["confidence"] >= thr)
        allsig.to_parquet(fout, index=False)
        mk.write_text(json.dumps({"tau": tau, "rows": len(allsig), "seconds": round(time.time() - t0, 1)}))
        taus.append(tau)
        log(f"[stage4] signals {spec['name']}: {len(allsig):,} rows in {time.time() - t0:.0f} s; tau={tau['tau']:.4f}"
            f"{' (DEGENERATE: ' + tau['degenerate'] + ')' if tau['degenerate'] else ''}")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(taus).to_csv(out / "frozen_tau.csv", index=False)


def cue_mass(cams: np.ndarray, offset: int, cue_size: int, img_px: int) -> np.ndarray:
    """Fraction of CAM mass inside the cue box, with the GRID x GRID map expanded to image pixels (nearest blocks)."""
    cell = img_px / ex.GRID
    w1 = np.clip(np.minimum(np.arange(1, ex.GRID + 1) * cell, offset + cue_size) - np.maximum(np.arange(ex.GRID) * cell, offset),
                 0, None) / cell                                     # overlap fraction of each cell row/col with the box
    W = np.outer(w1, w1).ravel()
    return cams @ W


# --------------------------------------------------------------------------- pilot ----------
def run_pilot(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> dict:
    """Time Grad-CAM, project the total GPU time, decide full sets vs one fixed 1,000-per-condition subset."""
    import torch
    out = p["out"]
    f = out / "compute_decision.json"
    if f.exists():
        return json.loads(f.read_text())
    dev = torch.device("cuda")
    specs = model_specs(cfg, p)
    spec = specs[0]
    model = load_model(spec, dev)
    imgs = cd.RowReader(cd.split_file(p["s2cache"], "pathmnist", "val_cal"))[np.arange(2048)]
    run_gradcam(model, imgs[:256], dev)
    torch.cuda.synchronize()
    t0 = time.time()
    run_gradcam(model, imgs, dev)
    torch.cuda.synchronize()
    per_img = (time.time() - t0) / len(imgs)
    total = sum(len(_read_len(s)) for sp in specs for s in _sets(cfg, p, df, sp, smp, None))
    proj_min = total * per_img / 60
    dec = {"seconds_per_image_128px": per_img, "images_total_full": int(total), "projected_minutes_full": round(proj_min, 1),
           "budget_minutes": cfg["stage4"]["gpu_budget_min"]}
    if proj_min <= cfg["stage4"]["gpu_budget_min"]:
        dec["decision"] = "full sets"
    else:
        rng_seed = cfg["stage4"]["seed"]
        keep = {}
        a = smp.assign(pos=np.arange(len(smp)))
        keep["AC"] = stratified_sample(a, 1000, ["label"], rng_seed)["uid"].tolist()
        for ds, split in (("pathmnist", "nct7k_test"), ("pathmnist", "id_test"), ("camelyon17", "id_test"),
                          ("camelyon17", "ood_c1"), ("camelyon17", "ood_c2")):
            fr = cd.split_frame(df, ds, split)
            keep[f"{ds}|{split}"] = stratified_sample(fr, 1000, ["label"], rng_seed)["uid"].tolist()
        dec["decision"] = "one fixed stratified 1,000-image-per-condition subset"
        dec["subset_uids"] = keep
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dec, indent=1))
    log(f"[stage4] Grad-CAM pilot: {1000 * per_img:.2f} ms/image, {total:,} images -> {proj_min:.0f} min "
        f"(budget {cfg['stage4']['gpu_budget_min']}) -> {dec['decision']}")
    return dec


def _read_len(s: dict):
    return s["pos"]



# --------------------------------------------------------------------------- tables ----------
def _auroc(y: np.ndarray, s: np.ndarray):
    from sklearn.metrics import roc_auc_score
    if y.sum() < 10 or (~y).sum() < 10:
        return np.nan
    return float(roc_auc_score(y, s))


def cam_randomisation_check(cfg: dict, p: dict, df: pd.DataFrame, n: int = 512) -> dict:
    """Adebayo et al. (2018) style: CAMs must change when the last layer's weights are randomised."""
    import copy
    import torch
    from scipy.stats import spearmanr
    dev = torch.device("cuda")
    spec = model_specs(cfg, p)[0]
    model = load_model(spec, dev)
    imgs = cd.RowReader(cd.split_file(p["s2cache"], spec["dataset"], "val_cal"))[np.arange(n)]
    base = run_gradcam(model, imgs, dev)
    rnd = copy.deepcopy(model)
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        rnd.net.fc.weight.copy_(torch.randn(rnd.net.fc.weight.shape, generator=g) * rnd.net.fc.weight.std().cpu())
        rnd.net.fc.bias.zero_()
    out = {}
    for ln in ex.LAYERS:
        from sfd.train import MEAN, STD  # noqa: F401
        tgt = torch.from_numpy(base["logits"].argmax(1))
        parts = []
        for x, a in _batches(imgs, None, None, 0, dev):
            parts.append(ex.gradcam_batch(rnd, x, target=tgt[a:a + len(x)].to(dev))[ln])
        r = np.concatenate(parts)
        rho = np.nanmean([spearmanr(base[ln][i], r[i]).statistic for i in range(n)])
        out[ln] = {"mean_js_orig_vs_randomised": float(ex.js_divergence(base[ln], r).mean()),
                   "mean_spearman_orig_vs_randomised": float(rho)}
    out["model"] = spec["name"]
    out["passed"] = bool(all(out[ln]["mean_spearman_orig_vs_randomised"] < 0.5 for ln in ex.LAYERS))
    return out


def run_tables(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = p["out"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    mon = pd.concat([pd.read_parquet(f) for f in sorted((out / "signals").glob("*.parquet"))], ignore_index=True)
    mon.to_parquet(out / "monitoring.parquet", index=False)                       # git-ignored (full table)
    groups = {"A (all corrupted)": (mon.family == "A") & (mon.severity > 0), "A clean": (mon.family == "A") & (mon.severity == 0),
              "B (ood_c1+ood_c2)": mon.family == "B", "C reversed": (mon.family == "C") & (mon.condition == "reversed"),
              "C removed": (mon.family == "C") & (mon.condition == "removed"), "X": mon.family == "X", "ID": mon.family == "ID"}
    rows = []
    for gname, gm in groups.items():
        for (ds, kind, seed), g in mon[gm].groupby(["dataset", "model_kind", "seed"]):
            err, sf = ~g["correct"].to_numpy(), g["sf_tau"].to_numpy()
            for sig in SIGNALS:
                s = g[sig].to_numpy()
                rows.append({"group": gname, "dataset": ds, "model_kind": kind, "seed": seed, "signal": sig,
                             "n": len(g), "error_rate": err.mean(), "sf_rate": sf.mean(),
                             "auroc_error": _auroc(err, s), "auroc_sf": _auroc(sf, s)})
    au = pd.DataFrame(rows)
    au.to_csv(out / "per_signal_auroc_per_seed.csv", index=False)
    aus = au.groupby(["group", "dataset", "model_kind", "signal"]).agg(
        n_seeds=("seed", "nunique"), error_rate=("error_rate", "mean"), sf_rate=("sf_rate", "mean"),
        auroc_error=("auroc_error", "mean"), auroc_error_std=("auroc_error", "std"),
        auroc_sf=("auroc_sf", "mean"), auroc_sf_std=("auroc_sf", "std")).reset_index().round(4)
    aus.to_csv(out / "per_signal_auroc.csv", index=False)

    a = mon[(mon.family == "A")]
    dose = a.groupby(["condition", "severity", "seed"])[[f"pct_{s}" for s in SIGNALS]].mean().reset_index()
    clean = dose[dose.condition == "clean"].drop(columns="condition")
    dr = []
    for cond, g in dose[dose.condition != "clean"].groupby("condition"):
        gg = pd.concat([clean, g.drop(columns="condition")]).groupby("severity").mean(numeric_only=True)
        for s in SIGNALS:
            v = gg[f"pct_{s}"].to_numpy()
            dr.append({"corruption": cond, "signal": s, **{f"sev{k}": round(float(x), 4) for k, x in zip(gg.index, v)},
                       "monotone_nondecreasing": bool(np.all(np.diff(v) >= -0.01))})
    dr = pd.DataFrame(dr)
    dr.to_csv(out / "dose_response.csv", index=False)

    cm = mon[mon.family == "C"].groupby(["model_kind", "condition", "seed"])[["cue_mass_l4", "cue_mass_l3"]].mean()
    cm = cm.groupby(["model_kind", "condition"]).agg(["mean", "std"]).round(4)
    cm.columns = ["_".join(c) for c in cm.columns]
    cm.reset_index().to_csv(out / "cue_mass.csv", index=False)

    tv = []
    for f in sorted((out / "refs").glob("*.npz")):
        z = np.load(f)
        for ln in ex.LAYERS:
            tv.append({"model": f.stem, "layer": ln, **{t: round(float(x), 4) for t, x in zip(ex.TRANSFORMS, z[f"{ln}_t_per_transform"])},
                       "ref_train_acc": round(float(z["ref_train_acc"]), 4), "lw_shrinkage": round(float(z["lw_shrinkage"]), 4)})
    pd.DataFrame(tv).to_csv(out / "transform_variability.csv", index=False)

    val = cam_randomisation_check(cfg, p, df)
    agree = {}                                                                     # recomputed vs saved predictions
    for kind, fam, cond, src in (("standard", "ID", "id_test", p["stage2"] / "preds" / "pathmnist_seed0_id_test.npz"),
                                 ("standard", "A", "clean", p["stage3"] / "preds" / "standard_seed0_A_clean_s0.npz"),
                                 ("shortcut", "C", "reversed", p["stage3"] / "preds" / "shortcut_seed0_C_reversed_s0.npz")):
        z = np.load(src)
        g = mon[(mon.model_kind == kind) & (mon.family == fam) & (mon.condition == cond) & (mon.seed == 0) & (mon.dataset == "pathmnist")]
        saved_pred = z["pred"] if "pred" in z.files else z["probs_cal"].argmax(1)   # stage-2 files store probs only
        saved = pd.Series(saved_pred, index=z["uid"].astype(str))
        agree[f"{kind}/{fam}/{cond}"] = float((saved.loc[g["uid"]].to_numpy() == g["pred"].to_numpy()).mean())
    sanity = {"rows": len(mon), "nan_in_signals": bool(mon[SIGNALS].isna().any().any()), "cam_randomisation": val,
              "pred_agreement_with_saved": agree,
              "dose_response_non_monotone": dr[~dr.monotone_nondecreasing][["corruption", "signal"]].values.tolist()}
    (out / "sanity_stage4.json").write_text(json.dumps(sanity, indent=1))

    # figures
    prim = [PRIMARY["uncertainty"], PRIMARY["representation"], PRIMARY["explanation"], "e_js_raw_l4", "r_knn", "u_entropy"]
    sel = aus[aus.signal.isin(prim) & aus.group.isin(["A (all corrupted)", "B (ood_c1+ood_c2)", "C reversed", "X"])]
    sel = sel[~((sel.group == "C reversed") & (sel.model_kind == "standard"))]
    lab = sel.group + " / " + sel.model_kind
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, col in enumerate(("auroc_error", "auroc_sf")):
        piv = sel.assign(lab=lab).pivot_table(index="lab", columns="signal", values=col)[prim]
        piv.plot.barh(ax=ax[k], legend=(k == 1))
        ax[k].axvline(0.5, color="k", lw=0.8); ax[k].set_xlim(0.3, 1.0); ax[k].set_title(f"descriptive {col} (mean over seeds)")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "figures" / "per_signal_auroc.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    d = a.groupby(["severity"])[[f"pct_{s}" for s in prim]].mean()
    for s in prim:
        ax.plot(d.index, d[f"pct_{s}"], marker="o", label=s)
    ax.set_xlabel("severity (0 = clean)"); ax.set_ylabel("mean val_cal percentile"); ax.legend(fontsize=7)
    ax.set_title("family A dose-response (pooled corruptions, all seeds)")
    fig.tight_layout(); fig.savefig(out / "figures" / "dose_response_A.png", dpi=130); plt.close(fig)
    md = ["# Stage 4 - descriptive signal results (NOT the Phase 5 tests)\n", "## Frozen tau\n",
          to_markdown(pd.read_csv(out / "frozen_tau.csv")), "\n## Cue mass (Fig 7 input)\n", to_markdown(cm.reset_index()),
          "\n## Primary signals: descriptive AUROC (mean over seeds)\n",
          to_markdown(aus[aus.signal.isin(list(PRIMARY.values()))]), "\n## Validity\n```\n" + json.dumps(sanity, indent=1) + "\n```\n"]
    (out / "summary_stage4.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[stage4] tables: {len(mon):,} rows; CAM randomisation passed={val['passed']}; agreement={agree}")
    return sanity
