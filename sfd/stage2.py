"""Stage 2: source classifiers (ResNet-18, ImageNet-pretrained) + temperature scaling + saved predictions.

Per dataset and seed (done-marker `outputs/stage2/markers/<dataset>_seed<k>.done`, honoured always):
  train (early stopping on val_es) -> predict every split (fp32) -> fit T on val_cal ONLY -> save
  logits / calibrated probs / 512-d features / labels / uid / patient per split -> marker.
Then (always) aggregate: metrics before/after calibration, seed stability, reliability diagrams, confusion
matrices and sanity checks. The frozen splits.parquet is only read; its hash is asserted unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfd import calibration as cal
from sfd.config import ROOT, resolve
from sfd.data import classify_data as cd
from sfd.data.audit import sha256_file, to_markdown
from sfd.sysinfo import ram_gb


def _paths(cfg: dict, out_dir: Path) -> dict:
    s2 = cfg["stage2"]
    cache = Path(s2["cache_dir"])
    return {"out": out_dir / "stage2", "stage1": out_dir / "stage1" / "splits.parquet",
            "cache": cache if cache.is_absolute() else ROOT / cache}


def _camelyon_npz(cfg: dict, out_dir: Path) -> Path:
    c = cfg["camelyon"]
    return (ROOT / c["subset_npz"]) if c.get("subset_npz") else out_dir / "stage1" / "cache" / "camelyon_subset_96.npz"


def _marker(out: Path, ds: str, seed: int) -> Path:
    return out / "markers" / f"{ds}_seed{seed}.done"


def eval_splits(ds: str) -> list[str]:
    return [s for s in cd.SPLITS[ds] if s != "train"]


def run(cfg: dict, out_dir: Path, resume: bool = False, log=print) -> dict:
    import torch
    from sfd.train import GpuGuard, predict, train_one

    p = _paths(cfg, out_dir)
    out = p["out"]
    for sub in ("markers", "preds", "logs", "ckpt", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    frozen_hash = sha256_file(p["stage1"])
    df = pd.read_parquet(p["stage1"])
    s2, tcfg = cfg["stage2"], cfg["train"]
    dev = torch.device("cuda")
    guard = GpuGuard(s2["guard"]["slowdown_margin_c"], s2["guard"]["min_free_ram_gb"], s2["guard"]["pause_s"],
                     s2["guard"]["max_pause_s"], log)
    log(f"[stage2] free RAM at start: {ram_gb()}")
    summary = {}
    for ds in s2["datasets"]:
        n_cls, size = cd.N_CLASSES[ds], cfg["data"]["pathmnist_size" if ds == "pathmnist" else "camelyon_size"]
        frames = {s: cd.split_frame(df, ds, s, tcfg.get("max_train_images") if s == "train" else None)
                  for s in cd.SPLITS[ds]}
        t0 = time.time()
        if ds == "pathmnist":
            cd.prepare_pathmnist(resolve(cfg, "data_dir") / f"pathmnist_{size}.npz", frames, p["cache"], size, log)
        else:
            cd.prepare_camelyon(_camelyon_npz(cfg, out_dir), frames, p["cache"], size, log)
        log(f"[stage2] {ds}: data cache ready ({time.time() - t0:.0f} s)")
        todo = [sd for sd in tcfg["seeds"] if not _marker(out, ds, sd).exists()]
        tr = va = None
        for seed in tcfg["seeds"]:
            mk = _marker(out, ds, seed)
            if mk.exists():
                log(f"[stage2] {ds} seed {seed}: done earlier -> skipped")
                continue
            if tr is None:                                                   # load once, keep on the GPU
                tr = cd.load_split(p["cache"], ds, "train", frames["train"])
                mode, free = s2.get("train_on_gpu", "auto"), torch.cuda.mem_get_info()[0]
                need = tr.images.nbytes + int(1.3 * 2**30)          # data + model/activations/workspace margin
                on_gpu = mode is True or (mode == "auto" and free >= need)
                log(f"[stage2] {ds}: train data {tr.images.nbytes / 2**30:.2f} GB, free VRAM {free / 2**30:.2f} GB, "
                    f"needs {need / 2**30:.2f} GB -> {'GPU-resident' if on_gpu else 'SSD memmap batches (CPU gather)'}")
                if on_gpu:
                    tr.images = cd.to_device(tr.images, dev)
                else:                                   # explicit row reads: no mapped pages in the working set
                    tr.images = cd.RowReader(cd.split_file(p["cache"], ds, "train"))
                va = cd.load_split(p["cache"], ds, "val_es", frames["val_es"])
                va.images = cd.to_device(va.images, dev)
                log(f"[stage2] {ds}: train {tuple(tr.images.shape)} on GPU, VRAM allocated "
                    f"{torch.cuda.memory_allocated() / 2**30:.2f} GB, free RAM {ram_gb().get('available')} GB")
            t1 = time.time()
            log(f"[stage2] {ds} seed {seed}: training")
            res = train_one(cfg, tr, va, n_cls, seed, out / "ckpt" / f"{ds}_seed{seed}",
                            out / "logs" / f"{ds}_seed{seed}.jsonl", guard, dev, log)
            model = res["model"]
            logits = {}
            for s in eval_splits(ds):
                d = cd.load_split(p["cache"], ds, s, frames[s])
                lo, ft = predict(model, d.images, dev, amp=False)
                logits[s] = (lo, ft, d)
            T = cal.fit_temperature(logits["val_cal"][0], logits["val_cal"][2].labels)
            T_prior, prior_gap = None, None
            if ds in s2["calibration"]["prior_reweight_datasets"]:
                yc, yi = logits["val_cal"][2].labels, logits["id_test"][2].labels
                pc, pi = np.bincount(yc, minlength=n_cls) / len(yc), np.bincount(yi, minlength=n_cls) / len(yi)
                prior_gap = float(50.0 * np.abs(pc - pi).sum())            # total-variation distance in points
                if prior_gap >= s2["calibration"].get("prior_reweight_min_gap_pts", 0.0):
                    T_prior = cal.fit_temperature(logits["val_cal"][0], yc, cal.prior_weights(yc, yi, n_cls))
                else:
                    log(f"[stage2] {ds}: val_cal vs id_test label-prior gap {prior_gap:.2f} pts < "
                        f"{s2['calibration']['prior_reweight_min_gap_pts']} pts -> no reweighted variant")
            for s, (lo, ft, d) in logits.items():
                arrays = dict(logits=lo, probs_cal=cal.probs(lo, T).astype(np.float32), features=ft,
                              labels=d.labels, uid=d.uid, patient=d.patient, T=np.float32(T))
                if T_prior is not None:
                    arrays["probs_cal_prior"] = cal.probs(lo, T_prior).astype(np.float32)
                np.savez(out / "preds" / f"{ds}_seed{seed}_{s}.npz", **arrays)
            mk.write_text(json.dumps({"dataset": ds, "seed": seed, "T": T, "T_prior": T_prior,
                                      "prior_gap_pts": prior_gap,
                                      "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                                      "best_val_loss": res["best_val_loss"], "seconds": round(time.time() - t1, 1)}))
            log(f"[stage2] {ds} seed {seed}: done in {time.time() - t1:.0f} s, best epoch {res['best_epoch']}, T={T:.3f}")
            del model, res, logits
            torch.cuda.empty_cache()
        summary[ds] = aggregate(cfg, ds, out, df, frames, log)
    assert sha256_file(p["stage1"]) == frozen_hash, "frozen splits.parquet was modified"
    return summary


# --------------------------------------------------------------------------- aggregation ------
def _auroc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(p.shape[1]))))


def aggregate(cfg: dict, ds: str, out: Path, df: pd.DataFrame, frames: dict, log=print) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ece_bins = cfg["stage2"]["calibration"]["ece_bins"]
    seeds = [s for s in cfg["train"]["seeds"] if _marker(out, ds, s).exists()]
    if not seeds:
        return {}
    rows, calib, sanity = [], [], {"dataset": ds, "seeds": seeds, "nan_found": False, "uid_order_ok": True,
                                   "pred_class_balance_id_test": {}, "id_acc": {}}
    id_logits = {}
    for seed in seeds:
        mk = json.loads(_marker(out, ds, seed).read_text())
        T, Tp = mk["T"], mk.get("T_prior")
        for s in eval_splits(ds):
            z = np.load(out / "preds" / f"{ds}_seed{seed}_{s}.npz")
            lo, y = z["logits"], z["labels"]
            if not (np.isfinite(lo).all() and np.isfinite(z["features"].astype(np.float32)).all()):
                sanity["nan_found"] = True
            if not np.array_equal(z["uid"], frames[s]["uid"].to_numpy(dtype=str)):
                sanity["uid_order_ok"] = False
            variants = {"uncalibrated": 1.0, "T_standard": T}
            if Tp is not None:
                variants["T_prior_reweighted"] = Tp
            for vname, tv in variants.items():
                pp = cal.probs(lo, tv)
                rows.append({"dataset": ds, "seed": seed, "split": s, "variant": vname, "T": round(tv, 4),
                             "n": len(y), "auroc": _auroc(pp, y), "nll": cal.nll(lo, y, T=tv),
                             "ece": cal.ece(pp, y, ece_bins), "brier": cal.brier(pp, y),
                             "acc": float((pp.argmax(1) == y).mean())})
            if s == "id_test":
                id_logits[seed] = (lo, y, T)
                pred = lo.argmax(1)
                sanity["pred_class_balance_id_test"][seed] = {
                    "pred_count": np.bincount(pred, minlength=lo.shape[1]).tolist(),
                    "true_count": np.bincount(y, minlength=lo.shape[1]).tolist()}
                sanity["id_acc"][seed] = float((pred == y).mean())
        calib.append({"dataset": ds, "seed": seed, "T_standard": round(T, 4),
                      "T_prior_reweighted": None if Tp is None else round(Tp, 4),
                      "best_epoch": mk["best_epoch"], "epochs_run": mk["epochs_run"],
                      "train_seconds": mk["seconds"]})
    m = pd.DataFrame(rows)
    m.to_csv(out / f"metrics_per_seed_{ds}.csv", index=False)
    pd.DataFrame(calib).to_csv(out / f"calibration_{ds}.csv", index=False)

    def pm(x):
        return f"{x.mean():.4f} Â± {x.std(ddof=1) if len(x) > 1 else 0:.4f}"
    summ = (m.groupby(["split", "variant"])[["acc", "auroc", "nll", "ece", "brier"]].agg(pm)
            .reset_index())
    summ.to_csv(out / f"metrics_summary_{ds}.csv", index=False)
    stab = m[m["variant"] == "T_standard"].groupby("split").agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"), acc_min=("acc", "min"), acc_max=("acc", "max"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), ece_mean=("ece", "mean"), ece_std=("ece", "std")
    ).round(4).reset_index()
    stab.to_csv(out / f"seed_stability_{ds}.csv", index=False)

    # figures: reliability (id_test, before/after) and confusion matrices (id_test, one per seed)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for k, (title, use_T) in enumerate([("before temperature scaling", False), ("after temperature scaling", True)]):
        ax[k].plot([0, 1], [0, 1], "k--", lw=1)
        for seed, (lo, y, T) in id_logits.items():
            c, a, n = cal.reliability_curve(cal.probs(lo, T if use_T else 1.0), y)
            ax[k].plot(c, a, "o-", ms=3, label=f"seed {seed}")
        ax[k].set_title(f"{ds} id_test, {title}", fontsize=9)
        ax[k].set_xlabel("confidence"); ax[k].set_ylabel("accuracy")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figures" / f"reliability_{ds}.png", dpi=130); plt.close(fig)
    n_cls = cd.N_CLASSES[ds]
    fig, axes = plt.subplots(1, len(id_logits), figsize=(3.2 * len(id_logits), 3.4), squeeze=False)
    for ax_, (seed, (lo, y, T)) in zip(axes[0], id_logits.items()):
        cm = np.zeros((n_cls, n_cls))
        np.add.at(cm, (y, lo.argmax(1)), 1)
        ax_.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
        ax_.set_title(f"seed {seed} acc {np.trace(cm) / cm.sum():.3f}", fontsize=8)
        ax_.set_xlabel("predicted"); ax_.set_ylabel("true")
    fig.tight_layout(); fig.savefig(out / f"figures/confusion_{ds}.png", dpi=130); plt.close(fig)

    min_acc = cfg["stage2"]["sanity"]["min_id_acc"].get(ds)
    sanity["min_id_acc_required"] = min_acc
    sanity["id_acc_ok"] = bool(min_acc is None or min(sanity["id_acc"].values()) >= min_acc)
    (out / f"sanity_{ds}.json").write_text(json.dumps(sanity, indent=1))
    (out / f"summary_{ds}.md").write_text(
        f"# Stage 2 - {ds} (seeds {seeds})\n\n## Mean Â± std over seeds\n\n{to_markdown(summ)}\n"
        f"## Seed stability (T_standard)\n\n{to_markdown(stab)}\n## Temperatures / training\n\n"
        f"{to_markdown(pd.DataFrame(calib))}\n", encoding="utf-8")
    log(f"[stage2] {ds}: aggregated {len(seeds)} seeds; sanity: nan={sanity['nan_found']} uid_order_ok="
        f"{sanity['uid_order_ok']} id_acc_ok={sanity['id_acc_ok']} (min id acc {min(sanity['id_acc'].values()):.4f})")
    return {"seeds": seeds, "sanity": sanity}

