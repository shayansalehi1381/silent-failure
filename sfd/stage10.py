"""Stage 10: post-hoc EXTENSIONS (docs/extension_plan_frozen.md). E1 = deep-ensemble uncertainty baseline from the 5 saved seeds
(per-seed probabilities reconstructed from the saved penultimate features and the frozen final layer; no retraining / GPU).
E2 (in-family vs out-of-family) uses saved tables only and lives in the notebook.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfd import stage4 as s4
from sfd import stage5 as s5
from sfd import stage7 as s7
from sfd import stage8 as s8
from sfd.signals import normalize as nz

KINDS = [("standard", "pathmnist"), ("standard", "camelyon17"), ("shortcutv2", "pathmnist")]   # the pools of families A, B, C(v2)
EPS = 1e-300
SIGS = ("msp", "ent", "mi")


# --------------------------------------------------------------------------- ensemble signals ---------
def softmax64(lo: np.ndarray, T: float) -> np.ndarray:
    z = lo.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(np.clip(p, EPS, 1)), 0.0), axis=1)


def ens_signals(probs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Signals of the mean-probability ensemble: log(1 - max p-bar) (summed non-max mass, stable), log predictive entropy, log mutual
    information (member disagreement). A one-member ensemble reduces to the single model (MI = 0 -> clipped)."""
    pb = np.mean(probs, axis=0)
    am = pb.argmax(1)
    nonmax = pb.sum(1) - pb[np.arange(len(pb)), am]
    h = entropy(pb)
    mi = np.maximum(h - np.mean([entropy(p) for p in probs], axis=0), 0.0)
    return {"msp": np.log(np.clip(nonmax, EPS, 1)), "ent": np.log(np.clip(h, EPS, None)), "mi": np.log(np.clip(mi, 1e-12, None))}


def check_alignment(stores: list[dict]) -> None:
    """Every seed of a model kind must carry the SAME rows in the SAME order (otherwise ensembling rows is meaningless)."""
    ref = stores[0]
    for s in stores[1:]:
        for k in ("fam", "cond", "sev", "uid", "label"):
            assert np.array_equal(ref[k], s[k]), f"rows differ across seeds in '{k}'"


def build_ens(cfg: dict, out_dir: Path, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch
    p = s4.P(cfg, out_dir)
    p["s6"] = out_dir / "stage6"
    specs = s4.model_specs(cfg, p)
    frames, checks = [], []
    for kind, ds in KINDS:
        sp = sorted([s for s in specs if s["kind"] == kind and s["dataset"] == ds], key=lambda s: s["seed"])
        st = [dict(np.load(p["s6"] / "rows" / f"{s['name']}.npz")) for s in sp]
        check_alignment(st)
        probs = []
        for s, z in zip(sp, st):
            sd = torch.load(s["ckpt"], map_location="cpu")
            lo = s8.reconstruct_logits(z["feat"], sd["net.fc.weight"].numpy(), sd["net.fc.bias"].numpy())
            dc = np.abs(s8._softmax_conf(lo, s["T"]) - z["conf"])
            assert np.quantile(dc, 0.999) < 1e-3 and dc.max() < 1e-2, f"logit reconstruction check failed for {s['name']}"
            checks.append({"model": s["name"], "conf_p999_abs_diff": float(np.quantile(dc, 0.999)), "conf_max_abs_diff": float(dc.max())})
            probs.append(softmax64(lo, s["T"]))
        vc = st[0]["fam"] == "VALCAL"
        fam = st[0]["fam"]
        for i, s in enumerate(sp):
            d = {"family": fam, "condition": st[0]["cond"], "severity": np.where(fam == "A", st[0]["sev"], 0).astype(np.int64),
                 "uid": st[0]["uid"], "model_kind": kind, "dataset": ds, "seed": s["seed"]}
            for tag, members in (("ens4", [j for j in range(len(sp)) if j != i]), ("ens5", list(range(len(sp))))):
                sg = ens_signals([probs[j] for j in members])
                for k in SIGS:
                    d[f"x_{tag}_{k}"] = nz.ecdf_percentile_ext(sg[k][vc], sg[k])
            frames.append(pd.DataFrame(d))
        log(f"[stage10] ensemble signals {kind}/{ds}: {len(fam):,} rows x {len(sp)} seeds")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(checks)


def merge_aug(df: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "condition", "severity", "model_kind", "dataset", "seed", "uid"]
    new = [c for c in aug.columns if c.startswith("x_")]
    a, b = df.copy(), aug.copy()
    for d in (a, b):
        d["severity"] = np.where(d["family"] == "A", d["severity"].fillna(0), 0).astype(np.int64)
        d["seed"] = d["seed"].astype(np.int64)
    return a.merge(b[keys + new], on=keys, how="left", validate="one_to_one")


# --------------------------------------------------------------------------- methods ---------------
def registry() -> dict:
    R = dict(s8.registry())
    for tag, pre in (("ens4", "E"), ("ens5", "E5")):
        base = [f"{tag}_msp", f"{tag}_ent"]
        R[f"{pre}_fuse"] = {"kind": "fit", "feats": base, "con": "nn"}
        R[f"{pre}_M1"] = {"kind": "fit", "feats": base + [s8.MAHA], "con": "nn"}
    for k in SIGS:
        R[f"E_{k}"] = {"kind": "single", "feat": f"ens4_{k}"}
    return R


NAMES = ["M0", "M1", "M2", "E_fuse", "E_msp", "E_ent", "E_mi", "E_M1", "E5_fuse", "E5_M1"]
PAIRS = [("E1", "E_fuse", "M0", False), ("E1", "M1", "E_fuse", False), ("E1", "E_M1", "E_fuse", False), ("E1", "E_M1", "M1", False),
         ("E1", "E_mi", "E_msp", False), ("E1", "E_ent", "E_msp", False), ("E1", "M1", "M0", False), ("E1", "M2", "M1", False),
         ("E1", "E5_fuse", "M0", False), ("E1", "M1", "E5_fuse", False), ("E1", "E5_M1", "E5_fuse", False), ("E1", "E_msp", "M0", False)]


def run(cfg: dict, out_dir: Path, log=print) -> None:
    s10 = cfg["stage10"]
    out = out_dir / "stage10"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    seeds, B = cfg["train"]["seeds"], s10["bootstrap_B"]
    assert tuple(s10["budgets"]) == tuple(s8.BUDGETS), "configs stage10 block differs from the frozen plan"
    t0 = time.time()
    f = out / "ens_features.parquet"
    if f.exists():
        aug = pd.read_parquet(f)
    else:
        aug, chk = build_ens(cfg, out_dir, log)
        aug.to_parquet(f)
        chk.to_csv(out / "ens_reconstruction_checks.csv", index=False)
    df = s5.load(cfg, out_dir)
    df = merge_aug(df, aug)
    df = df[[c for c in df.columns if c in s8.NEEDED or c.startswith("x_")]]
    log(f"[stage10] data ready ({time.time() - t0:.0f} s)")
    reg = registry()
    for H in ("A", "B", "C"):
        fo = out / "raw" / f"e1_{H}.csv"
        if fo.exists():
            continue
        ctx = s8.make_ctx(df, H, "sf_090", seeds)
        for s in seeds:                                                    # every ensemble feature present on the used pools
            assert ctx["te"][s][[c for c in ctx["xcols"] if "ens" in c]].notna().all().all(), "missing ensemble features (held-out)"
            assert ctx["tu"][s][[c for c in ctx["xcols"] if "ens" in c]].notna().all().all(), "missing ensemble features (tuning)"
        sc, wts = s8.score_methods(ctx, NAMES, seeds, reg=reg)
        s7.check_weights_match_stage5({"heldout": H, "label": "sf_090", "weights": [w for w in wts if w["model"] in ("M0", "M1", "M2")]}, out_dir)
        pd.DataFrame(wts).to_csv(out / "raw" / f"weights_{H}.csv", index=False)
        res = s8.evaluate(ctx["te"], sc, "sf_090", seeds, B)
        rows = s8.make_rows(res, {"heldout": H, "label": "sf_090", "c_kind": "shortcutv2", "level_col": "", "level": ""}, PAIRS)
        pd.DataFrame(rows).to_csv(fo, index=False)
        log(f"[stage10] E1 held-out {H} done ({time.time() - t0:.0f} s)")
        del ctx, sc, res
    tabs = pd.concat([pd.read_csv(out / "raw" / f"e1_{H}.csv") for H in "ABC"], ignore_index=True)
    tabs["table"] = "EXTENSION (post-hoc), not a main result"
    tabs.to_csv(out / "e1_ensemble_baseline.csv", index=False)
    _figure(tabs, out)
    log(f"[stage10] all done ({time.time() - t0:.0f} s)")


def _figure(t: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M0": "#0072B2", "M1": "#E69F00", "E_fuse": "#009E73", "E_M1": "#D55E00", "random": "#999999", "oracle": "#000000"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for a, H in zip(ax, "ABC"):
        for m, c in col.items():
            g = t[(t.kind == "level") & (t.heldout == H) & (t.quantity == m) & (t.metric == "resid_sf")].sort_values("budget")
            a.errorbar(100 * g.budget, 100 * g.estimate, yerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)],
                       fmt="--" if m == "oracle" else "-o", ms=3, capsize=2, color=c, label=m)
        a.set_title(f"EXTENSION E1 - held-out {H}")
        a.set_xlabel("review budget (%)")
    ax[0].set_ylabel("residual silent-failure rate (%)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "fig10_e1_referral.png", dpi=130)
    plt.close(fig)
