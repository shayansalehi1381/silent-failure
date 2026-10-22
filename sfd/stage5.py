"""Stage 5: sample-level silent-failure detection - nested detectors M0 < M1 < M2, held-out-family rotation,
paired two-level cluster bootstrap. Implements docs/analysis_plan_frozen.md (pre-registration) exactly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sfd.data.audit import to_markdown
from sfd.signals import normalize as nz

KEYS = ["family", "condition", "severity", "model_kind", "dataset", "seed", "uid"]
RAW = ["u_1msp", "u_entropy", "r_maha_pred", "r_knn", "e_js_norm_l4", "e_js_raw_l4", "e_pca_l4", "e_js_norm_l3"]
EXTRA = ["u_log1msp", "u_logent", "e_js_norm_l2"]
M0 = ["u_log1msp", "u_logent"]
DETECTORS = {"M0": M0, "M1": M0 + ["r_maha_pred"], "M2": M0 + ["r_maha_pred", "e_js_norm_l4"]}
SENSITIVITY = {                                    # (name, detector features replacing the primary M1/M2)
    "kNN instead of Mahalanobis": {"M1": M0 + ["r_knn"], "M2": M0 + ["r_knn", "e_js_norm_l4"]},
    "layer3 CAM": {"M2": M0 + ["r_maha_pred", "e_js_norm_l3"]},
    "layer2 CAM": {"M2": M0 + ["r_maha_pred", "e_js_norm_l2"]},
    "raw (unnormalised) CAM JS": {"M2": M0 + ["r_maha_pred", "e_js_raw_l4"]},
    "PCA-embedding CAM distance": {"M2": M0 + ["r_maha_pred", "e_pca_l4"]},
    "float 1-MSP/entropy": {"M0": ["u_1msp", "u_entropy"], "M1": ["u_1msp", "u_entropy", "r_maha_pred"],
                            "M2": ["u_1msp", "u_entropy", "r_maha_pred", "e_js_norm_l4"]},
}
LABELS_SENS = ["sf_095", "sf_tau", "sf_099"]


# --------------------------------------------------------------------------- data ------------
def load(cfg: dict, out_dir: Path) -> pd.DataFrame:
    """Monitoring rows + extra (log-scale uncertainty, layer2) + cluster ids + tail-extended val_cal percentiles."""
    s4 = out_dir / "stage4"
    cols = KEYS + ["patient", "label", "pred", "confidence", "correct", "sf_tau", "sf_090", "sf_095"] + RAW
    mon = pd.read_parquet(s4 / "monitoring.parquet", columns=cols)
    ex = pd.concat([pd.read_parquet(f, columns=KEYS + EXTRA + ["pred_check"]) for f in sorted((s4 / "signals_extra").glob("*.parquet"))],
                   ignore_index=True)
    df = mon.merge(ex, on=KEYS, how="left", validate="one_to_one")
    assert df[EXTRA].notna().all().all(), "extra signals missing for some rows"
    assert (df["pred"] == df["pred_check"]).mean() > 0.999, "extra pass predictions disagree with the monitoring table"
    splits = pd.read_parquet(out_dir / "stage1" / "splits.parquet", columns=["uid", "dataset", "group"])
    grp = splits[splits.dataset == "pathmnist"].set_index("uid")["group"]
    df["cluster"] = np.where(df["dataset"] == "camelyon17", "p" + df["patient"].astype(str),
                             "g" + df["uid"].map(grp).fillna(-1).astype(int).astype(str))
    # tail-extended percentiles and the stricter label, per model, from that model's val_cal rows only
    feats = RAW + EXTRA
    for c in feats:
        df["x_" + c] = np.nan
    df["sf_099"] = False
    tau99 = []
    for (ds, kind, seed), idx in df.groupby(["dataset", "model_kind", "seed"]).groups.items():
        g = df.loc[idx]
        vc = g[g.family == "VALCAL"]
        for c in feats:
            df.loc[idx, "x_" + c] = nz.ecdf_percentile_ext(vc[c].to_numpy(), g[c].to_numpy())
        t = _tau(vc["confidence"].to_numpy(), vc["correct"].to_numpy(), 0.99)
        df.loc[idx, "sf_099"] = (~g["correct"]) & (g["confidence"] >= t["tau"])
        tau99.append({"dataset": ds, "kind": kind, "seed": seed, **t})
    df.attrs["tau99"] = tau99
    return df


def _tau(conf, correct, target):
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order].astype(float)
    prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    valid = np.where(prec >= target)[0]
    if len(valid) == 0:
        return {"tau": 1.0 + 1e-9, "degenerate": "no threshold reaches the target"}
    tau = float(c[valid.max()])
    return {"tau": tau, "degenerate": "accepts every val_cal prediction" if tau <= c.min() + 1e-12 else None,
            "val_cal_acc": float(ok.mean())}


def family_pool(df: pd.DataFrame, fam: str, c_kind: str = "shortcutv2") -> pd.DataFrame:
    """Pre-specified evaluation pools (plan Sec 1)."""
    if fam == "A":
        m = (df.family == "A") & (df.model_kind == "standard")
    elif fam == "B":
        m = (df.dataset == "camelyon17") & (df.model_kind == "standard") & df.family.isin(["ID", "B"])
    elif fam == "C":
        m = (df.family == "C") & (df.model_kind == c_kind)
    elif fam == "X":
        m = (df.dataset == "pathmnist") & (df.model_kind == "standard") & df.family.isin(["ID", "X"])
    else:
        raise ValueError(fam)
    return df[m].assign(pool=fam)


def rotation(df: pd.DataFrame, heldout: str, c_kind: str = "shortcutv2") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tuning rows, held-out rows). Tuning never contains a row of the held-out family (asserted)."""
    fams = ["A", "B", "C"]
    tune_f = [f for f in fams if f != heldout] if heldout in fams else fams
    tune = pd.concat([family_pool(df, f, c_kind) for f in tune_f], ignore_index=True)
    test = family_pool(df, heldout, c_kind)
    assert heldout not in set(tune["pool"]), "held-out family leaked into tuning rows"
    assert not tune.index.empty
    return tune, test


# --------------------------------------------------------------------------- fusion ----------
def fit_nonneg_logreg(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Weighted logistic regression, weights >= 0 (monotone), intercept free, L2 on the weights. Returns [b0, b...]."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    w = np.asarray(w, np.float64) / np.sum(w)

    def f(theta):
        z = theta[0] + X @ theta[1:]
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        loss = -np.sum(w * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        g = w * (p - y)
        grad = np.concatenate([[g.sum()], X.T @ g + 2 * lam * theta[1:]])
        return loss + lam * np.sum(theta[1:] ** 2), grad
    theta0 = np.zeros(X.shape[1] + 1)
    bounds = [(None, None)] + [(0, None)] * X.shape[1]
    r = minimize(f, theta0, jac=True, bounds=bounds, method="L-BFGS-B", options={"maxiter": 500})
    return r.x


def family_balanced_weights(pool: pd.Series) -> np.ndarray:
    counts = pool.value_counts()
    return (1.0 / (len(counts) * counts.reindex(pool).to_numpy())).astype(np.float64)


def fused_scores(tune: pd.DataFrame, test: pd.DataFrame, feats: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    X = tune[["x_" + c for c in feats]].to_numpy()
    th = fit_nonneg_logreg(X, tune[label].to_numpy(), family_balanced_weights(tune["pool"]))
    return test[["x_" + c for c in feats]].to_numpy() @ th[1:] + th[0], th


# --------------------------------------------------------------------------- metrics ---------
class Ranked:
    """Pre-sorted score with tie groups, so weighted AUROC / AUPRC / AURC cost O(n) per bootstrap replicate."""

    def __init__(self, score: np.ndarray, y: np.ndarray):
        o = np.argsort(score, kind="mergesort")
        s = np.asarray(score)[o]
        self.order, self.y = o, np.asarray(y, bool)[o]
        self.start = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]       # first index of each tie group (ascending)

    def metrics(self, w: np.ndarray) -> tuple[float, float, float]:
        w = np.asarray(w, np.float64)[self.order]
        pw, nw = np.add.reduceat(w * self.y, self.start), np.add.reduceat(w * ~self.y, self.start)
        P, N = pw.sum(), nw.sum()
        if P <= 0 or N <= 0:
            return np.nan, np.nan, np.nan
        auroc = float(np.sum(pw * (np.cumsum(nw) - nw + 0.5 * nw)) / (P * N))
        # AUPRC (average precision): thresholds from high to low score, tie groups treated together
        # (tie groups with zero bootstrap weight contribute exactly 0; guard the 0/0 they would otherwise produce)
        pr, tw = pw[::-1], (pw + nw)[::-1]
        ctw = np.cumsum(tw)
        prec = np.divide(np.cumsum(pr), ctw, out=np.zeros_like(ctw), where=ctw > 0)
        ap = float(np.sum(pr * prec) / P)
        # AURC: accept lowest risk scores first; risk = positive rate among accepted
        cp, ct = np.cumsum(pw), np.cumsum(pw + nw)
        risk = np.divide(cp, ct, out=np.zeros_like(ct), where=ct > 0)
        aurc = float(np.sum((pw + nw) * risk) / (P + N))
        return auroc, ap, aurc


def boot_setup(test: pd.DataFrame, seeds: list[int]) -> dict:
    """Cluster index per row (shared across seeds) for the paired two-level bootstrap."""
    cl = pd.Index(sorted(test["cluster"].unique()))
    return {"clusters": cl, "row_cluster": {s: cl.get_indexer(test.loc[test.seed == s, "cluster"]) for s in seeds}}


def paired_bootstrap(scores: dict, ys: dict, setup: dict, seeds: list[int], B: int, rng_seed: int = 0) -> dict:
    """scores[model][seed] -> score array, ys[seed] -> labels. Returns per model the point estimate (mean over seeds)
    and the [B, 3] bootstrap matrix of (AUROC, AUPRC, AURC); all models share every resample (paired)."""
    rng = np.random.default_rng(rng_seed)
    ranked = {m: {s: Ranked(scores[m][s], ys[s]) for s in seeds} for m in scores}
    point = {m: np.nanmean([ranked[m][s].metrics(np.ones(len(ys[s]))) for s in seeds], 0) for m in scores}
    per_seed = {m: {s: ranked[m][s].metrics(np.ones(len(ys[s]))) for s in seeds} for m in scores}
    nC = len(setup["clusters"])
    boot = {m: np.empty((B, 3)) for m in scores}
    for b in range(B):
        sdraw = rng.choice(seeds, size=len(seeds), replace=True)
        cnt = np.bincount(rng.integers(0, nC, nC), minlength=nC).astype(np.float64)
        vals = {m: [] for m in scores}
        for s in sdraw:
            w = cnt[setup["row_cluster"][s]]
            for m in scores:
                vals[m].append(ranked[m][s].metrics(w))
        for m in scores:
            boot[m][b] = np.nanmean(vals[m], 0)
    return {"point": point, "boot": boot, "per_seed": per_seed}


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def boot_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(pv)
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


# --------------------------------------------------------------------------- analysis ---------
def evaluate(df: pd.DataFrame, heldout: str, detectors: dict, label: str, seeds: list[int], B: int,
             c_kind: str = "shortcutv2", subset=None) -> dict:
    """Fit each detector per seed on the tuning families, score the held-out pool, paired bootstrap."""
    tune, test = rotation(df, heldout, c_kind)
    if subset is not None:
        test = test[subset(test)]
    scores, ys, weights = {m: {} for m in detectors}, {}, []
    for s in seeds:
        tu, te = tune[tune.seed == s], test[test.seed == s]
        ys[s] = te[label].to_numpy()
        for m, feats in detectors.items():
            sc, th = fused_scores(tu, te, feats, label)
            scores[m][s] = sc
            weights.append({"heldout": heldout, "label": label, "model": m, "seed": s,
                            **{f"w_{c}": float(v) for c, v in zip(["intercept"] + feats, th)}})
    test = test[test.seed.isin(seeds)]
    res = paired_bootstrap(scores, ys, boot_setup(test, seeds), seeds, B)
    res.update(weights=weights, n=int(len(test) / len(seeds)), positives=float(np.mean([ys[s].mean() for s in seeds])))
    return res


def contrast_rows(res: dict, heldout: str, extra: dict) -> list[dict]:
    rows = []
    for m in res["point"]:
        for k, met in enumerate(("AUROC", "AUPRC", "AURC")):
            lo, hi = ci(res["boot"][m][:, k])
            rows.append({"heldout": heldout, **extra, "quantity": m, "metric": met, "estimate": res["point"][m][k],
                         "ci_low": lo, "ci_high": hi, "n_per_seed": res["n"], "sf_rate": res["positives"]})
    for a, b in (("M1", "M0"), ("M2", "M1")):
        if a in res["point"] and b in res["point"]:
            for k, met in enumerate(("AUROC", "AUPRC", "AURC")):
                d = res["boot"][a][:, k] - res["boot"][b][:, k]
                lo, hi = ci(d)
                jk = [res["per_seed"][a][s][k] - res["per_seed"][b][s][k] for s in res["per_seed"][a]]
                loo = [np.mean([x for j, x in enumerate(jk) if j != i]) for i in range(len(jk))]
                rows.append({"heldout": heldout, **extra, "quantity": f"{a}-{b}", "metric": f"d{met}",
                             "estimate": res["point"][a][k] - res["point"][b][k], "ci_low": lo, "ci_high": hi,
                             "p_boot": boot_p(d), "jackknife_min": float(np.min(loo)), "jackknife_max": float(np.max(loo)),
                             "n_per_seed": res["n"], "sf_rate": res["positives"]})
    return rows


def verdict(r: pd.Series) -> str:
    better_high = r["metric"] != "dAURC"
    if r["ci_low"] > 0:
        return "improves" if better_high else "worse"
    if r["ci_high"] < 0:
        return "worse" if better_high else "improves"
    return "inconclusive"


def inversion_diagnosis(df: pd.DataFrame, seeds: list[int]) -> dict:
    """Family C v2: single-signal AUROC pooled vs within each cue state; profile of the correct predictions under reversal."""
    from sklearn.metrics import roc_auc_score
    c = family_pool(df, "C")
    sig = ["u_log1msp", "r_maha_pred", "r_knn", "e_js_norm_l4"]
    out = {"auroc_sf090": {}, "reversed_profile": {}}
    for state in ("pooled", "aligned", "removed", "reversed"):
        g = c if state == "pooled" else c[c.condition == state]
        row = {"n_per_seed": int(len(g) / len(seeds)), "sf_rate": float(g.sf_090.mean()), "acc": float(g.correct.mean())}
        for s_ in sig:
            vals = [roc_auc_score(gg.sf_090, gg["x_" + s_]) for _, gg in g.groupby("seed") if 0 < gg.sf_090.mean() < 1]
            row[s_] = float(np.mean(vals)) if vals else None
        out["auroc_sf090"][state] = row
    r = c[c.condition == "reversed"]
    for name, g in (("correct", r[r.correct]), ("wrong_confident (sf_090)", r[r.sf_090]),
                    ("wrong_low_conf", r[~r.correct & ~r.sf_090])):
        out["reversed_profile"][name] = {
            "n_per_seed": float(len(g) / len(seeds)), "mean_confidence": float(g.confidence.mean()) if len(g) else None,
            "class_counts": {int(k): int(v) for k, v in g.label.value_counts().sort_index().items()},
            **{f"median_pct_{s_}": float(g["x_" + s_].median()) if len(g) else None for s_ in sig}}
    pr = out["auroc_sf090"]
    within = pr["reversed"]["u_log1msp"]
    out["reading"] = ("inversion holds WITHIN the reversed state (not a pooling artefact)" if within is not None and within < 0.5
                      else "inversion is driven by pooling across cue states")
    return out


def run(cfg: dict, out_dir: Path, log=print) -> None:
    import statsmodels.api as sm
    s5 = cfg["stage5"]
    out = out_dir / "stage5"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds, B = cfg["train"]["seeds"], s5["bootstrap_B"]
    t0 = time.time()
    df = load(cfg, out_dir)
    pd.DataFrame(df.attrs["tau99"]).to_csv(out / "tau99.csv", index=False)
    ties = {}
    for (ds, kind), g in df.groupby(["dataset", "model_kind"]):
        ties[f"{ds}/{kind}"] = {c: float(1 - g.groupby("seed")[c].nunique().sum() / len(g)) for c in ("u_log1msp", "u_1msp")}
    (out / "tie_report.json").write_text(json.dumps(ties, indent=1))
    log(f"[stage5] data loaded: {len(df):,} rows ({time.time() - t0:.0f} s)")

    # primary: Table 2 + Fig 4
    rows, weights, stab = [], [], []
    for H in ("A", "B", "C"):
        res = evaluate(df, H, DETECTORS, "sf_090", seeds, B)
        rows += contrast_rows(res, H, {"analysis": "primary", "label": "sf_090"})
        weights += res["weights"]
        for mdl, d in res["per_seed"].items():
            for sd, v in d.items():
                stab.append({"heldout": H, "model": mdl, "seed": sd, "AUROC": v[0], "AUPRC": v[1], "AURC": v[2]})
        log(f"[stage5] primary held-out {H} done ({time.time() - t0:.0f} s)")
    res = evaluate(df, "X", DETECTORS, "sf_090", seeds, B)
    rows += contrast_rows(res, "X", {"analysis": "supplement X", "label": "sf_090"})
    weights += res["weights"]
    prim = pd.DataFrame(rows)
    pd.DataFrame(weights).to_csv(out / "fusion_weights.csv", index=False)

    # sensitivity (one BH family across all deltas)
    sens = []
    for name, dets in SENSITIVITY.items():
        d = dict(DETECTORS, **dets)
        for H in ("A", "B", "C"):
            sens += contrast_rows(evaluate(df, H, d, "sf_090", seeds, B), H, {"analysis": "sensitivity", "variant": name})
    for lab in LABELS_SENS:
        for H in ("A", "B", "C"):
            sens += contrast_rows(evaluate(df, H, DETECTORS, lab, seeds, B), H, {"analysis": "sensitivity", "variant": f"label {lab}"})
    for H in ("A", "B", "C"):
        sens += contrast_rows(evaluate(df, H, DETECTORS, "sf_090", seeds, B, c_kind="shortcut"), H,
                              {"analysis": "sensitivity", "variant": "v1 weak shortcut for family C"})
    sens = pd.DataFrame(sens)
    m = sens.quantity.str.contains("-")
    sens.loc[m, "q_bh"] = bh(sens.loc[m, "p_boot"].to_numpy())
    sens.to_csv(out / "sensitivity.csv", index=False)
    log(f"[stage5] sensitivity done ({time.time() - t0:.0f} s)")

    # secondary breakdowns (each breakdown is its own BH family)
    sec = []
    def add(fam_name, H, key, values, col):
        part = []
        for v in values:
            r = evaluate(df, H, DETECTORS, "sf_090", seeds, B, subset=lambda t, v=v: t[col] == v)
            part += contrast_rows(r, H, {"analysis": fam_name, "level": str(v)})
        part = pd.DataFrame(part)
        mm = part.quantity.str.contains("-")
        part.loc[mm, "q_bh"] = bh(part.loc[mm, "p_boot"].to_numpy())
        sec.append(part)
    a = family_pool(df, "A")
    add("A by corruption", "A", "condition", sorted(a.condition.unique()), "condition")
    add("A by severity", "A", "severity", sorted(a.severity.unique()), "severity")
    add("B by condition", "B", "condition", ["id_test", "ood_c1", "ood_c2"], "condition")
    add("C by condition", "C", "condition", ["aligned", "removed", "reversed"], "condition")
    sec = pd.concat(sec, ignore_index=True)
    sec.to_csv(out / "secondary_breakdowns.csv", index=False)
    log(f"[stage5] secondary breakdowns done ({time.time() - t0:.0f} s)")

    # Fig 3: single-signal detection AUROC / AUPRC per held-out family (no fitting)
    f3 = []
    sigs = ["u_log1msp", "u_logent", "r_maha_pred", "r_knn", "e_js_norm_l4", "e_js_raw_l4", "e_pca_l4", "e_js_norm_l3", "e_js_norm_l2"]
    for H in ("A", "B", "C", "X"):
        test = family_pool(df, H)
        scores = {s_: {s: test.loc[test.seed == s, "x_" + s_].to_numpy() for s in seeds} for s_ in sigs}
        ys = {s: test.loc[test.seed == s, "sf_090"].to_numpy() for s in seeds}
        r = paired_bootstrap(scores, ys, boot_setup(test, seeds), seeds, s5["fig3_B"])
        for s_ in sigs:
            for k, met in ((0, "AUROC"), (1, "AUPRC")):
                lo, hi = ci(r["boot"][s_][:, k])
                f3.append({"family": H, "signal": s_, "metric": met, "estimate": r["point"][s_][k], "ci_low": lo, "ci_high": hi})
    pd.DataFrame(f3).to_csv(out / "fig3_data.csv", index=False)

    # incremental logistic regression (secondary, descriptive; per seed on the held-out family)
    lr = []
    for H in ("A", "B", "C"):
        tune, test = rotation(df, H)
        for s in seeds:
            tu, te = tune[tune.seed == s], test[test.seed == s]
            m0, _ = fused_scores(tu, te, DETECTORS["M0"], "sf_090")
            y = te.sf_090.to_numpy().astype(float)
            X0 = sm.add_constant(np.c_[m0])
            X1 = sm.add_constant(np.c_[m0, te.x_r_maha_pred])
            X2 = sm.add_constant(np.c_[m0, te.x_r_maha_pred, te.x_e_js_norm_l4])
            f0, f1, f2 = (sm.Logit(y, X).fit(disp=0, maxiter=200) for X in (X0, X1, X2))
            lr.append({"heldout": H, "seed": s, "LR_M1_vs_M0": 2 * (f1.llf - f0.llf), "wald_p_maha": float(f1.pvalues[2]),
                       "coef_maha": float(f1.params[2]), "LR_M2_vs_M1": 2 * (f2.llf - f1.llf),
                       "wald_p_cam": float(f2.pvalues[3]), "coef_cam": float(f2.params[3])})
    pd.DataFrame(lr).to_csv(out / "incremental_logit.csv", index=False)

    inv = inversion_diagnosis(df, seeds)
    (out / "inversion_diagnosis.json").write_text(json.dumps(inv, indent=1))

    # Table 2, seed stability, verdicts, figures
    prim["verdict"] = [verdict(r) if "-" in r.quantity else "" for _, r in prim.iterrows()]
    prim.to_csv(out / "table2.csv", index=False)
    st = pd.DataFrame(stab)
    st.to_csv(out / "seed_stability_per_seed.csv", index=False)
    st.groupby(["heldout", "model"])[["AUROC", "AUPRC", "AURC"]].agg(["mean", "std", "min", "max"]).round(4).to_csv(out / "seed_stability.csv")
    _figures(prim, pd.DataFrame(f3), out)
    _summary(prim, sens, sec, inv, ties, out)
    log(f"[stage5] all done ({time.time() - t0:.0f} s)")


def _figures(prim: pd.DataFrame, f3: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, met in enumerate(("AUROC", "AUPRC")):
        d = f3[f3.metric == met].pivot_table(index="signal", columns="family", values="estimate")
        d.plot.barh(ax=ax[k])
        ax[k].set_title(f"Fig 3: single-signal silent-failure {met} (sf_090)")
        if met == "AUROC":
            ax[k].axvline(0.5, color="k", lw=0.8)
    fig.tight_layout(); fig.savefig(out / "figures" / "fig3_single_signals.png", dpi=130); plt.close(fig)
    d = prim[prim.quantity.isin(["M1-M0", "M2-M1"]) & prim.heldout.isin(["A", "B", "C"])]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, met in enumerate(("dAUROC", "dAUPRC", "dAURC")):
        g = d[d.metric == met].reset_index(drop=True)
        lab = g.heldout + " " + g.quantity
        ax[k].errorbar(g.estimate, range(len(g)), xerr=[g.estimate - g.ci_low, g.ci_high - g.estimate], fmt="o", capsize=3)
        ax[k].set_yticks(range(len(g))); ax[k].set_yticklabels(lab); ax[k].axvline(0, color="k", lw=0.8)
        ax[k].set_title(f"Fig 4: {met} (95% paired CI)")
    fig.tight_layout(); fig.savefig(out / "figures" / "fig4_increments.png", dpi=130); plt.close(fig)


def _summary(prim, sens, sec, inv, ties, out: Path) -> None:
    t2 = prim[prim.heldout.isin(["A", "B", "C"])]
    fmt = lambda r: f"{r.estimate:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"  # noqa: E731
    lines = ["# Stage 5 - sample-level results (pre-registered plan: docs/analysis_plan_frozen.md)\n",
             "## Table 2 - held-out family, label sf_090 (point = mean over 5 seeds; paired cluster bootstrap 95% CI)\n"]
    tab = t2.assign(value=[fmt(r) for r in t2.itertuples()])[["heldout", "quantity", "metric", "value", "verdict"]]
    lines.append(to_markdown(tab))
    lines.append("\n## Does M2 improve over M1 out-of-family? (generated from the numbers)\n")
    for H in ("A", "B", "C"):
        r = t2[(t2.heldout == H) & (t2.quantity == "M2-M1") & (t2.metric == "dAUROC")].iloc[0]
        lines.append(f"- Held-out {H}: dAUROC(M2-M1) = {fmt(r)} -> **{r.verdict}**.")
    for H in ("A", "B", "C"):
        r = t2[(t2.heldout == H) & (t2.quantity == "M1-M0") & (t2.metric == "dAUROC")].iloc[0]
        lines.append(f"- Held-out {H}: dAUROC(M1-M0) = {fmt(r)} -> **{r.verdict}**.")
    s = sens[sens.metric == "dAUROC"]
    lines.append("\n## Sensitivity (dAUROC, BH q within the sensitivity family)\n")
    lines.append(to_markdown(s.assign(value=[fmt(r) for r in s.itertuples()])[["variant", "heldout", "quantity", "value", "q_bh"]].round(4)))
    lines.append("\n## Family C inversion diagnosis\n```\n" + json.dumps(inv, indent=1) + "\n```\n")
    lines.append("## Tie report (fraction of tied values per model kind)\n```\n" + json.dumps(ties, indent=1) + "\n```\n")
    (out / "summary_stage5.md").write_text("\n".join(lines), encoding="utf-8")

