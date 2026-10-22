"""Stage 8: ablations and sensitivity (+ an exploratory, post-hoc targeted-fusion analysis).
Implements docs/ablation_plan_frozen.md (pre-registration). Reuses the saved Phase 4-7 signals and per-row features: no
retraining, no GPU, no image re-extraction. Weights/gates are fitted only on the two tuning families (rotation).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from sfd import stage4 as s4
from sfd import stage5 as s5
from sfd import stage7 as s7
from sfd.data import classify_data as cd
from sfd.signals import normalize as nz
from sfd.signals import representation as rp
from sfd.signals import uncertainty as un

BUDGETS = s7.BUDGETS
SIZES = (500, 1000, 2000, "full")
GATE = 0.90
M0F = ["u_log1msp", "u_logent"]
MAHA, KNN, CAM4 = "r_maha_pred", "r_knn", "e_js_norm_l4"
CAM_VARIANTS = ["e_js_norm_l3", "e_js_norm_l2", "e_js_raw_l4", "e_pca_l4"]
AUC_MIN_EFFECT = 0.005
LEVEL_METRICS = ("resid_sf", "resid_err", "capture")
AUC_METRICS = ("auroc", "auprc", "aurc")
NEEDED = s5.KEYS + ["cluster", "correct", "confidence", "sf_090", "sf_095", "sf_tau", "sf_099"]


# --------------------------------------------------------------------------- augmented features -----
def _maha_pred(feat: np.ndarray, pred: np.ndarray, means: np.ndarray, prec: np.ndarray, chunk: int = 16384) -> np.ndarray:
    out = np.empty(len(feat), np.float32)
    P = prec.astype(np.float32)
    for a in range(0, len(feat), chunk):
        d = feat[a:a + chunk].astype(np.float32) - means[pred[a:a + chunk]].astype(np.float32)
        out[a:a + chunk] = np.sqrt(np.maximum(np.einsum("ij,ij->i", d @ P, d), 0))
    return out


def reconstruct_logits(feat: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return feat.astype(np.float32) @ W.T.astype(np.float32) + b.astype(np.float32)


def _softmax_conf(lo: np.ndarray, T: float) -> np.ndarray:
    z = lo / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).max(1)


def build_aug(cfg: dict, out_dir: Path, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model: Mahalanobis refit on nested subsets of the saved Phase 4 reference bank, T = 1 uncertainty features."""
    import torch
    from scipy.stats import spearmanr
    p = s4.P(cfg, out_dir)
    p["s6"] = out_dir / "stage6"
    stage1 = pd.read_parquet(p["stage1"])
    frames, checks = [], []
    perms: dict[str, np.ndarray] = {}
    for spec in s4.model_specs(cfg, p):
        t0 = time.time()
        z = np.load(p["s6"] / "rows" / f"{spec['name']}.npz")
        fam, pred, feat = z["fam"], z["pred"].astype(np.int64), z["feat"]
        vc = fam == "VALCAL"
        zr = np.load(p["out"] / "refs" / f"{spec['name']}.npz")
        bank, idx = zr["knn_bank"].astype(np.float32), zr["ref_uid_pos"]
        lab = cd.split_frame(stage1, spec["dataset"], "train")["label"].to_numpy(np.int64)[idx]
        perm = perms.setdefault(spec["dataset"], np.random.default_rng(0).permutation(len(bank)))
        d = {"family": fam, "condition": z["cond"], "severity": np.where(fam == "A", z["sev"], 0).astype(np.int64), "uid": z["uid"],
             "model_kind": spec["kind"], "dataset": spec["dataset"], "seed": spec["seed"]}
        chk = {"model": spec["name"]}
        for n in SIZES:
            sub = perm if n == "full" else perm[:n]
            g = rp.fit_class_gaussian(bank[sub], lab[sub], spec["n_classes"])
            m = _maha_pred(feat, pred, g["means"], g["precision"])
            d[f"x_r_maha_N{n}"] = nz.ecdf_percentile_ext(m[vc], m)
            chk[f"min_class_rows_N{n}"] = int(np.bincount(lab[sub], minlength=spec["n_classes"]).min())
            if n == "full":
                ii = np.random.default_rng(1).choice(len(m), min(30000, len(m)), replace=False)
                chk["spearman_full_refit_vs_stored"] = float(spearmanr(m[ii], z["maha"][ii])[0])
        sd = torch.load(spec["ckpt"], map_location="cpu")
        lo = reconstruct_logits(feat, sd["net.fc.weight"].numpy(), sd["net.fc.bias"].numpy())
        dc = np.abs(_softmax_conf(lo, spec["T"]) - z["conf"])
        chk["logit_conf_max_abs_diff"], chk["logit_conf_p999_abs_diff"] = float(dc.max()), float(np.quantile(dc, 0.999))
        # fp16 feature storage is amplified by 1/T for the saturated v2 models (T ~ 0.08): plan wording "max < 1e-3" relaxed to
        # p99.9 < 1e-3 and max < 1e-2 (deviation reported in the README)
        assert chk["logit_conf_p999_abs_diff"] < 1e-3 and chk["logit_conf_max_abs_diff"] < 1e-2,             f"logit reconstruction disagrees with stored confidence ({spec['name']})"
        assert (lo.argmax(1) == pred).mean() > 0.999
        for nm, fn in (("u_log1msp_T1", un.log_one_minus_msp), ("u_logent_T1", un.log_entropy)):
            v = fn(lo.astype(np.float64), 1.0)
            d["x_" + nm] = nz.ecdf_percentile_ext(v[vc], v)
        frames.append(pd.DataFrame(d))
        checks.append(chk)
        log(f"[stage8] augment {spec['name']}: {len(feat):,} rows ({time.time() - t0:.0f} s)")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(checks)


def augment(cfg: dict, out_dir: Path, df: pd.DataFrame, log=print) -> pd.DataFrame:
    f = out_dir / "stage8" / "aug_features.parquet"
    (out_dir / "stage8").mkdir(parents=True, exist_ok=True)
    if f.exists():
        aug = pd.read_parquet(f)
    else:
        aug, checks = build_aug(cfg, out_dir, log)
        aug.to_parquet(f)
        checks.to_csv(out_dir / "stage8" / "augment_checks.csv", index=False)
    new = [c for c in aug.columns if c.startswith("x_")]
    keys = ["family", "condition", "severity", "model_kind", "dataset", "seed", "uid"]
    a, b = df.copy(), aug.copy()
    for d in (a, b):
        d["severity"] = np.where(d["family"] == "A", d["severity"].fillna(0), 0).astype(np.int64)
        d["seed"] = d["seed"].astype(np.int64)
    m = a.merge(b[keys + new], on=keys, how="left", validate="one_to_one")
    assert m[new].notna().all().all(), "augmented features missing for some rows"
    return m


# --------------------------------------------------------------------------- fitting -----------------
def fit_logreg(X, y, w, lam: float = 1e-3, nonneg: bool = True) -> np.ndarray:
    """Weighted logistic regression, L2 on the weights, intercept free. nonneg=True is the Phase 5 fusion (asserted equal in tests);
    nonneg=False lets every weight take either sign (ablation G2 and exploratory variants)."""
    X, y, w = np.asarray(X, np.float64), np.asarray(y, np.float64), np.asarray(w, np.float64)
    w = w / w.sum()

    def f(theta):
        z = theta[0] + X @ theta[1:]
        pr = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        loss = -np.sum(w * (y * np.log(np.clip(pr, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - pr, 1e-12, 1))))
        g = w * (pr - y)
        return loss + lam * np.sum(theta[1:] ** 2), np.concatenate([[g.sum()], X.T @ g + 2 * lam * theta[1:]])
    bounds = [(None, None)] + [((0, None) if nonneg else (None, None))] * X.shape[1]
    return minimize(f, np.zeros(X.shape[1] + 1), jac=True, bounds=bounds, method="L-BFGS-B", options={"maxiter": 500}).x


def gated_scores(sg: np.ndarray, m0: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Single ranking score: every gated case outranks every non-gated case; gated cases are ordered by `sg`, non-gated by the
    plain M0 score `m0` (the fill rule when the gated pool is smaller than the budget). Ties are preserved (average ranks)."""
    out = np.empty(len(sg))
    out[gate] = len(sg) + rankdata(sg[gate])
    out[~gate] = rankdata(m0[~gate])
    return out


def registry() -> dict:
    R: dict[str, dict] = {}
    for con in ("nn", "sf"):
        pre = "" if con == "nn" else "sf_"
        fit = lambda feats, **kw: {"kind": "fit", "feats": feats, "con": con, **kw}
        R[pre + "M0"] = fit(M0F)
        R[pre + "M1"] = fit(M0F + [MAHA])
        R[pre + "M2"] = fit(M0F + [MAHA, CAM4])
        R[pre + "M0c"] = fit(M0F + [CAM4])
        for c in CAM_VARIANTS:
            R[f"{pre}M1+{c}"] = fit(M0F + [MAHA, c])
        R[pre + "M1knn"] = fit(M0F + [KNN])
        R[pre + "M2knn"] = fit(M0F + [KNN, CAM4])
        for n in SIZES:
            R[f"{pre}M1_N{n}"] = fit(M0F + [f"r_maha_N{n}"])
    R["M0_T1"] = {"kind": "fit", "feats": ["u_log1msp_T1", "u_logent_T1"], "con": "nn"}
    R["M1_T1"] = {"kind": "fit", "feats": ["u_log1msp_T1", "u_logent_T1", MAHA], "con": "nn"}
    for f in ("u_log1msp", "u_logent", MAHA, KNN, CAM4):
        R["s_" + f] = {"kind": "single", "feat": f}
    R["G_M0"] = {"kind": "fit", "feats": M0F, "con": "nn", "gate": GATE}
    R["G_M1"] = {"kind": "fit", "feats": M0F + [MAHA], "con": "nn", "gate": GATE}
    R["G_sfM1"] = {"kind": "fit", "feats": M0F + [MAHA], "con": "sf", "gate": GATE}
    R["G_sfM2"] = {"kind": "fit", "feats": M0F + [MAHA, CAM4], "con": "sf", "gate": GATE}
    return R


def pairs_main() -> list[tuple]:
    """(group, a, b, primary) contrasts of the pre-registered groups G1-G6 and the exploratory table (E)."""
    P = []
    P += [("G1", "M1", "M0", False), ("G1", "M0c", "M0", False), ("G1", "M2", "M1", False)]
    P += [("G1", "s_" + f, "s_u_log1msp", False) for f in ("u_logent", MAHA, KNN, CAM4)]
    P += [("G2", "sf_M0", "M0", False), ("G2", "sf_M1", "M1", True), ("G2", "sf_M2", "M2", False),
          ("G2", "sf_M1", "sf_M0", True), ("G2", "sf_M2", "sf_M1", True)]
    for pre in ("", "sf_"):
        P += [("G3", f"{pre}M1+{c}", pre + "M1", False) for c in CAM_VARIANTS]
        P += [("G3", f"{pre}M1+{c}", pre + "M2", False) for c in CAM_VARIANTS]
        P += [("G4", pre + "M1knn", pre + "M1", False), ("G4", pre + "M2knn", pre + "M2", False)]
        P += [("G5", f"{pre}M1_N{n}", f"{pre}M1_Nfull", False) for n in SIZES[:-1]]
        P += [("G5", f"{pre}M1_N{n}", pre + "M0", False) for n in SIZES]
        P += [("G5", f"{pre}M1_Nfull", pre + "M1", False)]
    P += [("G6", "M0_T1", "M0", False), ("G6", "M1_T1", "M1", False)]
    P += [("G9", "M1", "M0", False), ("G9", "M2", "M1", False), ("G9", "sf_M1", "M0", False)]
    for g in ("G_M0", "G_M1", "G_sfM1", "G_sfM2", "sf_M0", "sf_M1", "sf_M2"):
        P += [("E", g, "M0", False), ("E", g, "M1", False), ("E", g, "random", False)]
    P += [("E", "M0", "random", False), ("E", "M1", "random", False)]
    return P


GROUP_NAMES = {
    "G1": ["M0", "M1", "M0c", "M2", "s_u_log1msp", "s_u_logent", "s_" + MAHA, "s_" + KNN, "s_" + CAM4],
    "G2": ["M0", "M1", "M2", "sf_M0", "sf_M1", "sf_M2"],
    "G3": [n for pre in ("", "sf_") for n in [pre + "M1", pre + "M2"] + [f"{pre}M1+{c}" for c in CAM_VARIANTS]],
    "G4": [pre + n for pre in ("", "sf_") for n in ("M1", "M2", "M1knn", "M2knn")],
    "G5": ["M0", "M1"] + [f"{pre}M1_N{n}" for pre in ("", "sf_") for n in SIZES],
    "G6": ["M0", "M0_T1", "M1", "M1_T1"],
    "E": ["random", "oracle", "M0", "M1", "sf_M0", "sf_M1", "sf_M2", "G_M0", "G_M1", "G_sfM1", "G_sfM2"],
}
MAIN_NAMES = sorted({n for v in GROUP_NAMES.values() for n in v if n not in ("random", "oracle")})


# --------------------------------------------------------------------------- contexts ---------------
def make_ctx(df: pd.DataFrame, heldout: str, label: str, seeds: list[int], c_kind: str = "shortcutv2") -> dict:
    tune, test = s5.rotation(df, heldout, c_kind)
    xcols = [c for c in df.columns if c.startswith("x_")]
    ctx = {"heldout": heldout, "label": label, "c_kind": c_kind, "te": {}, "tu": {}, "xcols": xcols}
    keep_te = ["seed", "uid", "cluster", "correct", "confidence", "condition", "severity", label] + ["x_" + c for c in []] + xcols
    keep_tu = ["uid", "family", "condition", "pool", "confidence", label] + xcols
    for s in seeds:
        te = test[test.seed == s]
        tu = tune[tune.seed == s]
        assert heldout not in set(tu["pool"]), "held-out family leaked into tuning rows"
        if s == seeds[0]:
            key = lambda d: d["family"].astype(str) + "|" + d["condition"].astype(str) + "|" + d["uid"].astype(str)
            assert not (set(key(tu)) & set(key(te))), "tuning and held-out rows overlap"
        ctx["te"][s] = te[list(dict.fromkeys(keep_te))].reset_index(drop=True)
        ctx["tu"][s] = tu[keep_tu].reset_index(drop=True)
    return ctx


def score_methods(ctx: dict, names: list[str], seeds: list[int], reg: dict | None = None) -> tuple[dict, list[dict]]:
    """Fit every requested method per seed on the TUNING rows only and score the held-out pool. Gated methods fall back to the plain
    NN M0 score outside the gate."""
    R = reg if reg is not None else registry()
    label = ctx["label"]
    scores = {n: {} for n in names}
    weights = []
    need_m0 = any(R[n].get("gate") for n in names)
    for s in seeds:
        te, tu = ctx["te"][s], ctx["tu"][s]
        w_all = s5.family_balanced_weights(tu["pool"])
        cache = {}
        def fit_one(n):
            spec = R[n]
            xs = ["x_" + c for c in spec["feats"]]
            m = np.ones(len(tu), bool) if not spec.get("gate") else tu["confidence"].to_numpy() >= spec["gate"]
            w = w_all if m.all() else s5.family_balanced_weights(tu.loc[m, "pool"])
            th = fit_logreg(tu.loc[m, xs].to_numpy(), tu.loc[m, label].to_numpy(), w, nonneg=spec["con"] == "nn")
            weights.append({"heldout": ctx["heldout"], "label": label, "model": n, "seed": s,
                            **{f"w_{c}": float(v) for c, v in zip(["intercept"] + spec["feats"], th)}})
            return te[xs].to_numpy() @ th[1:] + th[0]
        m0 = None
        if need_m0:
            m0 = fit_one("M0") if "M0" not in names else None
        for n in names:
            spec = R[n]
            if spec["kind"] == "single":
                scores[n][s] = te["x_" + spec["feat"]].to_numpy()
                continue
            sc = fit_one(n)
            cache[n] = sc
            if spec.get("gate"):
                scores[n][s] = None
            else:
                scores[n][s] = sc
        if need_m0:
            m0s = cache["M0"] if "M0" in cache else m0
            gate = te["confidence"].to_numpy() >= GATE
            for n in names:
                if R[n].get("gate"):
                    scores[n][s] = gated_scores(cache[n], m0s, gate)
    return scores, weights


# --------------------------------------------------------------------------- bootstrap ---------------
def evaluate(te: dict, scores: dict, label: str, seeds: list[int], B: int, budgets=BUDGETS, refs: bool = True,
             rng_seed: int = 0) -> dict:
    """Joint paired two-level cluster bootstrap of every method (and the random / oracle references)."""
    names = list(scores)
    budgets = np.asarray(budgets, np.float64)
    ytab = {s: (te[s][label].to_numpy(np.float64), (~te[s]["correct"]).to_numpy(np.float64)) for s in seeds}
    pol, rk = {}, {}
    for n in names:
        pol[n], rk[n] = {}, {}
        for s in seeds:
            pol[n][s] = s7.Policy(scores[n][s], ytab[s][0], ytab[s][1])
            rk[n][s] = s5.Ranked(scores[n][s], ytab[s][0] > 0)
    if refs:
        for r, fn in (("random", lambda s: s7.random_score(len(ytab[s][0]))), ("oracle", lambda s: s7.oracle_score(ytab[s][0]))):
            pol[r] = {s: s7.Policy(fn(s), ytab[s][0], ytab[s][1]) for s in seeds}
    allm = list(pol)
    test = pd.concat([te[s] for s in seeds], ignore_index=True)
    setup = s5.boot_setup(test, seeds)
    nC = len(setup["clusters"])
    ones = {s: np.ones(len(ytab[s][0])) for s in seeds}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_seed = {m: {s: (pol[m][s].stats(ones[s], budgets)[0],
                            np.array(rk[m][s].metrics(ones[s])) if m in rk else np.full(3, np.nan)) for s in seeds} for m in allm}
        point = {m: (np.nanmean([per_seed[m][s][0] for s in seeds], 0), np.nanmean([per_seed[m][s][1] for s in seeds], 0)) for m in allm}
        rng = np.random.default_rng(rng_seed)
        bl = {m: np.empty((B, len(budgets), 3)) for m in allm}
        ba = {m: np.empty((B, 3)) for m in allm}
        for b in range(B):
            sdraw = rng.choice(seeds, size=len(seeds), replace=True)
            cnt = np.bincount(rng.integers(0, nC, nC), minlength=nC).astype(np.float64)
            accl = {m: [] for m in allm}
            acca = {m: [] for m in names}
            for s in sdraw:
                w = cnt[setup["row_cluster"][s]]
                for m in allm:
                    accl[m].append(pol[m][s].stats(w, budgets)[0])
                    if m in rk:
                        acca[m].append(rk[m][s].metrics(w))
            for m in allm:
                bl[m][b] = np.nanmean(accl[m], 0)
                ba[m][b] = np.nanmean(acca[m], 0) if m in acca else np.nan
    return {"names": allm, "point": point, "per_seed": per_seed, "bl": bl, "ba": ba, "budgets": budgets, "seeds": seeds,
            "n": int(np.mean([len(te[s]) for s in seeds])), "prev": float(np.mean([ytab[s][0].mean() for s in seeds]))}


def verdict_auc(d: float, lo: float, hi: float) -> str:
    if not np.isfinite(d):
        return ""
    if abs(d) < AUC_MIN_EFFECT:
        return "no practical difference"
    return "improves" if lo > 0 else "worse" if hi < 0 else "inconclusive"


def make_rows(res: dict, base: dict, pairs: list[tuple], level_names: list[str] | None = None) -> list[dict]:
    rows = []
    bs = res["budgets"]
    common = {**base, "n_per_seed": res["n"], "prevalence": res["prev"]}
    for m in (level_names or res["names"]):
        for i, b in enumerate(bs):
            for k, nm in enumerate(LEVEL_METRICS):
                lo, hi = s5.ci(res["bl"][m][:, i, k])
                rows.append({**common, "kind": "level", "group": "", "quantity": m, "budget": b, "metric": nm,
                             "estimate": res["point"][m][0][i, k], "ci_low": lo, "ci_high": hi})
        if np.isfinite(res["point"][m][1]).any():
            for k, nm in enumerate(AUC_METRICS):
                lo, hi = s5.ci(res["ba"][m][:, k])
                rows.append({**common, "kind": "level", "group": "", "quantity": m, "budget": np.nan, "metric": nm,
                             "estimate": res["point"][m][1][k], "ci_low": lo, "ci_high": hi})
    for grp, a, c, prim in pairs:
        if a not in res["point"] or c not in res["point"]:
            continue
        specs = [(i, k, nm, "bl") for i in range(len(bs)) for k, nm in enumerate(LEVEL_METRICS)]
        if np.isfinite(res["point"][a][1]).all() and np.isfinite(res["point"][c][1]).all():
            specs += [(None, k, nm, "ba") for k, nm in enumerate(AUC_METRICS)]
        for i, k, nm, which in specs:
            if which == "bl":
                d = res["bl"][a][:, i, k] - res["bl"][c][:, i, k]
                est = res["point"][a][0][i, k] - res["point"][c][0][i, k]
                comp = res["point"][c][0][i, k]
                jk = [res["per_seed"][a][s][0][i, k] - res["per_seed"][c][s][0][i, k] for s in res["seeds"]]
            else:
                d = res["ba"][a][:, k] - res["ba"][c][:, k]
                est = res["point"][a][1][k] - res["point"][c][1][k]
                comp = res["point"][c][1][k]
                jk = [res["per_seed"][a][s][1][k] - res["per_seed"][c][s][1][k] for s in res["seeds"]]
            lo, hi = s5.ci(d)
            loo = [np.nanmean([x for j, x in enumerate(jk) if j != q]) for q in range(len(jk))]
            v = s7.verdict(est, lo, hi) if nm == "resid_sf" else verdict_auc(est, lo, hi) if nm in ("auroc", "auprc") else ""
            rows.append({**common, "kind": "contrast", "group": grp, "primary": bool(prim), "quantity": f"{a}-{c}",
                         "budget": bs[i] if i is not None else np.nan, "metric": "d" + nm, "estimate": est, "ci_low": lo, "ci_high": hi,
                         "p_boot": s5.boot_p(d), "jackknife_min": float(np.min(loo)), "jackknife_max": float(np.max(loo)),
                         "comparator_value": comp, "verdict": v})
    return rows


def subset(ctx_te: dict, scores: dict, col: str, value) -> tuple[dict, dict]:
    masks = {s: (te[col].astype(str) == str(value)).to_numpy() for s, te in ctx_te.items()}
    return ({s: te[masks[s]].reset_index(drop=True) for s, te in ctx_te.items()},
            {m: {s: v[masks[s]] for s, v in d.items()} for m, d in scores.items()})


# --------------------------------------------------------------------------- orchestration -----------
LEVEL_NAMES = ["M0", "M1", "M2", "sf_M1", "sf_M2"]
LEVEL_PAIRS = [("G7", "M1", "M0", False), ("G7", "M2", "M1", False), ("G7", "sf_M1", "M1", False),
               ("G8", "M1", "M0", False), ("G8", "M2", "M1", False), ("G8", "sf_M1", "M0", False), ("G8", "sf_M2", "sf_M1", False)]
LABEL_NAMES = ["M0", "M1", "M2", "sf_M1"]
LABEL_PAIRS = [("G9", "M1", "M0", False), ("G9", "M2", "M1", False), ("G9", "sf_M1", "M0", False)]


def _save(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run(cfg: dict, out_dir: Path, steps: str = "augment,main,labels,tables", log=print) -> None:
    s8 = cfg["stage8"]
    out = out_dir / "stage8"
    for d in ("raw", "tables", "figures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    seeds, B = cfg["train"]["seeds"], s8["bootstrap_B"]
    assert tuple(s8["budgets"]) == tuple(BUDGETS) and s8["gate"] == GATE and list(s8["sizes"]) == [500, 1000, 2000, "full"], \
        "configs stage8 block differs from the frozen plan constants"
    t0 = time.time()
    steps = steps.split(",")
    need_data = any(s in steps for s in ("augment", "main", "labels"))
    if need_data:
        df = s5.load(cfg, out_dir)
        log(f"[stage8] data loaded: {len(df):,} rows ({time.time() - t0:.0f} s)")
        df = augment(cfg, out_dir, df, log)
        df = df[[c for c in df.columns if c in NEEDED or c.startswith("x_")]]
        log(f"[stage8] augmented ({time.time() - t0:.0f} s)")
    if "main" in steps:
        for H in ("A", "B", "C"):
            f = out / "raw" / f"main_{H}.csv"
            if f.exists():
                continue
            ctx = make_ctx(df, H, "sf_090", seeds)
            sc, wts = score_methods(ctx, MAIN_NAMES, seeds)
            s7.check_weights_match_stage5({"heldout": H, "label": "sf_090", "weights": [w for w in wts if w["model"] in ("M0", "M1", "M2")]},
                                          out_dir)
            pd.DataFrame(wts).to_csv(out / "raw" / f"weights_{H}.csv", index=False)
            res = evaluate(ctx["te"], sc, "sf_090", seeds, B)
            rows = make_rows(res, {"heldout": H, "label": "sf_090", "c_kind": "shortcutv2", "level_col": "", "level": ""}, pairs_main())
            log(f"[stage8] main held-out {H} pooled done ({time.time() - t0:.0f} s)")
            # G7 (family A) and G8 (family C) levels: the policy is re-applied inside each level, same rotation weights
            lvls = []
            if H == "A":
                a_pool = ctx["te"][seeds[0]]
                lvls = [("condition", v) for v in sorted(a_pool.condition.unique())] + [("severity", v) for v in sorted(a_pool.severity.unique())]
            if H == "C":
                lvls = [("condition", v) for v in ("aligned", "removed", "reversed")]
            for col, v in lvls:
                te_l, sc_l = subset(ctx["te"], {n: sc[n] for n in LEVEL_NAMES}, col, v)
                r = evaluate(te_l, sc_l, "sf_090", seeds, B)
                rows += make_rows(r, {"heldout": H, "label": "sf_090", "c_kind": "shortcutv2", "level_col": col, "level": str(v)}, LEVEL_PAIRS,
                                  level_names=LEVEL_NAMES + ["random", "oracle"])
            gsh = float(np.mean([(ctx["te"][s]["confidence"] >= GATE).mean() for s in seeds]))
            rows.append({"kind": "info", "heldout": H, "label": "sf_090", "metric": "gated_share", "estimate": gsh})
            _save(rows, f)
            log(f"[stage8] main held-out {H} + levels done ({time.time() - t0:.0f} s)")
            if H == "C":                                           # v1 weak-shortcut supplement (G8)
                f1 = out / "raw" / "main_C_v1.csv"
                if not f1.exists():
                    ctx1 = make_ctx(df, "C", "sf_090", seeds, "shortcut")
                    sc1, _ = score_methods(ctx1, LEVEL_NAMES, seeds)
                    rows1 = []
                    for col, v in [("", "pooled")] + [("condition", s_) for s_ in ("aligned", "removed", "reversed")]:
                        te_l, sc_l = (ctx1["te"], sc1) if col == "" else subset(ctx1["te"], sc1, col, v)
                        r = evaluate(te_l, sc_l, "sf_090", seeds, B)
                        rows1 += make_rows(r, {"heldout": "C", "label": "sf_090", "c_kind": "shortcut", "level_col": col, "level": v},
                                           LEVEL_PAIRS, level_names=LEVEL_NAMES + ["random", "oracle"])
                    _save(rows1, f1)
                    log(f"[stage8] C v1 done ({time.time() - t0:.0f} s)")
            del ctx, sc, res
    if "labels" in steps:
        for H in ("A", "B", "C"):
            for lab in ("sf_095", "sf_tau", "sf_099"):
                f = out / "raw" / f"label_{H}_{lab}.csv"
                if f.exists():
                    continue
                ctx = make_ctx(df, H, lab, seeds)
                sc, _ = score_methods(ctx, LABEL_NAMES, seeds)
                res = evaluate(ctx["te"], sc, lab, seeds, B)
                _save(make_rows(res, {"heldout": H, "label": lab, "c_kind": "shortcutv2", "level_col": "", "level": ""}, LABEL_PAIRS,
                                level_names=LABEL_NAMES + ["random", "oracle"]), f)
                log(f"[stage8] labels {H} {lab} done ({time.time() - t0:.0f} s)")
    if "tables" in steps:
        build_tables(out_dir)
        log(f"[stage8] tables + figures + summary done ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------- tables / figures ----------
def load_raw(out: Path) -> pd.DataFrame:
    fs = sorted((out / "raw").glob("main_*.csv")) + sorted((out / "raw").glob("label_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def _bh_group(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    m = (t.kind == "contrast") & (t.metric == "dresid_sf") & (~t.primary.fillna(False).astype(bool))
    t.loc[m, "q_bh"] = s5.bh(t.loc[m, "p_boot"].to_numpy())
    return t


def build_tables(out_dir: Path) -> None:
    out = out_dir / "stage8"
    raw = load_raw(out)
    for c in ("level_col", "level"):
        raw[c] = raw[c].fillna("").astype(str)
    raw["primary"] = raw["primary"].fillna(False).astype(bool)
    main = (raw.level_col == "") & (raw.label == "sf_090") & (raw.c_kind == "shortcutv2")
    is_lv, is_ct = raw.kind == "level", raw.kind == "contrast"
    tabs = {}
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        tabs[g] = _bh_group(raw[main & ((is_lv & raw.quantity.isin(GROUP_NAMES[g])) | (is_ct & (raw.group == g)))].assign(group=g))
    lvl_nm = LEVEL_NAMES + ["random", "oracle"]
    tabs["G7"] = _bh_group(raw[(raw.heldout == "A") & (raw.level_col != "") & ((is_lv & raw.quantity.isin(lvl_nm)) | (is_ct & (raw.group == "G7")))].assign(group="G7"))
    tabs["G8"] = _bh_group(raw[(raw.heldout == "C") & ((raw.level_col != "") | (raw.c_kind == "shortcut")) &
                               ((is_lv & raw.quantity.isin(lvl_nm)) | (is_ct & (raw.group == "G8")))].assign(group="G8"))
    lab_nm = LABEL_NAMES + ["random", "oracle"]
    tabs["G9"] = _bh_group(raw[(raw.level_col == "") & (raw.c_kind == "shortcutv2") & ((is_lv & raw.quantity.isin(lab_nm)) | (is_ct & (raw.group == "G9")))].assign(group="G9"))
    ex = raw[main & ((is_lv & raw.quantity.isin(GROUP_NAMES["E"])) | (is_ct & (raw.group == "E")))].assign(group="E")
    ex.insert(0, "table", "exploratory, post-hoc")
    tabs["E"] = ex
    ct = raw[is_ct]
    for g, t in tabs.items():
        t.to_csv(out / "tables" / (f"ablation_{g}.csv" if g != "E" else "exploratory_targeted.csv"), index=False)
    jk = ct[ct.metric.isin(["dresid_sf", "dauroc"]) & main[ct.index]][["group", "heldout", "quantity", "budget", "metric", "estimate", "jackknife_min", "jackknife_max"]]
    jk.to_csv(out / "tables" / "seed_jackknife_G10.csv", index=False)
    gs = raw[raw.kind == "info"][["heldout", "metric", "estimate"]]
    gs.to_csv(out / "tables" / "gated_share.csv", index=False)
    _figures(raw, tabs, out)
    _summary(raw, tabs, gs, out)


def _pick(t, quantity, H, metric, budget=None, **kw):
    m = (t.heldout == H) & (t.quantity == quantity) & (t.metric == metric)
    if budget is not None:
        m &= np.isclose(t.budget.astype(float), budget)
    for k, v in kw.items():
        m &= t[k].astype(str) == str(v)
    r = t[m]
    return r.iloc[0] if len(r) else None


def _figures(raw: pd.DataFrame, tabs: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = raw[(raw.level_col.fillna("") == "") & (raw.label == "sf_090") & (raw.c_kind == "shortcutv2")]
    ct = main[main.kind == "contrast"]
    sel = [("G1", "M1-M0"), ("G1", "M2-M1"), ("G2", "sf_M1-sf_M0"), ("G2", "sf_M2-sf_M1"), ("G2", "sf_M1-M1"), ("G4", "M1knn-M1"),
           ("G3", "M1+e_js_norm_l3-M1"), ("G3", "M1+e_js_norm_l2-M1"), ("G5", "M1_N500-M1_Nfull"), ("G6", "M0_T1-M0")]
    fig, ax = plt.subplots(2, 3, figsize=(17, 7))
    for j, H in enumerate(("A", "B", "C")):
        for row, (met, bud, ttl) in enumerate((("dresid_sf", 0.20, "residual silent-failure rate (20% budget)"), ("dauroc", None, "AUROC"))):
            for i, (g, q) in enumerate(sel):
                r = _pick(ct[ct.group == g], q, H, met, bud)
                if r is None:
                    continue
                s = 100 if row == 0 else 1
                ax[row, j].errorbar(s * r.estimate, i, xerr=[[s * (r.estimate - r.ci_low)], [s * (r.ci_high - r.estimate)]], fmt="o", color="#1f77b4", capsize=2)
            ax[row, j].axvline(0, color="k", lw=0.7)
            ax[row, j].set_yticks(range(len(sel)))
            ax[row, j].set_yticklabels([q for _, q in sel], fontsize=7)
            ax[row, j].set_title(f"held-out {H}: delta {ttl}", fontsize=8)
            ax[row, j].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig8a_incremental_deltas.png", dpi=130)
    plt.close(fig)

    def level_plot(H, col, levels, fname, c_kind="shortcutv2", methods=("random", "M0", "M1", "M2", "sf_M1"), xlabel=""):
        d = raw[(raw.kind == "level") & (raw.heldout == H) & (raw.level_col.fillna("") == col) & (raw.c_kind == c_kind) & (raw.label == "sf_090")]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        colr = {"random": "#888888", "M0": "#1f77b4", "M1": "#d95f02", "M2": "#2ca02c", "sf_M1": "#9467bd", "sf_M2": "#8c564b"}
        for m in methods:
            for a, (met, bud) in zip(ax, (("resid_sf", 0.20), ("auroc", None))):
                if met == "auroc" and m == "random":
                    continue
                pts = []
                for x, v in enumerate(levels):
                    r = _pick(d, m, H, met, bud, level=v)
                    if r is not None:
                        pts.append((x, r.estimate, r.ci_low, r.ci_high))
                if pts:
                    p = np.array(pts)
                    s = 100 if met == "resid_sf" else 1
                    a.plot(p[:, 0], s * p[:, 1], "-o", ms=3, color=colr[m], label=m)
                    a.fill_between(p[:, 0], s * p[:, 2], s * p[:, 3], color=colr[m], alpha=0.12, lw=0)
        for a, t in zip(ax, ("residual silent-failure rate (%) at 20% budget", "silent-failure detection AUROC")):
            a.set_xticks(range(len(levels)))
            a.set_xticklabels(levels, rotation=30, fontsize=8)
            a.set_title(f"{H}{'' if c_kind == 'shortcutv2' else ' (v1)'}: {t}", fontsize=9)
            a.set_xlabel(xlabel)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / fname, dpi=130)
        plt.close(fig)
    level_plot("A", "severity", ["0", "1", "2", "3", "4", "5"], "fig8b_severity_A.png", xlabel="MedMNIST-C severity (0 = clean)")
    level_plot("C", "condition", ["aligned", "removed", "reversed"], "fig8c_shortcut_states.png", methods=("random", "M0", "M1", "M2", "sf_M1", "sf_M2"),
               xlabel="shortcut cue state (v2)")
    level_plot("C", "condition", ["aligned", "removed", "reversed"], "fig8c_shortcut_states_v1.png", c_kind="shortcut",
               methods=("random", "M0", "M1", "M2", "sf_M1", "sf_M2"), xlabel="shortcut cue state (v1)")
    e = tabs["E"]
    e = e[e.kind == "level"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    colr = {"random": "#888888", "oracle": "#000000", "M0": "#1f77b4", "M1": "#d95f02", "sf_M1": "#9467bd", "G_M0": "#17becf", "G_M1": "#e377c2", "G_sfM1": "#bcbd22"}
    for a, H in zip(ax, ("A", "B", "C")):
        for m, c in colr.items():
            g = e[(e.heldout == H) & (e.quantity == m) & (e.metric == "resid_sf")].sort_values("budget")
            a.plot(100 * g.budget, 100 * g.estimate, "-o" if m != "oracle" else "--", ms=3, color=c, label=m)
            a.fill_between(100 * g.budget, 100 * g.ci_low, 100 * g.ci_high, color=c, alpha=0.1, lw=0)
        a.set_title(f"exploratory, post-hoc - held-out {H}", fontsize=9)
        a.set_xlabel("review budget (%)")
    ax[0].set_ylabel("residual silent-failure rate (%)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig8d_exploratory_gated.png", dpi=130)
    plt.close(fig)


def _fmt(r, pt=True) -> str:
    s = 100 if pt else 1
    return f"{s * r.estimate:+.2f}{' pt' if pt else ''} [{s * r.ci_low:+.2f}, {s * r.ci_high:+.2f}]"


def _summary(raw: pd.DataFrame, tabs: dict, gs: pd.DataFrame, out: Path) -> None:
    L = ["# Stage 8 - ablations and sensitivity (generated from the computed numbers)", "",
         "Plan: docs/ablation_plan_frozen.md. Endpoint: residual silent-failure rate (sf_090) at exact budgets; differences in percentage "
         "points (negative = fewer residual silent failures); 95% paired cluster-bootstrap CIs; P1-P3 are the declared primary contrasts "
         "(unadjusted); all other contrasts are secondary (BH within the group, `q_bh`).", ""]
    def para(g, title, pairs):
        t = tabs[g]
        t = t[(t.kind == "contrast") & (t.metric == "dresid_sf")]
        L.append(f"## {g}: {title}")
        L.append("")
        L.append("| contrast | held-out | 5% | 10% | 20% |")
        L.append("|---|---|---|---|---|")
        for q in pairs:
            for H in ("A", "B", "C"):
                cells = []
                for b in BUDGETS:
                    r = _pick(t, q, H, "dresid_sf", b)
                    cells.append(f"{_fmt(r)} {r.verdict}" if r is not None else "-")
                if any(c != "-" for c in cells):
                    L.append(f"| {q} | {H} | " + " | ".join(cells) + " |")
        L.append("")
        n = t[t.quantity.isin(pairs)]
        cnt = n.verdict.value_counts().to_dict()
        L.append(f"Verdict counts over these contrasts (3 held-out x 3 budgets): {cnt}. ")
        L.append("")
    para("G1", "signal groups", ["M1-M0", "M0c-M0", "M2-M1", "s_r_maha_pred-s_u_log1msp", "s_e_js_norm_l4-s_u_log1msp", "s_r_knn-s_u_log1msp"])
    para("G2", "fusion constraint (P1 = sf_M1-sf_M0, P2 = sf_M2-sf_M1, P3 = sf_M1-M1)", ["sf_M0-M0", "sf_M1-M1", "sf_M2-M2", "sf_M1-sf_M0", "sf_M2-sf_M1"])
    para("G3", "CAM layer / form", [f"{pre}M1+{c}-{pre}M1" for pre in ("", "sf_") for c in CAM_VARIANTS])
    para("G4", "Mahalanobis vs kNN", ["M1knn-M1", "M2knn-M2", "sf_M1knn-sf_M1", "sf_M2knn-sf_M2"])
    para("G5", "reference size", [f"{pre}M1_N{n}-{pre}M1_Nfull" for pre in ("", "sf_") for n in SIZES[:-1]] + ["M1_Nfull-M1"])
    para("G6", "temperature scaling (before - after)", ["M0_T1-M0", "M1_T1-M1"])
    e = tabs["E"]
    L += ["## Exploratory, post-hoc: targeted fusion (NOT a main-table result)", "",
          "Gated share (calibrated confidence >= 0.90) of each pool: " + ", ".join(f"{r.heldout} {100 * r.estimate:.1f}%" for r in gs[gs.metric == "gated_share"].itertuples()),
          "", "| held-out | budget | random | M0 | M1 | sf_M1 | G-M0 | G-M1 | G-sf M1 | G-sf M2 | oracle |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    lv = e[(e.kind == "level") & (e.metric == "resid_sf")]
    for H in ("A", "B", "C"):
        for b in BUDGETS:
            cells = []
            for m in ("random", "M0", "M1", "sf_M1", "G_M0", "G_M1", "G_sfM1", "G_sfM2", "oracle"):
                r = _pick(lv, m, H, "resid_sf", b)
                cells.append(f"{100 * r.estimate:.2f} [{100 * r.ci_low:.2f}, {100 * r.ci_high:.2f}]" if r is not None else "-")
            L.append(f"| {H} | {int(b * 100)}% | " + " | ".join(cells) + " |")
    L += ["", "Targeted contrasts (residual rate, points; exploratory):", ""]
    cx = e[(e.kind == "contrast") & (e.metric == "dresid_sf")]
    for q in ("G_M0-M0", "G_M1-M1", "G_sfM1-M1", "G_sfM1-M0", "G_sfM2-M1", "sf_M1-M1"):
        for H in ("A", "B", "C"):
            cells = []
            for b in BUDGETS:
                r = _pick(cx, q, H, "dresid_sf", b)
                cells.append(f"{_fmt(r)} {r.verdict}" if r is not None else "-")
            L.append(f"- {q}, held-out {H}: " + "; ".join(f"{int(b * 100)}%: {c}" for b, c in zip(BUDGETS, cells)))
    (out / "summary_stage8.md").write_text("\n".join(L), encoding="utf-8")
