"""Stage 7: referral policy (accept / review / refer) under a fixed human-review budget.
Implements docs/referral_plan_frozen.md (pre-registration) exactly. Scores, fusion weights and the cluster bootstrap are the
Phase 5 ones (`sfd.stage5`); no val_cal-derived threshold and no held-out label is used anywhere.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sfd import stage5 as s5

BUDGETS = (0.05, 0.10, 0.20)                                       # primary workloads
GRID = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
METRICS = ("resid_sf", "resid_err", "capture")
MIN_EFFECT = 0.001                                                 # |d| below 0.1 point = no practical difference
WORKLOAD_TOL = 0.01
CONTRASTS = (("M1", "M0"), ("M2", "M1"), ("M0", "random"), ("M1", "random"), ("M2", "random"))
SENS_DETECTORS = {                                                 # variants replacing the primary M1 / M2 (plan Sec 7)
    "kNN instead of Mahalanobis": {"M1": s5.M0 + ["r_knn"], "M2": s5.M0 + ["r_knn", "e_js_norm_l4"]},
    "layer3 CAM": {"M2": s5.M0 + ["r_maha_pred", "e_js_norm_l3"]},
    "layer2 CAM": {"M2": s5.M0 + ["r_maha_pred", "e_js_norm_l2"]},
}


# --------------------------------------------------------------------------- exact-budget policy -----
class Policy:
    """Descending-score tie groups of one (method, seed). `stats(w, budgets)` flags the top-b share of the WEIGHT mass
    (a tie group straddling the boundary is flagged fractionally, i.e. ties are broken uniformly at random), so the
    flagged mass is exactly b * sum(w) for every method, the random baseline and the oracle."""

    def __init__(self, score: np.ndarray, y_sf: np.ndarray, y_err: np.ndarray):
        _, self.gid = np.unique(-np.asarray(score, np.float64), return_inverse=True)   # group 0 = highest score
        self.gid = self.gid.reshape(-1)
        self.G = int(self.gid.max()) + 1
        self.y_sf, self.y_err = np.asarray(y_sf, np.float64), np.asarray(y_err, np.float64)

    def _groups(self, w):
        g = self.gid
        return (np.bincount(g, weights=w, minlength=self.G), np.bincount(g, weights=w * self.y_sf, minlength=self.G),
                np.bincount(g, weights=w * self.y_err, minlength=self.G))

    def stats(self, w: np.ndarray, budgets) -> tuple[np.ndarray, float]:
        """-> ([nb, 3] = residual sf rate, residual error rate, sf capture) and AURC."""
        gw, gp, ge = self._groups(np.asarray(w, np.float64))
        W, P, E = gw.sum(), gp.sum(), ge.sum()
        b = np.asarray(budgets, np.float64)
        cw, cp, ce = np.cumsum(gw), np.cumsum(gp), np.cumsum(ge)
        m = b * W
        idx = np.minimum(np.searchsorted(cw, m, side="left"), self.G - 1)
        g_ = gw[idx]
        frac = np.clip(np.divide(m - (cw[idx] - g_), g_, out=np.zeros_like(m), where=g_ > 0), 0.0, 1.0)
        cap_p = cp[idx] - gp[idx] + frac * gp[idx]
        cap_e = ce[idx] - ge[idx] + frac * ge[idx]
        rest = W - m
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.stack([(P - cap_p) / rest, (E - cap_e) / rest, cap_p / P if P > 0 else np.full_like(m, np.nan)], 1)
        # AURC: accept the lowest scores first (reverse of the descending groups); risk = sf rate among accepted
        pw, tw = gp[::-1], gw[::-1]
        ct = np.cumsum(tw)
        risk = np.divide(np.cumsum(pw), ct, out=np.zeros_like(ct), where=ct > 0)
        return out, float(np.sum(tw * risk) / W) if W > 0 else np.nan

    def boundary_tie_share(self, budgets) -> np.ndarray:
        """Share of rows in the tie group that straddles each budget boundary (0 if the boundary falls between groups)."""
        gw = np.bincount(self.gid, minlength=self.G).astype(np.float64)
        cw = np.cumsum(gw)
        m = np.asarray(budgets) * cw[-1]
        idx = np.minimum(np.searchsorted(cw, m, side="left"), self.G - 1)
        inside = (cw[idx] - gw[idx] < m - 1e-9) & (cw[idx] > m + 1e-9) & (gw[idx] > 1)
        return np.where(inside, gw[idx] / cw[-1], 0.0)


def flag_fractions(score: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Row-level view of the policy (used to test it): per-row flag fraction in [0, 1], sum(w * f) == b * sum(w)."""
    score, w = np.asarray(score, np.float64), np.asarray(w, np.float64)
    _, gid = np.unique(-score, return_inverse=True)
    gid = gid.reshape(-1)
    gw = np.bincount(gid, weights=w)
    cw = np.cumsum(gw)
    m = b * w.sum()
    f_g = np.clip((m - (cw - gw)) / np.where(gw > 0, gw, 1.0), 0.0, 1.0)
    return f_g[gid]


def oracle_score(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, np.float64)


def random_score(n: int) -> np.ndarray:
    return np.zeros(n)


# --------------------------------------------------------------------------- scores --------------------
def fit_scores(df: pd.DataFrame, heldout: str, detectors: dict, label: str, seeds: list[int], c_kind: str = "shortcutv2") -> dict:
    """Phase 5 rotation: per seed and detector, non-negative logistic fusion tuned ONLY on the tuning families, scoring the
    held-out pool. Also scores the tuning rows (for the secondary threshold-transfer protocol)."""
    tune, test = s5.rotation(df, heldout, c_kind)
    out = {"test": {}, "score": {m: {} for m in detectors}, "tscore": {m: {} for m in detectors},
           "tw": {}, "weights": [], "heldout": heldout, "label": label}
    for s in seeds:
        tu, te = tune[tune.seed == s].reset_index(drop=True), test[test.seed == s].reset_index(drop=True)
        assert heldout not in set(tu["pool"]), "held-out family leaked into tuning rows"
        if s == seeds[0]:                                                                  # no (family, condition, uid) row in both
            key = lambda d: d["family"].astype(str) + "|" + d["condition"].astype(str) + "|" + d["uid"].astype(str)
            assert not (set(key(tu)) & set(key(te))), "tuning and held-out rows overlap"
        out["test"][s] = te[["seed", "cluster", "correct", "condition", "severity", label]]
        out["tw"][s] = s5.family_balanced_weights(tu["pool"])
        for m, feats in detectors.items():
            X = tu[["x_" + c for c in feats]].to_numpy()
            th = s5.fit_nonneg_logreg(X, tu[label].to_numpy(), out["tw"][s])
            out["score"][m][s] = te[["x_" + c for c in feats]].to_numpy() @ th[1:] + th[0]
            out["tscore"][m][s] = X @ th[1:] + th[0]
            out["weights"].append({"heldout": heldout, "label": label, "model": m, "seed": s,
                                   **{f"w_{c}": float(v) for c, v in zip(["intercept"] + feats, th)}})
    return out


def check_weights_match_stage5(sc: dict, out_dir: Path, tol: float = 1e-6) -> None:
    """The primary rotation weights must be the Phase 5 ones (earlier outputs untouched, same procedure)."""
    ref = pd.read_csv(out_dir / "stage5" / "fusion_weights.csv")
    ref = ref[(ref.heldout == sc["heldout"]) & (ref.label == sc["label"])]
    mine = pd.DataFrame(sc["weights"])
    m = ref.merge(mine, on=["heldout", "label", "model", "seed"], suffixes=("_ref", ""))
    assert len(m) == len(ref) and len(m) > 0, "stage 5 weights not found for the primary configuration"
    for c in [c for c in ref.columns if c.startswith("w_")]:
        a, b = m[c + "_ref"].to_numpy(float), m[c].to_numpy(float)
        ok = (np.isnan(a) & np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) < tol)
        assert ok.all(), f"stage 7 fusion weights differ from stage 5 in {c}"


def subset_scores(sc: dict, col: str | None = None, value=None) -> dict:
    """Restrict the held-out pool to rows with te[col] == value (a breakdown level); None = whole pool."""
    if col is None:
        return sc
    masks = {s: (te[col].astype(str) == str(value)).to_numpy() for s, te in sc["test"].items()}
    return {**sc, "test": {s: te[masks[s]] for s, te in sc["test"].items()},
            "score": {m: {s: v[masks[s]] for s, v in d.items()} for m, d in sc["score"].items()}}


# --------------------------------------------------------------------------- bootstrap -----------------
def make_policies(sc: dict, methods: list[str], seeds: list[int], with_refs: bool = True) -> dict:
    pol: dict[str, dict] = {m: {} for m in methods}
    if with_refs:
        pol["random"], pol["oracle"] = {}, {}
    for s in seeds:
        te = sc["test"][s]
        y_sf, y_err = te[sc["label"]].to_numpy(), (~te["correct"]).to_numpy()
        for m in methods:
            pol[m][s] = Policy(sc["score"][m][s], y_sf, y_err)
        if with_refs:
            pol["random"][s] = Policy(random_score(len(te)), y_sf, y_err)
            pol["oracle"][s] = Policy(oracle_score(y_sf), y_sf, y_err)
    return pol


def paired_bootstrap(pol: dict, sc: dict, seeds: list[int], budgets, B: int, rng_seed: int = 0) -> dict:
    """Same two-level cluster resampling as Phase 5 (seeds, then clusters; one draw shared by every method).
    The policy is re-applied to each weighted resample, so the workload is `budgets` in every replicate."""
    budgets = np.asarray(budgets, np.float64)
    test = pd.concat([sc["test"][s] for s in seeds], ignore_index=True)
    setup = s5.boot_setup(test, seeds)
    nC, ms = len(setup["clusters"]), list(pol)
    ones = {s: np.ones(len(sc["test"][s])) for s in seeds}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_seed = {m: {s: pol[m][s].stats(ones[s], budgets) for s in seeds} for m in ms}
        point = {m: (np.nanmean([per_seed[m][s][0] for s in seeds], 0), float(np.nanmean([per_seed[m][s][1] for s in seeds])))
                 for m in ms}
        rng = np.random.default_rng(rng_seed)
        boot = {m: np.empty((B, len(budgets), 3)) for m in ms}
        boot_aurc = {m: np.empty(B) for m in ms}
        for b in range(B):
            sdraw = rng.choice(seeds, size=len(seeds), replace=True)
            cnt = np.bincount(rng.integers(0, nC, nC), minlength=nC).astype(np.float64)
            acc = {m: ([], []) for m in ms}
            for s in sdraw:
                w = cnt[setup["row_cluster"][s]]
                for m in ms:
                    st, au = pol[m][s].stats(w, budgets)
                    acc[m][0].append(st)
                    acc[m][1].append(au)
            for m in ms:
                boot[m][b] = np.nanmean(acc[m][0], 0)
                boot_aurc[m][b] = np.nanmean(acc[m][1])
    ties = {m: np.mean([pol[m][s].boundary_tie_share(budgets) for s in seeds], 0) for m in ms}
    return {"point": point, "per_seed": per_seed, "boot": boot, "boot_aurc": boot_aurc, "budgets": budgets,
            "n": int(np.mean([len(sc["test"][s]) for s in seeds])),
            "prev": float(np.mean([sc["test"][s][sc["label"]].mean() for s in seeds])), "ties": ties}


def verdict(d: float, lo: float, hi: float) -> str:
    """Generated from the numbers (plan Sec 5). d < 0 = fewer residual silent failures."""
    if not np.isfinite(d):
        return ""
    if abs(d) < MIN_EFFECT:
        return "no practical difference"
    if hi < 0:
        return "reduces"
    if lo > 0:
        return "increases"
    return "inconclusive"


def result_rows(res: dict, heldout: str, extra: dict, budgets_keep=None, contrasts=CONTRASTS) -> list[dict]:
    rows = []
    names = ["resid_sf", "resid_err", "capture"]
    bs = res["budgets"]
    keep = [i for i, b in enumerate(bs) if budgets_keep is None or any(abs(b - k) < 1e-9 for k in budgets_keep)]
    base = {"heldout": heldout, **extra, "n_per_seed": res["n"], "prevalence": res["prev"]}
    for m in res["point"]:
        for i in keep:
            for k, nm in enumerate(names):
                lo, hi = s5.ci(res["boot"][m][:, i, k])
                rows.append({**base, "budget": bs[i], "quantity": m, "metric": nm, "estimate": res["point"][m][0][i, k],
                             "ci_low": lo, "ci_high": hi, "boundary_tie_share": res["ties"][m][i]})
        lo, hi = s5.ci(res["boot_aurc"][m])
        rows.append({**base, "budget": np.nan, "quantity": m, "metric": "aurc", "estimate": res["point"][m][1], "ci_low": lo, "ci_high": hi})
    for a, c in contrasts:
        if a not in res["point"] or c not in res["point"]:
            continue
        for i in keep:
            for k, nm in enumerate(names):
                d = res["boot"][a][:, i, k] - res["boot"][c][:, i, k]
                lo, hi = s5.ci(d)
                est = res["point"][a][0][i, k] - res["point"][c][0][i, k]
                jk = [res["per_seed"][a][s][0][i, k] - res["per_seed"][c][s][0][i, k] for s in res["per_seed"][a]]
                loo = [np.nanmean([x for j, x in enumerate(jk) if j != q]) for q in range(len(jk))]
                comp = res["point"][c][0][i, k]
                rows.append({**base, "budget": bs[i], "quantity": f"{a}-{c}", "metric": "d" + nm, "estimate": est, "ci_low": lo, "ci_high": hi,
                             "p_boot": s5.boot_p(d), "jackknife_min": float(np.min(loo)), "jackknife_max": float(np.max(loo)),
                             "comparator_value": comp, "relative_change": est / comp if comp else np.nan,
                             "verdict": verdict(est, lo, hi) if nm == "resid_sf" else ""})
        d = res["boot_aurc"][a] - res["boot_aurc"][c]
        lo, hi = s5.ci(d)
        est = res["point"][a][1] - res["point"][c][1]
        rows.append({**base, "budget": np.nan, "quantity": f"{a}-{c}", "metric": "daurc", "estimate": est, "ci_low": lo, "ci_high": hi,
                     "p_boot": s5.boot_p(d), "verdict": verdict(est, lo, hi)})
    return rows


def tier_rows(res: dict, heldout: str, extra: dict, budgets=BUDGETS) -> list[dict]:
    """Accept / review / refer: refer = top B/2 by score, review = the next B/2. Share of all silent failures per tier."""
    bs = list(np.round(res["budgets"], 6))
    rows = []
    for B_ in budgets:
        i_b, i_h = bs.index(round(B_, 6)), bs.index(round(B_ / 2, 6))
        for m in res["point"]:
            cap = res["boot"][m][:, :, 2]
            refer, review = cap[:, i_h], cap[:, i_b] - cap[:, i_h]
            for tier, x, pt in (("refer (top B/2)", refer, res["point"][m][0][i_h, 2]),
                                ("review (next B/2)", review, res["point"][m][0][i_b, 2] - res["point"][m][0][i_h, 2])):
                lo, hi = s5.ci(x)
                rows.append({"heldout": heldout, **extra, "budget": B_, "quantity": m, "tier": tier, "workload_share": B_ / 2,
                             "sf_share_captured": pt, "ci_low": lo, "ci_high": hi})
    return rows


# --------------------------------------------------------------------------- threshold transfer -------
def weighted_threshold(score: np.ndarray, w: np.ndarray, b: float) -> float:
    """Minimal t such that the weighted mass with score >= t is >= b (flag score >= t)."""
    o = np.argsort(-score, kind="mergesort")
    cw = np.cumsum(np.asarray(w, np.float64)[o]) / np.sum(w)
    k = min(int(np.searchsorted(cw, b - 1e-12, side="left")), len(cw) - 1)
    return float(np.asarray(score)[o][k])


def threshold_transfer(sc: dict, methods: list[str], seeds: list[int], budgets, B: int, rng_seed: int = 0) -> list[dict]:
    """Secondary protocol: threshold tuned on the pooled tuning families, applied to the held-out family."""
    test = pd.concat([sc["test"][s] for s in seeds], ignore_index=True)
    setup = s5.boot_setup(test, seeds)
    nC = len(setup["clusters"])
    bud = np.asarray(budgets, np.float64)
    flags, sf, err = {m: {} for m in methods}, {}, {}
    for s in seeds:
        te = sc["test"][s]
        sf[s], err[s] = te[sc["label"]].to_numpy(np.float64), (~te["correct"]).to_numpy(np.float64)
        for m in methods:
            t = np.array([weighted_threshold(sc["tscore"][m][s], sc["tw"][s], b) for b in bud])
            flags[m][s] = (sc["score"][m][s][:, None] >= t[None, :]).astype(np.float64)

    def stat(w_by_seed):
        out = {}
        for m in methods:
            vals = []
            for s, w in w_by_seed:
                F, W = flags[m][s], w.sum()
                fl = w @ F
                psf, perr = (w * sf[s]) @ F, (w * err[s]) @ F
                P, E = (w * sf[s]).sum(), (w * err[s]).sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals.append(np.stack([fl / W, (P - psf) / (W - fl), (E - perr) / (W - fl), psf / P if P > 0 else np.full(len(bud), np.nan)], 1))
            out[m] = np.nanmean(vals, 0)
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = stat([(s, np.ones(len(sf[s]))) for s in seeds])
        rng = np.random.default_rng(rng_seed)
        boot = {m: np.empty((B, len(bud), 4)) for m in methods}
        for b in range(B):
            sdraw = rng.choice(seeds, size=len(seeds), replace=True)
            cnt = np.bincount(rng.integers(0, nC, nC), minlength=nC).astype(np.float64)
            r = stat([(s, cnt[setup["row_cluster"][s]]) for s in sdraw])
            for m in methods:
                boot[m][b] = r[m]
    prev = float(np.mean([sf[s].mean() for s in seeds]))
    rows = []
    for m in methods:
        for i, b_ in enumerate(bud):
            for k, nm in enumerate(("realised_workload", "resid_sf", "resid_err", "capture")):
                lo, hi = s5.ci(boot[m][:, i, k])
                est = point[m][i, k]
                rows.append({"heldout": sc["heldout"], "label": sc["label"], "budget": b_, "quantity": m, "metric": nm, "estimate": est,
                             "ci_low": lo, "ci_high": hi, "random_resid_sf_same_workload": prev,
                             "workload_flag": bool(abs(point[m][i, 0] - b_) > WORKLOAD_TOL) if nm == "realised_workload" else None})
    for a, c in (("M1", "M0"), ("M2", "M1")):
        for i, b_ in enumerate(bud):
            d = boot[a][:, i, 1] - boot[c][:, i, 1]
            lo, hi = s5.ci(d)
            est = point[a][i, 1] - point[c][i, 1]
            dw = point[a][i, 0] - point[c][i, 0]
            rows.append({"heldout": sc["heldout"], "label": sc["label"], "budget": b_, "quantity": f"{a}-{c}", "metric": "dresid_sf",
                         "estimate": est, "ci_low": lo, "ci_high": hi, "workload_difference": dw,
                         "workload_matched": bool(abs(dw) <= WORKLOAD_TOL), "verdict": verdict(est, lo, hi)})
    return rows


# --------------------------------------------------------------------------- orchestration ------------
def _bh_family(df: pd.DataFrame) -> pd.DataFrame:
    m = df.metric.eq("dresid_sf") & df.quantity.isin(["M1-M0", "M2-M1"])
    df.loc[m, "q_bh"] = s5.bh(df.loc[m, "p_boot"].to_numpy())
    return df


def run(cfg: dict, out_dir: Path, log=print) -> None:
    s7 = cfg["stage7"]
    out = out_dir / "stage7"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds, B = cfg["train"]["seeds"], s7["bootstrap_B"]
    assert tuple(s7["budgets"]) == BUDGETS and tuple(s7["grid"]) == GRID and s7["min_effect"] == MIN_EFFECT \
        and s7["workload_tolerance"] == WORKLOAD_TOL, "configs/*.yaml stage7 block differs from the frozen plan constants"
    t0 = time.time()
    df = s5.load(cfg, out_dir)
    log(f"[stage7] data loaded: {len(df):,} rows ({time.time() - t0:.0f} s)")
    DET = s5.DETECTORS
    grid = list(s7["grid"])
    fig6, table3, tiers, thr, ties, wts = [], [], [], [], [], []

    # ---- primary: exact-budget ranking, M0/M1/M2 vs random vs oracle ----
    cache = {}
    for H in ("A", "B", "C", "X"):
        sc = fit_scores(df, H, DET, "sf_090", seeds)
        check_weights_match_stage5(sc, out_dir)
        cache[H] = sc
        wts += sc["weights"]
        pol = make_policies(sc, list(DET), seeds)
        res = paired_bootstrap(pol, sc, seeds, grid, B)
        extra = {"analysis": "primary" if H != "X" else "supplement X", "label": "sf_090"}
        rows = result_rows(res, H, extra)
        df_rows = pd.DataFrame(rows)
        fig6.append(df_rows[df_rows.metric.isin(["resid_sf", "resid_err", "capture"]) & ~df_rows.quantity.str.contains("-")])
        table3 += result_rows(res, H, extra, budgets_keep=BUDGETS)
        tiers += tier_rows(res, H, extra)
        for m in res["ties"]:
            for i, b_ in enumerate(grid):
                ties.append({"heldout": H, "method": m, "budget": b_, "boundary_tie_share": float(res["ties"][m][i])})
        # secondary: threshold transfer on the primary label
        thr += threshold_transfer(sc, list(DET), seeds, BUDGETS, B)
        log(f"[stage7] primary + threshold transfer held-out {H} done ({time.time() - t0:.0f} s)")
    pd.DataFrame(wts).to_csv(out / "fusion_weights.csv", index=False)
    f6 = pd.concat(fig6, ignore_index=True)
    f6.to_csv(out / "fig6_data.csv", index=False)
    t3 = pd.DataFrame(table3)
    t3.to_csv(out / "table3_referral.csv", index=False)
    pd.DataFrame(tiers).to_csv(out / "tiers.csv", index=False)
    pd.DataFrame(thr).to_csv(out / "threshold_transfer.csv", index=False)
    pd.DataFrame(ties).to_csv(out / "boundary_ties.csv", index=False)

    # ---- sensitivities (one BH family over M1-M0 / M2-M1 at the primary budgets) ----
    sens = []
    contr = (("M1", "M0"), ("M2", "M1"))
    def sens_run(H, label, dets, c_kind, variant, sc=None, methods=("M0", "M1", "M2")):
        sc = sc or fit_scores(df, H, dets, label, seeds, c_kind)
        pol = make_policies(sc, list(methods), seeds, with_refs=False)
        res = paired_bootstrap(pol, sc, seeds, BUDGETS, B)
        sens.extend(result_rows(res, H, {"analysis": "sensitivity", "variant": variant, "label": label}, contrasts=contr))
    for name, dets in SENS_DETECTORS.items():
        for H in ("A", "B", "C"):
            sens_run(H, "sf_090", dict(DET, **dets), "shortcutv2", name)
        log(f"[stage7] sensitivity '{name}' done ({time.time() - t0:.0f} s)")
    for lab in ("sf_095", "sf_tau"):
        for H in ("A", "B", "C"):
            sens_run(H, lab, DET, "shortcutv2", f"label {lab}")
        log(f"[stage7] sensitivity label {lab} done ({time.time() - t0:.0f} s)")
    sens_run("C", "sf_090", DET, "shortcut", "v1 weak shortcut for family C")
    sens_run("X", "sf_090", DET, "shortcutv2", "X supplement (weights tuned on A+B+C)", sc=cache["X"])
    sens = _bh_family(pd.DataFrame(sens))
    sens = sens[sens.metric.isin(["dresid_sf", "dresid_err", "dcapture"]) | sens.metric.isin(["resid_sf"])]
    sens.to_csv(out / "sensitivity.csv", index=False)

    # ---- breakdowns (each its own BH family; policy re-applied inside the level, same rotation weights) ----
    brk = []
    a_pool = s5.family_pool(df, "A")
    levels = {"A by corruption": ("A", "condition", sorted(a_pool.condition.unique())),
              "A by severity": ("A", "severity", sorted(a_pool.severity.unique())),
              "B by condition": ("B", "condition", ["id_test", "ood_c1", "ood_c2"]),
              "C by condition": ("C", "condition", ["aligned", "removed", "reversed"])}
    for fam_name, (H, col, vals) in levels.items():
        part = []
        for v in vals:
            sub = subset_scores(cache[H], col, v)
            pol = make_policies(sub, ["M0", "M1", "M2"], seeds, with_refs=False)
            res = paired_bootstrap(pol, sub, seeds, BUDGETS, B)
            part += result_rows(res, H, {"analysis": fam_name, "level": str(v), "label": "sf_090"}, contrasts=contr)
        part = _bh_family(pd.DataFrame(part))
        brk.append(part)
        log(f"[stage7] breakdown '{fam_name}' done ({time.time() - t0:.0f} s)")
    pd.concat(brk, ignore_index=True).to_csv(out / "breakdowns.csv", index=False)

    jk = t3[t3.jackknife_min.notna()][["heldout", "budget", "quantity", "metric", "estimate", "jackknife_min", "jackknife_max"]]
    jk.to_csv(out / "seed_jackknife.csv", index=False)
    _figure(f6, out)
    _summary(t3, pd.DataFrame(tiers), pd.DataFrame(thr), sens, pd.DataFrame(ties), out)
    log(f"[stage7] all done ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------- reporting ----------------
def _figure(f6: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"random": "#888888", "M0": "#1f77b4", "M1": "#d95f02", "M2": "#2ca02c", "oracle": "#000000"}
    fams = [("A", "A: MedMNIST-C"), ("B", "B: Camelyon hospitals"), ("C", "C: shortcut v2"), ("X", "X: NCT-CRC-HE-7K")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    d = f6[(f6.metric == "resid_sf") & f6.analysis.isin(["primary", "supplement X"])]
    for a, (H, name) in zip(ax, fams):
        for m in ("random", "M0", "M1", "M2", "oracle"):
            g = d[(d.heldout == H) & (d.quantity == m)].sort_values("budget")
            a.plot(100 * g.budget, 100 * g.estimate, "-o" if m != "oracle" else "--", ms=3, color=col[m], label=m)
            a.fill_between(100 * g.budget, 100 * g.ci_low, 100 * g.ci_high, color=col[m], alpha=0.15, lw=0)
        a.set_title(name, fontsize=10)
        a.set_xlabel("review budget (% of cases leaving automatic acceptance)")
    ax[0].set_ylabel("residual silent-failure rate among accepted (%)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig6_residual_vs_budget.png", dpi=130)
    plt.close(fig)


def _f(x, p=4):
    return f"{x:+.{p}f}"


def _summary(t3: pd.DataFrame, tiers: pd.DataFrame, thr: pd.DataFrame, sens: pd.DataFrame, ties: pd.DataFrame, out: Path) -> None:
    L = ["# Stage 7 - referral policy (generated from the computed numbers)", "",
         "Plan: docs/referral_plan_frozen.md. Residual silent-failure rate (sf_090) among accepted cases at an EXACT review budget B; "
         "mean over 5 seeds; 95% paired two-level cluster-bootstrap CIs (B=2000). Differences: negative = fewer residual silent failures.", ""]
    for H, name in (("A", "A (MedMNIST-C)"), ("B", "B (Camelyon hospitals)"), ("C", "C (shortcut v2)"), ("X", "X (supplement)")):
        g = t3[(t3.heldout == H)]
        L += [f"## Held-out {name}: prevalence of silent failures in the pool = {100 * g.prevalence.iloc[0]:.2f}%", "",
              "| budget | random | M0 | M1 | M2 | oracle | M1-M0 | M2-M1 | M1-random |", "|---|---|---|---|---|---|---|---|---|"]
        for b in BUDGETS:
            def lv(q):
                r = g[(g.budget == b) & (g.quantity == q) & (g.metric == "resid_sf")].iloc[0]
                return f"{100 * r.estimate:.2f}% [{100 * r.ci_low:.2f}, {100 * r.ci_high:.2f}]"
            def ct(q):
                r = g[(g.budget == b) & (g.quantity == q) & (g.metric == "dresid_sf")].iloc[0]
                return f"{100 * r.estimate:+.2f} pt [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}] {r.verdict}"
            L.append(f"| {int(b * 100)}% | " + " | ".join([lv("random"), lv("M0"), lv("M1"), lv("M2"), lv("oracle"), ct("M1-M0"), ct("M2-M1"), ct("M1-random")]) + " |")
        L.append("")
    L += ["## Answer to the primary question (generated)", ""]
    for H in ("A", "B", "C"):
        for q in ("M1-M0", "M2-M1"):
            vs = []
            for b in BUDGETS:
                r = t3[(t3.heldout == H) & (t3.budget == b) & (t3.quantity == q) & (t3.metric == "dresid_sf")].iloc[0]
                vs.append(f"{int(b * 100)}%: {100 * r.estimate:+.2f} pt [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}] ({r.verdict})")
            L.append(f"- held-out {H}, {q}: " + "; ".join(vs))
    L += ["", "## Secondary: threshold transfer (realised workload; flagged if |realised - B| > 1 point)", ""]
    w = thr[thr.metric == "realised_workload"]
    L.append("| held-out | method | B | realised workload | flag |")
    L.append("|---|---|---|---|---|")
    for _, r in w.iterrows():
        L.append(f"| {r.heldout} | {r.quantity} | {int(r.budget * 100)}% | {100 * r.estimate:.1f}% [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}] | {'FLAG' if r.workload_flag else ''} |")
    nflag = int(w.workload_flag.sum())
    L += ["", f"{nflag} of {len(w)} (held-out, method, budget) cells deviate by more than 1 point.", ""]
    (out / "summary_stage7.md").write_text("\n".join(L), encoding="utf-8")
