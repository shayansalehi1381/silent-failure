"""Stage 6: batch (window) level monitoring. Implements docs/batch_plan_frozen.md.

Steps: extract (GPU, per-image arrays) -> windows (CPU, window statistics + thresholds + FAR) -> analysis (power, delay,
bootstrap, tables). Per-image arrays and window statistics are git-ignored; summaries are committed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfd import stage4 as s4
from sfd.config import ROOT
from sfd.data import classify_data as cd
from sfd.signals import explanation as ex
from sfd.signals import representation as rp
from sfd.signals import uncertainty as un

STATS = ["u_mean", "u_p90", "u_ks", "r_mmd", "e_cam_l4", "r_maha_mean", "r_knn_mean", "e_cam_l3", "e_cam_l2", "e_img_mean"]
FUSE = ["u_mean", "u_p90", "u_ks", "r_mmd", "e_cam_l4"]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def P(cfg: dict, out_dir: Path) -> dict:
    p = s4.P(cfg, out_dir)
    p["s6"] = out_dir / "stage6"
    return p


# --------------------------------------------------------------------------- extraction ------
def extract(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print) -> None:
    """Per model: for every Stage-4 set (val_cal, ID, shifted) save features, CAMs (l4, l3: 8x8; l2: 16x16),
    prediction, calibrated confidence, stable-log uncertainty and the per-image Phase-4 signals."""
    import torch
    out = p["s6"] / "rows"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    off = cfg["stage3"]["shortcut"]["offset_px"]
    for spec in s4.model_specs(cfg, p):
        f = out / f"{spec['name']}.npz"
        if f.exists():
            continue
        t0 = time.time()
        ref = s4.load_ref(p, spec)
        z2 = np.load(p["out"] / "refs_l2" / f"{spec['name']}.npz")
        model = s4.load_model(spec, dev)
        pal = s4.sc.palette_for(spec["cue"]) if spec["cue"] else None
        cols = {k: [] for k in ("fam", "cond", "sev", "uid", "label", "pred", "conf", "ulog", "maha", "knn", "ejs",
                                "feat", "cam4", "cam3", "cam2")}
        for s in s4._sets(cfg, p, df, spec, smp, None):
            if (s4.ram_gb().get("available") or 99) < 2.0:
                raise RuntimeError("free RAM < 2 GB during extraction")
            imgs = s4._read(s)
            R = s4.run_gradcam(model, imgs, dev, s["cue"], pal, s["cue_off"])
            sig = s4.compute_signals(R, spec["T"], ref)
            R2 = s4.run_gradcam(model, imgs, dev, s["cue"], pal, s["cue_off"], layers=("layer2",), grid=16)
            n = len(sig)
            cols["fam"] += [s["family"]] * n
            cols["cond"] += [s["condition"]] * n
            cols["sev"].append(np.full(n, s["severity"], np.int16))
            cols["uid"].append(np.asarray(s["uid"], dtype=str))
            cols["label"].append(s["labels"].astype(np.int16))
            cols["pred"].append(sig["pred"].to_numpy().astype(np.int16))
            cols["conf"].append(sig["confidence"].to_numpy().astype(np.float32))
            cols["ulog"].append(un.log_one_minus_msp(R["logits"], spec["T"]).astype(np.float32))
            cols["maha"].append(sig["r_maha_pred"].to_numpy().astype(np.float32))
            cols["knn"].append(sig["r_knn"].to_numpy().astype(np.float32))
            cols["ejs"].append(sig["e_js_norm_l4"].to_numpy().astype(np.float32))
            cols["feat"].append(R["features"].astype(np.float16))
            cols["cam4"].append(R["layer4"].astype(np.float16))
            cols["cam3"].append(R["layer3"].astype(np.float16))
            cols["cam2"].append(R2["layer2"].astype(np.float16))
        arr = {k: (np.asarray(v, dtype=str) if k in ("fam", "cond") else np.concatenate(v)) for k, v in cols.items()}
        arr["T"] = np.float32(spec["T"])
        np.savez(f, **arr)
        log(f"[stage6] extract {spec['name']}: {len(arr['uid']):,} rows ({time.time() - t0:.0f} s)")
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- windows -----------
def cell_rng(model_seed: int, cell: str, size: int) -> np.random.Generator:
    """Deterministic RNG keyed by (model seed, cell name, window size)."""
    import hashlib
    key = int.from_bytes(hashlib.sha256(f"{cell}|{size}".encode()).digest()[:4], "little")
    return np.random.default_rng([int(model_seed), key, int(size)])


def _choose(pool: np.ndarray, k: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """[W, k] row indices drawn WITHOUT replacement inside each window (windows are independent draws)."""
    if k == 0:
        return np.zeros((W, 0), np.int64)
    assert 0 < k <= len(pool), f"pool of {len(pool)} rows cannot give {k} distinct rows"
    pos = np.argpartition(rng.random((W, len(pool))), k - 1, axis=1)[:, :k]
    return pool[pos]


def sample_windows(shift_pool: np.ndarray | None, id_pool: np.ndarray, f: float, n: int, W: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """[W, n] row indices: round(f*n) rows from shift_pool, the rest from id_pool (disjoint pools assumed)."""
    m = int(round(f * n)) if shift_pool is not None else 0
    parts = [_choose(shift_pool, m, W, rng), _choose(id_pool, n - m, W, rng)]
    idx = np.concatenate(parts, axis=1)
    return idx[np.arange(W)[:, None], np.argsort(rng.random((W, n)), axis=1)], m      # shuffle row order in the window


def ks_stat(x: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Two-sample KS statistic of each window row of x [W, n] against a reference sample (sorted 1-D)."""
    xs = np.sort(x, axis=1)
    n, m = x.shape[1], len(ref_sorted)
    Fr = np.searchsorted(ref_sorted, xs, side="right") / m
    Fl = np.searchsorted(ref_sorted, xs, side="left") / m
    hi = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    return np.maximum(np.abs(hi - Fr).max(1), np.abs(lo - Fl).max(1))


class MMD:
    """Unbiased RBF-kernel MMD^2 between windows and a fixed reference (torch; GPU when available)."""

    def __init__(self, ref: np.ndarray, sigma: float, device: str | None = None):
        import torch
        self.t = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.R = torch.from_numpy(np.asarray(ref, np.float32)).to(self.dev)
        self.g = 1.0 / (2.0 * sigma ** 2)
        self.rn = (self.R ** 2).sum(1)
        m = len(self.R)
        K = torch.exp(-self.g * torch.clamp(self.rn[:, None] + self.rn[None] - 2 * self.R @ self.R.T, min=0))
        self.c_ref = float((K.sum() - torch.diagonal(K).sum()) / (m * (m - 1)))

    def __call__(self, X: np.ndarray, chunk: int = 50) -> np.ndarray:
        """X: [W, n, D] -> [W] unbiased MMD^2."""
        torch, out = self.t, []
        for a in range(0, len(X), chunk):
            x = torch.from_numpy(np.asarray(X[a:a + chunk], np.float32)).to(self.dev)
            W, n, D = x.shape
            xn = (x ** 2).sum(2)
            d2 = torch.clamp(xn[:, :, None] + self.rn[None, None] - 2 * (x.reshape(W * n, D) @ self.R.T).reshape(W, n, -1), min=0)
            cross = torch.exp(-self.g * d2).mean((1, 2))
            G = torch.einsum("wnd,wmd->wnm", x, x)
            Kw = torch.exp(-self.g * torch.clamp(xn[:, :, None] + xn[:, None, :] - 2 * G, min=0))
            within = (Kw.sum((1, 2)) - torch.einsum("wnn->w", Kw)) / (n * (n - 1))
            out.append((within + self.c_ref - 2 * cross).cpu().numpy())
        return np.concatenate(out)


class ModelCtx:
    """Per-model constants for window statistics: val_cal references, MMD reference, class CAM references."""

    def __init__(self, cfg: dict, p: dict, spec: dict, store: dict):
        vc = store["fam"] == "VALCAL"
        self.store, self.spec = store, spec
        self.ref_ul = np.sort(store["ulog"][vc])
        z = np.load(p["out"] / "refs" / f"{spec['name']}.npz")
        bank = z["knn_bank"][:2000].astype(np.float32)                  # first 2,000 rows of the train reference
        vf = store["feat"][vc][np.random.default_rng(0).choice(vc.sum(), min(1000, vc.sum()), replace=False)].astype(np.float32)
        sq = (vf ** 2).sum(1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * vf @ vf.T, 0))   # no [n, n, D] broadcast (2 GB)
        self.sigma = float(np.median(d[np.triu_indices(len(vf), 1)]))      # median heuristic on val_cal ONLY
        self.mmd = MMD(bank, self.sigma)
        z2 = np.load(p["out"] / "refs_l2" / f"{spec['name']}.npz")
        self.ref_cam = {"cam4": z["layer4_mean_cam"], "cam3": z["layer3_mean_cam"], "cam2": z2["layer2_mean_cam"]}
        self.C = spec["n_classes"]

    def stats(self, idx: np.ndarray, chunk: int = 100) -> pd.DataFrame:
        s = self.store
        rows = []
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            ul = s["ulog"][ii]
            pred = s["pred"][ii]
            onehot = (pred[..., None] == np.arange(self.C)).mean(1)               # [W, C] predicted-class shares
            d = {"u_mean": ul.mean(1), "u_p90": np.percentile(ul, 90, axis=1), "u_ks": ks_stat(ul, self.ref_ul),
                 "r_mmd": self.mmd(s["feat"][ii]), "r_maha_mean": s["maha"][ii].mean(1), "r_knn_mean": s["knn"][ii].mean(1),
                 "e_img_mean": s["ejs"][ii].mean(1), "err": (pred != s["label"][ii]).mean(1), "conf": s["conf"][ii].mean(1)}
            for key, tag in (("cam4", "l4"), ("cam3", "l3"), ("cam2", "l2")):
                mean_cam = s[key][ii].astype(np.float32).mean(1)
                mix = onehot @ self.ref_cam[key]
                d[f"e_cam_{tag}"] = ex.js_divergence(mean_cam, mix)
            rows.append(pd.DataFrame(d))
        return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- per-model windows ---
def cells_for(store: dict, spec: dict, smp: pd.DataFrame, seed: int) -> dict:
    """Named cells: {name: dict(family, shift_pool, id_pool, fractions, tags)} per the frozen plan Sec 1."""
    fam, cond, sev, uid = store["fam"], store["cond"], store["sev"], store["uid"]
    cells = {}
    if spec["kind"] == "standard" and spec["dataset"] == "pathmnist":
        in_smp = np.isin(uid, smp["uid"].to_numpy())
        id_A = np.flatnonzero((fam == "ID") & ~in_smp)
        id_X = np.flatnonzero(fam == "ID")
        cells["A|src"] = dict(family="A", shift=None, id=id_A, fr=[0.0])
        for k in range(1, 6):
            cells[f"A|sev{k}"] = dict(family="A", shift=np.flatnonzero((fam == "A") & (sev == k)), id=id_A, fr=FRACTIONS)
        for c in sorted(set(cond[(fam == "A") & (sev > 0)])):
            for k in range(1, 6):
                cells[f"Acorr|{c}|sev{k}"] = dict(family="Acorr", shift=np.flatnonzero((fam == "A") & (cond == c) & (sev == k)),
                                                  id=id_A, fr=[1.0], tags=dict(corruption=c, severity=k))
        cells["X|src"] = dict(family="X", shift=None, id=id_X, fr=[0.0])
        cells["X|nct7k"] = dict(family="X", shift=np.flatnonzero(fam == "X"), id=id_X, fr=FRACTIONS)
    elif spec["kind"] == "standard":
        id_B = np.flatnonzero(fam == "ID")
        cells["B|src"] = dict(family="B", shift=None, id=id_B, fr=[0.0])
        for c in ("ood_c1", "ood_c2"):
            cells[f"B|{c}"] = dict(family="B", shift=np.flatnonzero(cond == c), id=id_B, fr=FRACTIONS)
    else:                                                                       # shortcut models: family C halves
        sm = smp["uid"].to_numpy()
        half = np.random.default_rng(0).permutation(len(sm)) < len(sm) // 2          # fixed seeded halves
        h1, h2 = set(sm[half]), set(sm[~half])
        al = (cond == "aligned") & (fam == "C")
        id_C = np.flatnonzero(al & np.isin(uid, list(h1)))
        cells["C|src"] = dict(family="C", shift=None, id=id_C, fr=[0.0])
        for c in ("reversed", "removed"):
            cells[f"C|{c}"] = dict(family="C", shift=np.flatnonzero((cond == c) & (fam == "C") & np.isin(uid, list(h2))),
                                   id=id_C, fr=FRACTIONS)
    return cells


def far_pool(store: dict, spec: dict) -> np.ndarray:
    fam, cond = store["fam"], store["cond"]
    if spec["kind"] == "standard":
        return np.flatnonzero(fam == "ID")
    return np.flatnonzero((cond == "aligned") & (fam == "C"))


def uid_half(uids: np.ndarray) -> np.ndarray:
    """Amendment A1: fixed row-level split of ID rows, sha256(uid) mod 2 (0 = calibration half, 1 = evaluation half)."""
    import hashlib
    return np.array([int(hashlib.sha256(u.encode()).hexdigest()[:8], 16) % 2 for u in np.asarray(uids, dtype=str)])


def run_model_windows(cfg: dict, p: dict, spec: dict, smp: pd.DataFrame, sizes: list[int], log=print,
                      variant: str = "pre") -> None:
    w = cfg["stage6"]
    out = p["s6"] / ("windows" if variant == "pre" else "windows_a1")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{spec['name']}.parquet"
    if f.exists():
        return
    t0 = time.time()
    z = np.load(p["s6"] / "rows" / f"{spec['name']}.npz")
    store = {k: z[k] for k in z.files if k != "T"}
    ctx = ModelCtx(cfg, p, spec, store)
    seed = spec["seed"]
    frames, idx_store = [], {}
    vc = np.flatnonzero(store["fam"] == "VALCAL")
    far = far_pool(store, spec)
    cells = cells_for(store, spec, smp, seed)
    if variant == "a1":                                    # thresholds from the calibration half, evaluation on the other
        h = uid_half(store["uid"][far])
        vc, far = far[h == 0], far[h == 1]
        for c in cells.values():
            c["id"] = c["id"][uid_half(store["uid"][c["id"]]) == 1]
    for n in sizes:
        meta = []
        # source-only windows from val_cal (thresholds) and held-out ID windows (actual FAR)
        for role, pool, W in (("valcal", vc, w["n_valcal_windows"]), ("far", far, w["n_far_windows"])):
            idx, _ = sample_windows(None, pool, 0.0, n, W, cell_rng(seed, role, n))
            st = ctx.stats(idx)
            st.insert(0, "wid", np.arange(W))
            meta.append(st.assign(role=role, family=role, cell=role, fraction=0.0, size=n, n_shift=0))
            idx_store[f"{role}|{n}"] = idx.astype(np.int32)
        for name, c in cells.items():
            if n != w["size"] and (c["family"] == "Acorr" or name == "C|removed"):
                continue                                                     # sensitivities use primary cells only
            for fr in c["fr"]:
                idx, m = sample_windows(c["shift"], c["id"], fr, n, w["windows_per_cell"], cell_rng(seed, f"{name}|{fr}", n))
                st = ctx.stats(idx)
                st.insert(0, "wid", np.arange(len(idx)))
                tags = c.get("tags", {})
                meta.append(st.assign(role="cell", family=c["family"], cell=name, fraction=fr, size=n, n_shift=m,
                                      corruption=tags.get("corruption"), severity=tags.get("severity")))
                idx_store[f"{name}|{fr}|{n}"] = idx.astype(np.int32)
        frames += meta
        log(f"[stage6] windows {spec['name']} size {n}: {sum(len(m) for m in meta):,} windows ({time.time() - t0:.0f} s)")
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "model", spec["name"])
    df["kind"], df["dataset"], df["seed"], df["sigma"] = spec["kind"], spec["dataset"], seed, ctx.sigma
    df.to_parquet(f, index=False)
    np.savez(out / f"{spec['name']}_idx.npz", **idx_store)
    np.savez(out / f"{spec['name']}_rowmeta.npz", uid=store["uid"], fam=store["fam"], cond=store["cond"])


def run_windows(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print, variant: str = "pre") -> None:
    w = cfg["stage6"]
    sizes = [w["size"]] + list(w["sensitivity_sizes"])
    for spec in s4.model_specs(cfg, p):
        run_model_windows(cfg, p, spec, smp, sizes, log, variant)


# --------------------------------------------------------------------------- analysis -----------
def load_windows(p: dict, variant: str = "pre") -> pd.DataFrame:
    d = "windows" if variant == "pre" else "windows_a1"
    return pd.concat([pd.read_parquet(f) for f in sorted((p["s6"] / d).glob("*.parquet"))], ignore_index=True)


def thresholds(W: pd.DataFrame) -> pd.DataFrame:
    """Per (model, size): alarm thresholds (95th pct), silent-window bounds and fusion standardisation, ALL from the
    model's val_cal source-only windows (role == 'valcal'); nothing else is read."""
    v = W[W.role == "valcal"]
    g = v.groupby(["model", "size"])
    thr = g[STATS].quantile(0.95).add_prefix("thr_")
    b = pd.DataFrame({"err_ub": g["err"].quantile(0.95), "conf_lo": g["conf"].quantile(0.025), "conf_hi": g["conf"].quantile(0.975)})
    mu, sd = g[FUSE].mean().add_prefix("mu_"), g[FUSE].std().add_prefix("sd_")
    return pd.concat([thr, b, mu, sd], axis=1).reset_index()


H_SEL = {"A": lambda d: (d.dataset == "pathmnist") & (d.kind == "standard"), "X": lambda d: (d.dataset == "pathmnist") & (d.kind == "standard"),
         "B": lambda d: (d.dataset == "camelyon17") & (d.kind == "standard"), "C": lambda d: d.kind.isin(["shortcutv2", "shortcut"])}
TUNE = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"], "X": ["A", "B", "C"]}
PRIMARY_CELL = lambda d: ~d.family.eq("Acorr") & ~d.cell.eq("C|removed")  # noqa: E731


def build_eval(W: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    """Evaluation frame: one row per (window, family H it is evaluated for) with thresholds, labels and individual alarms."""
    cells = W[W.role == "cell"].assign(H=lambda d: d.family.replace({"Acorr": "A"}))
    src = W[W.role.isin(["valcal", "far"])]
    E = pd.concat([cells] + [src[H_SEL[H](src)].assign(H=H) for H in "ABCX"], ignore_index=True)
    E = E.merge(thr, on=["model", "size"], how="left", validate="many_to_one")
    for c in STATS:
        E["alarm_" + c] = E[c] > E["thr_" + c]
    sh = E.n_shift > 0
    E["shifted"] = sh
    E["silent"] = sh & (E.err > E.err_ub) & (E.conf >= E.conf_lo) & (E.conf <= E.conf_hi)
    return E


def add_combined(E: pd.DataFrame, lam: float = 1e-3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined monitor via the Phase-5 rotation: non-negative logistic fusion tuned on the TWO tuning families' windows
    (label n_shift > 0), per seed and size; threshold = 95th pct of the fused score on the held-out model's val_cal windows."""
    from sfd.stage5 import family_balanced_weights, fit_nonneg_logreg
    Z = E[FUSE].to_numpy() - E[["mu_" + c for c in FUSE]].to_numpy()
    E = E.copy()
    E[["z_" + c for c in FUSE]] = Z / np.maximum(E[["sd_" + c for c in FUSE]].to_numpy(), 1e-12)
    zc = ["z_" + c for c in FUSE]
    E["fused"], E["alarm_comb"] = np.nan, False
    wt = []
    for H, kinds in (("A", ["standard"]), ("X", ["standard"]), ("B", ["standard"]), ("C", ["shortcutv2", "shortcut"])):
        for size in sorted(E["size"].unique()):
            for seed in sorted(E.seed.unique()):
                tr = E[(E.role == "cell") & (E["size"] == size) & (E.seed == seed) & E.H.isin(TUNE[H]) & PRIMARY_CELL(E)
                       & ((E.H != "C") | (E.kind == "shortcutv2"))]
                if tr.empty:
                    continue
                th = fit_nonneg_logreg(tr[zc].to_numpy(), (tr.n_shift > 0).to_numpy(), family_balanced_weights(tr["H"]), lam)
                for kind in kinds:
                    m = (E.H == H) & (E["size"] == size) & (E.seed == seed) & (E.kind == kind)
                    if not (m & (E.role == "valcal")).any():
                        continue
                    E.loc[m, "fused"] = E.loc[m, zc].to_numpy() @ th[1:] + th[0]
                    vt = np.quantile(E.loc[m & (E.role == "valcal"), "fused"], 0.95)
                    E.loc[m, "alarm_comb"] = E.loc[m, "fused"] > vt
                    wt.append({"H": H, "kind": kind, "size": size, "seed": seed, "thr": vt, "intercept": th[0],
                               **{f"w_{c}": v for c, v in zip(FUSE, th[1:])}})
    return E, pd.DataFrame(wt)


ALARMS = ["u_mean", "u_p90", "u_ks", "r_mmd", "e_cam_l4", "comb", "r_maha_mean", "r_knn_mean", "e_cam_l3", "e_cam_l2", "e_img_mean"]


# --------------------------------------------------------------------------- cluster bootstrap ---
def cluster_map(out_dir: Path) -> dict:
    sp = pd.read_parquet(out_dir / "stage1" / "splits.parquet", columns=["uid", "dataset", "group"])
    return dict(zip(sp["uid"], sp["dataset"] + ":" + sp["group"].astype(str)))


def composition(p: dict, R: pd.DataFrame, cmap: dict, variant: str = "pre"):
    """Sparse [windows, clusters] matrix of the share of each cluster's rows in each window (rows follow R's order)."""
    import scipy.sparse as sps
    clusters, rows_i, cols_j, vals = {}, [], [], []
    cache = {}
    for i, (model, cell, fr, size, wid) in enumerate(zip(R.model, R.cell, R.fraction, R["size"], R.wid)):
        if model not in cache:
            d_ = "windows" if variant == "pre" else "windows_a1"
            z = np.load(p["s6"] / d_ / f"{model}_idx.npz")
            rm = np.load(p["s6"] / d_ / f"{model}_rowmeta.npz")
            cache = {model: ({k: z[k] for k in z.files}, rm["uid"])}
        idx_d, uid = cache[model]
        key = f"{cell}|{fr}|{size}" if cell not in ("valcal", "far") else f"{cell}|{size}"
        cl = [cmap[u] for u in uid[idx_d[key][int(wid)]]]
        for c, n in pd.Series(cl).value_counts().items():
            rows_i.append(i); cols_j.append(clusters.setdefault(c, len(clusters))); vals.append(n / len(cl))
    return sps.csr_matrix((vals, (rows_i, cols_j)), shape=(len(R), len(clusters)))


def boot_power(R: pd.DataFrame, Mx, masks: dict, alarm_cols: list[str], B: int, seeds: list[int], rng_seed: int = 0) -> dict:
    """Paired two-level bootstrap of weighted power. Window weight = mean cluster multiplicity of its rows x its seed's
    multiplicity (identical for every statistic: paired). Returns {mask: [B, n_alarm_cols]} plus point estimates."""
    rng = np.random.default_rng(rng_seed)
    nC = Mx.shape[1]
    A = R[alarm_cols].to_numpy(np.float32)
    sidx = np.searchsorted(seeds, R.seed.to_numpy())
    out = {k: np.empty((B, len(alarm_cols))) for k in masks}
    for a in range(0, B, 250):
        Bc = min(250, B - a)
        cnt = rng.multinomial(nC, np.full(nC, 1 / nC), size=Bc).T.astype(np.float32)             # [clusters, Bc]
        smult = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=Bc).T.astype(np.float32)   # [seeds, Bc]
        Wm = (Mx @ cnt) * smult[sidx]
        for k, m in masks.items():
            num = Wm[m].T @ A[m]
            den = Wm[m].sum(0)
            out[k][a:a + Bc] = num / np.maximum(den, 1e-12)[:, None]
    out["_weights"] = None
    return out


def point_power(R: pd.DataFrame, masks: dict, alarm_cols: list[str]) -> dict:
    A = R[alarm_cols].to_numpy(np.float64)
    return {k: (A[m].mean(0) if m.sum() else np.full(len(alarm_cols), np.nan)) for k, m in masks.items()}


def km_median(d: np.ndarray, event: np.ndarray, w: np.ndarray, H: int) -> np.ndarray:
    """Weighted Kaplan-Meier median delay for each weight column of w [n, Bc]; NaN when the survival never reaches 0.5."""
    S, med = np.ones(w.shape[1]), np.full(w.shape[1], np.nan)
    for t in range(1, H + 1):
        atrisk = (w * (d >= t)[:, None]).sum(0)
        ev = (w * ((d == t) & event)[:, None]).sum(0)
        S = S * (1 - np.divide(ev, atrisk, out=np.zeros_like(ev), where=atrisk > 0))
        med = np.where(np.isnan(med) & (S <= 0.5), t, med)
    return med


# --------------------------------------------------------------------------- orchestration ------
DELAY_STATS = ["u_mean", "u_p90", "u_ks", "r_mmd", "e_cam_l4", "comb"]
CONTRASTS = {"primary": [("comb", "u_mean"), ("r_mmd", "u_mean"), ("e_cam_l4", "r_mmd")],
             "cam-layer": [("e_cam_l3", "r_mmd"), ("e_cam_l2", "r_mmd")],
             "variants": [("r_knn_mean", "u_mean"), ("r_maha_mean", "u_mean"), ("e_img_mean", "r_mmd")]}


def _cells_masks(R: pd.DataFrame, min_silent: int, primary_only: bool = True) -> tuple[dict, pd.DataFrame]:
    """Pooled and per-cell masks over R's rows (shifted windows; silent windows in cells with >= min_silent silent)."""
    prim = PRIMARY_CELL(R).to_numpy()
    sh = R.shifted.to_numpy() & prim
    per = R[R.shifted & prim].groupby(["cell", "fraction"]).agg(n=("wid", "size"), n_silent=("silent", "sum"),
                                                                  share_silent=("silent", "mean")).reset_index()
    per["qualifies"] = per.n_silent >= min_silent
    q = {(c, f) for c, f, ok in zip(per.cell, per.fraction, per.qualifies) if ok}
    inq = np.array([(c, f) in q for c, f in zip(R.cell, R.fraction)])
    masks = {"silent_pooled": sh & R.silent.to_numpy() & inq, "shifted_pooled": sh}
    return masks, per


def run_analysis(cfg: dict, p: dict, df: pd.DataFrame, smp: pd.DataFrame, log=print, variant: str = "pre") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sfd.stage5 import bh
    out = p["s6"] if variant == "pre" else p["s6"] / "a1"
    out.mkdir(parents=True, exist_ok=True)
    w6 = cfg["stage6"]
    B, seeds, min_sil = w6["bootstrap_B"], sorted(cfg["train"]["seeds"]), w6["min_silent_windows"]
    t0 = time.time()
    W = load_windows(p, variant)
    thr = thresholds(W)
    thr.to_csv(out / "thresholds_silent_bounds.csv", index=False)
    E, wt = add_combined(build_eval(W, thr))
    wt.to_csv(out / "combined_weights.csv", index=False)
    log(f"[stage6] windows loaded: {len(W):,}; evaluation rows {len(E):,} ({time.time() - t0:.0f} s)")

    # ---- actual false-alarm rate on held-out ID windows
    acols = ["alarm_" + c for c in STATS] + ["alarm_comb"]
    far = E[E.role == "far"].groupby(["H", "kind", "size"])[acols].mean()
    far.columns = [c.replace("alarm_", "") for c in far.columns]
    farl = far.reset_index().melt(id_vars=["H", "kind", "size"], var_name="stat", value_name="far")
    ps = E[E.role == "far"].groupby(["H", "kind", "size", "seed"])[acols].mean().reset_index()
    ps.columns = [c.replace("alarm_", "") for c in ps.columns]
    rng_ = ps.melt(id_vars=["H", "kind", "size", "seed"], var_name="stat", value_name="far").groupby(["H", "kind", "size", "stat"])["far"].agg(["min", "max"])
    farl = farl.merge(rng_.reset_index(), on=["H", "kind", "size", "stat"])
    lo, hi = w6["far_tolerance"]
    farl["flag_outside_3_7pct"] = (farl.far < lo) | (farl.far > hi)
    farl.round(4).to_csv(out / "far_table.csv", index=False)

    # ---- power per cell (point estimates, mean over windows and seeds)
    pcols = ["alarm_" + c for c in ALARMS]
    cellsE = E[(E.role == "cell") & E.shifted]
    rows = []
    for (H, kind, size, cell, fr), g in cellsE.groupby(["H", "kind", "size", "cell", "fraction"]):
        sil = g[g.silent]
        r = {"H": H, "kind": kind, "size": size, "cell": cell, "fraction": fr, "n_windows": len(g), "n_silent": len(sil),
             "share_silent": len(sil) / len(g), "err_rate_mean": g.err.mean(), "conf_mean": g.conf.mean()}
        for c in ALARMS:
            r["power_all_" + c] = g["alarm_" + c].mean()
            r["power_silent_" + c] = sil["alarm_" + c].mean() if len(sil) >= min_sil else np.nan
        rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(out / "power_cells.csv", index=False)
    log(f"[stage6] FAR + power tables done ({time.time() - t0:.0f} s)")

    # ---- bootstrap contrasts + delays per (H, kind, size)
    cmap = cluster_map(p["s6"].parent)
    crow, drow, srow = [], [], []
    streams_cfg = w6["streams"]
    onset_cells = {"A": [f"A|sev{k}" for k in (1, 3, 5)], "B": ["B|ood_c1", "B|ood_c2"], "C": ["C|reversed"], "X": ["X|nct7k"]}
    for (H, kind, size), R in E[E.role == "cell"].groupby(["H", "kind", "size"]):
        R = R.sort_values(["model", "cell", "fraction", "wid"]).reset_index(drop=True)
        Mx = composition(p, R, cmap, variant)
        masks, per = _cells_masks(R, min_sil)
        extra = {}
        if size == w6["size"]:
            prim = PRIMARY_CELL(R).to_numpy()
            for (cell, fr), _ in R[R.shifted & prim].groupby(["cell", "fraction"]):
                m = ((R.cell == cell) & (R.fraction == fr)).to_numpy()
                extra[f"cell|{cell}|{fr}|all"] = m
                extra[f"cell|{cell}|{fr}|silent"] = m & R.silent.to_numpy()
            if H == "A":
                for c in sorted(R.corruption.dropna().unique()):
                    extra[f"corruption|{c}|silent"] = ((R.corruption == c) & R.silent).to_numpy()
                for k in range(1, 6):
                    extra[f"severity|{k}|silent"] = ((R.cell == f"A|sev{k}") & R.silent & R.shifted).to_numpy()
        allm = {**masks, **extra}
        pe = point_power(R, allm, pcols)
        bt = boot_power(R, Mx, allm, pcols, B, seeds)
        names = ["alarm_" + c for c in ALARMS]
        ix = {c: i for i, c in enumerate(ALARMS)}
        for mname in allm:
            n_m = int(allm[mname].sum())
            for fam_c, pairs in CONTRASTS.items():
                for a, b in pairs:
                    d = bt[mname][:, ix[a]] - bt[mname][:, ix[b]]
                    lo_, hi_ = np.nanpercentile(d, [2.5, 97.5])
                    if mname == "silent_pooled" or mname == "shifted_pooled" or mname.startswith(("corruption", "severity")) or fam_c == "primary":
                        crow.append({"H": H, "kind": kind, "size": size, "mask": mname, "n_windows": n_m, "set": fam_c,
                                     "a": a, "b": b, "power_a": pe[mname][ix[a]], "power_b": pe[mname][ix[b]],
                                     "diff": pe[mname][ix[a]] - pe[mname][ix[b]], "ci_low": lo_, "ci_high": hi_,
                                     "p_boot": float(min(1, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))})
            if mname in ("silent_pooled", "shifted_pooled"):
                for c in ALARMS:
                    lo_, hi_ = np.nanpercentile(bt[mname][:, ix[c]], [2.5, 97.5])
                    srow.append({"H": H, "kind": kind, "size": size, "mask": mname, "stat": c, "power": pe[mname][ix[c]],
                                 "ci_low": lo_, "ci_high": hi_, "n_windows": n_m})
        log(f"[stage6] bootstrap {H}/{kind}/size {size}: {len(allm)} masks ({time.time() - t0:.0f} s)")
        # detection delay (size == primary only)
        if size == w6["size"] and H in onset_cells and (kind in ("standard", "shortcutv2")):
            rng = np.random.default_rng(1)
            pos = {(s, c, f): np.flatnonzero(((R.seed == s) & (R.cell == c) & (R.fraction == f)).to_numpy())
                   for s in seeds for c in R.cell.unique() for f in sorted(R.fraction.unique())}
            Acol = R[["alarm_" + c for c in DELAY_STATS]].to_numpy()
            src_cell = f"{H if H != 'X' else 'X'}|src"
            groups = []
            for cell in onset_cells[H]:
                for onset in streams_cfg["onsets"]:
                    sp, dd, ev, pre = [], [], [], []
                    for s in seeds:
                        srcp = pos[(s, src_cell, 0.0)]
                        for _ in range(streams_cfg["per_cell"]):
                            a_pre = rng.choice(srcp, streams_cfg["pre"])
                            if onset == "ramp":
                                post = np.concatenate([rng.choice(pos[(s, cell, f)], 6) for f in (0.1, 0.25, 0.5, 0.75, 1.0)])
                            else:
                                post = rng.choice(pos[(s, cell, float(onset))], streams_cfg["horizon"])
                            allp = np.concatenate([a_pre, post])
                            sp.append(allp)
                            al = Acol[post]                                  # [horizon, S]
                            first = np.where(al.any(0), al.argmax(0) + 1, streams_cfg["horizon"] + 1)
                            dd.append(first); ev.append(al.any(0)); pre.append(Acol[a_pre].mean(0))
                    groups.append({"cell": cell, "onset": str(onset), "pos": np.array(sp), "d": np.array(dd), "ev": np.array(ev),
                                   "pre": np.mean(pre, 0)})
            rngb = np.random.default_rng(0)
            nC, Hh = Mx.shape[1], streams_cfg["horizon"]
            meds = {i: [] for i in range(len(groups))}
            for a in range(0, B, 250):
                Bc = min(250, B - a)
                cnt = rngb.multinomial(nC, np.full(nC, 1 / nC), size=Bc).T.astype(np.float32)
                sm = rngb.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=Bc).T.astype(np.float32)
                Wm = (Mx @ cnt) * sm[np.searchsorted(seeds, R.seed.to_numpy())]
                for gi, g in enumerate(groups):
                    sw = Wm[g["pos"]].mean(1)                                        # [streams, Bc]
                    meds[gi].append(np.stack([km_median(g["d"][:, k], g["ev"][:, k], sw, Hh) for k in range(len(DELAY_STATS))], 1))
            for gi, g in enumerate(groups):
                bm = np.concatenate(meds[gi])
                for k, st in enumerate(DELAY_STATS):
                    pt = km_median(g["d"][:, k], g["ev"][:, k], np.ones((len(g["d"]), 1)), Hh)[0]
                    lo_, hi_ = (np.nanpercentile(bm[:, k], [2.5, 97.5]) if np.isfinite(bm[:, k]).sum() > 0.5 * len(bm) else (np.nan, np.nan))
                    drow.append({"H": H, "kind": kind, "cell": g["cell"], "onset": g["onset"], "stat": st,
                                 "median_delay": pt, "ci_low": lo_, "ci_high": hi_, "censored_share": float(1 - g["ev"][:, k].mean()),
                                 "pre_onset_alarm_rate": float(g["pre"][k]), "n_streams": len(g["d"]), "horizon": Hh})
        log(f"[stage6] delays {H}/{kind}/size {size} done ({time.time() - t0:.0f} s)")
    C = pd.DataFrame(crow)
    # BH only inside the prespecified secondary families
    fam = np.select([C["size"] != w6["size"], C["set"].isin(["cam-layer"]), C["mask"].str.startswith("corruption"),
                     C["mask"].str.startswith("severity")], ["window size", "CAM layer", "per-corruption", "per-severity"], "")
    C["bh_family"] = fam
    C["q_bh"] = np.nan
    for f in ["window size", "CAM layer", "per-corruption", "per-severity"]:
        m = (C.bh_family == f).to_numpy()
        if m.any():
            C.loc[m, "q_bh"] = bh(C.loc[m, "p_boot"].to_numpy())
    C["verdict"] = np.where(C.ci_low > 0, "improves", np.where(C.ci_high < 0, "worse", "inconclusive"))
    C.round(5).to_csv(out / "contrasts.csv", index=False)
    pd.DataFrame(srow).round(5).to_csv(out / "power_pooled.csv", index=False)
    pd.DataFrame(drow).round(3).to_csv(out / "detection_delay.csv", index=False)
    # seed spread of pooled silent power
    sp_rows = []
    for (H, kind, size, seed), g in E[(E.role == "cell")].groupby(["H", "kind", "size", "seed"]):
        masks, _ = _cells_masks(g.reset_index(drop=True), min_sil)
        gg = g.reset_index(drop=True)
        m = masks["silent_pooled"]
        sp_rows.append({"H": H, "kind": kind, "size": size, "seed": seed, "n_silent_windows": int(m.sum()),
                        **{c: (gg.loc[m, "alarm_" + c].mean() if m.sum() else np.nan) for c in ALARMS}})
    pd.DataFrame(sp_rows).round(4).to_csv(out / "seed_spread.csv", index=False)
    _fig5(out, plt)
    _summary6(out, farl, C, pd.DataFrame(srow), pd.DataFrame(drow), w6)
    log(f"[stage6] analysis done ({time.time() - t0:.0f} s)")


def _fig5(out: Path, plt) -> None:
    pc = pd.read_csv(out / "power_cells.csv")
    pc = pc[(pc["size"] == 100) & (pc.kind.isin(["standard", "shortcutv2"]))]
    fams = [("A", "A|sev3"), ("B", "B|ood_c2"), ("C", "C|reversed"), ("X", "X|nct7k")]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for j, (H, cell) in enumerate(fams):
        g = pc[(pc.H == H) & (pc.cell == cell)].sort_values("fraction")
        for c in ("u_mean", "u_ks", "r_mmd", "e_cam_l4", "comb"):
            ax[0, j].plot(g.fraction, g["power_all_" + c], marker="o", label=c)
            ax[1, j].plot(g.fraction, g["power_silent_" + c], marker="o", label=c)
        ax[0, j].set_title(f"{cell}: power, all shifted windows"); ax[1, j].set_title(f"{cell}: power, silent windows")
        ax[1, j].set_xlabel("shifted fraction")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figures_fig5_power.png", dpi=130); plt.close(fig)
    dd = pd.read_csv(out / "detection_delay.csv")
    if len(dd):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        g = dd[dd.onset == "1.0"].assign(label=lambda d: d.cell)
        piv = g.pivot_table(index="label", columns="stat", values="median_delay")
        piv.plot.bar(ax=ax)
        ax.set_ylabel("median delay (windows; NaN = more than half censored)"); ax.set_title("Detection delay, abrupt onset f = 1.0")
        fig.tight_layout(); fig.savefig(out / "figures_fig5_delay.png", dpi=130); plt.close(fig)


def _summary6(out, farl, C, S, D, w6) -> None:
    from sfd.data.audit import to_markdown
    L = ["# Stage 6 - batch-level monitoring (plan: docs/batch_plan_frozen.md)\n"]
    far100 = farl[farl["size"] == w6["size"]].pivot_table(index=["H", "kind"], columns="stat", values="far").round(3)
    L += ["## Actual false-alarm rate on held-out ID windows (size 100; nominal 5%, tolerance 3-7%)\n", to_markdown(far100.reset_index())]
    flagged = farl[(farl["size"] == w6["size"]) & farl.flag_outside_3_7pct]
    L += [f"\nOutside tolerance: {len(flagged)} of {len(farl[farl['size'] == w6['size']])} (statistic, family) cells.\n"]
    c = C[(C["size"] == w6["size"]) & (C["mask"] == "silent_pooled") & (C["set"] == "primary")]
    L += ["## Primary contrasts: power on silent windows, held-out family (95% CI)\n",
          to_markdown(c.assign(estimate=[f"{r.diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in c.itertuples()])[["H", "kind", "a", "b", "power_a", "power_b", "estimate", "verdict", "n_windows"]].round(3))]
    s = S[(S["size"] == w6["size"]) & (S["mask"] == "silent_pooled") & S.stat.isin(DELAY_STATS)]
    L += ["\n## Power on silent windows, pooled (95% CI)\n", to_markdown(s.round(3))]
    (out / "summary_stage6.md").write_text("\n".join(L), encoding="utf-8")

