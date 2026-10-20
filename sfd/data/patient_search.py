"""Phase 1b: seeded search over patient-disjoint assignments of the Camelyon source patients.

Goal: val_es / val_cal / id_test with tumour fractions close to each other (and to the OOD ~42%) while
keeping the planned sizes, >= min_patients per non-train split and (preferably) all source hospitals in
every non-train split. `split_counts` reproduces the sampler allocation of `splits.balanced_sample`
exactly (a test checks it against the real planner), so thousands of assignments can be scored cheaply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sfd.data.splits import _allocate

SPLITS = ["train", "val_es", "val_cal", "id_test"]


def patient_table(meta: pd.DataFrame, source_hospitals) -> pd.DataFrame:
    src = meta[meta["hospital"].isin(source_hospitals)]
    t = src.groupby(["patient", "hospital", "label"]).size().unstack("label", fill_value=0)
    t.columns = [f"n{c}" for c in t.columns]
    t["n_slides"] = src.groupby("patient")["slide"].nunique()
    return t.reset_index().sort_values("patient").reset_index(drop=True)


def split_counts(avail: np.ndarray, n: int) -> tuple[int, int, np.ndarray]:
    """(size, tumour, per-hospital size) that balanced_sample(group=hospital, label balanced) yields.
    avail: [hospital, 2] patch counts of (non-tumour, tumour) in the split, hospitals ascending."""
    avail = np.asarray(avail, dtype=int)
    if n >= avail.sum():
        return int(avail.sum()), int(avail[:, 1].sum()), avail.sum(1)
    per_h = np.zeros(len(avail), int)
    size = tum = 0
    idx = np.where(avail.sum(1) > 0)[0]
    q = _allocate(avail[idx].sum(1), n)
    for h, qq in zip(idx, q):
        row = avail[h]
        present = [l for l in (0, 1) if row[l] > 0]
        a = np.array([row[l] for l in present])
        lq = np.minimum(a, qq // len(present))
        lq = lq + _allocate(a - lq, qq - lq.sum())
        size += int(lq.sum())
        per_h[h] = int(lq.sum())
        if 1 in present:
            tum += int(lq[present.index(1)])
    return size, tum, per_h


class Scorer:
    def __init__(self, pt: pd.DataFrame, sizes: dict, ood_frac: float, min_patients: int = 3,
                 require_all_hospitals: bool = True, min_share: float = 0.0):
        self.pt, self.sizes, self.ood = pt, sizes, ood_frac
        self.min_patients, self.strict, self.min_share = min_patients, require_all_hospitals, min_share
        self.hosp = sorted(pt["hospital"].unique())
        self.hidx = pt["hospital"].map({h: i for i, h in enumerate(self.hosp)}).to_numpy()
        self.n01 = pt[["n0", "n1"]].to_numpy(int)

    def stats(self, a: np.ndarray) -> dict:
        out = {}
        for k, s in enumerate(SPLITS):
            m = a == k
            av = np.zeros((len(self.hosp), 2), int)
            np.add.at(av, self.hidx[m], self.n01[m])
            size, tum, per_h = split_counts(av, self.sizes[s])
            out[s] = {"patients": int(m.sum()), "hospitals": int((av.sum(1) > 0).sum()), "size": size,
                      "tumour": tum, "frac": tum / max(size, 1), "min_share": float(per_h.min() / max(size, 1))}
        return out

    def score(self, a: np.ndarray) -> tuple[float, dict]:
        st = self.stats(a)
        f = [st[s]["frac"] for s in SPLITS[1:]]
        gap = max(f) - min(f)
        j = gap + 0.5 * max(abs(x - self.ood) for x in f) + 0.25 * abs(st["train"]["frac"] - 0.5)
        for s in SPLITS:
            j += 10 * max(0, self.sizes[s] - st[s]["size"]) / self.sizes[s]
        for s in SPLITS[1:]:
            j += 1.0 * max(0, self.min_patients - st[s]["patients"])
            if self.strict:
                j += 1.0 * (len(self.hosp) - st[s]["hospitals"])
            if self.min_share:                      # every hospital must contribute at least this share
                j += 2.0 * max(0.0, self.min_share - st[s]["min_share"])
        if st["train"]["patients"] < self.min_patients:
            j += 1.0
        if self.strict:                                       # train must also cover every source hospital
            j += 1.0 * (len(self.hosp) - st["train"]["hospitals"])
        return j, st


def search(scorer: Scorer, seed: int, restarts: int = 30, iters: int = 6000, keep: int = 2000) -> list:
    """Simulated annealing with random restarts; returns [(score, assignment tuple)] of the best distinct states."""
    rng = np.random.default_rng(seed)
    n = len(scorer.pt)
    seen: dict[tuple, float] = {}
    for _ in range(restarts):
        a = rng.integers(0, 4, n)
        cur, _ = scorer.score(a)
        for it in range(iters):
            temp = 0.08 * (1 - it / iters) + 0.002
            b = a.copy()
            if rng.random() < 0.7:
                i = rng.integers(n)
                b[i] = (b[i] + rng.integers(1, 4)) % 4
            else:
                i, k = rng.integers(n, size=2)
                b[i], b[k] = b[k], b[i]
            sc, _ = scorer.score(b)
            if sc <= cur or rng.random() < np.exp((cur - sc) / temp):
                a, cur = b, sc
                seen[tuple(int(x) for x in a)] = sc
    best = sorted(seen.items(), key=lambda kv: kv[1])[:keep]
    return [(sc, np.array(k)) for k, sc in best]


def top_distinct(cands: list, k: int, min_hamming: int = 3) -> list:
    """Top-k candidates that differ in at least `min_hamming` patients from each other."""
    out = []
    for sc, a in cands:
        if all((a != b).sum() >= min_hamming for _, b in out):
            out.append((sc, a))
        if len(out) == k:
            break
    return out
