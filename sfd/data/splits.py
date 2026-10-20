"""Group-aware stratified splitting and proportional stratified sampling (deterministic given a seed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_stratified_split(groups, strata, fractions: dict[str, float], seed: int,
                           weights=None, min_groups: int = 0) -> np.ndarray:
    """Assign every row to a split so that each GROUP lands in exactly one split.

    Greedy: groups (largest first, ties broken by a seeded shuffle) go to the split whose per-stratum
    counts move closest to fraction * stratum_total. `weights` gives each row's contribution (default 1).
    `min_groups` guarantees every split at least that many groups (moves the smallest group of the
    split holding the most groups). Returns an array of split names aligned with the input rows.
    """
    groups, strata = np.asarray(groups), np.asarray(strata)
    w = np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=float)
    names = list(fractions)
    f = np.array([fractions[n] for n in names], float)
    f = f / f.sum()
    ug, inv = np.unique(groups, return_inverse=True)
    us, sinv = np.unique(strata, return_inverse=True)
    C = np.zeros((len(ug), len(us)))
    np.add.at(C, (inv, sinv), w)
    T = f[:, None] * C.sum(0)[None, :]                       # target counts [split, stratum]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ug))
    order = order[np.argsort(-C[order].sum(1), kind="stable")]
    cur = np.zeros_like(T)
    assign = np.full(len(ug), -1)
    cost = lambda x: (((x - T) / (T + 1.0)) ** 2).sum(1)      # noqa: E731  per-split cost
    for g in order:
        delta = cost(cur + C[g][None, :]) - cost(cur)
        s = int(np.argmin(delta))
        assign[g] = s
        cur[s] += C[g]
    if min_groups:
        size = C.sum(1)
        for s in range(len(names)):
            while (assign == s).sum() < min_groups:
                counts = np.bincount(assign, minlength=len(names))
                donor = int(np.argmax(counts))
                if counts[donor] <= min_groups:
                    raise ValueError("not enough groups to give every split min_groups")
                cand = np.where(assign == donor)[0]
                assign[cand[np.argmin(size[cand])]] = s
    return np.array(names, dtype=object)[assign[inv]]


def _allocate(avail: np.ndarray, n: int) -> np.ndarray:
    """Proportional integer allocation of n over strata (largest remainder), capped by availability."""
    avail = np.asarray(avail, dtype=int)
    n = int(min(n, avail.sum()))
    exact = n * avail / max(avail.sum(), 1)
    q = np.minimum(np.floor(exact).astype(int), avail)
    rem = n - q.sum()
    for i in np.argsort(-(exact - np.floor(exact)), kind="stable"):
        if rem <= 0:
            break
        if q[i] < avail[i]:
            q[i] += 1
            rem -= 1
    while rem > 0:                                            # strata exhausted: spread the rest
        room = avail - q
        i = int(np.argmax(room))
        take = int(min(room[i], rem))
        q[i] += take
        rem -= take
    return q


def stratified_sample(df: pd.DataFrame, n: int, strata_cols: list[str], seed: int) -> pd.DataFrame:
    """Sample n rows with proportional allocation over strata (capped by availability)."""
    if n >= len(df):
        return df.copy()
    keys = df[strata_cols].astype(str).agg("|".join, axis=1)
    cnt = keys.value_counts().sort_index()
    quota = _allocate(cnt.values, n)
    rng = np.random.default_rng(seed)
    picked = []
    for k, q in zip(cnt.index, quota):
        idx = np.sort(df.index[keys.values == k].to_numpy())
        picked.extend(rng.choice(idx, size=int(q), replace=False).tolist())
    return df.loc[sorted(picked)].copy()


def balanced_sample(df: pd.DataFrame, n: int, group_col: str, label_col: str, seed: int) -> pd.DataFrame:
    """Sample n rows: groups (e.g. hospital or slide) proportionally to availability, and within each
    group as close to 50/50 between the labels as availability allows (shortfall goes to the other label)."""
    if n >= len(df):
        return df.copy()
    g_cnt = df[group_col].value_counts().sort_index()
    g_quota = _allocate(g_cnt.values, n)
    rng = np.random.default_rng(seed)
    picked = []
    for g, q in zip(g_cnt.index, g_quota):
        sub = df[df[group_col] == g]
        labs = sorted(sub[label_col].unique())
        avail = np.array([(sub[label_col] == l).sum() for l in labs])
        lq = np.minimum(avail, q // len(labs))                 # equal share, capped
        lq = lq + _allocate(avail - lq, q - lq.sum())          # remainder to labels with room
        for l, k in zip(labs, lq):
            idx = np.sort(sub.index[sub[label_col] == l].to_numpy())
            picked.extend(rng.choice(idx, size=int(k), replace=False).tolist())
    return df.loc[sorted(picked)].copy()
