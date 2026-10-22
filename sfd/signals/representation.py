"""Representation-drift signals in the 512-d penultimate feature space (higher = more atypical).

Reference = class-conditional Gaussian means with ONE shared covariance, shrunk with Ledoit-Wolf, fitted on source
TRAINING features only (Lee et al. 2018). Primary score: Mahalanobis distance to the PREDICTED class; variant: min over
classes. kNN variant: distance to the k-th nearest reference feature after L2 normalisation (Sun et al. 2022, k=10).
"""
from __future__ import annotations

import numpy as np


def fit_class_gaussian(feats: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    from sklearn.covariance import LedoitWolf
    f = np.asarray(feats, np.float64)
    means = np.stack([f[labels == c].mean(0) for c in range(n_classes)])
    centred = f - means[labels]
    lw = LedoitWolf().fit(centred)
    return {"means": means.astype(np.float32), "precision": lw.precision_.astype(np.float32),
            "shrinkage": float(lw.shrinkage_)}


def mahalanobis_all(feats: np.ndarray, means: np.ndarray, precision: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """[N, C] Mahalanobis distances to every class mean (shared precision)."""
    f = np.asarray(feats, np.float64)
    m = np.asarray(means, np.float64)
    P = np.asarray(precision, np.float64)
    out = np.empty((len(f), len(m)))
    for a in range(0, len(f), chunk):
        d = f[a:a + chunk, None, :] - m[None]                          # [n, C, D]
        out[a:a + chunk] = np.sqrt(np.maximum(np.einsum("ncd,de,nce->nc", d, P, d), 0))
    return out


def mahalanobis_scores(feats, pred, ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """(distance to predicted class [primary], min distance over classes [variant])."""
    d = mahalanobis_all(feats, ref["means"], ref["precision"])
    return d[np.arange(len(d)), pred], d.min(1)


def l2n(x) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_distance(feats, bank, k: int = 10, device: str = "cuda", chunk: int = 2048, exclude_self: bool = False) -> np.ndarray:
    """L2 distance to the k-th nearest neighbour in `bank` (both L2-normalised). exclude_self skips the first match
    (used when the queries ARE the bank)."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    B = torch.from_numpy(l2n(bank)).to(dev)
    Q = l2n(feats)
    kk = k + 1 if exclude_self else k
    out = np.empty(len(Q), np.float32)
    for a in range(0, len(Q), chunk):
        q = torch.from_numpy(Q[a:a + chunk]).to(dev)
        sim = q @ B.T                                                   # cosine similarity
        top = torch.topk(sim, kk, dim=1).values[:, -1]
        out[a:a + chunk] = torch.sqrt(torch.clamp(2 - 2 * top, min=0)).cpu().numpy()
    return out
