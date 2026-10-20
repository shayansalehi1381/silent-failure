"""Near-duplicate detection: cheap image fingerprints + exact radius search + connected components.

Fingerprint = block-averaged colour thumbnail (default 16x16x3, values in [0,1]). Distance = RMS
difference between thumbnails (0 = identical, 1 = black vs white). Pairs within `max_rms` are found
by an exact chunked brute-force search (GPU if available), clusters are connected components of the
pair graph at a chosen threshold. Exact duplicates (equal SHA-256 of the pixel bytes) have RMS 0.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def thumbnails(imgs: np.ndarray, size: int = 16, chunk: int = 2048) -> np.ndarray:
    """imgs uint8 NxHxWx3 -> float32 N x (size*size*3) block means in [0,1]."""
    n, h, w, c = imgs.shape
    assert h % size == 0 and w % size == 0, f"{h}x{w} not divisible by {size}"
    fh, fw = h // size, w // size
    out = np.empty((n, size * size * c), dtype=np.float32)
    for a in range(0, n, chunk):
        x = imgs[a:a + chunk].reshape(-1, size, fh, size, fw, c).mean((2, 4), dtype=np.float32)
        out[a:a + chunk] = x.reshape(len(x), -1) / 255.0
    return out


def exact_hashes(imgs: np.ndarray) -> np.ndarray:
    """First 8 bytes of SHA-256 of every image's pixel bytes, as uint64."""
    return np.fromiter((int.from_bytes(hashlib.sha256(im.tobytes()).digest()[:8], "little") for im in imgs),
                       dtype=np.uint64, count=len(imgs))


def radius_pairs(a: np.ndarray, b: np.ndarray | None = None, max_rms: float = 0.05, chunk: int = 2048,
                 device: str = "auto", cap: int = 30_000_000):
    """All pairs (i, j, rms) with rms <= max_rms.

    b is None: self-join, i < j within `a`.  b given: cross-join, i indexes `a`, j indexes `b`.
    Stops collecting once `cap` pairs were found and reports saturated=True (counts are then lower bounds).
    """
    import torch
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       ("cpu" if device == "auto" else device))
    A = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    B = A if b is None else torch.from_numpy(np.ascontiguousarray(b)).to(dev)
    dim = A.shape[1]
    thr2 = (max_rms ** 2) * dim
    Bn = (B * B).sum(1)
    cols = torch.arange(B.shape[0], device=dev)
    I, J, R, found, saturated = [], [], [], 0, False
    for s in range(0, A.shape[0], chunk):
        x = A[s:s + chunk]
        d2 = (x * x).sum(1)[:, None] + Bn[None, :] - 2.0 * (x @ B.T)
        m = d2 <= thr2
        if b is None:
            m &= cols[None, :] > torch.arange(s, s + len(x), device=dev)[:, None]
        nz = m.nonzero()
        if len(nz):
            I.append((nz[:, 0] + s).cpu().numpy())
            J.append(nz[:, 1].cpu().numpy())
            R.append(torch.sqrt(d2[m].clamp_min(0) / dim).cpu().numpy())
            found += len(nz)
            if found >= cap:
                saturated = True
                break
        del d2, m
    if not I:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32), False
    return np.concatenate(I), np.concatenate(J), np.concatenate(R).astype(np.float32), saturated


def cluster_labels(n: int, i: np.ndarray, j: np.ndarray, rms: np.ndarray, thr: float) -> np.ndarray:
    """Connected components over pairs with rms <= thr; every image gets a cluster id (singletons too)."""
    k = rms <= thr
    g = coo_matrix((np.ones(k.sum(), dtype=np.int8), (i[k], j[k])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def sensitivity_table(n: int, i, j, rms, thresholds, labels=None, saturated: bool = False):
    """Threshold sensitivity: pairs, non-singleton clusters, images inside them, largest cluster."""
    rows = []
    for t in thresholds:
        cl = cluster_labels(n, i, j, rms, t)
        sizes = np.bincount(cl)
        multi = sizes > 1
        row = {"rms_threshold": t, "n_pairs": int((rms <= t).sum()), "n_clusters": int(len(sizes)),
               "n_dup_clusters": int(multi.sum()), "n_images_in_dup_clusters": int(sizes[multi].sum()),
               "max_cluster_size": int(sizes.max()), "pairs_saturated": bool(saturated)}
        if labels is not None and multi.any():
            lab = np.asarray(labels)
            order = np.argsort(cl, kind="stable")
            first = np.r_[0, np.cumsum(sizes)[:-1]]
            mn = np.minimum.reduceat(lab[order], first)
            mx = np.maximum.reduceat(lab[order], first)
            row["n_label_mixed_clusters"] = int(((mn != mx) & multi).sum())
        rows.append(row)
    return rows
