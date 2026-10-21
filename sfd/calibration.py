"""Temperature scaling and calibration metrics (NLL, ECE, Brier), plus prior-reweighted fitting.

Temperature T > 0 divides the logits; it is fitted by minimising the (optionally weighted) NLL of
softmax(logits / T) on the calibration split ONLY. The fit is a 1-D convex problem in log T
(NLL is convex in 1/T), solved with a bounded scalar minimiser, so the result is deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def nll(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None, T: float = 1.0) -> float:
    lp = log_softmax(np.asarray(logits, dtype=np.float64) / T, axis=1)
    ll = -lp[np.arange(len(labels)), labels]
    return float(np.average(ll, weights=weights))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None,
                    bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """argmin_T weighted NLL(softmax(logits / T)); bounds guard against degenerate fits."""
    res = minimize_scalar(lambda lt: nll(logits, labels, weights, float(np.exp(lt))),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded",
                          options={"xatol": 1e-8})
    return float(np.exp(res.x))


def prior_weights(labels_fit: np.ndarray, labels_target: np.ndarray, n_classes: int) -> np.ndarray:
    """Importance weights w(y) = p_target(y) / p_fit(y) per sample of the fitting split (label-shift correction)."""
    pf = np.bincount(labels_fit, minlength=n_classes) / len(labels_fit)
    pt = np.bincount(labels_target, minlength=n_classes) / len(labels_target)
    w = np.where(pf > 0, pt / np.maximum(pf, 1e-12), 0.0)
    return w[labels_fit]


def probs(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / T, axis=1)


def ece(p: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(tot)


def brier(p: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of sum_k (p_k - y_k)^2."""
    y = np.zeros_like(p)
    y[np.arange(len(labels)), labels] = 1.0
    return float(((p - y) ** 2).sum(1).mean())


def reliability_curve(p: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """(bin confidence, bin accuracy, bin count) of top-label confidence for a reliability diagram."""
    conf, acc = p.max(1), (p.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    c, a, n = [], [], []
    for b in range(n_bins):
        m = idx == b
        c.append(conf[m].mean() if m.any() else np.nan)
        a.append(acc[m].mean() if m.any() else np.nan)
        n.append(int(m.sum()))
    return np.array(c), np.array(a), np.array(n)


def calibration_metrics(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> dict:
    p = probs(logits, T)
    return {"nll": nll(logits, labels, T=T), "ece": ece(p, labels), "brier": brier(p, labels),
            "acc": float((p.argmax(1) == labels).mean())}
