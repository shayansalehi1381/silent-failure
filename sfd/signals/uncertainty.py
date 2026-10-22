"""Uncertainty signals (higher = more uncertain). Primary: 1 - MSP and entropy of the CALIBRATED probabilities;
sensitivity: energy computed from logits / T (Liu et al. 2020: E = -T_e * logsumexp(logits / T_e), here on logits / T)."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def one_minus_msp(p: np.ndarray) -> np.ndarray:
    return 1.0 - p.max(1)


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(1)


def energy(logits: np.ndarray, T: float) -> np.ndarray:
    """Negative log-sum-exp of the temperature-scaled logits (higher = more uncertain / more OOD-like)."""
    return -logsumexp(np.asarray(logits, np.float64) / T, axis=1)


def log_one_minus_msp(logits: np.ndarray, T: float) -> np.ndarray:
    """log(1 - MSP) of softmax(logits / T), computed as logsumexp(non-max) - logsumexp(all): finite and tie-free even
    when float probabilities saturate to exactly 1 (e.g. shortcut models with T ~ 0.12). Monotone in 1 - MSP."""
    z = np.asarray(logits, np.float64) / T
    mx = z.argmax(1)
    others = z.copy()
    others[np.arange(len(z)), mx] = -np.inf
    return logsumexp(others, axis=1) - logsumexp(z, axis=1)


def log_entropy(logits: np.ndarray, T: float) -> np.ndarray:
    """log of the predictive entropy, computed in log space: H = sum_j p_j * (-log p_j), so
    log H = logsumexp_j(log p_j + log(-log p_j)). For the top class -log p_max = -log1p(-u), u = 1 - MSP, whose log is
    taken from log(1 - MSP) directly when u is tiny, so saturated predictions keep distinct, finite values."""
    z = np.asarray(logits, np.float64) / T
    lp = z - logsumexp(z, axis=1, keepdims=True)
    mx = lp.argmax(1)
    l1m = log_one_minus_msp(logits, T)
    u = np.exp(l1m)
    with np.errstate(divide="ignore"):
        term = lp + np.log(np.maximum(-lp, 1e-300))
    neg_lp_max = np.where(u < 1e-8, u, -np.log1p(-np.minimum(u, 1 - 1e-16)))
    term[np.arange(len(z)), mx] = lp[np.arange(len(z)), mx] + np.where(u < 1e-8, l1m, np.log(np.maximum(neg_lp_max, 1e-300)))
    return logsumexp(term, axis=1)
