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
