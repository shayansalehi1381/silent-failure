"""Signal normalisation against the SOURCE val_cal distribution only (Blueprint 4.3): empirical-CDF percentile and
robust z-score (median / 1.4826 MAD). Shifted data are never used here."""
from __future__ import annotations

import numpy as np


def ecdf_percentile(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fraction of reference values <= x (mid-rank for ties), in [0, 1]."""
    r = np.sort(np.asarray(ref, np.float64))
    x = np.asarray(x, np.float64)
    lo = np.searchsorted(r, x, side="left")
    hi = np.searchsorted(r, x, side="right")
    return (lo + hi) / (2.0 * len(r))


def robust_z(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    r = np.asarray(ref, np.float64)
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med))
    return (np.asarray(x, np.float64) - med) / max(mad, 1e-12)


def ecdf_percentile_ext(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Tail-extended val_cal percentile: identical to the mid-rank ECDF inside [min, max] of the reference; outside it a
    smooth monotone extension keeps the ORDER of extreme values instead of collapsing them to 0 or 1 (avoids ties):
    above max: 1 - (1/(2n)) / (1 + (x - max) / s); below min: (1/(2n)) / (1 + (min - x) / s), s = reference IQR
    (hyperbolic tails never underflow, so even extreme values stay distinct)."""
    r = np.sort(np.asarray(ref, np.float64))
    x = np.asarray(x, np.float64)
    n = len(r)
    out = ecdf_percentile(r, x)
    s = max(np.subtract(*np.percentile(r, [75, 25])), 1e-12)
    hi, lo = x > r[-1], x < r[0]
    out = np.where(hi, 1 - 1 / ((2 * n) * (1 + np.maximum(x - r[-1], 0) / s)), out)
    out = np.where(lo, 1 / ((2 * n) * (1 + np.maximum(r[0] - x, 0) / s)), out)
    return out

