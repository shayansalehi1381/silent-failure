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
