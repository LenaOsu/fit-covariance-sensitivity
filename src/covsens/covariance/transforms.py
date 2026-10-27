"""Controlled modifications of a covariance matrix.

Every transform works on the decomposition  Sigma = D R D  with D = diag(sigma)
and R the correlation matrix, so "changing correlations" and "changing
uncertainties" stay two separate, interpretable operations.
"""
from __future__ import annotations

import numpy as np


def split(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma, correlation). Zero-variance parameters get a unit diagonal."""
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    safe = np.where(sigma > 0, sigma, 1.0)
    corr = cov / np.outer(safe, safe)
    np.fill_diagonal(corr, 1.0)
    return sigma, corr


def join(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return corr * np.outer(sigma, sigma)


def interpolate_correlation(cov: np.ndarray, alpha: float) -> np.ndarray:
    """R_alpha = (1 - alpha) I + alpha R, with alpha in [0, 1].

    alpha = 1 -> reference, alpha = 0 -> fully decorrelated. A convex mix of two
    PSD matrices is PSD, so no repair is ever needed.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    sigma, corr = split(cov)
    return join(sigma, (1 - alpha) * np.eye(len(cov)) + alpha * corr)


def decorrelate_groups(cov: np.ndarray, groups: dict[str, np.ndarray],
                       a: str, b: str, repair: bool = True) -> np.ndarray:
    """Zero the correlations between parameter groups ``a`` and ``b``.

    Correlations *inside* each group are kept. Hand-editing a correlation matrix
    can break positive semi-definiteness, so the result is repaired by default.
    """
    sigma, corr = split(cov)
    ia, ib = groups[a], groups[b]
    corr = corr.copy()
    corr[np.ix_(ia, ib)] = 0.0
    corr[np.ix_(ib, ia)] = 0.0
    out = join(sigma, corr)
    return ensure_psd(out) if repair else out


def inflate_group(cov: np.ndarray, groups: dict[str, np.ndarray],
                  group: str, factor: float) -> np.ndarray:
    """Multiply the standard deviations of one group by ``factor`` (R unchanged)."""
    sigma, corr = split(cov)
    sigma = sigma.copy()
    sigma[groups[group]] *= factor
    return join(sigma, corr)


def ensure_psd(cov: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Nearest-PSD repair by eigenvalue clipping, preserving the diagonal.

    1. eigendecompose, 2. clip negative eigenvalues, 3. rebuild,
    4. rescale so the variances are exactly the original ones.
    """
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    if w.min() >= floor:
        return cov
    fixed = (v * np.clip(w, floor, None)) @ v.T
    d_old = np.sqrt(np.diag(cov))
    d_new = np.sqrt(np.diag(fixed))
    fixed = fixed * np.outer(d_old / d_new, d_old / d_new)
    return 0.5 * (fixed + fixed.T)


def apply_scenario(cov: np.ndarray, groups: dict[str, np.ndarray], spec: dict) -> np.ndarray:
    """Build a modified covariance from a config entry (see configs/demo.yaml)."""
    kind = spec["type"]
    if kind == "interpolate":
        return interpolate_correlation(cov, spec["alpha"])
    if kind == "decorrelate_groups":
        a, b = spec["groups"]
        return decorrelate_groups(cov, groups, a, b)
    if kind == "inflate":
        return inflate_group(cov, groups, spec["group"], spec["factor"])
    raise ValueError(f"Unknown scenario type: {kind!r}")
