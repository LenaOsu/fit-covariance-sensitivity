"""Linear (first-order) error propagation:  Sigma_N = J Sigma_theta J^T."""
from __future__ import annotations

import numpy as np

from covsens.data import Propagated, ResponseFn


def numerical_jacobian(fn: ResponseFn, theta: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Central finite differences, J_ij = dN_i / dtheta_j, in one vectorised call.

    Works for any ``fn`` that accepts a (m, n_params) batch, so the same code
    would apply to a real response (splines, re-weighting...).
    """
    n = len(theta)
    shifts = step * np.eye(n)
    plus = fn(theta[None, :] + shifts)    # (n, n_bins)
    minus = fn(theta[None, :] - shifts)
    return ((plus - minus) / (2 * step)).T  # (n_bins, n)


def propagate_linear(nominal: np.ndarray, jac: np.ndarray, cov: np.ndarray) -> Propagated:
    """Propagate a parameter covariance to the bin covariance."""
    return Propagated(mean=nominal, cov=jac @ cov @ jac.T, method="linear")
