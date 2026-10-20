"""Sanity checks on a covariance matrix."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CovReport:
    symmetric: bool
    min_eigenvalue: float
    is_psd: bool
    condition_number: float

    def __str__(self) -> str:
        return (f"symmetric={self.symmetric}  min_eig={self.min_eigenvalue:.3e}  "
                f"psd={self.is_psd}  cond={self.condition_number:.2e}")


def check_covariance(cov: np.ndarray, rtol: float = 1e-9) -> CovReport:
    """Symmetry and positive semi-definiteness, with a *relative* tolerance.

    ``eigvalsh`` is used (not ``eigvals``) because the matrix is symmetric:
    it is faster and returns real eigenvalues.
    """
    symmetric = bool(np.allclose(cov, cov.T, rtol=1e-8, atol=1e-12))
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    tol = rtol * max(abs(eig).max(), 1e-300)
    min_eig = float(eig.min())
    pos = eig[eig > tol]
    cond = float(eig.max() / pos.min()) if pos.size else float("inf")
    return CovReport(symmetric, min_eig, bool(min_eig >= -tol), cond)
