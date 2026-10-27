
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExponentialResponse:
    n0: np.ndarray   # (n_bins,)          nominal contents at theta = 0
    a: np.ndarray    # (n_bins, n_params) linear sensitivities
    b: np.ndarray    # (n_bins, n_params) quadratic sensitivities

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        t = np.atleast_2d(theta)
        expo = t @ self.a.T + 0.5 * (t**2) @ self.b.T
        out = self.n0 * np.exp(expo)
        return out[0] if np.ndim(theta) == 1 else out

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Analytic dN_i / dtheta_j at ``theta`` (for testing)."""
        n = self(theta)
        return n[:, None] * (self.a + self.b * theta[None, :])
