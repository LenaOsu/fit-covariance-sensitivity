"""Monte Carlo propagation: sample parameters, evaluate the model, look at the spread."""
from __future__ import annotations

import numpy as np

from covsens.data import Propagated, ResponseFn


def sample_gaussian(mean: np.ndarray, cov: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Draw n samples from N(mean, cov).

    theta = mean + L z with cov = L L^T (Cholesky) and z ~ N(0, I).
    Cholesky needs a strictly positive-definite matrix; for a singular
    (but PSD) one, e.g. after fixing a parameter, we fall back to an
    eigendecomposition: theta = mean + V sqrt(lambda) z.

    The seed is fixed per call so that reference and modified covariances use the
    *same* z: differences between scenarios are then not drowned in MC noise.
    """
    z = np.random.default_rng(seed).standard_normal((n, len(mean)))
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        factor = v * np.sqrt(np.clip(w, 0.0, None))
    return mean + z @ factor.T


def propagate_toys(fn: ResponseFn, mean: np.ndarray, cov: np.ndarray,
                   n: int = 20000, seed: int = 1) -> Propagated:
    toys = fn(sample_gaussian(mean, cov, n, seed))    # (n, n_bins)
    return Propagated(mean=toys.mean(axis=0), cov=np.cov(toys, rowvar=False), method="toys")
