
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ResponseFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HistSpec:
    """One histogram (a sample and a variable) inside the flat bin vector.

    All histograms are concatenated in a single vector of length n_bins_total.
    ``start:stop`` says where this histogram lives in that vector.
    """

    sample: str
    variable: str
    edges: np.ndarray
    start: int
    stop: int

    @property
    def n_bins(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def label(self) -> str:
        return f"{self.sample} | {self.variable}"


@dataclass
class FitResult:
    """What a fitter gives you: best-fit values, covariance, parameter names."""

    theta_hat: np.ndarray            # (n_params,)
    cov: np.ndarray                  # (n_params, n_params)
    param_names: list[str]
    groups: dict[str, np.ndarray]    # group name -> parameter indices

    @property
    def n_params(self) -> int:
        return len(self.theta_hat)


@dataclass
class Dataset:
    """A fit result plus the model linking parameters to binned predictions."""

    fit: FitResult
    layout: list[HistSpec]
    response: ResponseFn             # theta (n,) or (m, n)  ->  bins (B,) or (m, B)

    @property
    def n_bins(self) -> int:
        return self.layout[-1].stop


@dataclass
class Propagated:
    """Prediction mean and covariance across all bins, for one covariance."""

    mean: np.ndarray                 # (n_bins,)
    cov: np.ndarray                  # (n_bins, n_bins)
    method: str                      # "linear" or "toys"

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    @property
    def corr(self) -> np.ndarray:
        s = self.sigma
        with np.errstate(invalid="ignore", divide="ignore"):
            c = self.cov / np.outer(s, s)
        return np.nan_to_num(c)
