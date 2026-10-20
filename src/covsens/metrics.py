"""Sensitivity metrics: how much does a covariance change move the predictions?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from covsens.data import HistSpec, Propagated


def sigma_ratio(ref: Propagated, mod: Propagated) -> np.ndarray:
    """sigma_mod / sigma_ref per bin. >1 means the modification inflates the error."""
    return mod.sigma / ref.sigma


def per_bin_table(layout: list[HistSpec], ref: Propagated, mod: Propagated,
                  scenario: str) -> pd.DataFrame:
    """Tidy table, one row per bin (easy to sort, filter, export)."""
    rows = []
    ratio = sigma_ratio(ref, mod)
    for h in layout:
        for i in range(h.n_bins):
            k = h.start + i
            rows.append({
                "scenario": scenario, "sample": h.sample, "variable": h.variable,
                "bin": i, "bin_lo": h.edges[i], "bin_hi": h.edges[i + 1],
                "n_ref": ref.mean[k], "sigma_ref": ref.sigma[k], "sigma_mod": mod.sigma[k],
                "rel_ref_pct": 100 * ref.sigma[k] / ref.mean[k],
                "rel_mod_pct": 100 * mod.sigma[k] / mod.mean[k],
                "sigma_ratio": ratio[k],
            })
    return pd.DataFrame(rows)


def correlation_shift(ref: Propagated, mod: Propagated) -> float:
    """Frobenius norm of (corr_mod - corr_ref) divided by the number of bins."""
    return float(np.linalg.norm(mod.corr - ref.corr) / len(ref.mean))


def group_decomposition(jac: np.ndarray, cov: np.ndarray,
                        groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split the per-bin variance by parameter group.

    The variance is  sum_{g,h} J_g Sigma_gh J_h^T. The diagonal blocks (g = h)
    are each group's own contribution; everything else is the 'cross' term,
    i.e. the part that exists only because of correlations *between* groups.
    It can be negative: that is cancellation.
    """
    total = np.einsum("ij,jk,ik->i", jac, cov, jac)
    out = {}
    for g, ix in groups.items():
        jg = jac[:, ix]
        out[g] = np.einsum("ij,jk,ik->i", jg, cov[np.ix_(ix, ix)], jg)
    out["cross-group"] = total - sum(out.values())
    return out
