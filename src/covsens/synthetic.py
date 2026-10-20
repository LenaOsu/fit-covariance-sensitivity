"""Synthetic benchmark: a fit result and response with a known ground truth.

Design choices (each mimics a feature of real neutrino-oscillation fits):
* three parameter groups: flux, cross-section ("xsec") and detector ("det");
* correlations inside each group, plus a flux<->xsec anti-correlation, which is
  what a near-detector fit creates because both change the event rate;
* flux acts on momentum, xsec acts on normalisation and slope, detector
  parameters act locally in momentum or angle;
* two samples, two variables (Pmu, CosTheta), 20 bins each -> 80 bins.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from covsens.data import Dataset, FitResult, HistSpec
from covsens.response import ExponentialResponse


def _ar1(n: int, rho: float) -> np.ndarray:
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def make_synthetic_dataset(
    seed: int = 7,
    n_flux: int = 20,
    n_xsec: int = 12,
    n_det: int = 16,
    n_bins: int = 20,
    anticorr_strength: float = 1.0,
) -> tuple[Dataset, ExponentialResponse]:
    rng = np.random.default_rng(seed)
    samples = ["CC0pi_0p", "CC0pi_Np"]
    variables = {
        "Pmu": np.linspace(0.0, 5.0, n_bins + 1),
        "CosTheta": np.linspace(-1.0, 1.0, n_bins + 1),
    }

    # ---- layout: flat vector of all bins -------------------------------
    layout: list[HistSpec] = []
    start = 0
    for s in samples:
        for v, edges in variables.items():
            layout.append(HistSpec(s, v, edges, start, start + n_bins))
            start += n_bins
    total_bins = start

    # ---- parameters ----------------------------------------------------
    n_params = n_flux + n_xsec + n_det
    names = (
        [f"flux_{k:02d}" for k in range(n_flux)]
        + [f"xsec_{k:02d}" for k in range(n_xsec)]
        + [f"det_{k:02d}" for k in range(n_det)]
    )
    groups = {
        "flux": np.arange(0, n_flux),
        "xsec": np.arange(n_flux, n_flux + n_xsec),
        "det": np.arange(n_flux + n_xsec, n_params),
    }

    # ---- covariance (PSD by construction: blocks + rank-one term) ------
    blocks = np.zeros((n_params, n_params))
    for g, rho in (("flux", 0.85), ("xsec", 0.4), ("det", 0.25)):
        ix = groups[g]
        blocks[np.ix_(ix, ix)] = _ar1(len(ix), rho)
    u = np.zeros(n_params)
    u[groups["flux"]] = 0.9
    u[groups["xsec"]] = -1.0          # opposite sign => anti-correlation
    raw = blocks + anticorr_strength * np.outer(u, u)
    d = np.sqrt(np.diag(raw))
    corr = raw / np.outer(d, d)
    sigma = np.empty(n_params)
    sigma[groups["flux"]] = 0.30 * rng.uniform(0.8, 1.2, n_flux)
    sigma[groups["xsec"]] = 0.50 * rng.uniform(0.8, 1.2, n_xsec)
    sigma[groups["det"]] = 0.40 * rng.uniform(0.8, 1.2, n_det)
    cov = corr * np.outer(sigma, sigma)
    theta_hat = 0.25 * sigma * rng.standard_normal(n_params)

    # ---- response: sensitivities a_ij per bin and parameter ------------
    a = np.zeros((total_bins, n_params))
    det_center = rng.uniform(0.0, 1.0, (n_det, len(variables)))
    det_amp = 0.03 * rng.standard_normal((n_det, len(variables)))
    n0 = np.zeros(total_bins)
    for h in layout:
        x = (h.centers - h.edges[0]) / (h.edges[-1] - h.edges[0])  # in [0, 1]
        is_p = h.variable == "Pmu"
        is_np = h.sample.endswith("Np")
        vi = list(variables).index(h.variable)
        rows = h.slice
        # nominal spectrum
        if is_p:
            p = h.centers
            shape = p**2 * np.exp(-p / 0.45)
        else:
            shape = np.exp(2.2 * h.centers)
        n0[rows] = (15000 if is_np else 40000) * shape / shape.sum()
        # flux: smooth bumps along momentum, near-flat in angle
        for k in range(n_flux):
            c = k / (n_flux - 1)
            if is_p:
                a[rows, k] = (0.8 if is_np else 1.0) * 0.030 * np.exp(-0.5 * ((x - c) / 0.18) ** 2)
            else:
                a[rows, k] = 0.012 + 0.008 * (c - 0.5) * (2 * x - 1)
        # xsec: normalisation plus a tilt that changes sign along the group
        tilts = np.linspace(-1.0, 1.0, n_xsec)
        for k in range(n_xsec):
            amp = 0.022 * (1.5 if is_np else 1.0)
            a[rows, n_flux + k] = amp * (1.0 + 0.6 * tilts[k] * (2 * x - 1))
        # detector: localised bumps, random sign
        for k in range(n_det):
            a[rows, n_flux + n_xsec + k] = det_amp[k, vi] * np.exp(
                -0.5 * ((x - det_center[k, vi]) / 0.15) ** 2
            )
    b = 0.2 * a   # quadratic term: mild non-linearity (few-percent effect on sigma)

    response = ExponentialResponse(n0=n0, a=a, b=b)
    fit = FitResult(theta_hat=theta_hat, cov=cov, param_names=names, groups=groups)
    return Dataset(fit=fit, layout=layout, response=response), response


def save_dataset(path: str | Path, ds: Dataset, response: ExponentialResponse) -> None:
    """Store everything in one portable .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "layout": [
            {"sample": h.sample, "variable": h.variable, "edges": h.edges.tolist(),
             "start": h.start, "stop": h.stop}
            for h in ds.layout
        ],
        "groups": {g: ix.tolist() for g, ix in ds.fit.groups.items()},
        "param_names": ds.fit.param_names,
    }
    np.savez(
        path,
        theta_hat=ds.fit.theta_hat, cov=ds.fit.cov,
        n0=response.n0, a=response.a, b=response.b,
        meta=json.dumps(meta),
    )
