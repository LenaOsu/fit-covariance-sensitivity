
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from covsens import plotting as P
from covsens.covariance import apply_scenario, check_covariance
from covsens.covariance.transforms import interpolate_correlation, split
from covsens.io import load_dataset
from covsens.metrics import correlation_shift, group_decomposition, per_bin_table, sigma_ratio
from covsens.propagation import numerical_jacobian, propagate_linear, propagate_toys

log = logging.getLogger("covsens")


def run(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg.get("output_dir", "outputs"))
    figs, tabs = out / "figures", out / "tables"
    tabs.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg["data"])
    fit, layout = ds.fit, ds.layout
    n_toys, seed = cfg["toys"]["n"], cfg["toys"]["seed"]

    # 1. validate the input covariance -------------------------------------
    rep = check_covariance(fit.cov)
    log.info("reference covariance: %s", rep)
    if not (rep.symmetric and rep.is_psd):
        raise ValueError(f"Reference covariance is not a valid covariance: {rep}")

    # 2. reference propagation (linear + toys) ------------------------------
    n_ref = ds.response(fit.theta_hat)
    jac = numerical_jacobian(ds.response, fit.theta_hat)
    ref_lin = propagate_linear(n_ref, jac, fit.cov)
    ref_toys = propagate_toys(ds.response, fit.theta_hat, fit.cov, n_toys, seed)
    P.plot_validation(layout, ref_lin, ref_toys, n_toys, figs / "05_validation_linear_vs_toys.png")
    nonlin = np.abs(ref_toys.sigma / ref_lin.sigma - 1)
    log.info("linear vs toys: max |dsigma/sigma| = %.2f%% (MC precision %.2f%%)",
             100 * nonlin.max(), 100 / np.sqrt(2 * n_toys))

    P.plot_variance_decomposition(layout, group_decomposition(jac, fit.cov, fit.groups),
                                  figs / "06_variance_decomposition.png")

    # 3. scenarios -----------------------------------------------------------
    ratios: dict[str, np.ndarray] = {}
    tables, summary = [], []
    headline = cfg["headline_scenario"]
    for spec in cfg["scenarios"]:
        name = spec["name"]
        cov_mod = apply_scenario(fit.cov, fit.groups, spec)
        r = check_covariance(cov_mod)
        if not r.is_psd:
            raise ValueError(f"Scenario {name} produced a non-PSD covariance: {r}")
        mod_lin = propagate_linear(n_ref, jac, cov_mod)
        mod_toys = propagate_toys(ds.response, fit.theta_hat, cov_mod, n_toys, seed)
        ratio = sigma_ratio(ref_lin, mod_lin)
        ratios[name] = ratio
        tables.append(per_bin_table(layout, ref_lin, mod_lin, name))
        summary.append({
            "scenario": name,
            "mean_sigma_ratio": ratio.mean(), "min_sigma_ratio": ratio.min(),
            "max_sigma_ratio": ratio.max(),
            "bin_corr_shift": correlation_shift(ref_lin, mod_lin),
            "toys_vs_linear_ratio_diff_pct": 100 * np.abs(
                sigma_ratio(ref_toys, mod_toys) - ratio).max(),
        })
        if name == headline:
            title = f"Reference vs '{name}' (linear propagation)"
            P.plot_prediction_bands(layout, ref_lin, mod_lin, title, figs / "01_headline_prediction_bands.png")
            P.plot_param_correlations(split(fit.cov)[1], split(cov_mod)[1], fit.groups,
                                      "Parameter correlations", figs / "02_parameter_correlations.png")
            P.plot_bin_correlations(layout, ref_lin, mod_lin,
                                    "Bin-to-bin correlations of the predictions",
                                    figs / "03_bin_correlations.png")
            log.info("headline '%s': sigma ratio mean %.3f (range %.3f-%.3f)",
                     name, ratio.mean(), ratio.min(), ratio.max())

    P.plot_sensitivity_map(layout, ratios, figs / "04_sensitivity_map.png")

    # 4. alpha sweep -----------------------------------------------------------
    sw = cfg["sweep"]
    alphas = np.linspace(0, 1, sw["n_alpha"])
    sweep = {h.label: np.zeros((len(alphas), h.n_bins)) for h in layout}
    for i, a in enumerate(alphas):
        lin = propagate_linear(n_ref, jac, interpolate_correlation(fit.cov, a))
        r = sigma_ratio(ref_lin, lin)
        for h in layout:
            sweep[h.label][i] = r[h.slice]
    P.plot_alpha_sweep(alphas, sweep, figs / "07_alpha_sweep.png")

    # 5. tables ------------------------------------------------------------------
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(tabs / "per_bin_sensitivity.csv", index=False)
    pd.DataFrame(summary).to_csv(tabs / "scenario_summary.csv", index=False)
    top = (full[full.scenario == headline].sort_values("sigma_ratio", ascending=False).head(10))
    top.to_csv(tabs / "top10_sensitive_bins.csv", index=False)
    log.info("tables written to %s", tabs)
    return {"summary": pd.DataFrame(summary), "top": top, "nonlinearity_max": float(nonlin.max())}
