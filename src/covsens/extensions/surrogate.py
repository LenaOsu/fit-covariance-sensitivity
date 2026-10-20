"""EXTENSION: can a machine-learning surrogate replace the expensive response N(theta)?

Why this exists
---------------
The core of covsens needs the response N(theta) evaluated many times (Jacobian, toys).
In a real analysis one evaluation can cost minutes (re-weighting millions of events).
A *surrogate* is a cheap model trained on a limited number of evaluations that imitates
N(theta). The question is not "does ML fit well?" but the one that matters for this project:

    Does the surrogate give the right per-bin uncertainty -- and the right
    sigma_mod / sigma_ref -- when the covariance is modified?

Why the second part is hard
---------------------------
The surrogate is trained on parameters drawn from the *reference* covariance. A modified
covariance (e.g. xsec uncertainties doubled) sends the parameters into regions the
surrogate never saw. Linear models extrapolate gracefully; tree models cannot extrapolate
at all. The benchmark below measures exactly this.

Everything here is a benchmark against the exact toys, which are available because the
benchmark response is known.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError("The surrogate extension needs scikit-learn: pip install -e '.[ml]'") from exc

from covsens.covariance import apply_scenario
from covsens.data import Dataset
from covsens.io import load_dataset
from covsens.propagation import numerical_jacobian, propagate_linear, sample_gaussian

LINEAR = "linear J$\\Sigma$J$^T$"
MODEL_NAMES = ("ridge", "poly2_ridge", "mlp", "rf")
COLORS = {"ridge": "#1f78b4", "poly2_ridge": "#33a02c", "mlp": "#ff7f00", "rf": "#e31a1c",
          LINEAR: "#6b6b6b"}


def make_models(seed: int = 0) -> dict:
    """Four surrogate families. Inputs and outputs are standardised for every model:
    forgetting this is what made the first prototype's MLP score a negative R^2."""
    def wrap(reg):
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), reg), transformer=StandardScaler())

    alphas = np.logspace(-3, 3, 13)
    return {
        "ridge": wrap(RidgeCV(alphas=alphas)),
        "poly2_ridge": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                    StandardScaler(), RidgeCV(alphas=alphas)),
            transformer=StandardScaler()),
        "mlp": wrap(MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=800,
                                 early_stopping=True, n_iter_no_change=20, random_state=seed)),
        "rf": wrap(RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1,
                                         random_state=seed)),
    }


def run_benchmark(ds: Dataset, covs: dict[str, np.ndarray], train_sizes: list[int],
                  n_test: int = 20000, seed: int = 11, plot_n: int | None = None):
    """Train every surrogate on the reference covariance, test on every covariance.

    Returns (table, detail). ``table`` has one row per (model, n_train, scenario).
    ``detail`` keeps per-bin sigmas at ``plot_n`` training points for plotting.
    """
    fit = ds.fit
    ref_cov = covs["reference"]
    n_ref = ds.response(fit.theta_hat)
    plot_n = plot_n or train_sizes[-1]

    # ground truth on independent test samples (same z for every scenario)
    truth = {}
    for name, cov in covs.items():
        theta = sample_gaussian(fit.theta_hat, cov, n_test, seed + 1)
        n_true = ds.response(theta)
        truth[name] = (theta, n_true.std(axis=0, ddof=1))
    true_ratio = {k: truth[k][1] / truth["reference"][1] for k in covs}

    rows, detail = [], {"truth_ratio": true_ratio, "sigma_pred": {}, "sigma_true": {k: v[1] for k, v in truth.items()}}

    def add(model, n_train, scenario, sig, sig_ref):
        e = np.abs(sig / truth[scenario][1] - 1)
        r = np.abs((sig / sig_ref) / true_ratio[scenario] - 1)
        rows.append({"model": model, "n_train": n_train, "scenario": scenario,
                     "sigma_err_median_pct": 100 * np.median(e), "sigma_err_max_pct": 100 * e.max(),
                     "ratio_err_median_pct": 100 * np.median(r), "ratio_err_max_pct": 100 * r.max()})

    # baseline: linear propagation costs 2 * n_params evaluations (central differences)
    jac = numerical_jacobian(ds.response, fit.theta_hat)
    lin_sig = {k: propagate_linear(n_ref, jac, c).sigma for k, c in covs.items()}
    for k in covs:
        add(LINEAR, 2 * fit.n_params, k, lin_sig[k], lin_sig["reference"])
    detail["sigma_pred"][(LINEAR, plot_n)] = lin_sig

    # surrogates: training pool drawn from the REFERENCE covariance only
    pool = sample_gaussian(fit.theta_hat, ref_cov, max(train_sizes), seed)
    y_pool = ds.response(pool) / n_ref - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in train_sizes:
            for name, model in make_models(seed).items():
                model.fit(pool[:n], y_pool[:n])
                sig = {}
                for k in covs:
                    pred = n_ref * (1.0 + model.predict(truth[k][0]))
                    sig[k] = pred.std(axis=0, ddof=1)
                for k in covs:
                    add(name, n, k, sig[k], sig["reference"])
                if n == plot_n:
                    detail["sigma_pred"][(name, n)] = sig
    return pd.DataFrame(rows), detail


# ----------------------------------------------------------------------------- figures
def plot_learning_curves(table: pd.DataFrame, scenarios: list[str], path) -> None:
    """Error on sigma vs number of training evaluations, one panel per covariance."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.2), sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        sub = table[table.scenario == sc]
        for m in MODEL_NAMES:
            d = sub[sub.model == m].sort_values("n_train")
            ax.plot(d.n_train, d.sigma_err_median_pct, "o-", color=COLORS[m], label=m, lw=1.8)
        lin = sub[sub.model == LINEAR].iloc[0]
        ax.axhline(lin.sigma_err_median_pct, color=COLORS[LINEAR], ls="--", lw=1.5,
                   label=f"{LINEAR} ({int(lin.n_train)} evals)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(sc if sc != "reference" else "reference covariance (in-distribution)")
        ax.set_xlabel("training evaluations of N($\\theta$)")
        ax.grid(alpha=0.25, which="both")
    np.atleast_1d(axes)[0].set_ylabel("median error on $\\sigma$ per bin [%]")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Surrogates trained on the reference covariance, tested on each covariance", fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_prediction(layout, detail, scenarios: list[str], plot_n: int, path) -> None:
    """Per-bin sigma_mod/sigma_ref: exact toys vs each surrogate."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 4.2))
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        ax.plot(detail["truth_ratio"][sc], color="k", lw=3, label="exact toys")
        for (m, n), sig in detail["sigma_pred"].items():
            if n != plot_n:
                continue
            ax.plot(sig[sc] / sig["reference"], color=COLORS[m], lw=1.6,
                    ls="--" if m == LINEAR else "-", label=m)
        for h in layout[1:]:
            ax.axvline(h.start - 0.5, color="k", lw=0.5, alpha=0.4)
        ax.axhline(1, color="grey", lw=0.8, ls=":")
        ax.set_title(sc)
        ax.set_xlabel("bin index (all histograms concatenated)")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("$\\sigma_{mod}/\\sigma_{ref}$")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Can a surrogate (trained on {plot_n} evaluations) predict the effect of a covariance change?",
                 fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(config_path: str | Path) -> pd.DataFrame:
    """Entry point used by ``covsens surrogate``."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    sc = cfg["surrogate"]
    ds = load_dataset(cfg["data"])
    specs = {s["name"]: s for s in cfg["scenarios"]}
    covs = {"reference": ds.fit.cov}
    for name in sc["scenarios"]:
        if name != "reference":
            covs[name] = apply_scenario(ds.fit.cov, ds.fit.groups, specs[name])
    out = Path(cfg.get("output_dir", "outputs"))
    table, detail = run_benchmark(ds, covs, sc["train_sizes"], sc["n_test"], sc["seed"], sc["plot_n"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tables" / "surrogate_benchmark.csv", index=False)
    plot_learning_curves(table, list(covs), out / "figures" / "S1_surrogate_learning_curves.png")
    plot_ratio_prediction(ds.layout, detail, [k for k in covs if k != "reference"], sc["plot_n"],
                          out / "figures" / "S2_surrogate_ratio_prediction.png")
    return table
