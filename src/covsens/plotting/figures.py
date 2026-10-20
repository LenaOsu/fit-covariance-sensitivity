"""All figures. Each function takes arrays/objects and a path; no computation of physics here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from covsens.data import HistSpec, Propagated

REF, MOD, GREY = "#1f4e79", "#d95f02", "#6b6b6b"
GROUP_COLORS = {"flux": "#1f78b4", "xsec": "#33a02c", "det": "#984ea3", "cross-group": "#e31a1c"}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _xlabel(h: HistSpec) -> str:
    return "p$_\\mu$ [GeV]" if h.variable == "Pmu" else "cos $\\theta_\\mu$"


def plot_prediction_bands(layout, ref: Propagated, mod: Propagated, title: str, path) -> None:
    """Hero figure: prediction with +-1 sigma bands (ref vs modified) and the sigma ratio."""
    fig, axes = plt.subplots(2, len(layout), figsize=(4.2 * len(layout), 6.2),
                             gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    for col, h in enumerate(layout):
        s = h.slice
        n, e = ref.mean[s], h.edges
        top, bot = axes[0, col], axes[1, col]
        top.stairs(n + ref.sigma[s], baseline=n - ref.sigma[s], edges=e, fill=True,
                   color=REF, alpha=0.30, label="reference $\\pm1\\sigma$")
        top.stairs(n + mod.sigma[s], edges=e, baseline=None, color=MOD, lw=1.6,
                   label="modified $\\pm1\\sigma$")
        top.stairs(n - mod.sigma[s], edges=e, baseline=None, color=MOD, lw=1.6)
        top.stairs(n, edges=e, baseline=None, color=REF, lw=1.4, label="best-fit prediction")
        top.set_title(h.label, fontsize=10)
        top.set_xticklabels([])
        if col == 0:
            top.set_ylabel("events / bin")
        ratio = mod.sigma[s] / ref.sigma[s]
        bot.stairs(ratio, edges=e, baseline=None, color=MOD, lw=1.8)
        bot.axhline(1, color=GREY, lw=1, ls="--")
        bot.set_xlabel(_xlabel(h))
        if col == 0:
            bot.set_ylabel("$\\sigma_{mod}/\\sigma_{ref}$")
        top.sharex(bot)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, y=0.995, fontsize=12)
    _save(fig, path)


def plot_param_correlations(corr_ref, corr_mod, groups, title: str, path) -> None:
    """Parameter correlation matrices: reference, modified, difference."""
    diff = corr_mod - corr_ref
    lim = max(abs(diff).max(), 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    panels = [(corr_ref, "reference", -1, 1), (corr_mod, "modified", -1, 1),
              (diff, "modified $-$ reference", -lim, lim)]
    bounds = [ix[0] for ix in groups.values()] + [list(groups.values())[-1][-1] + 1]
    centers = [(ix[0] + ix[-1]) / 2 for ix in groups.values()]
    for ax, (m, name, lo, hi) in zip(axes, panels):
        im = ax.imshow(m, cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="nearest")
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="k", lw=0.8)
            ax.axvline(b - 0.5, color="k", lw=0.8)
        ax.set_xticks(centers, list(groups), rotation=0)
        ax.set_yticks(centers, list(groups))
        ax.grid(False)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def plot_bin_correlations(layout, ref: Propagated, mod: Propagated, title: str, path) -> None:
    """Bin-to-bin correlation of the predictions (all samples and variables stacked)."""
    diff = mod.corr - ref.corr
    lim = max(abs(diff).max(), 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    panels = [(ref.corr, "reference", -1, 1), (mod.corr, "modified", -1, 1),
              (diff, "modified $-$ reference", -lim, lim)]
    centers = [(h.start + h.stop - 1) / 2 for h in layout]
    names = [f"{h.sample}\n{h.variable}" for h in layout]
    for ax, (m, name, lo, hi) in zip(axes, panels):
        im = ax.imshow(m, cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="nearest")
        for h in layout[1:]:
            ax.axhline(h.start - 0.5, color="k", lw=0.8)
            ax.axvline(h.start - 0.5, color="k", lw=0.8)
        ax.set_xticks(centers, names, fontsize=7)
        ax.set_yticks(centers, names, fontsize=7)
        ax.grid(False)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def plot_alpha_sweep(alphas, ratios: dict[str, np.ndarray], path) -> None:
    """Sensitivity curve: mean sigma_mod/sigma_ref over bins vs alpha (R_alpha = (1-a)I + aR)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    cmap = plt.get_cmap("tab10")
    for i, (label, r) in enumerate(ratios.items()):       # r: (n_alpha, n_bins_of_hist)
        ax.plot(alphas, r.mean(axis=1), color=cmap(i), lw=2, label=label)
        ax.fill_between(alphas, r.min(axis=1), r.max(axis=1), color=cmap(i), alpha=0.12)
    ax.axhline(1, color=GREY, ls="--", lw=1)
    ax.set_xlabel("$\\alpha$   (0 = fully decorrelated, 1 = reference)")
    ax.set_ylabel("$\\sigma_{mod}/\\sigma_{ref}$ (mean over bins, band = min-max)")
    ax.set_title("How much do parameter correlations matter?")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_validation(layout, lin: Propagated, toys: Propagated, n_toys: int, path) -> None:
    """Linear propagation vs Monte Carlo toys: the cross-check of the method."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.get_cmap("tab10")
    for i, h in enumerate(layout):
        a1.scatter(lin.sigma[h.slice], toys.sigma[h.slice], s=22, color=cmap(i), label=h.label)
    m = max(lin.sigma.max(), toys.sigma.max()) * 1.05
    a1.plot([0, m], [0, m], color=GREY, ls="--", lw=1)
    a1.set_xlabel("$\\sigma$ linear  ($J\\Sigma J^T$)")
    a1.set_ylabel("$\\sigma$ toys")
    a1.legend(fontsize=7)
    a1.set_title("Per-bin uncertainty")
    mc_err = 1 / np.sqrt(2 * n_toys)
    for i, h in enumerate(layout):
        a2.plot(np.arange(h.start, h.stop),
                toys.sigma[h.slice] / lin.sigma[h.slice] - 1, color=cmap(i), lw=1.5)
    a2.axhspan(-mc_err, mc_err, color=GREY, alpha=0.25, label=f"MC precision ($\\pm${100*mc_err:.2f}%)")
    a2.axhline(0, color=GREY, lw=1)
    for h in layout[1:]:
        a2.axvline(h.start - 0.5, color="k", lw=0.6, alpha=0.5)
    a2.set_xlabel("bin index (all histograms concatenated)")
    a2.set_ylabel("$\\sigma_{toys}/\\sigma_{lin} - 1$")
    a2.set_title("Residual = non-linearity of the response")
    a2.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_sensitivity_map(layout, ratios: dict[str, np.ndarray], path) -> None:
    """Scenarios (rows) x bins (columns): where does each modification bite?"""
    names = list(ratios)
    mat = np.vstack([ratios[n] for n in names])
    lo, hi = min(mat.min(), 0.95), max(mat.max(), 1.05)
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(names) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(1.0, lo, hi),
                   interpolation="nearest")
    for h in layout[1:]:
        ax.axvline(h.start - 0.5, color="k", lw=1)
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks([(h.start + h.stop - 1) / 2 for h in layout],
                  [f"{h.sample}\n{h.variable}" for h in layout])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="$\\sigma_{mod}/\\sigma_{ref}$", pad=0.01)
    ax.set_title("Sensitivity map: effect of each covariance modification, bin by bin")
    fig.tight_layout()
    _save(fig, path)


def plot_variance_decomposition(layout, contrib: dict[str, np.ndarray], path) -> None:
    """Fraction of each bin's variance coming from each group (+ the cross-group term).

    Negative bars = cancellation induced by anti-correlation between groups.
    """
    fig, axes = plt.subplots(1, len(layout), figsize=(4.2 * len(layout), 3.9), sharey=True)
    total = sum(contrib.values())
    for ax, h in zip(np.atleast_1d(axes), layout):
        x = np.arange(h.n_bins)
        pos = np.zeros(h.n_bins)
        neg = np.zeros(h.n_bins)
        for g, v in contrib.items():
            frac = v[h.slice] / total[h.slice]
            bottom = np.where(frac >= 0, pos, neg)
            ax.bar(x, frac, bottom=bottom, width=0.85, color=GROUP_COLORS.get(g, GREY),
                   label=g)
            pos += np.clip(frac, 0, None)
            neg += np.clip(frac, None, 0)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(1, color=GREY, lw=0.8, ls=":")
        ax.set_title(h.label, fontsize=9)
        ax.set_xlabel("bin")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    np.atleast_1d(axes)[0].set_ylabel("fraction of bin variance")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle("Where does each bin's uncertainty come from? (reference covariance)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, path)
