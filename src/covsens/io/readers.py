"""Readers: turn files on disk into the objects defined in ``covsens.data``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from covsens.data import Dataset, FitResult, HistSpec
from covsens.response import ExponentialResponse


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset written by ``covsens.synthetic.save_dataset``."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    layout = [
        HistSpec(m["sample"], m["variable"], np.array(m["edges"]), m["start"], m["stop"])
        for m in meta["layout"]
    ]
    fit = FitResult(
        theta_hat=z["theta_hat"],
        cov=z["cov"],
        param_names=list(meta["param_names"]),
        groups={g: np.array(ix) for g, ix in meta["groups"].items()},
    )
    return Dataset(fit=fit, layout=layout, response=ExponentialResponse(z["n0"], z["a"], z["b"]))


def read_gundam_fit(path: str | Path) -> FitResult:
    """EXPERIMENTAL: read best-fit values and covariance from a GUNDAM ROOT file.

    Ported from the first prototype of this project; it was NOT exercised in the
    synthetic demo (no collaboration file is shipped). Needs ``pip install uproot``.

    Parameters are grouped by the text before the first "/" in the covariance
    axis labels (e.g. "Flux Systematics/..."). A *response model* (or Jacobian)
    must still be supplied separately: a covariance alone cannot give predictions.
    """
    try:
        import uproot
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading ROOT files needs `pip install uproot`.") from exc

    f = uproot.open(path)
    cov_obj = f["FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D"]
    cov = np.asarray(cov_obj.values())
    labels = list(cov_obj.axes[0].labels())
    if labels != list(cov_obj.axes[1].labels()):
        raise ValueError("Covariance axes have different labels.")

    best: dict[str, float] = {}
    for key in f["FitterEngine/postFit/Hesse/errors"].keys(cycle=False, recursive=False):
        hist = f[f"FitterEngine/postFit/Hesse/errors/{key}/values/postFitErrors_TH1D"]
        for name, val in zip(hist.axes[0].labels(), hist.values()):
            short = name.split("/")[-1]
            if short in best:
                raise ValueError(f"Duplicate parameter name across groups: {short}")
            best[short] = float(val)

    short_names = [lab.split("/")[-1] for lab in labels]
    missing = [n for n in short_names if n not in best]
    if missing:
        raise KeyError(f"{len(missing)} covariance parameters have no best-fit value, e.g. {missing[:3]}")

    group_of = [lab.split("/")[0] if "/" in lab else "unknown" for lab in labels]
    groups = {g: np.array([i for i, x in enumerate(group_of) if x == g]) for g in dict.fromkeys(group_of)}
    return FitResult(np.array([best[n] for n in short_names]), cov, short_names, groups)
