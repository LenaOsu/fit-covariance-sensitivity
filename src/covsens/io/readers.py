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

