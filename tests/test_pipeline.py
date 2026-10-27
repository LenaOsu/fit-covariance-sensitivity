import numpy as np
import yaml

from covsens.pipeline import run
from covsens.synthetic import make_synthetic_dataset, save_dataset


def test_end_to_end_runs_and_control_scenario_is_neutral(tmp_path):
    ds, resp = make_synthetic_dataset(seed=1, n_flux=6, n_xsec=4, n_det=5, n_bins=6)
    save_dataset(tmp_path / "d.npz", ds, resp)
    cfg = {
        "data": str(tmp_path / "d.npz"), "output_dir": str(tmp_path / "out"),
        "toys": {"n": 3000, "seed": 1}, "sweep": {"n_alpha": 3},
        "headline_scenario": "decorr",
        "scenarios": [
            {"name": "decorr", "type": "decorrelate_groups", "groups": ["flux", "xsec"]},
            {"name": "control", "type": "decorrelate_groups", "groups": ["flux", "det"]},
        ],
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    res = run(tmp_path / "c.yaml")
    s = res["summary"].set_index("scenario")
    assert np.isclose(s.loc["control", "mean_sigma_ratio"], 1.0)   # no flux-det correlation exists
    assert s.loc["decorr", "mean_sigma_ratio"] > 1.0                # anti-correlation was removed
    assert (tmp_path / "out" / "figures" / "01_headline_prediction_bands.png").exists()
