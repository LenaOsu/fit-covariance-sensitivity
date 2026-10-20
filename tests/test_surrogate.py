import numpy as np
import pytest

pytest.importorskip("sklearn")

from covsens.covariance.transforms import inflate_group
from covsens.extensions.surrogate import LINEAR, run_benchmark


def test_benchmark_structure_and_sanity(small_dataset):
    ds, _ = small_dataset
    covs = {"reference": ds.fit.cov,
            "inflate": inflate_group(ds.fit.cov, ds.fit.groups, "xsec", 2.0)}
    table, detail = run_benchmark(ds, covs, train_sizes=[100, 400], n_test=4000, seed=3, plot_n=400)

    assert {"ridge", "poly2_ridge", "mlp", "rf", LINEAR} <= set(table.model)
    assert set(table.scenario) == set(covs)
    # the ratio is defined relative to the reference, so it is exactly 0 error there
    assert (table[table.scenario == "reference"].ratio_err_median_pct == 0).all()
    # a linear surrogate must do well in-distribution on a mildly non-linear response
    ridge = table[(table.model == "ridge") & (table.n_train == 400) & (table.scenario == "reference")]
    assert ridge.sigma_err_median_pct.iloc[0] < 10
    assert np.all(np.isfinite(table.select_dtypes("number").to_numpy()))
    assert (LINEAR, 400) in detail["sigma_pred"]
