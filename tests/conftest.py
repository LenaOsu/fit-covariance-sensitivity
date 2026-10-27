import pytest

from covsens.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def small_dataset():
    ds, resp = make_synthetic_dataset(seed=3, n_flux=6, n_xsec=4, n_det=5, n_bins=8)
    return ds, resp
