import numpy as np
import pytest

from covsens.covariance import apply_scenario, check_covariance
from covsens.covariance.transforms import (
    decorrelate_groups,
    ensure_psd,
    inflate_group,
    interpolate_correlation,
    split,
)


def test_interpolation_endpoints(small_dataset):
    cov = small_dataset[0].fit.cov
    assert np.allclose(interpolate_correlation(cov, 1.0), cov)
    zero = interpolate_correlation(cov, 0.0)
    assert np.allclose(zero, np.diag(np.diag(cov)))


@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_interpolation_preserves_psd_and_variances(small_dataset, alpha):
    cov = small_dataset[0].fit.cov
    mod = interpolate_correlation(cov, alpha)
    assert check_covariance(mod).is_psd
    assert np.allclose(np.diag(mod), np.diag(cov))


def test_group_decorrelation_zeroes_only_the_requested_block(small_dataset):
    fit = small_dataset[0].fit
    mod = decorrelate_groups(fit.cov, fit.groups, "flux", "xsec")
    ref_corr, mod_corr = split(fit.cov)[1], split(mod)[1]
    fl, xs = fit.groups["flux"], fit.groups["xsec"]
    assert check_covariance(mod).is_psd
    assert np.allclose(np.diag(mod), np.diag(fit.cov))
    # flux-xsec block is (close to) zero after repair, intra-flux correlations survive
    assert abs(mod_corr[np.ix_(fl, xs)]).max() < 0.2 * abs(ref_corr[np.ix_(fl, xs)]).max() + 1e-9
    assert np.allclose(mod_corr[np.ix_(fl, fl)], ref_corr[np.ix_(fl, fl)], atol=0.1)


def test_ensure_psd_repairs_and_keeps_diagonal():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert not check_covariance(bad).is_psd
    fixed = ensure_psd(bad)
    assert check_covariance(fixed).is_psd
    assert np.allclose(np.diag(fixed), np.diag(bad))


def test_inflate_scales_sigma_not_correlation(small_dataset):
    fit = small_dataset[0].fit
    mod = inflate_group(fit.cov, fit.groups, "det", 3.0)
    s0, c0 = split(fit.cov)
    s1, c1 = split(mod)
    assert np.allclose(c0, c1)
    assert np.allclose(s1[fit.groups["det"]], 3 * s0[fit.groups["det"]])


def test_unknown_scenario_raises(small_dataset):
    fit = small_dataset[0].fit
    with pytest.raises(ValueError):
        apply_scenario(fit.cov, fit.groups, {"type": "nope"})
