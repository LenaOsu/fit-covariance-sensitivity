import numpy as np

from covsens.covariance import check_covariance


def test_valid_covariance_passes():
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    assert check_covariance(cov).is_psd


def test_non_psd_is_detected():
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])   # correlation 2 is impossible
    assert not check_covariance(cov).is_psd


def test_asymmetry_is_detected():
    assert not check_covariance(np.array([[1.0, 0.3], [0.0, 1.0]])).symmetric
