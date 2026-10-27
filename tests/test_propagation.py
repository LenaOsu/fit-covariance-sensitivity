import numpy as np

from covsens.metrics import group_decomposition
from covsens.propagation import numerical_jacobian, propagate_linear, propagate_toys, sample_gaussian


def test_linear_propagation_hand_computed():
    # N = 1*t0 + 2*t1 ; var = 1*1 + 4*4 + 2*1*2*0.5 = 19
    jac = np.array([[1.0, 2.0]])
    cov = np.array([[1.0, 0.5], [0.5, 4.0]])
    out = propagate_linear(np.zeros(1), jac, cov)
    assert np.isclose(out.cov[0, 0], 19.0)


def test_numerical_jacobian_matches_analytic(small_dataset):
    ds, resp = small_dataset
    th = ds.fit.theta_hat
    assert np.allclose(numerical_jacobian(resp, th), resp.jacobian(th), rtol=1e-5, atol=1e-8)


def test_toys_agree_with_linear_for_a_linear_model():
    rng = np.random.default_rng(0)
    jac = rng.normal(size=(5, 4))
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + 0.1 * np.eye(4)
    fn = lambda t: np.atleast_2d(t) @ jac.T + 10.0
    lin = propagate_linear(fn(np.zeros(4))[0], jac, cov)
    toys = propagate_toys(fn, np.zeros(4), cov, n=200_000, seed=2)
    assert np.allclose(toys.sigma, lin.sigma, rtol=0.02)


def test_toys_and_linear_agree_to_few_percent_on_mild_nonlinearity(small_dataset):
    ds, resp = small_dataset
    f = ds.fit
    jac = numerical_jacobian(resp, f.theta_hat)
    lin = propagate_linear(resp(f.theta_hat), jac, f.cov)
    toys = propagate_toys(resp, f.theta_hat, f.cov, n=100_000, seed=5)
    assert np.max(np.abs(toys.sigma / lin.sigma - 1)) < 0.10


def test_sampling_reproduces_covariance(small_dataset):
    f = small_dataset[0].fit
    th = sample_gaussian(f.theta_hat, f.cov, 200_000, seed=1)
    assert np.allclose(np.cov(th, rowvar=False), f.cov, atol=0.02 * np.abs(f.cov).max())


def test_sampling_handles_singular_covariance():
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])    # rank 1: Cholesky fails, eigh fallback works
    th = sample_gaussian(np.zeros(2), cov, 1000, seed=0)
    assert np.allclose(th[:, 0], th[:, 1])


def test_group_decomposition_sums_to_total(small_dataset):
    ds, resp = small_dataset
    f = ds.fit
    jac = numerical_jacobian(resp, f.theta_hat)
    parts = group_decomposition(jac, f.cov, f.groups)
    total = np.diag(jac @ f.cov @ jac.T)
    assert np.allclose(sum(parts.values()), total)
