"""Unit tests for inversion_methods.manipulation.lognormal_transformations."""

import numpy as np
import pytest

from inversion_methods.manipulation.lognormal_transformations import (
    lognormal_mean_stdev,
    lognormal_mode_stdev,
    lognormal_median_stdev,
    state_vector_mu_transform,
    build_Wb,
    covariance_lognormal_transform,
    update_log_normal_prior,
    state_percentiles,
)


def test_lognormal_mean_stdev_round_trip():
    mu_true, sigma_true = 0.4, 0.6
    mean = np.exp(mu_true + 0.5 * sigma_true ** 2)
    var = np.exp(2 * mu_true + sigma_true ** 2) * (np.exp(sigma_true ** 2) - 1)
    stdev = np.sqrt(var)

    mu, sigma = lognormal_mean_stdev(mean, stdev)
    assert mu == pytest.approx(mu_true, rel=1e-8)
    assert sigma == pytest.approx(sigma_true, rel=1e-8)


def test_lognormal_median_stdev_round_trip():
    mu_true, sigma_true = -0.3, 0.5
    median = np.exp(mu_true)
    var = np.exp(2 * mu_true + sigma_true ** 2) * (np.exp(sigma_true ** 2) - 1)
    stdev = np.sqrt(var)

    mu, sigma = lognormal_median_stdev(median, stdev)
    assert mu == pytest.approx(mu_true, rel=1e-8)
    assert sigma == pytest.approx(sigma_true, rel=1e-6)


def test_lognormal_mode_stdev_round_trip():
    mu_true, sigma_true = 0.2, 0.4
    mode = np.exp(mu_true - sigma_true ** 2)
    var = np.exp(2 * mu_true + sigma_true ** 2) * (np.exp(sigma_true ** 2) - 1)
    stdev = np.sqrt(var)

    mu, sigma = lognormal_mode_stdev(mode, stdev)
    assert mu == pytest.approx(mu_true, rel=1e-4)
    assert sigma == pytest.approx(sigma_true, rel=1e-4)


def test_state_vector_mu_transform_exponentiates_only_lognormal_blocks():
    xprior = {"pdf": "lognormal"}
    bcprior = {"pdf": "normal"}
    rprior = {"pdf": "lognormal"}
    nbasis, nbc, nr = 2, 2, 2

    z_mu = np.array([0.1, 0.2, 1.0, -1.0, 0.3, 0.4])
    z = state_vector_mu_transform(z_mu, xprior, bcprior, rprior, nbasis, nbc, nr)

    np.testing.assert_allclose(z[:nbasis], np.exp(z_mu[:nbasis]))
    np.testing.assert_allclose(z[nbasis:nbasis + nbc], z_mu[nbasis:nbasis + nbc])  # untouched (normal)
    np.testing.assert_allclose(z[-nr:], np.exp(z_mu[-nr:]))


def test_build_wb_matches_expected_diagonal():
    xprior = {"pdf": "lognormal"}
    bcprior = {"pdf": "normal"}
    rprior = {"pdf": "lognormal"}
    nbasis, nbc = 2, 2

    zf = np.array([2.0, 3.0, 5.0, 7.0, 1.5, 2.5])  # already in natural space
    Wb = build_Wb(zf, xprior, bcprior, rprior, nbasis, nbc)

    expected_diag = np.array([2.0, 3.0, 1.0, 1.0, 1.5, 2.5])
    np.testing.assert_allclose(np.diag(Wb), expected_diag)
    assert np.count_nonzero(Wb - np.diag(np.diag(Wb))) == 0  # purely diagonal


def test_covariance_lognormal_transform_matches_manual_formula():
    mean_normal = 0.0
    stdev_normal = 0.3
    cov_lognormal = np.array([[0.5, 0.1], [0.1, 0.5]])

    cov_normal, prec_normal = covariance_lognormal_transform(cov_lognormal, mean_normal, stdev_normal)

    mean_ln = np.exp(mean_normal + 0.5 * stdev_normal ** 2)
    expected_offdiag = np.log(1 + cov_lognormal[0, 1] / mean_ln ** 2)
    assert cov_normal[0, 1] == pytest.approx(expected_offdiag)
    assert cov_normal[1, 0] == pytest.approx(expected_offdiag)
    np.testing.assert_allclose(np.diag(cov_normal), stdev_normal ** 2)
    np.testing.assert_allclose(prec_normal, np.linalg.inv(cov_normal))


def test_update_log_normal_prior_from_mean_and_stdev():
    prior = {"pdf": "lognormal", "mean": 10.0, "stdev": 2.0}
    update_log_normal_prior(prior)
    assert set(prior) == {"pdf", "mu", "sigma"}

    mu, sigma = lognormal_mean_stdev(10.0, 2.0)
    assert prior["mu"] == pytest.approx(mu)
    assert prior["sigma"] == pytest.approx(sigma)


def test_update_log_normal_prior_passthrough_for_mu_sigma():
    prior = {"pdf": "lognormal", "mu": 1.0, "sigma": 0.5}
    update_log_normal_prior(prior)
    assert prior == {"pdf": "lognormal", "mu": 1.0, "sigma": 0.5}


def test_update_log_normal_prior_ignores_normal_pdf():
    prior = {"pdf": "normal", "mu": 1.0, "sigma": 0.5}
    update_log_normal_prior(prior)
    assert prior == {"pdf": "normal", "mu": 1.0, "sigma": 0.5}


def test_update_log_normal_prior_rejects_incompatible_keys():
    prior = {"pdf": "lognormal", "mean": 10.0}  # stdev missing
    with pytest.raises(ValueError):
        update_log_normal_prior(prior)


def test_state_percentiles_lognormal_and_normal():
    mus = np.array([0.0, 1.0])
    sigmas = np.array([0.5, 0.2])

    out68_ln, _out95_ln = state_percentiles(mus, sigmas, {"pdf": "lognormal"}, len(mus))
    np.testing.assert_allclose(out68_ln[0], np.exp(mus - sigmas))
    np.testing.assert_allclose(out68_ln[1], np.exp(mus + sigmas))

    out68_n, _out95_n = state_percentiles(mus, sigmas, {"pdf": "normal"}, len(mus))
    np.testing.assert_allclose(out68_n[0], mus - sigmas)
    np.testing.assert_allclose(out68_n[1], mus + sigmas)
