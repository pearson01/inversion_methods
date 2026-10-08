"""Unit tests for inversion_methods.bristau.sigma_exc."""

from types import SimpleNamespace

import numpy as np
import pytest

from inversion_methods.bristau.sigma_exc import (
    sigma_exc_scheme_select,
    initialise_sigma2_exc,
    update_sigma2_exc,
    sigma2_exc_log_posterior,
    sample_sigma2_exc,
)


def _iid_ar1_args(n):
    """tau / prev_index_flat / gap_flat arguments that switch the AR(1) term off."""
    return 0.0, np.full(n, -1), np.zeros(n)


def test_scheme_select_fixed_numeric():
    config = SimpleNamespace(sigma_exc=25.0)
    scheme, fixed = sigma_exc_scheme_select(config)
    assert scheme == "fixed additive"
    assert fixed == pytest.approx(625.0)


def test_scheme_select_none_is_fixed_zero():
    config = SimpleNamespace(sigma_exc=None)
    scheme, fixed = sigma_exc_scheme_select(config)
    assert scheme == "fixed additive"
    assert fixed == 0.0


def test_initialise_sigma2_exc_uses_fixed_value_when_fixed():
    """The first FFBS sweep must use the fixed value, not the sampler's starting guess."""
    assert initialise_sigma2_exc("fixed additive", 0.0, 400.0) == 0.0
    assert initialise_sigma2_exc("global additive", None, 400.0) == 400.0


def test_scheme_select_squares_negative_input_rather_than_rejecting():
    """
    sigma_exc is a standard deviation, so it's squared before the
    negativity check runs -- meaning that check can never trigger, and a
    negative input is silently treated the same as its magnitude.
    """
    config = SimpleNamespace(sigma_exc=-2.0)
    scheme, fixed = sigma_exc_scheme_select(config)
    assert scheme == "fixed additive"
    assert fixed == pytest.approx(4.0)


def test_scheme_select_rejects_non_finite():
    config = SimpleNamespace(sigma_exc=float("nan"))
    with pytest.raises(ValueError):
        sigma_exc_scheme_select(config)


def test_scheme_select_global_additive_string():
    config = SimpleNamespace(sigma_exc=" Global  Additive ")
    scheme, fixed = sigma_exc_scheme_select(config)
    assert scheme == "global additive"
    assert fixed is None


def test_scheme_select_rejects_unknown_scheme():
    config = SimpleNamespace(sigma_exc="banana")
    with pytest.raises(ValueError):
        sigma_exc_scheme_select(config)


def test_update_sigma2_exc_fixed_is_static():
    result = update_sigma2_exc(
        state_residuals=np.zeros(10), sigma_obs=np.ones(10),
        sigma2_exc_aprior=None, sigma2_exc_bprior=None, sigma2_exc_max=None,
        sigma_exc_scheme="fixed additive", sigma2_exc_current=42.0, fixed_sigma2_exc=42.0,
        tau=0.0, prev_index_flat=np.full(10, -1), gap_flat=np.zeros(10),
    )
    assert result == pytest.approx(42.0)


def test_update_sigma2_exc_global_additive_moves_and_is_bounded():
    rng = np.random.default_rng(1)
    residuals = rng.normal(scale=3.0, size=200)
    sigma_obs = np.ones(200)

    draws = [
        update_sigma2_exc(
            residuals, sigma_obs, sigma2_exc_aprior=1.0, sigma2_exc_bprior=1.0,
            sigma2_exc_max=100.0, sigma_exc_scheme="global additive",
            sigma2_exc_current=10.0, fixed_sigma2_exc=None,
            tau=0.0, prev_index_flat=np.full(200, -1), gap_flat=np.zeros(200),
        )
        for _ in range(20)
    ]
    assert all(0.0 < d < 100.0 for d in draws)
    assert len(set(np.round(draws, 6))) > 1  # actually varies, not stuck


def test_sigma2_exc_log_posterior_rejects_out_of_bounds():
    r = np.ones(5)
    sigma_obs = np.ones(5)
    with np.errstate(divide="ignore"):
        s_at_zero = np.log(0.0)  # sigma2_exc == 0 exactly, i.e. <= 0
    assert sigma2_exc_log_posterior(s_at_zero, r, sigma_obs, 1.0, 1.0, 10.0, *_iid_ar1_args(5)) == -np.inf
    assert sigma2_exc_log_posterior(np.log(20.0), r, sigma_obs, 1.0, 1.0, 10.0, *_iid_ar1_args(5)) == -np.inf  # >= max


def test_sample_sigma2_exc_prefers_variance_matching_residuals():
    """The posterior should favour sigma2_exc values near the true residual variance."""
    true_var = 9.0
    rng = np.random.default_rng(2)
    residuals = rng.normal(scale=np.sqrt(true_var), size=500)
    sigma_obs = np.zeros(500)  # isolate sigma2_exc as the only error source

    draws = np.array([
        sample_sigma2_exc(10.0, residuals, sigma_obs, 1.0, 1.0, 100.0, *_iid_ar1_args(500))
        for _ in range(30)
    ])
    assert abs(np.median(draws) - true_var) < 3.0


def _ar1_dense_loglik(r, sigma_obs, sigma2_exc, tau, prev_index_flat, gap_flat):
    """Reference: full-covariance Gaussian log density for the same-site AR(1)/OU model."""
    n = len(r)
    v = sigma2_exc + sigma_obs**2
    corr = np.eye(n)
    # Correlations compose multiplicatively along each site's chain of predecessors.
    for i in range(n):
        j, rho = i, 1.0
        while prev_index_flat[j] >= 0:
            rho *= np.exp(-gap_flat[j] / tau)
            j = prev_index_flat[j]
            corr[i, j] = corr[j, i] = rho
    cov = corr * np.sqrt(np.outer(v, v))
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (logdet + r @ np.linalg.solve(cov, r))


def _two_site_layout():
    # Interleaved sites with irregular gaps, as in the flattened residual vector.
    site = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    time = np.array([0.0, 1.0, 2.0, 7.0, 9.0, 10.0, 12.0, 30.0])
    prev_index, gap, last = np.full(8, -1), np.zeros(8), {}
    for i, (s, t) in enumerate(zip(site, time)):
        if s in last:
            prev_index[i], gap[i] = last[s], t - time[last[s]]
        last[s] = i
    return prev_index, gap


def test_sigma2_exc_log_posterior_tau_zero_matches_iid_form():
    rng = np.random.default_rng(3)
    r = rng.normal(size=8)
    sigma_obs = rng.uniform(0.5, 2.0, size=8)
    prev_index, gap = _two_site_layout()
    s = np.log(2.5)

    lp = sigma2_exc_log_posterior(s, r, sigma_obs, 2.0, 3.0, 10.0, 0.0, prev_index, gap)

    err = 2.5 + sigma_obs**2
    z = 2.5 / 10.0
    expected = -0.5 * np.sum(np.log(err) + r**2 / err) + np.log(z) + 2.0 * np.log1p(-z) + s
    assert lp == pytest.approx(expected)


def test_sigma2_exc_log_posterior_matches_dense_ar1_likelihood():
    """
    Differences in log posterior between sigma2_exc values must match the
    full-covariance AR(1) likelihood (flat prior, Jacobian removed), including
    heteroscedastic sigma_obs, where the innovation itself depends on sigma2_exc.
    """
    rng = np.random.default_rng(4)
    r = rng.normal(scale=2.0, size=8)
    sigma_obs = rng.uniform(0.5, 2.0, size=8)
    prev_index, gap = _two_site_layout()
    tau = 5.0

    def lp(s2):
        return sigma2_exc_log_posterior(np.log(s2), r, sigma_obs, 1.0, 1.0, 100.0, tau, prev_index, gap) - np.log(s2)

    for a, b in [(0.5, 3.0), (1.0, 8.0), (4.0, 20.0)]:
        dense_diff = (_ar1_dense_loglik(r, sigma_obs, a, tau, prev_index, gap)
                      - _ar1_dense_loglik(r, sigma_obs, b, tau, prev_index, gap))
        assert lp(a) - lp(b) == pytest.approx(dense_diff)


def test_sample_sigma2_exc_ar1_recovers_variance_from_correlated_residuals():
    """Strongly autocorrelated residuals at one site: posterior should still centre on the truth."""
    true_var, tau, n = 9.0, 24.0, 2000
    rng = np.random.default_rng(5)
    phi = np.exp(-1.0 / tau)
    u = np.empty(n)
    u[0] = rng.normal()
    for i in range(1, n):
        u[i] = phi * u[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    residuals = np.sqrt(true_var) * u
    prev_index = np.arange(n) - 1
    gap = np.ones(n)
    gap[0] = 0.0

    draws, current = [], 10.0
    for _ in range(200):
        current = sample_sigma2_exc(current, residuals, np.zeros(n), 1.0, 1.0, 100.0, tau, prev_index, gap, rng=rng)
        draws.append(current)
    assert abs(np.median(draws) - true_var) < 1.5
