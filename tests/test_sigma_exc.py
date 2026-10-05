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
        )
        for _ in range(20)
    ]
    assert all(0.0 < d < 100.0 for d in draws)
    assert len(set(np.round(draws, 6))) > 1  # actually varies, not stuck


def test_sigma2_exc_log_posterior_rejects_out_of_bounds():
    r2 = np.ones(5)
    sigma_obs = np.ones(5)
    with np.errstate(divide="ignore"):
        s_at_zero = np.log(0.0)  # sigma2_exc == 0 exactly, i.e. <= 0
    assert sigma2_exc_log_posterior(s_at_zero, r2, sigma_obs, 1.0, 1.0, 10.0) == -np.inf
    assert sigma2_exc_log_posterior(np.log(20.0), r2, sigma_obs, 1.0, 1.0, 10.0) == -np.inf  # >= max


def test_sample_sigma2_exc_prefers_variance_matching_residuals():
    """The posterior should favour sigma2_exc values near the true residual variance."""
    true_var = 9.0
    rng = np.random.default_rng(2)
    residuals = rng.normal(scale=np.sqrt(true_var), size=500)
    sigma_obs = np.zeros(500)  # isolate sigma2_exc as the only error source

    draws = np.array([
        sample_sigma2_exc(10.0, residuals ** 2, sigma_obs, alpha_prior=1.0, beta_prior=1.0, sigma2_exc_max=100.0)
        for _ in range(30)
    ])
    assert abs(np.median(draws) - true_var) < 3.0
