"""Unit tests for inversion_methods.bristau.tau_resid."""

from types import SimpleNamespace

import numpy as np
import pytest

from inversion_methods.bristau.tau_resid import (
    tau_scheme_select,
    tau_trace_params,
    initialise_tau,
    prepare_tau_resid_indexing,
    whiten_observations,
    prepare_tau_sampler_inputs,
    sample_tau,
    update_tau_resid,
)


def test_scheme_select_fixed_default_disables_feature():
    scheme, fixed_tau = tau_scheme_select(SimpleNamespace(tau_resid=0.0))
    assert scheme == "fixed"
    assert fixed_tau == pytest.approx(0.0)


def test_scheme_select_rejects_negative_or_bad_type():
    with pytest.raises(ValueError):
        tau_scheme_select(SimpleNamespace(tau_resid=-1.0))
    with pytest.raises(TypeError):
        tau_scheme_select(SimpleNamespace(tau_resid=[1, 2]))


def test_scheme_select_normalises_global_string():
    scheme, fixed_tau = tau_scheme_select(SimpleNamespace(tau_resid="  Global "))
    assert scheme == "global"
    assert fixed_tau is None


def test_trace_params_and_initialise():
    assert tau_trace_params("fixed") == (["global"], 1)
    assert initialise_tau("fixed", fixed_tau=3.5, initial_tau=6.0) == pytest.approx(3.5)
    assert initialise_tau("global", fixed_tau=None, initial_tau=6.0) == pytest.approx(6.0)


def _two_site_example():
    # Two sites, observations spread across 2 periods, irregular timing:
    # period 0: site A @ t=0h, site B @ t=1h, site A @ t=5h
    # period 1: site B @ t=8h, site A @ t=30h
    Y_dic = {0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0])}
    Hz_dic = {
        0: np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        1: np.array([[2.0, 0.0], [0.0, 2.0]]),
    }
    sigma_obs_dic = {0: np.array([0.1, 0.1, 0.1]), 1: np.array([0.1, 0.1])}
    Ytime_dic = {
        0: np.array([0, 1, 5], dtype="datetime64[h]"),
        1: np.array([8, 30], dtype="datetime64[h]"),
    }
    siteindicator_dic = {0: np.array([0, 1, 0]), 1: np.array([1, 0])}
    return Y_dic, Hz_dic, sigma_obs_dic, Ytime_dic, siteindicator_dic


def test_prepare_tau_resid_indexing_finds_correct_same_site_predecessor_across_periods():
    Y_dic, Hz_dic, sigma_obs_dic, Ytime_dic, siteindicator_dic = _two_site_example()

    prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic, gap_flat, prev_index_flat = prepare_tau_resid_indexing(
        Y_dic, Hz_dic, Ytime_dic, siteindicator_dic, nperiod=2,
    )

    # Each site's first-ever observation has no predecessor.
    np.testing.assert_array_equal(has_prev_dic[0], [False, False, True])
    np.testing.assert_array_equal(gap_dic[0], [0.0, 0.0, 5.0])  # site A: t=5 - t=0

    # Period 1: site B's predecessor is period-0 row 1 (t=1, Y=2.0, gap=7h);
    # site A's predecessor is period-0 row 2 (t=5, Y=3.0, gap=25h) -- not row 0.
    np.testing.assert_array_equal(has_prev_dic[1], [True, True])
    np.testing.assert_allclose(gap_dic[1], [7.0, 25.0])
    np.testing.assert_allclose(prev_Y_dic[1], [2.0, 3.0])

    # Flat indices (order: p0r0, p0r1, p0r2, p1r0, p1r1).
    np.testing.assert_array_equal(prev_index_flat, [-1, -1, 0, 1, 2])
    np.testing.assert_allclose(gap_flat, [0.0, 0.0, 5.0, 7.0, 25.0])


def test_whiten_observations_is_exact_noop_when_tau_is_zero():
    Y_dic, Hz_dic, sigma_obs_dic, Ytime_dic, siteindicator_dic = _two_site_example()
    prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic, _, _ = prepare_tau_resid_indexing(
        Y_dic, Hz_dic, Ytime_dic, siteindicator_dic, nperiod=2,
    )

    Y_out, Hz_out, err_var_out = whiten_observations(
        Y_dic, Hz_dic, sigma_obs_dic, prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic,
        sigma2_rep=2.0, tau=0.0, nperiod=2,
    )

    for t in range(2):
        np.testing.assert_array_equal(Y_out[t], Y_dic[t])
        np.testing.assert_array_equal(Hz_out[t], Hz_dic[t])
        np.testing.assert_allclose(err_var_out[t], 2.0 + sigma_obs_dic[t] ** 2)


def test_whiten_observations_matches_manual_gls_formula():
    Y_dic, Hz_dic, sigma_obs_dic, Ytime_dic, siteindicator_dic = _two_site_example()
    prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic, _, _ = prepare_tau_resid_indexing(
        Y_dic, Hz_dic, Ytime_dic, siteindicator_dic, nperiod=2,
    )

    tau, sigma2_rep = 4.0, 0.0
    Y_out, Hz_out, err_var_out = whiten_observations(
        Y_dic, Hz_dic, sigma_obs_dic, prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic,
        sigma2_rep=sigma2_rep, tau=tau, nperiod=2,
    )

    # Period 1, row 0: site B, gap=7h since its predecessor (period 0, row 1).
    phi = np.exp(-7.0 / tau)
    expected_Y = 4.0 - phi * 2.0
    expected_H = np.array([2.0, 0.0]) - phi * np.array([0.0, 1.0])
    expected_err_var = (sigma2_rep + 0.1**2) * (1 - phi**2)

    assert Y_out[1][0] == pytest.approx(expected_Y)
    np.testing.assert_allclose(Hz_out[1][0], expected_H)
    assert err_var_out[1][0] == pytest.approx(expected_err_var)

    # Rows with no predecessor are untouched.
    assert Y_out[0][0] == pytest.approx(Y_dic[0][0])
    assert err_var_out[0][0] == pytest.approx(sigma2_rep + sigma_obs_dic[0][0] ** 2)


def test_sample_tau_recovers_strong_correlation():
    """
    Build standardised residuals with strong, short-range AR(1) correlation
    (true tau small relative to the observation spacing) and check the
    sampler concentrates away from the prior/boundary toward small tau,
    as opposed to a sequence with no correlation at all.
    """
    rng = np.random.default_rng(3)
    n = 400
    gap = np.full(n, 1.0)  # 1-hour spacing throughout
    has_prev = np.ones(n, dtype=bool)
    has_prev[0] = False

    true_tau = 2.0
    true_phi = np.exp(-gap / true_tau)
    standardised = np.zeros(n)
    for i in range(1, n):
        standardised[i] = true_phi[i] * standardised[i - 1] + rng.normal(scale=np.sqrt(1 - true_phi[i] ** 2))
    prev_standardised = np.roll(standardised, 1)
    prev_standardised[0] = 0.0

    draws = [
        sample_tau(3.0, tau_max=48.0, standardised=standardised, prev_standardised=prev_standardised,
                   has_prev=has_prev, gap=gap, alpha_prior=1.5, beta_prior=1.5)
        for _ in range(30)
    ]

    assert all(0.0 < d < 48.0 for d in draws)
    # Should concentrate well below the prior's midpoint (24h), reflecting
    # the short true correlation length, not just track the prior.
    assert np.median(draws) < 10.0


def test_prepare_tau_sampler_inputs_aligns_predecessor_correctly():
    state_residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma_obs = np.full(5, 0.1)
    prev_index_flat = np.array([-1, -1, 0, 1, 2])

    standardised, prev_standardised, has_prev = prepare_tau_sampler_inputs(
        state_residuals, sigma_obs, sigma2_rep=0.0, prev_index_flat=prev_index_flat,
    )

    np.testing.assert_array_equal(has_prev, [False, False, True, True, True])
    # standardised[2]'s predecessor is standardised[0]; standardised[4]'s is standardised[2].
    assert prev_standardised[2] == pytest.approx(standardised[0])
    assert prev_standardised[4] == pytest.approx(standardised[2])


def test_update_tau_resid_fixed_scheme_never_calls_sampler():
    """
    Regression test for the tau_max=None crash: with tau_scheme == "fixed",
    update_tau_resid must return fixed_tau directly and never touch
    sample_tau/tau_max, even if tau_max is None (as it would be for any
    caller that only sets tau_resid_max when actually using "global").
    """
    result = update_tau_resid(
        state_residuals=np.array([1.0, -1.0]), sigma_obs=np.array([0.1, 0.1]),
        tau_aprior=None, tau_bprior=None, tau_max=None, tau_scheme="fixed",
        tau_current=0.0, fixed_tau=0.0, sigma2_rep_current=1.0,
        obs_prev_index_flat=np.array([-1, 0]), obs_gap_flat=np.array([0.0, 1.0]),
    )
    assert result == pytest.approx(0.0)
