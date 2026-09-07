"""
Unit tests for inversion_methods.bristau.kappa_x.

The `update_kappa_x` tests are regression tests for a real bug: three
branches used `==` instead of `=` when writing the sampled kappa value,
so the Gibbs sampler silently pinned kappa_x at 0 (or crashed) regardless
of the data. That bug has since been fixed in this branch; these tests
guard against it being reintroduced.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from inversion_methods.bristau.kappa_x import (
    kappa_x_scheme_select,
    kappa_x_trace_params,
    initialise_kappa_x_vector,
    update_kappa_x,
    kappa_max,
    kappa_log_posterior,
    sample_kappa,
)


def _synthetic_ar1_samples(true_kappa, n_steps, n_series, sigma_process=0.02, seed=1):
    """
    Build zmusample-style trajectories x_t = kappa*x_{t-1} + (1-kappa)*r + noise,
    with a constant zero reference trajectory r, in the [x..., r] column
    layout expected by sample_kappa/update_kappa_x.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros((n_steps, 1))
    x = np.zeros((n_steps, n_series))
    x[0] = rng.normal(size=n_series)
    for t in range(1, n_steps):
        x[t] = true_kappa * x[t - 1] + rng.normal(scale=sigma_process, size=n_series)
    return np.hstack([x, r])


# ---------------------------------------------------------------------------
# kappa_x_scheme_select
# ---------------------------------------------------------------------------


def test_scheme_select_fixed_numeric():
    config = SimpleNamespace(kappa_x=0.3, nxout=2)
    scheme, fixed = kappa_x_scheme_select(config)
    assert scheme == "fixed"
    assert fixed == pytest.approx(0.3)


def test_scheme_select_fixed_rejects_negative():
    config = SimpleNamespace(kappa_x=-0.1, nxout=2)
    with pytest.raises(ValueError):
        kappa_x_scheme_select(config)


def test_scheme_select_normalises_whitespace_and_case():
    config = SimpleNamespace(kappa_x="  Inner   Outer ", nxout=2)
    scheme, fixed = kappa_x_scheme_select(config)
    assert scheme == "inner outer"
    assert fixed is None


def test_scheme_select_rejects_unknown_string():
    config = SimpleNamespace(kappa_x="banana", nxout=2)
    with pytest.raises(ValueError):
        kappa_x_scheme_select(config)


def test_scheme_select_rejects_bad_type():
    config = SimpleNamespace(kappa_x=[1, 2], nxout=2)
    with pytest.raises(TypeError):
        kappa_x_scheme_select(config)


def test_inner_outer_requires_outer_basis_functions():
    config = SimpleNamespace(kappa_x="inner outer", nxout=0)
    with pytest.raises(ValueError):
        kappa_x_scheme_select(config)


# ---------------------------------------------------------------------------
# kappa_x_trace_params / initialise_kappa_x_vector
# ---------------------------------------------------------------------------


def test_trace_params_fixed_and_global():
    assert kappa_x_trace_params("fixed") == (["global"], 1)
    assert kappa_x_trace_params("global") == (["global"], 1)


def test_trace_params_inner_outer():
    assert kappa_x_trace_params("inner outer") == (["outer", "inner"], 2)


def test_initialise_fixed_vector_uses_fixed_value():
    vec = initialise_kappa_x_vector("fixed", fixed_kappa_x=0.42, initial_kappa_x=0.5, n_kappa_x_parameters=2)
    np.testing.assert_allclose(vec, [0.42, 0.42])


def test_initialise_global_and_inner_outer_use_initial_value():
    np.testing.assert_allclose(initialise_kappa_x_vector("global", None, 0.5, 1), [0.5])
    np.testing.assert_allclose(initialise_kappa_x_vector("inner outer", None, 0.5, 2), [0.5, 0.5])


# ---------------------------------------------------------------------------
# update_kappa_x -- the bug regression tests
# ---------------------------------------------------------------------------


def test_update_kappa_x_fixed_returns_fixed_value_unchanged():
    result = update_kappa_x(
        zmusample=None, zmusample_out=None, zmusample_in=None,
        sigma2_qx_bf_current=None, kappa_x_aprior=None, kappa_x_bprior=None,
        nxout=1, nxin=2, kappa_x_max=None, kappa_x_scheme="fixed",
        kappa_x_vector_current=np.array([0.5, 0.5]), fixed_kappa_x=0.37,
    )
    assert result == pytest.approx(0.37)


def test_update_kappa_x_inner_outer_actually_updates():
    """
    Regression test for the `==` vs `=` bug: a correct implementation must
    return values coming from `sample_kappa`, not the zero-initialised
    array `update_kappa_x` starts from. A real slice-sampled float will
    (essentially) never come back out as exactly 0.0.
    """
    nxout, nxin = 3, 4
    zmusample_out = _synthetic_ar1_samples(true_kappa=0.8, n_steps=60, n_series=nxout, seed=1)
    zmusample_in = _synthetic_ar1_samples(true_kappa=0.8, n_steps=60, n_series=nxin, seed=2)
    sigma2_qx_bf_current = np.full(nxout + nxin, 0.01)

    sample = update_kappa_x(
        zmusample=None,
        zmusample_out=zmusample_out,
        zmusample_in=zmusample_in,
        sigma2_qx_bf_current=sigma2_qx_bf_current,
        kappa_x_aprior=2.0,
        kappa_x_bprior=2.0,
        nxout=nxout,
        nxin=nxin,
        kappa_x_max=kappa_max(nperiod=60, c=2),
        kappa_x_scheme="inner outer",
        kappa_x_vector_current=np.array([0.0, 0.0]),  # matches the sampler's zero-init before iteration 0
        fixed_kappa_x=None,
    )

    assert sample.shape == (2,)
    assert sample[0] != 0.0
    assert sample[1] != 0.0
    assert 0.0 < sample[0] < 1.0
    assert 0.0 < sample[1] < 1.0


def test_update_kappa_x_global_scheme_runs_without_nameerror():
    """
    Regression test: the `global` branch referenced `kappa_x_sample` before
    it had ever been assigned, raising NameError for any run using
    kappa_x="global" with nxout=0.
    """
    nxin = 5
    zmusample = _synthetic_ar1_samples(true_kappa=0.6, n_steps=40, n_series=nxin, seed=3)
    sigma2_qx_bf_current = np.full(nxin, 0.01)

    sample = update_kappa_x(
        zmusample=zmusample,
        zmusample_out=None,
        zmusample_in=None,
        sigma2_qx_bf_current=sigma2_qx_bf_current,
        kappa_x_aprior=2.0,
        kappa_x_bprior=2.0,
        nxout=0,
        nxin=nxin,
        kappa_x_max=kappa_max(nperiod=40, c=2),
        kappa_x_scheme="global",
        kappa_x_vector_current=np.array([0.0]),
        fixed_kappa_x=None,
    )

    assert np.isfinite(sample)
    assert 0.0 < float(sample) < 1.0


# ---------------------------------------------------------------------------
# kappa_max
# ---------------------------------------------------------------------------


def test_kappa_max_basic_formula():
    assert kappa_max(nperiod=100, c=2) == pytest.approx(0.98)


def test_kappa_max_rejects_c_greater_than_nperiod():
    with pytest.raises(ValueError):
        kappa_max(nperiod=1, c=2)


def test_kappa_max_boundary_c_equals_nperiod():
    assert kappa_max(nperiod=2, c=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# kappa_log_posterior / sample_kappa
# ---------------------------------------------------------------------------


def test_kappa_log_posterior_rejects_out_of_bounds():
    dev_curr = np.zeros((5, 2))
    dev_prev = np.zeros((5, 2))
    rmusample = np.zeros((6, 1))
    sigma2 = np.array([0.1, 0.1])
    assert kappa_log_posterior(0.0, 0.9, dev_curr, dev_prev, rmusample, sigma2, 2.0, 2.0) == -np.inf
    assert kappa_log_posterior(1.0, 0.9, dev_curr, dev_prev, rmusample, sigma2, 2.0, 2.0) == -np.inf


def test_kappa_log_posterior_prefers_true_kappa():
    """The log posterior should peak near the kappa that actually generated the data."""
    true_kappa = 0.7
    n_steps, n_series = 200, 3
    ar1 = _synthetic_ar1_samples(true_kappa, n_steps, n_series, sigma_process=0.02, seed=7)
    rmusample = ar1[:, -1:]
    dev = ar1[:, :-1] - rmusample
    dev_prev, dev_curr = dev[:-1], dev[1:]
    sigma2 = np.full(n_series, 0.02 ** 2)

    logp_true = kappa_log_posterior(true_kappa, 0.99, dev_curr, dev_prev, rmusample, sigma2, 1.0, 1.0)
    logp_far = kappa_log_posterior(0.1, 0.99, dev_curr, dev_prev, rmusample, sigma2, 1.0, 1.0)
    assert logp_true > logp_far


def test_sample_kappa_recovers_true_persistence():
    true_kappa = 0.75
    n_steps, n_series = 300, 4
    ar1 = _synthetic_ar1_samples(true_kappa, n_steps, n_series, sigma_process=0.02, seed=11)
    sigma2 = np.full(n_series, 0.02 ** 2)

    draws = np.array([
        sample_kappa(ar1, sigma2, kappa_current=0.5, kappa_max=0.995,
                     kappa_aprior=1.0, kappa_bprior=1.0, nbasis=n_series)
        for _ in range(50)
    ])

    assert np.all((draws > 0) & (draws < 0.995))
    assert abs(np.mean(draws) - true_kappa) < 0.15
