"""
System-level regression tests for
inversion_methods.bristau.inversion_bristau.augmented_ffbs_mxkf_gibbs_double_slice,
using small fully-synthetic InversionInput objects (no OpenGHG / real data
store needed).

These exercise real configuration combinations end-to-end and were used
while building this suite to independently confirm three bugs beyond the
kappa_x `==`/`=` typo (which is now fixed -- see the first test below,
which is a *passing* confirmation that the fix holds through the full
Gibbs loop, not just the unit-level `update_kappa_x` call):

1. `augmented_backward_sampler` does `nbc = int(config.nbc)` unconditionally
   (bristau_filter_sampler.py). When `use_bc=False`, the real pipeline
   (`bristau_monthly_dictionaries`) sets `nbc=None`, and `int(None)` raises
   TypeError. Every use_bc=False inversion currently crashes.

2. In `augmented_ffbs_mxkf_gibbs_double_slice`, the guard that decides
   whether `sigma2_rep_prior` needs parsing checks
   `sigma_qx_scheme != "fixed additive"` -- but `sigma_qx_scheme` can never
   equal that string (it belongs to sigma_rep's scheme vocabulary, not
   sigma_qx's). The condition is therefore always True, so a fixed numeric
   `sigma_rep` still requires a valid `sigma2_rep_prior`/`sigma_rep_max`,
   contradicting the documented "(only read if sigma_rep is a string)"
   contract in the example .ini files.

3. The prior-variance construction only includes the outer-basis-function
   block when `nxout > 0 and n_kappa_x_parameters == 2`. If a user sets
   `nxout > 0` (outer regions configured) but chooses a fixed/global
   (single-parameter) `kappa_x` scheme, the outer block is silently
   dropped and `zprior_sigma2s` ends up shorter than `zprior_mus`,
   crashing with a shape-broadcast error on the first Gibbs iteration.

Each bug test below currently fails (documenting the bug); fixing the
underlying code should turn it green.
"""

import numpy as np
import pytest

from inversion_methods.bristau.inversion_bristau import (
    InversionInput,
    augmented_ffbs_mxkf_gibbs_double_slice,
)


def _make_inversion_input(
    nperiod=3, nbasis=4, nxout=2, nr=2, ny_per_period=6, iterations=8,
    sigma_rep=5.0, sigma2_rep_prior=None, sigma_rep_max=100.0,
    sigma_qx=0.02, kappa_x="inner outer", nbc=0, seed=0,
):
    rng = np.random.default_rng(seed)
    Y_dic, sigma_obs_dic, Ytime_dic, Hz_dic, Hx_dic, siteindicator_dic = {}, {}, {}, {}, {}, {}
    for t in range(nperiod):
        Hx = rng.uniform(0.1, 1.0, size=(ny_per_period, nbasis))
        Hr = np.zeros((ny_per_period, nr))
        Hz_dic[t] = np.hstack([Hx, Hr])
        Hx_dic[t] = Hx
        true_x = rng.uniform(0.8, 1.2, size=nbasis)
        Y_dic[t] = Hx @ true_x + rng.normal(scale=0.1, size=ny_per_period)
        sigma_obs_dic[t] = np.full(ny_per_period, 0.1)
        Ytime_dic[t] = np.arange(ny_per_period)
        siteindicator_dic[t] = np.zeros(ny_per_period)

    xprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}
    rprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}

    return InversionInput(
        Y_dic=Y_dic, sigma_obs_dic=sigma_obs_dic, Ytime_dic=Ytime_dic, Hz_dic=Hz_dic,
        Hx_dic=Hx_dic, Hbc_dic={}, siteindicator_dic=siteindicator_dic,
        nperiod=nperiod, nbc=nbc, nxout=nxout, nr=nr, nbasis=nbasis,
        xprior=xprior, bcprior=None, rprior=rprior,
        sigma2_rep_prior=sigma2_rep_prior,
        sigma2_qx_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        sigma_qbc=0.0, sigma_qr=0.1, kappa_x_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        iterations=iterations, inner_group_id=None, ningroup=None,
        sigma_rep=sigma_rep, sigma_rep_max=sigma_rep_max,
        sigma_qx=sigma_qx, sigma_qx_max=0.5,
        kappa_x=kappa_x, kappa_x_minfold=1, kappa_bc=None, kappa_r=0.0,
    )


def test_gibbs_sampler_learns_kappa_x_inner_outer_end_to_end():
    """
    Confirms the kappa_x `==`/`=` fix holds through the full Gibbs loop:
    kappa_x_trace must show real variation, not be pinned at its
    zero/initial value. `sigma_qx` is fixed here to isolate this from the
    separate "inner outer" sigma_qx shape bug covered in test_sigma_qx.py.
    """
    config = _make_inversion_input(
        kappa_x="inner outer", nxout=2, nbc=0, sigma_qx=0.02,
        sigma2_rep_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
    )
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    kappa_x_trace = outputs[5]

    assert kappa_x_trace.shape[1] == 2
    assert not np.any(kappa_x_trace == 0.0)
    assert kappa_x_trace.std(axis=0).min() > 0.0
    assert len(np.unique(kappa_x_trace[:, 0])) > 1


def test_gibbs_sampler_supports_use_bc_false():
    """
    Bug regression: use_bc=False (nbc=None) crashes inside
    augmented_backward_sampler's unconditional `int(config.nbc)`.
    """
    config = _make_inversion_input(
        kappa_x=0.3, nxout=0, sigma_qx=0.02, nbc=None,
        sigma2_rep_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
    )
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    xtrace = outputs[0]
    assert np.all(np.isfinite(xtrace))


def test_gibbs_sampler_fixed_sigma_rep_does_not_require_sigma2_rep_prior():
    """
    Bug regression: the guard deciding whether sigma2_rep_prior must be
    parsed checks `sigma_qx_scheme != "fixed additive"` instead of
    `sigma_rep_scheme != "fixed additive"`. sigma_qx_scheme can never equal
    that string, so the guard is always True and a fixed numeric sigma_rep
    still requires a (valid) sigma2_rep_prior/sigma_rep_max, contradicting
    the documented contract.
    """
    config = _make_inversion_input(
        kappa_x=0.3, nxout=0, sigma_qx=0.02, nbc=0,
        sigma_rep=5.0, sigma2_rep_prior=None, sigma_rep_max=None,
    )
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    xtrace = outputs[0]
    assert np.all(np.isfinite(xtrace))


def test_gibbs_sampler_supports_outer_basis_functions_with_fixed_kappa():
    """
    Bug regression: the prior-variance construction only appends the
    outer-basis-function block when
    `nxout > 0 and n_kappa_x_parameters == 2`. With nxout > 0 but a fixed
    (single-parameter) kappa_x scheme, the outer block is dropped and
    zprior_sigma2s ends up shorter than zprior_mus, crashing with a
    shape-broadcast error on the very first Gibbs iteration.
    """
    config = _make_inversion_input(
        kappa_x=0.3, nxout=2, sigma_qx=0.02, nbc=0,
        sigma2_rep_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
    )
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    xtrace = outputs[0]
    assert xtrace.shape[2] == config.nbasis
    assert np.all(np.isfinite(xtrace))


def test_gibbs_sampler_supports_sigma_qx_inner_outer_with_fixed_kappa():
    """
    Bug regression: update_sigma2_qx only assigned kappa_xout_current inside
    its n_kappa_x_parameters == 2 branch. With a fixed (single-parameter)
    kappa_x scheme and sigma_qx="inner outer" (or "inner outer country"),
    kappa_xout_current was referenced unbound, raising UnboundLocalError on
    the first Gibbs iteration. A single fixed/global kappa_x should be
    usable with sigma_qx inference of any dimensionality.
    """
    config = _make_inversion_input(
        kappa_x=0.3, nxout=2, sigma_qx="inner outer", nbc=0,
        sigma2_rep_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
    )
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    xtrace, sigma2_qx_trace = outputs[0], outputs[4]
    assert sigma2_qx_trace.shape[1] == 2
    assert np.all(np.isfinite(xtrace))
    assert np.all(np.isfinite(sigma2_qx_trace))


def _make_inversion_input_with_timestamps(
    nperiod=3, nbasis=4, nr=2, ny_per_period=20, iterations=8, seed=0,
    tau_resid="global", tau_resid_prior=None, tau_resid_max=48.0,
):
    """
    Like _make_inversion_input, but with real datetime64 timestamps spread
    across two alternating sites, needed to exercise the AR(1)/OU
    same-site residual-correlation feature (ar1_resid.py).
    """
    rng = np.random.default_rng(seed)
    Y_dic, sigma_obs_dic, Ytime_dic, Hz_dic, Hx_dic, siteindicator_dic = {}, {}, {}, {}, {}, {}
    base_time = np.datetime64("2020-01-01T00:00")
    for t in range(nperiod):
        Hx = rng.uniform(0.1, 1.0, size=(ny_per_period, nbasis))
        Hr = np.zeros((ny_per_period, nr))
        Hz_dic[t] = np.hstack([Hx, Hr])
        Hx_dic[t] = Hx
        true_x = rng.uniform(0.8, 1.2, size=nbasis)
        Y_dic[t] = Hx @ true_x + rng.normal(scale=0.1, size=ny_per_period)
        sigma_obs_dic[t] = np.full(ny_per_period, 0.1)
        hours = t * 30 * 24 + np.arange(ny_per_period) * 3  # 3-hourly, per period offset by 30 days
        Ytime_dic[t] = base_time + hours.astype("timedelta64[h]")
        siteindicator_dic[t] = np.arange(ny_per_period) % 2  # two alternating sites

    xprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}
    rprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}

    return InversionInput(
        Y_dic=Y_dic, sigma_obs_dic=sigma_obs_dic, Ytime_dic=Ytime_dic, Hz_dic=Hz_dic,
        Hx_dic=Hx_dic, Hbc_dic={}, siteindicator_dic=siteindicator_dic,
        nperiod=nperiod, nbc=0, nxout=0, nr=nr, nbasis=nbasis,
        xprior=xprior, bcprior=None, rprior=rprior,
        sigma2_rep_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        sigma2_qx_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        sigma_qbc=0.0, sigma_qr=0.1, kappa_x_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        iterations=iterations, inner_group_id=None, ningroup=None,
        sigma_rep=5.0, sigma_rep_max=100.0,
        sigma_qx=0.02, sigma_qx_max=0.5,
        kappa_x=0.3, kappa_x_minfold=1, kappa_bc=None, kappa_r=0.0,
        tau_resid=tau_resid,
        tau_resid_prior=tau_resid_prior or {"pdf": "beta", "shape": 1.5, "scale": 1.5},
        tau_resid_max=tau_resid_max,
    )


def test_gibbs_sampler_default_tau_resid_is_an_exact_noop():
    """
    tau_resid defaults to 0.0 (fixed): the AR(1)/OU whitening step must be a
    strict no-op, so this feature cannot silently change results for any
    existing caller that doesn't opt in.
    """
    config = _make_inversion_input_with_timestamps(tau_resid=0.0, tau_resid_prior=None, tau_resid_max=None)
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    tau_trace, tau_trace_labels = outputs[8], outputs[9]

    assert tau_trace_labels == ["global"]
    np.testing.assert_array_equal(tau_trace, 0.0)
    assert np.all(np.isfinite(outputs[0]))


def test_gibbs_sampler_learns_tau_resid_global_end_to_end():
    """
    With tau_resid='global', tau should be sampled (not stuck at its
    initial value) and stay within (0, tau_resid_max), and the run must
    still produce finite emissions/hyperparameter traces.
    """
    config = _make_inversion_input_with_timestamps(tau_resid="global")
    outputs = augmented_ffbs_mxkf_gibbs_double_slice(config)
    tau_trace, tau_trace_labels = outputs[8], outputs[9]

    assert tau_trace_labels == ["global"]
    assert tau_trace.shape == (config.iterations - int(0.2 * config.iterations), 1)
    assert np.all((tau_trace > 0) & (tau_trace < config.tau_resid_max))
    assert len(np.unique(tau_trace)) > 1
    assert np.all(np.isfinite(outputs[0]))  # xtrace
