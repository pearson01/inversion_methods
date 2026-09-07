"""Unit tests for inversion_methods.bristau.bristau_filter_sampler core math."""

import numpy as np
import pytest
from scipy.linalg import cholesky

from inversion_methods.bristau.bristau_filter_sampler import (
    augmented_forecast_model,
    analysis_covariance_update,
    iterative_analysis_update,
    iterative_augmented_mxkf,
    augmented_backward_sampler,
    amxkf_inputs,
)


def _naive_F_aug(nbasis, nbc, nxout, nr, kappa_out, kappa_in, kappa_r, rprior_mu):
    """Explicit dense transition matrix matching augmented_forecast_model's docstring."""
    nstate = nbasis + nbc + nr
    F = np.zeros((nstate, nstate))
    r_out_idx, r_in_idx = nbasis + nbc, nbasis + nbc + 1

    for i in range(nxout):
        F[i, i] = kappa_out
        F[i, r_out_idx] = 1 - kappa_out
    for i in range(nxout, nbasis):
        F[i, i] = kappa_in
        F[i, r_in_idx] = 1 - kappa_in
    for i in range(nbasis, nbasis + nbc):
        F[i, i] = 1.0  # persistent boundary conditions

    F[r_out_idx, r_out_idx] = kappa_r
    F[r_in_idx, r_in_idx] = kappa_r

    b = np.zeros(nstate)
    b[r_out_idx] = (1 - kappa_r) * rprior_mu
    b[r_in_idx] = (1 - kappa_r) * rprior_mu

    return F, b


def test_augmented_forecast_model_matches_naive_matrix_construction():
    nbasis, nbc, nxout, nr = 5, 2, 2, 2
    kappa_out, kappa_in, kappa_r = 0.7, 0.4, 0.1
    rprior_mu = 1.3
    nstate = nbasis + nbc + nr

    rng = np.random.default_rng(21)
    z_mu = rng.normal(size=nstate)
    A = rng.normal(size=(nstate, nstate))
    P_aug = A @ A.T + nstate * np.eye(nstate)
    Q_aug = rng.uniform(0.01, 0.05, size=nstate)

    zf_mu, Pf_aug = augmented_forecast_model(
        z_mu, P_aug, kappa_out, kappa_in, Q_aug, nbasis, nbc, nxout, nr, kappa_r, rprior_mu,
    )

    F, b = _naive_F_aug(nbasis, nbc, nxout, nr, kappa_out, kappa_in, kappa_r, rprior_mu)
    zf_naive = F @ z_mu + b
    Pf_naive = F @ P_aug @ F.T + np.diag(Q_aug)

    np.testing.assert_allclose(zf_mu, zf_naive, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(Pf_aug, Pf_naive, rtol=1e-8, atol=1e-10)


def test_augmented_forecast_model_boundary_conditions_are_persistent():
    nbasis, nbc, nxout, nr = 3, 2, 1, 2
    nstate = nbasis + nbc + nr
    rng = np.random.default_rng(22)
    z_mu = rng.normal(size=nstate)
    P_aug = np.eye(nstate)
    Q_aug = np.zeros(nstate)

    zf_mu, _ = augmented_forecast_model(z_mu, P_aug, 0.5, 0.5, Q_aug, nbasis, nbc, nxout, nr, kappa_r=0.0, rprior=0.0)

    np.testing.assert_allclose(zf_mu[nbasis:nbasis + nbc], z_mu[nbasis:nbasis + nbc])


def test_analysis_covariance_update_matches_joseph_form():
    nz, ny = 4, 6
    rng = np.random.default_rng(22)
    A = rng.normal(size=(nz, nz))
    Pf = A @ A.T + nz * np.eye(nz)
    H_hat = rng.normal(size=(ny, nz))
    r = rng.uniform(0.5, 2.0, size=ny)
    K = rng.normal(size=(nz, ny)) * 0.1

    Pa = analysis_covariance_update(K, H_hat, Pf, r)

    I_KH = np.eye(nz) - K @ H_hat
    expected = I_KH @ Pf @ I_KH.T + K @ np.diag(r) @ K.T

    np.testing.assert_allclose(Pa, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Pa, Pa.T)


def test_iterative_analysis_update_matches_closed_form_linear_kalman_filter():
    """
    With all-normal priors, state_vector_mu_transform/build_Wb become
    identity operations, so the MXKF analysis step should reduce exactly to
    a standard linear Kalman filter update.
    """
    nz, ny = 4, 15
    rng = np.random.default_rng(23)

    A = rng.normal(size=(nz, nz))
    Pf = A @ A.T + nz * np.eye(nz)
    zf_mu = rng.normal(size=nz)
    H = rng.normal(size=(ny, nz))
    r = rng.uniform(0.5, 2.0, size=ny)
    Y = H @ zf_mu + rng.normal(scale=np.sqrt(r))

    xprior = {"pdf": "normal"}
    rprior = {"pdf": "normal"}
    nbasis, nbc, nr = 2, 0, 2

    za_mu, Pa, converged = iterative_analysis_update(
        zf_mu, za_mu_current=zf_mu.copy(), Y=Y, r=r, r_inv=1.0 / r, Pf=Pf, H=H,
        xprior=xprior, bcprior=None, rprior=rprior, nbasis=nbasis, nbc=nbc, nr=nr,
    )

    K_naive = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + np.diag(r))
    za_naive = zf_mu + K_naive @ (Y - H @ zf_mu)
    Pa_naive = (np.eye(nz) - K_naive @ H) @ Pf

    assert converged
    np.testing.assert_allclose(za_mu, za_naive, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(Pa, Pa_naive, rtol=1e-4, atol=1e-6)


def _make_amxkf_inputs_no_bc(nperiod=3, nbasis=3, nxout=1, nr=2, ny_per_period=8, seed=31):
    rng = np.random.default_rng(seed)
    nz = nbasis + nr  # no boundary conditions

    xprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}
    rprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}

    Y_dic, sigma_obs_dic, Ytime_dic, Hz_dic, siteindicator_dic = {}, {}, {}, {}, {}
    for t in range(nperiod):
        Hx = rng.uniform(0.1, 1.0, size=(ny_per_period, nbasis))
        Hr = np.zeros((ny_per_period, nr))
        Hz_dic[t] = np.hstack([Hx, Hr])
        true_x = rng.uniform(0.8, 1.2, size=nbasis)
        Y_dic[t] = Hx @ true_x + rng.normal(scale=0.1, size=ny_per_period)
        sigma_obs_dic[t] = np.full(ny_per_period, 0.1)
        Ytime_dic[t] = np.arange(ny_per_period)
        siteindicator_dic[t] = np.zeros(ny_per_period)

    return amxkf_inputs(
        Y_dic=Y_dic, sigma_obs_dic=sigma_obs_dic, Ytime_dic=Ytime_dic, Hz_dic=Hz_dic,
        siteindicator_dic=siteindicator_dic, nbasis=nbasis, zprior_mus=np.zeros(nz),
        zprior_sigma2s=np.full(nz, 1.0), forecast_noise=np.full(nz, 0.01), nperiod=nperiod,
        sigma2_rep=0.01, kappa_x_vector=np.array([0.5, 0.5]), xprior=xprior, bcprior=None,
        rprior=rprior, nbc=None, nr=nr, nxout=nxout,
    )


def test_filter_and_backward_sampler_run_end_to_end_with_zero_boundary_conditions():
    """Sanity check that the filter + RTS-style backward sampler produce finite output."""
    inputs = _make_amxkf_inputs_no_bc()
    inputs.nbc = 0  # int zero, not None -- see test_integration_gibbs.py for the nbc=None bug
    filtered = iterative_augmented_mxkf(inputs)

    z_mu, z, residuals = augmented_backward_sampler(filtered, rng=np.random.default_rng(99))

    assert z_mu.shape == (inputs.nperiod, inputs.nbasis + inputs.nr)
    assert np.all(np.isfinite(z_mu))
    assert np.all(np.isfinite(z))
    assert residuals.shape[0] == inputs.nperiod * 8
