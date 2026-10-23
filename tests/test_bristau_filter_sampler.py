"""Unit tests for inversion_methods.bristau.bristau_filter_sampler core math."""

import numpy as np
import pytest
from scipy.linalg import cholesky
from scipy.optimize import minimize

from inversion_methods.bristau.bristau_filter_sampler import (
    augmented_forecast_model,
    iterative_analysis_update,
    iterative_augmented_mxkf,
    augmented_backward_sampler,
    amxkf_inputs,
    _analysis_cost_change,
    _backtracking_line_search,
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

    r_inv = 1.0 / r
    G = H.T @ (r_inv[:, None] * H)
    g = H.T @ (r_inv * Y)

    za_mu, Pa, _L, converged = iterative_analysis_update(
        zf_mu, za_mu_current=zf_mu.copy(), G=G, g=g, Pf=Pf,
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
        sigma2_exc=0.01, kappa_x_vector=np.array([0.5, 0.5]), xprior=xprior, bcprior=None,
        rprior=rprior, nbc=None, nr=nr, nxout=nxout, iteration=0, print_chain_progress=False,
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


def _lognormal_problem_and_observations(seed=41, nbasis=5, nbc=0, nr=2, ny=30):
    """
    Keyword arguments for iterative_analysis_update with all-lognormal
    priors, plus the Y, H and r_inv that G and g were built from.
    """
    rng = np.random.default_rng(seed)
    nz = nbasis + nbc + nr
    A = rng.normal(size=(nz, nz))
    Pf = A @ A.T / nz + 0.05 * np.eye(nz)
    zf_mu = rng.normal(scale=0.3, size=nz)
    H = np.hstack([rng.uniform(0.1, 1.0, size=(ny, nbasis)), np.zeros((ny, nr))])
    r_inv = 1.0 / rng.uniform(0.5, 2.0, size=ny)
    Y = H @ np.exp(rng.normal(scale=0.5, size=nz)) + rng.normal(scale=0.3, size=ny)
    G = H.T @ (r_inv[:, None] * H)
    g = H.T @ (r_inv * Y)
    prior = {"pdf": "lognormal"}
    problem = dict(zf_mu=zf_mu, za_mu_current=zf_mu.copy(), G=G, g=g, Pf=Pf,
                   xprior=prior, bcprior=None, rprior=prior, nbasis=nbasis, nbc=nbc, nr=nr)
    return problem, Y, H, r_inv


def _lognormal_analysis_problem(**kwargs):
    problem, _Y, _H, _r_inv = _lognormal_problem_and_observations(**kwargs)
    return problem


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_iterative_analysis_update_information_and_woodbury_forms_agree(seed):
    """
    The two analysis forms are algebraically identical, so with lognormal
    priors (non-trivial, iteration-dependent Wb) they must follow the same
    Gauss-Newton path and return the same mean, covariance and flag.
    """
    problem = _lognormal_analysis_problem(seed=seed)

    za_info, Pa_info, L_info, conv_info = iterative_analysis_update(**problem, analysis_form="information")
    za_wood, Pa_wood, L_wood, conv_wood = iterative_analysis_update(**problem, analysis_form="woodbury")

    np.testing.assert_allclose(za_info, za_wood, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Pa_info, Pa_wood, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(L_info, L_wood)
    assert conv_info == conv_wood


def test_iterative_analysis_update_defaults_to_information_form():
    problem = _lognormal_analysis_problem()

    default = iterative_analysis_update(**problem)
    explicit = iterative_analysis_update(**problem, analysis_form="information")

    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a, b)


def test_iterative_analysis_update_rejects_unknown_analysis_form():
    with pytest.raises(ValueError, match="analysis_form"):
        iterative_analysis_update(**_lognormal_analysis_problem(), analysis_form="kalman")


# ---------------------------------------------------------------------------
# The Gauss-Newton loop in iterative_analysis_update must find the minimum of
# the analysis cost function
#
#     J(u) = (u - zf)' Pf^-1 (u - zf) + (Y - H z(u))' R^-1 (Y - H z(u))
#
# wherever it starts. An earlier version accepted a step only if it lowered
# the unweighted observation misfit, so a starting estimate that already fitted
# the observations better than the minimum of J (e.g. a warm start from the
# previous Gibbs iteration) barely moved, yet was still reported as converged.
# ---------------------------------------------------------------------------


def _observation_misfit(u, Y, H, r_inv):
    """(Y - H z)' R^-1 (Y - H z) for all-lognormal priors, z = exp(u)."""
    residual = Y - H @ np.exp(u)
    return residual @ (r_inv * residual)


def _analysis_cost(u, zf_mu, Pf, Y, H, r_inv):
    """J(u) for all-lognormal priors, evaluated directly from Y and H."""
    departure = u - zf_mu
    return departure @ np.linalg.solve(Pf, departure) + _observation_misfit(u, Y, H, r_inv)


def _minimise_analysis_cost_directly(zf_mu, Pf, Y, H, r_inv):
    """Reference minimiser of J, from a general-purpose optimiser."""
    def gradient(u):
        z = np.exp(u)
        residual = Y - H @ z
        return 2 * np.linalg.solve(Pf, u - zf_mu) - 2 * z * (H.T @ (r_inv * residual))

    cost = lambda u: _analysis_cost(u, zf_mu, Pf, Y, H, r_inv)

    return minimize(cost, zf_mu, jac=gradient, method="BFGS", options={"gtol": 1e-10}).x


@pytest.mark.parametrize("analysis_form", ["information", "woodbury"])
def test_iterative_analysis_update_reaches_kalman_analysis_from_overfitted_start(analysis_form):
    """
    Linear (all-normal) problem with a tight forecast covariance, started from
    the observation-only least-squares fit. That start fits the observations
    better than the Kalman analysis does, so a step test on the observation
    misfit alone rejects every step towards the right answer.
    """
    rng = np.random.default_rng(1)
    nbasis, nbc, nr, ny = 4, 0, 2, 40
    nz = nbasis + nr

    H = np.hstack([rng.uniform(0.2, 1.0, (ny, nbasis)), np.zeros((ny, nr))])
    r_inv = np.full(ny, 1 / 0.3**2)
    Y = H[:, :nbasis] @ np.array([1.5, 0.7, 1.2, 0.9]) + rng.normal(0, 0.3, ny)
    G = H.T @ (r_inv[:, None] * H)
    g = H.T @ (r_inv * Y)

    zf_mu = np.ones(nz)
    Pf = np.diag(np.r_[np.full(nbasis, 0.05**2), np.full(nr, 0.1)])
    prior = {"pdf": "normal"}

    x_least_squares = np.linalg.lstsq(H[:, :nbasis], Y, rcond=None)[0]
    overfitted_start = np.r_[x_least_squares, zf_mu[nbasis:]]

    za_mu, Pa, _L, converged = iterative_analysis_update(
        zf_mu, za_mu_current=overfitted_start, G=G, g=g, Pf=Pf,
        xprior=prior, bcprior=None, rprior=prior, nbasis=nbasis, nbc=nbc, nr=nr,
        analysis_form=analysis_form,
    )

    Pa_exact = np.linalg.inv(np.linalg.inv(Pf) + G)
    za_exact = zf_mu + Pa_exact @ (g - G @ zf_mu)

    assert converged
    np.testing.assert_allclose(za_mu, za_exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Pa, Pa_exact, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("analysis_form", ["information", "woodbury"])
@pytest.mark.parametrize("start", ["forecast", "overfitted", "far"])
def test_iterative_analysis_update_finds_minimum_of_cost_with_lognormal_priors(start, analysis_form):
    problem, Y, H, r_inv = _lognormal_problem_and_observations(seed=41)
    zf_mu, Pf = problem["zf_mu"], problem["Pf"]

    reference = _minimise_analysis_cost_directly(zf_mu, Pf, Y, H, r_inv)

    if start == "forecast":
        problem["za_mu_current"] = zf_mu.copy()

    elif start == "overfitted":
        # Best fit to the observations alone (bounded to keep exp(u) sensible).
        misfit = lambda u: _observation_misfit(u, Y, H, r_inv)
        problem["za_mu_current"] = minimize(misfit, reference, method="L-BFGS-B", bounds=[(-3, 3)] * len(zf_mu)).x
        assert misfit(problem["za_mu_current"]) < misfit(reference)

    elif start == "far":
        problem["za_mu_current"] = zf_mu + 1.0

    # Tight tolerance and plenty of iterations: this tests where the loop
    # ends up, not how quickly it gets there.
    za_mu, _Pa, _L, converged = iterative_analysis_update(**problem, max_inner_iters=50, tol=1e-6, analysis_form=analysis_form)

    assert converged
    np.testing.assert_allclose(za_mu, reference, atol=1e-5)


def test_iterative_analysis_update_reports_not_converged_when_out_of_iterations():
    problem = _lognormal_analysis_problem(seed=41)
    problem["za_mu_current"] = problem["zf_mu"] + 1.0

    *_, converged = iterative_analysis_update(**problem, max_inner_iters=1)

    assert not converged


def test_analysis_cost_change_matches_direct_evaluation():
    problem, Y, H, r_inv = _lognormal_problem_and_observations(seed=3)
    zf_mu, Pf, G, g = problem["zf_mu"], problem["Pf"], problem["G"], problem["g"]

    rng = np.random.default_rng(4)
    u_old = zf_mu + rng.normal(scale=0.2, size=zf_mu.size)
    u_new = u_old + rng.normal(scale=0.05, size=zf_mu.size)

    # All blocks are lognormal and there are no boundary conditions, so
    # state_vector_mu_transform is exp().
    change = _analysis_cost_change(u_new, u_old, zf_mu, np.linalg.cholesky(Pf), G, g, np.exp)

    expected = _analysis_cost(u_new, zf_mu, Pf, Y, H, r_inv) - _analysis_cost(u_old, zf_mu, Pf, Y, H, r_inv)

    np.testing.assert_allclose(change, expected, rtol=1e-9)


def test_backtracking_line_search_halves_step_until_cost_decreases():
    # The cost only goes down for steps shorter than 0.3 in every component.
    cost_change = lambda u_new, u_old: -1.0 if np.max(np.abs(u_new - u_old)) < 0.3 else 1.0

    step = _backtracking_line_search(np.zeros(3), np.ones(3), cost_change)

    np.testing.assert_allclose(step, np.full(3, 0.25))


def test_backtracking_line_search_returns_none_when_no_step_lowers_cost():
    cost_never_decreases = lambda u_new, u_old: 1.0

    assert _backtracking_line_search(np.zeros(3), np.ones(3), cost_never_decreases) is None
