import re
from pathlib import Path
import warnings

import numpy as np
from dataclasses import dataclass
from scipy.linalg import cho_solve, solve_triangular

from inversion_methods.manipulation.matrix_identities import kalman_gain_woodbury_from_gram, covariance_from_woodbury_factor, precision_from_cholesky, kalman_gain_information_from_gram, covariance_from_information_factor, _symmetrize_square, _cholesky_with_repair, _sample_gaussian_cholesky
from inversion_methods.manipulation.lognormal_transformations import state_vector_mu_transform, build_Wb


@dataclass
class amxkf_inputs:
    Y_dic: dict
    sigma_obs_dic: dict
    Ytime_dic: dict
    Hz_dic: dict
    siteindicator_dic: dict
    nbasis: int
    zprior_mus: np.ndarray
    zprior_sigma2s : np.ndarray
    forecast_noise: np.ndarray
    nperiod: int
    sigma2_exc: float
    kappa_x_vector: np.ndarray
    xprior: dict
    bcprior: dict
    rprior: dict
    nxout: int
    nr: int
    iteration: int
    kappa_r: float = 0.0
    nbc: int | None = None
    za_mu_warmstart: np.ndarray | None = None
    err_var_dic: dict | None = None
    Y_dic_raw: dict | None = None
    Hz_dic_raw: dict | None = None
    verbose: bool = True
    analysis_form: str = "information"


@dataclass
class abs_inputs:
    zf_mu: np.ndarray
    # Pf: np.ndarray
    Pf_chol: np.ndarray
    za_mu: np.ndarray
    Pa: np.ndarray
    nperiod: int
    Y_dic: dict
    Hz_dic: dict
    nbasis: int
    nxout: int
    nbc: int
    xprior: dict
    bcprior: dict
    rprior: dict
    nr: int
    kappa_xin: float
    kappa_xout: float | None
    kappa_r: float = 0.0



def augmented_forecast_model(
    z_mu,
    P_aug,
    kappa_out,
    kappa_in,
    Q_aug,
    nbasis,
    nbc,
    nxout,
    nr,
    kappa_r=0.0,
    rprior=0.0,
):
    """
    Vectorised forecast model for the augmented MXKF.

    The augmented state is ordered as:

        z = [x_out, x_in, x_bc, r_out, r_in]

    where:

        x_out : outer basis coefficients, length nout
        x_in  : inner basis coefficients, length nbasis - nout
        x_bc  : optional boundary-condition states, length nbc
        r_out : scalar reference state driving x_out
        r_in  : scalar reference state driving x_in

    The basis coefficients relax toward their respective reference states:

        x_out(t + 1) = kappa_out * x_out(t) + (1 - kappa_out) * r_out(t)

        x_in(t + 1) = kappa_in * x_in(t) + (1 - kappa_in) * r_in(t)

    The reference states relax toward rprior:

        r_out(t + 1) = kappa_r * r_out(t) + (1 - kappa_r) * rprior

        r_in(t + 1) = kappa_r * r_in(t) + (1 - kappa_r) * rprior

    This is an affine Gaussian forecast model:

        zf_mu = F_aug @ z_mu + b
        Pf_aug = F_aug @ P_aug @ F_aug.T + Q_aug

    where b contains the deterministic prior contributions for r_out
    and r_in.

    Parameters
    ----------
    z_mu : ndarray, shape (nstate,)
        Current augmented-state mean.

    P_aug : ndarray, shape (nstate, nstate)
        Current augmented-state covariance matrix.

    kappa_out : float
        Persistence parameter for the outer basis coefficients.

        Values between zero and one cause the outer coefficients to
        relax toward r_out.

    kappa_in : float
        Persistence parameter for the inner basis coefficients.

        Values between zero and one cause the inner coefficients to
        relax toward r_in.

    Q_aug : ndarray
        Forecast-noise covariance. This can be either:

        - a full matrix with shape (nstate, nstate), or
        - a vector with shape (nstate,), interpreted as its diagonal.

    nbasis : int
        Total number of basis coefficients.

    nbc : int or None, default=None
        Number of boundary-condition states.

    nxout : int, default=6
        Number of outer basis coefficients.

    kappa_r : float, default=0.0
        Persistence parameter for both scalar reference states.

        Typical interpretations are:

        - kappa_r = 0:
          reference states move immediately to rprior;

        - 0 < kappa_r < 1:
          reference states relax exponentially toward rprior;

        - kappa_r = 1:
          reference states are persistent and rprior has no effect.

    rprior : float, default=0.0
        Prior or equilibrium value toward which both r_out and r_in
        relax.

        This is treated as a fixed deterministic input, not as an
        uncertain component of the augmented state.

    Returns
    -------
    zf_mu : ndarray, shape (nstate,)
        Forecast augmented-state mean.

    Pf_aug : ndarray, shape (nstate, nstate)
        Forecast augmented-state covariance.
    """
    # ------------------------------------------------------------------
    # Dimensions and input validation
    # ------------------------------------------------------------------
    nbc = 0 if nbc is None else nbc
    nxin = nbasis - nxout
    nstate = nbasis + nbc + nr

    z_mu = np.asarray(z_mu)
    P_aug = np.asarray(P_aug)

    # Use a floating-point-compatible dtype for the calculations.
    dtype = np.result_type(
        z_mu.dtype,
        P_aug.dtype,
        kappa_out,
        kappa_in,
        kappa_r,
        rprior,
        np.float64,
    )

    z_mu = z_mu.astype(dtype, copy=False)
    P_aug = P_aug.astype(dtype, copy=False)

    # ------------------------------------------------------------------
    # State indices
    # ------------------------------------------------------------------
    iout = slice(0, nxout)
    iin = slice(nxout, nbasis)
    ibc = slice(nbasis, nbasis + nbc)
    ir = slice(nbasis + nbc, nstate)
    irout = slice(nbasis + nbc, nstate - 1)
    irin = slice(-1, nstate)
    # rout = nbasis + nbc

    alpha_out = 1.0 - kappa_out
    alpha_in = 1.0 - kappa_in
    alpha_r = 1.0 - kappa_r

    # ------------------------------------------------------------------
    # Mean forecast
    #
    #     zf_mu = F_aug @ z_mu + b
    # ------------------------------------------------------------------
    zf_mu = np.empty(nstate, dtype=dtype)

    # Outer basis coefficients relax toward the current outer
    # reference state.
    zf_mu[iout] = (
        kappa_out * z_mu[iout]
        + alpha_out * z_mu[irout]
    )

    # Inner basis coefficients relax toward the current inner
    # reference state.
    zf_mu[iin] = (
        kappa_in * z_mu[iin]
        + alpha_in * z_mu[irin]
    )

    # Boundary-condition states are persistent.
    if nbc:
        zf_mu[ibc] = z_mu[ibc]

    # Reference states relax toward the fixed prior value.
    zf_mu[irout] = (
        kappa_r * z_mu[irout]
        + alpha_r * rprior
    )

    zf_mu[irin] = (
        kappa_r * z_mu[irin]
        + alpha_r * rprior
    )

    # ------------------------------------------------------------------
    # Covariance forecast
    #
    #     Pf_aug = F_aug @ P_aug @ F_aug.T
    #
    # The prior contribution:
    #
    #     (1 - kappa_r) * rprior
    #
    # is deterministic. It shifts the forecast mean but does not add
    # uncertainty, so it does not appear in the covariance calculation.
    #
    # Apply F_aug to the rows first and then apply F_aug.T to the
    # columns. This avoids constructing the full Jacobian.
    # ------------------------------------------------------------------

    # Left multiplication:
    #
    #     FP = F_aug @ P_aug
    #
    FP = P_aug.copy()

    # Outer coefficient rows.
    FP[iout, :] = (kappa_out * P_aug[iout, :] + alpha_out * P_aug[irout, :])

    # Inner coefficient rows.
    FP[iin, :] = (kappa_in * P_aug[iin, :] + alpha_in * P_aug[irin, :])

    # Boundary-condition rows remain unchanged because those states
    # use an identity transition.

    # Reference-state rows are scaled by kappa_r.
    FP[ir, :] = kappa_r * P_aug[ir, :]

    # Right multiplication:
    #
    #     Pf_aug = FP @ F_aug.T
    #
    Pf_aug = FP.copy()

    # Outer coefficient columns.
    Pf_aug[:, iout] = (kappa_out * FP[:, iout] + alpha_out * FP[:, irout])

    # Inner coefficient columns.
    Pf_aug[:, iin] = (kappa_in * FP[:, iin] + alpha_in * FP[:, irin])

    # Boundary-condition columns remain unchanged.

    # Reference-state columns are scaled by kappa_r.
    Pf_aug[:, ir] = kappa_r * FP[:, ir]
    # Pf_aug[:, irin] = kappa_r * FP[:, irin]

    # ------------------------------------------------------------------
    # Add forecast-noise covariance
    # ------------------------------------------------------------------
    Q_aug = np.asarray(Q_aug, dtype=dtype)

    if Q_aug.ndim == 1:
        if Q_aug.shape != (nstate,):
            raise ValueError(f"Diagonal Q_aug must have shape ({nstate},), but has shape {Q_aug.shape}.")

        Pf_aug[np.diag_indices(nstate)] += Q_aug

    elif Q_aug.ndim == 2:
        if Q_aug.shape != (nstate, nstate):
            raise ValueError(f"Full Q_aug must have shape ({nstate}, {nstate}), but has shape {Q_aug.shape}.")

        Pf_aug += Q_aug

    else:
        raise ValueError("Q_aug must be either a one-dimensional diagonal vector or a two-dimensional covariance matrix.")

    # Remove small floating-point asymmetries.
    Pf_aug = 0.5 * (Pf_aug + Pf_aug.T)

    return zf_mu, Pf_aug


def _analysis_cost_change(u_new, u_old, zf_mu, Pf_chol, G, g, to_natural_space):
    """
    Change in the analysis cost function, J(u_new) - J(u_old).

    The analysis mean is the optimisation-space state u that minimises

        J(u) = (u - zf)' Pf^-1 (u - zf)  +  (Y - H z)' R^-1 (Y - H z),    z = z(u)

    i.e. the departure from the forecast plus the misfit to the observations,
    each weighted by its own error covariance. Both terms can be evaluated
    from nz-sized quantities alone, without the (ny, nz) observation operator:

    - Forecast term: with Pf = L L', (u - zf)' Pf^-1 (u - zf) = |L^-1 (u - zf)|^2.

    - Observation term: expanding it with G = H' R^-1 H and g = H' R^-1 Y
      gives Y' R^-1 Y - 2 g' z + z' G z. The first part is the same for every
      z, so the change in this term between two states is

          (z_new - z_old)' (G (z_new + z_old) - 2 g).

      Computing the change directly, rather than J at each point and then
      subtracting, avoids losing precision against the large Y' R^-1 Y when
      the two states are close together.

    Parameters
    ----------
    u_new, u_old : numpy.ndarray
        The two states, in optimisation space.
    zf_mu : numpy.ndarray
        Forecast mean, in optimisation space.
    Pf_chol : numpy.ndarray
        Lower Cholesky factor of the forecast covariance Pf.
    G, g : numpy.ndarray
        ``H' R^-1 H`` and ``H' R^-1 Y`` for this period.
    to_natural_space : callable
        Maps an optimisation-space state u to the natural-space state z(u).
    """
    forecast_term_new = np.sum(solve_triangular(Pf_chol, u_new - zf_mu, lower=True) ** 2)
    forecast_term_old = np.sum(solve_triangular(Pf_chol, u_old - zf_mu, lower=True) ** 2)

    z_new = to_natural_space(u_new)
    z_old = to_natural_space(u_old)
    observation_term_change = (z_new - z_old) @ (G @ (z_new + z_old) - 2.0 * g)

    return (forecast_term_new - forecast_term_old) + observation_term_change


def _anderson_step(za_mu, gn_step, previous_za_mu, previous_gn_step):
    """
    First-order Anderson acceleration of the Gauss-Newton iteration.

    Each Gauss-Newton iteration is a fixed-point map, u -> u + gn_step(u).
    Combining the last two iterates extrapolates towards the fixed point,
    which usually needs fewer iterations than plain Gauss-Newton.

    Returns the proposed step, or None if the extrapolation is ill-conditioned
    (the two Gauss-Newton steps are almost the same) or more than 10 times
    longer than the plain Gauss-Newton step. The caller still has to check
    that the step lowers the analysis cost before using it.
    """
    change_in_step = gn_step - previous_gn_step
    denominator = change_in_step @ change_in_step

    # Relative guard: when both steps are tiny the denominator is mostly
    # rounding error.
    if denominator <= 1e-14 * max(gn_step @ gn_step, 1e-30):
        return None

    gamma = (gn_step @ change_in_step) / denominator
    step = gn_step - gamma * ((za_mu - previous_za_mu) + change_in_step)

    # Trust region: reject extrapolations much longer than the Gauss-Newton step.
    if np.linalg.norm(step) > 10 * np.linalg.norm(gn_step):
        return None

    return step


def _backtracking_line_search(za_mu, gn_step, cost_change, max_tries=10):
    """
    Return the longest of gn_step, gn_step / 2, gn_step / 4, ... (max_tries
    candidates, down to gn_step / 512 by default) that does not increase the
    analysis cost, or None if none of them does.

    The Gauss-Newton step points downhill on the analysis cost, so a short
    enough step always lowers it unless za_mu is already at the minimum.

    cost_change(u_new, u_old) must return J(u_new) - J(u_old).
    """
    step = gn_step

    for _ in range(max_tries):
        if cost_change(za_mu + step, za_mu) <= 0:
            return step

        step = 0.5 * step

    return None


def iterative_analysis_update(zf_mu, za_mu_current, G, g, Pf, xprior, bcprior, rprior, nbasis, nbc, nr, max_inner_iters=5, tol=1e-02, analysis_form="information"):

    """
    Analysis update for one filter time step, iterated with Gauss-Newton to
    allow for the non-linear (lognormal) state transform.

    The analysis mean is the minimiser of the analysis cost function

        J(u) = (u - zf)' Pf^-1 (u - zf)  +  (Y - H z(u))' R^-1 (Y - H z(u))

    where u is the state in optimisation space, zf the forecast mean, and
    z(u) the natural-space state from state_vector_mu_transform. Lognormal
    blocks make z(u) non-linear, so J is minimised iteratively. Each
    Gauss-Newton iteration linearises z(u) about the current estimate and
    solves the resulting linear-Gaussian problem exactly (an ordinary Kalman
    update). That solution is the "Gauss-Newton target", and the move from
    the current estimate to it is the "Gauss-Newton step".

    Each iteration:

    1. Compute the Gauss-Newton target from the current estimate.
    2. If the Gauss-Newton step is smaller than ``tol`` in every component,
       the estimate is a fixed point of the iteration, i.e. the minimum of J:
       move to the target and stop, converged.
    3. Otherwise, move to a new estimate, but only if it lowers J. From the
       second iteration onwards a first-order Anderson-accelerated step is
       tried first. If that is unavailable or does not lower J, the
       Gauss-Newton step is used, halved up to 9 times until it lowers J.
       If nothing lowers J, stop without converging.

    Accepting steps on J, rather than on the observation misfit alone, makes
    the result independent of the starting estimate: a warm start that fits
    the observations better than the minimum of J still moves to it.

    For all-normal priors z(u) is linear, so the first Gauss-Newton target is
    already the exact Kalman analysis.

    Parameters
    ----------
    zf_mu : numpy.ndarray
        Forecast mean in optimisation space.
    za_mu_current : numpy.ndarray
        Starting estimate of the analysis mean in optimisation space, e.g. a
        warm start from the previous Gibbs iteration.
    G : numpy.ndarray
        State-space Gram matrix ``H.T @ diag(r_inv) @ H``, precomputed by the
        caller (see iterative_augmented_mxkf) since it depends only on the
        fixed H/r_inv for this period, not on Pf or the estimate.
    g : numpy.ndarray
        State-space vector ``H.T @ diag(r_inv) @ Y``, precomputed alongside G
        for the same reason.
    Pf : numpy.ndarray
        Forecast-error covariance matrix.
    xprior, bcprior, rprior : dict
        Prior information used by the state transformation and its
        linearisation.
    nbasis : int
        Number of basis coefficients.
    nbc : int
        Number of boundary-condition components.
    nr : int
        Number of relaxation terms. Currently either 1 or 2.
    max_inner_iters : int, optional
        Maximum number of Gauss-Newton iterations. Default is 5.
    tol : float, optional
        Convergence tolerance on the largest component of the Gauss-Newton
        step. Default is ``1e-2``.
    analysis_form : {"information", "woodbury"}, optional
        How the Kalman gain and analysis covariance are computed. Both are
        algebraically identical. "information" (the default) factorises
        the posterior precision ``Pf^-1 + diag(Wb) @ G @ diag(Wb)``
        directly and is usually faster; "woodbury" uses the Woodbury
        system ``I + L.T @ diag(Wb) @ G @ diag(Wb) @ L`` and is the more
        robust choice if Pf becomes badly conditioned.

    Returns
    -------
    za_mu : numpy.ndarray
        Analysis mean in optimisation space.
    Pa : numpy.ndarray
        Analysis-error covariance, linearised at the returned mean.
    Pf_chol : numpy.ndarray
        Lower Cholesky factor of Pf (reused by the backward sampler).
    converged : bool
        True if the Gauss-Newton step fell below ``tol`` within
        ``max_inner_iters`` iterations.
    """

    Pf_chol, _, _ = _cholesky_with_repair(Pf)

    if analysis_form == "information":
        # Pf is fixed across the inner loop, so its inverse is formed once
        # here and reused by every gain evaluation below.
        Pf_inv = precision_from_cholesky(Pf_chol)
        kalman_gain = lambda Wb, h: kalman_gain_information_from_gram(Pf_inv, G, Wb, h)
        analysis_covariance = covariance_from_information_factor

    elif analysis_form == "woodbury":
        kalman_gain = lambda Wb, h: kalman_gain_woodbury_from_gram(Pf_chol, G, Wb, h)
        analysis_covariance = lambda factor: covariance_from_woodbury_factor(Pf_chol, factor)

    else:
        raise ValueError(f"Unknown analysis_form: {analysis_form!r}. Expected 'information' or 'woodbury'.")

    def to_natural_space(u):
        return state_vector_mu_transform(u, xprior, bcprior, rprior, nbasis, nbc, nr)

    def gauss_newton_target(u):
        """
        Minimiser of J with z linearised about u, z(v) ~ z(u) + diag(Wb) (v - u).
        Also returns the factorisation from the gain calculation, from which
        the analysis covariance at u is formed.
        """
        z_lin = to_natural_space(u)
        Wb = build_Wb(z_lin, xprior, bcprior, rprior, nbasis, nbc)

        # H.T @ diag(r_inv) @ d for the linearised innovation
        #   d = Y - H @ z_lin - H @ diag(Wb) @ (zf_mu - u),
        # written with the precomputed G and g so that the (ny, nz)
        # observation operator is never needed here.
        h = g - G @ (z_lin + Wb * (zf_mu - u))

        Kd, gain_factor = kalman_gain(Wb, h)

        return zf_mu + Kd, gain_factor

    def cost_change(u_new, u_old):
        return _analysis_cost_change(u_new, u_old, zf_mu, Pf_chol, G, g, to_natural_space)

    za_mu = za_mu_current
    converged = False

    # The previous estimate and its Gauss-Newton step, for Anderson acceleration.
    previous_za_mu = None
    previous_gn_step = None

    for _ in range(max_inner_iters):

        gn_target, _ = gauss_newton_target(za_mu)
        gn_step = gn_target - za_mu

        if np.max(np.abs(gn_step)) < tol:
            za_mu = gn_target
            converged = True
            break

        # Try the Anderson-accelerated step first, keeping it only if it
        # lowers J; otherwise fall back to a backtracked Gauss-Newton step.
        step = None

        if previous_gn_step is not None:
            step = _anderson_step(za_mu, gn_step, previous_za_mu, previous_gn_step)

            if step is not None and cost_change(za_mu + step, za_mu) > 0:
                step = None

        if step is None:
            step = _backtracking_line_search(za_mu, gn_step, cost_change)

        if step is None:
            # No step lowers J. Stop here rather than move to a worse
            # estimate; converged stays False.
            break

        previous_za_mu, previous_gn_step = za_mu, gn_step
        za_mu = za_mu + step

    # Analysis covariance, from the linearisation at the returned mean.
    _, gain_factor = gauss_newton_target(za_mu)
    Pa = analysis_covariance(gain_factor)

    return za_mu, Pa, Pf_chol, converged


def iterative_augmented_mxkf(config: amxkf_inputs) -> abs_inputs:

    nx = config.nbasis

    kappa_xin = config.kappa_x_vector[-1]

    if config.nxout > 0 and len(config.kappa_x_vector) == 2:

        kappa_xout = config.kappa_x_vector[0]

    elif config.nxout > 0 and len(config.kappa_x_vector) == 1:
        kappa_xout = kappa_xin

    else:

        kappa_xout = -np.inf
    
    if config.nbc:
        nz = nx + config.nbc + config.nr
    else:
        nz = nx + config.nr

    zf_mu = np.zeros((config.nperiod, nz))
    Pf = np.zeros((config.nperiod, nz, nz))
    Pf_chol = np.zeros((config.nperiod, nz, nz))
    za_mu = np.zeros((config.nperiod, nz))
    Pa = np.zeros((config.nperiod, nz, nz))

    # ------------------------------------------------------------------
    # Precompute the per-period state-space Gram reduction, G_t = H_t.T @
    # diag(r_inv_t) @ H_t and g_t = H_t.T @ diag(r_inv_t) @ Y_t, before the
    # sequential filter recursion below.
    #
    # This is the dominant cost of the forward filter (O(ny_t * nz**2) per
    # period), but unlike zf_mu/Pf/za_mu/Pa it does not depend on the
    # recursion at all -- only on the fixed H_t/Y_t and this iteration's
    # (shared, scalar, unless a period-varying err_var_dic is supplied)
    # r_inv_t. Computing every period's G_t/g_t here, in one pass with no
    # cross-period dependency, rather than recomputing it once per period
    # interleaved inside the recursive loop, is a pure reordering of the
    # same arithmetic -- it changes nothing about the numbers produced.
    # ------------------------------------------------------------------
    G_list = [None] * config.nperiod
    g_list = [None] * config.nperiod

    for t in range(config.nperiod):
        H_t = config.Hz_dic[t]
        Y_t = config.Y_dic[t]

        if config.err_var_dic is not None:
            # Precomputed, e.g. by an AR(1)/OU residual-correlation whitening
            # step. Falls back to the plain i.i.d. formula below when absent,
            # so callers that don't use that feature are unaffected.
            err_var_inv = 1 / config.err_var_dic[t]
        else:
            sigma_obs = config.sigma_obs_dic[t]
            err_var_inv = 1 / (config.sigma2_exc + sigma_obs**2)

        HTrinv = H_t.T * err_var_inv[None, :]
        G_list[t] = HTrinv @ H_t
        g_list[t] = HTrinv @ Y_t

    za_mu_current = zf_mu.copy() if config.za_mu_warmstart is None else config.za_mu_warmstart.copy()
    converge_count = 0

    for t in range(config.nperiod):

        if t == 0:
            zf_mu[t] = config.zprior_mus
            Pf[t] = np.diag(config.zprior_sigma2s)
        else:
            zf_mu[t], Pf[t] = augmented_forecast_model(za_mu[t-1], Pa[t-1], kappa_xout, kappa_xin, config.forecast_noise, config.nbasis, config.nbc, config.nxout,  config.nr, config.kappa_r, config.rprior["mu"])
            # zf_mu[t], Pf[t] = slow_augmented_forecast_model(za_mu[t-1], Pa[t-1], config.F_aug, Q_aug)


        za_mu[t], Pa[t], Pf_chol[t], converged = iterative_analysis_update(zf_mu[t], za_mu_current[t], G_list[t], g_list[t], Pf[t], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc, config.nr, analysis_form=config.analysis_form)

        if converged:
            converge_count += 1
                

    if config.verbose and config.iteration % 50 == 0:
        print(f"            {converge_count} convergences in {config.nperiod} MXKF time steps")

    # The backward sampler computes residuals (Y_dic - Hz_dic @ z) purely as a
    # diagnostic fed to the sigma2_exc/sigma2_qx Gibbs updates. Those updates
    # assume the residual's marginal variance is sigma2_exc + sigma_obs**2, so
    # they need the *raw* (unwhitened) Y/Hz, not whatever was used internally
    # above to drive the Kalman gain. Defaults to the same dict when no
    # whitening is in use, i.e. unchanged behaviour.
    Y_dic_for_residuals = config.Y_dic_raw if config.Y_dic_raw is not None else config.Y_dic
    Hz_dic_for_residuals = config.Hz_dic_raw if config.Hz_dic_raw is not None else config.Hz_dic

    return abs_inputs(zf_mu=zf_mu,
                        #  Pf=Pf,
                         Pf_chol=Pf_chol,
                         za_mu=za_mu,
                         Pa=Pa,
                         kappa_xout=kappa_xout,
                         kappa_xin=kappa_xin,
                         nperiod=config.nperiod,
                         Y_dic=Y_dic_for_residuals,
                         Hz_dic=Hz_dic_for_residuals,
                         nbasis=config.nbasis,
                         nbc=config.nbc,
                         nxout=config.nxout,
                         xprior=config.xprior,
                         bcprior=config.bcprior,
                         rprior=config.rprior,
                         nr=config.nr,
                         kappa_r=config.kappa_r,
                         )


def augmented_backward_sampler(
    config: abs_inputs,
    rng=None,
):
    """
    Backward simulation sampler for an augmented MXKF state.

    Supported state layouts
    -----------------------
    When config.nr == 2:

        z_mu = [x_out, x_in, bc, r_out, r_in]

    When config.nr == 1:

        z_mu = [x_in, bc, r_in]

    In the one-reference-state case, config.nxout must be zero because
    there is no r_out state to drive outer basis coefficients.

    The basis-coefficient transitions are:

        x_out(t + 1) = kappa_out * x_out(t) + (1 - kappa_out) * r_out(t)

        x_in(t + 1) = kappa_in * x_in(t) + (1 - kappa_in) * r_in(t)

    The reference-state transitions are:

        r(t + 1) = kappa_r * r(t) + (1 - kappa_r) * rprior

    The additive rprior term affects the forecast mean but not the
    transition Jacobian used by the backward sampler. Therefore, the
    reference-state block of the transition Jacobian is:

        kappa_r * I

    Boundary-condition states are persistent:

        bc(t + 1) = bc(t)

    The structured transition is used to calculate:

        M1 = Pa[t] @ F.T

    without explicitly constructing the full transition matrix.

    Expected config attributes
    --------------------------
    nperiod
    nbasis
    nbc
    nxout
    nr
    kappa_xout
    kappa_xin
    za_mu
    zf_mu
    Pa
    Pf
    Y_dic
    Hz_dic
    xprior
    bcprior
    rprior

    Optional config attributes
    --------------------------
    kappa_r

        Reference-state persistence parameter. If absent, it defaults
        to zero.

    Parameters
    ----------
    config : abs_inputs
        Filter outputs, transition parameters, and transformation
        information.

    rng : numpy.random.Generator or None, default=None
        Random-number generator. If None, a new generator is created.

    Returns
    -------
    z_mu : ndarray
        Sampled states in transformed or Gaussian space.

    z : ndarray
        States after applying state_vector_mu_transform.

    state_residuals : ndarray
        Residuals concatenated in chronological order.
    """
    if rng is None:
        rng = np.random.default_rng()

    # -------------------------------------------------------------
    # State dimensions and validation
    # -------------------------------------------------------------
    nperiod = int(config.nperiod)
    nbasis = int(config.nbasis)
    nxout = int(config.nxout)
    nr = int(config.nr)

    nbc = 0 if config.nbc is None else int(config.nbc)


    nxin = nbasis - nxout
    nstate = nbasis + nbc + nr

    # -------------------------------------------------------------
    # State slices
    # -------------------------------------------------------------
    ix_out = slice(0, nxout)
    ix_in = slice(nxout, nbasis)

    #  ibc not necessary as it behaves as a random walk
    # ibc = slice(nbasis, nbasis + nbc)

    r_start = nbasis + nbc
    ir = slice(r_start, r_start + nr)

    if nr == 1:
        # State layout:
        #
        #     [x_in, bc, r_in]
        #
        ir_out = None
        ir_in = r_start

    else:
        # State layout:
        #
        #     [x_out, x_in, bc, r_out, r_in]
        #
        ir_out = r_start
        ir_in = r_start + 1

    # -------------------------------------------------------------
    # Cache transition parameters
    # -------------------------------------------------------------
    kappa_out = float(config.kappa_xout)
    kappa_in = float(config.kappa_xin)

    # Preserve the previous kappa_r=0 behaviour if config.kappa_r
    # has not been supplied.
    kappa_r = float(config.kappa_r)

    one_minus_kappa_out = 1.0 - kappa_out
    one_minus_kappa_in = 1.0 - kappa_in

    # -------------------------------------------------------------
    # Cache state and covariance arrays
    # -------------------------------------------------------------
    Pa = np.asarray(config.Pa)
    Pf_chol = np.asarray(config.Pf_chol)
    za_mu = np.asarray(config.za_mu)
    zf_mu = np.asarray(config.zf_mu)

    Y_dic = config.Y_dic
    Hz_dic = config.Hz_dic

    # -------------------------------------------------------------
    # Allocate outputs
    # -------------------------------------------------------------
    z_mu = np.empty_like(za_mu)
    z = np.empty_like(z_mu)

    # Store residuals separately and concatenate once at the end.
    residuals = [None] * nperiod

    # -------------------------------------------------------------
    # Sample final state
    # -------------------------------------------------------------
    z_mu[-1] = _sample_gaussian_cholesky(mean=za_mu[-1], covariance=Pa[-1], rng=rng,)

    z[-1] = state_vector_mu_transform(z_mu[-1], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc, config.nr)

    residuals[-1] = (Y_dic[nperiod - 1] - Hz_dic[nperiod - 1] @ z[-1])

    # -------------------------------------------------------------
    # Backward simulation
    # -------------------------------------------------------------
    for t in range(nperiod - 2, -1, -1):
        Pa_t = Pa[t]
        Pf_factor = (Pf_chol[t + 1], True)

        # ---------------------------------------------------------
        # Structured calculation:
        #
        #     M1 = Pa_t @ F.T
        #
        # Starting from Pa_t copies the persistent BC columns.
        # The basis and reference columns are then overwritten with
        # their corresponding structured transition calculations.
        # ---------------------------------------------------------
        M1 = Pa_t.copy()

        # Outer basis coefficients exist only when nxout > 0.
        #
        # The validation above guarantees that r_out also exists in
        # this case.
        if nxout > 0:
            np.multiply(Pa_t[:, ix_out], kappa_out, out=M1[:, ix_out],)

            M1[:, ix_out] += one_minus_kappa_out* Pa_t[:, ir_out, None]

        np.multiply(Pa_t[:, ix_in], kappa_in,out=M1[:, ix_in],)

        M1[:, ix_in] += one_minus_kappa_in * Pa_t[:, ir_in, None]

        # Every reference-state column is scaled by kappa_r.
        #
        # This handles both:
        #
        #     nr = 1: [r_in]
        #     nr = 2: [r_out, r_in]
        np.multiply(Pa_t[:, ir], kappa_r, out=M1[:, ir],)

        # ---------------------------------------------------------
        # RTS backward gain
        #
        #     A = M1 @ inv(Pf_tp1)
        #
        # Because Pf_tp1 is symmetric:
        #
        #     A.T = solve(Pf_tp1, M1.T)
        # ---------------------------------------------------------
        # Pf_L, _, _ = _cholesky_with_repair(Pf_tp1)
        # Pf_factor = (Pf_L, True)

        A_transpose = cho_solve(Pf_factor, M1.T, overwrite_b=False, check_finite=False,)

        # ---------------------------------------------------------
        # Conditional mean
        # ---------------------------------------------------------
        state_difference = z_mu[t + 1] - zf_mu[t + 1]

        mu_t = za_mu[t] + A_transpose.T @ state_difference

        # ---------------------------------------------------------
        # Conditional covariance
        #
        #     cov_t = Pa_t - A @ Pf_tp1 @ A.T
        #
        # Using M1 = A @ Pf_tp1 gives:
        #
        #     cov_t = Pa_t - M1 @ A.T
        # ---------------------------------------------------------
        cov_t = Pa_t - M1 @ A_transpose

        # Remove floating-point asymmetry before sampling.
        cov_t = _symmetrize_square(cov_t)

        # ---------------------------------------------------------
        # Draw conditional state
        # ---------------------------------------------------------
        z_mu[t] = _sample_gaussian_cholesky(mean=mu_t, covariance=cov_t, rng=rng,)

        z[t] = state_vector_mu_transform(z_mu[t], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc, config.nr)

        residuals[t] = Y_dic[t] - Hz_dic[t] @ z[t]

    # Preserve chronological ordering.
    state_residuals = np.concatenate(residuals)

    return z_mu, z, state_residuals