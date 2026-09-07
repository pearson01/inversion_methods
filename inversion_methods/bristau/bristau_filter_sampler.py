import re
from pathlib import Path
import warnings

import numpy as np
from dataclasses import dataclass
from scipy.linalg import cho_solve

from inversion_methods.manipulation.matrix_identities import kalman_gain_woodbury_from_cholesky, covariance_from_woodbury_factor, _symmetrize_square, _cholesky_with_repair, _sample_gaussian_cholesky
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
    sigma2_rep: float
    kappa_x_vector: np.ndarray
    xprior: dict
    bcprior: dict
    rprior: dict
    nxout: int
    nr: int
    kappa_r: float = 0.0
    nbc: int | None = None
    za_mu_warmstart: np.ndarray | None = None


@dataclass
class abs_inputs:
    zf_mu: np.ndarray
    Pf: np.ndarray
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


def analysis_covariance_update(K, H_hat, Pf, r):
    
    KH = K @ H_hat
    I_KH = np.eye(Pf.shape[0]) - KH

    Pa = I_KH @ Pf @ I_KH.T

    sqrt_r = np.sqrt(r)
    K_sqrt_r = K * sqrt_r[np.newaxis, :]
    Pa += K_sqrt_r @ K_sqrt_r.T

    Pa = 0.5 * (Pa + Pa.T)

    return Pa


def iterative_analysis_update(zf_mu, za_mu_current, Y, r, r_inv, Pf, H, xprior, bcprior, rprior, nbasis, nbc, nr, max_inner_iters=5, tol=1e-02):

    """
    Compute an iterative nonlinear analysis update with Anderson acceleration.

    This function accounts for the tangent linear approximation in the MXKF, requiring that za ~= zf.
    As this isn't true for "unexpected" fluxes, we perform iterative state linearisations until za ~= zf
    is true within a threshold tolerance.

    Each inner iteration linearises the transformed state, constructs an
    effective observation operator, and computes a Kalman-style Gauss-Newton
    proposal. First-order Anderson acceleration is attempted when possible and
    accepted only if it does not increase the observation-space squared-error
    misfit. Otherwise, a backtracked Gauss-Newton step is used.

    Parameters
    ----------
    zf_mu : numpy.ndarray
        Forecast mean in optimisation space.
    za_mu_current : numpy.ndarray
        Initial analysis-mean estimate in optimisation space.
    Y : numpy.ndarray
        Observation vector.
    r : numpy.ndarray
        Observation-error covariance matrix.
    r_inv : numpy.ndarray
        Inverse observation-error covariance matrix.
    Pf : numpy.ndarray
        Forecast-error covariance matrix.
    H : numpy.ndarray
        Observation operator for the transformed state.
    xprior, bcprior, rprior : array-like
        Prior information used by the state transformation and its
        linearisation.
    nbasis : int
        Number of basis coefficients.
    nbc : int
        Number of boundary-condition components.
    nr : int
        Number of relaxation terms. Currently either 1 or 2.
    max_inner_iters : int, optional
        Maximum number of inner iterations. Default is 5.
    tol : float, optional
        Convergence tolerance for the maximum absolute update. Default is
        ``1e-2``.

    Returns
    -------
    za_mu_current : numpy.ndarray
        Updated analysis mean.
    Pa : numpy.ndarray
        Analysis-error covariance based on the final linearisation.
    converged : bool
        Whether the accepted update satisfied the convergence tolerance.
    """


    f_prev, za_prev = None, None

    converged = False

    L, _, _ = _cholesky_with_repair(Pf)

    for _ in range(max_inner_iters):

        z_lin = state_vector_mu_transform(za_mu_current, xprior, bcprior, rprior, nbasis, nbc, nr)
        Wb = build_Wb(z_lin, xprior, bcprior, rprior, nbasis, nbc)
        H_hat = H @ Wb

        d = Y - H @ z_lin - H_hat @ (zf_mu - za_mu_current)
        K, S_factor = kalman_gain_woodbury_from_cholesky(L, H_hat, r_inv)
        za_proposal = zf_mu + K @ d   # this is g(za_current)

        f_curr = za_proposal - za_mu_current   # fixed-point residual

        current_misfit = np.sum((Y - H @ z_lin) ** 2)

        # --- Anderson(1) extrapolation, once we have a previous iterate ---
        za_aa = None
        if f_prev is not None:
            df = f_curr - f_prev
            denom = df @ df
            # relative threshold: guard against denom dominated by roundoff when f_curr, f_prev are tiny
            if denom > 1e-14 * max(f_curr @ f_curr, 1e-30):
                gamma = (f_curr @ df) / denom
                candidate = za_mu_current + f_curr - gamma * ((za_mu_current - za_prev) + df)
                # trust region: reject extrapolations much larger than the raw GN step itself
                if np.linalg.norm(candidate - za_mu_current) <= 10 * np.linalg.norm(f_curr):
                    za_aa = candidate

        f_prev, za_prev = f_curr, za_mu_current

        # --- try the accelerated step first; fall back to plain (backtracked) Newton step ---
        if za_aa is not None:
            z_aa = state_vector_mu_transform(za_aa, xprior, bcprior, rprior, nbasis, nbc, nr)
            aa_misfit = np.sum((Y - H @ z_aa) ** 2)
        else:
            aa_misfit = np.inf

        if aa_misfit <= current_misfit:
            za_new = za_aa
            step = za_new - za_mu_current
        else:
            # fallback: your existing backtracked Gauss-Newton step
            step = f_curr
            alpha = 1.0
            for _ in range(10):
                za_trial = za_mu_current + alpha * step
                z_trial = state_vector_mu_transform(za_trial, xprior, bcprior, rprior, nbasis, nbc, nr)
                trial_misfit = np.sum((Y - H @ z_trial) ** 2)
                if trial_misfit <= current_misfit or alpha < 1e-4:
                    break
                alpha *= 0.5
            za_new = za_trial
            step = za_new - za_mu_current

        if np.max(np.abs(step)) < tol:
            za_mu_current = za_new
            converged = True
            break

        za_mu_current = za_new

    Pa = covariance_from_woodbury_factor(L, S_factor)

    return za_mu_current, Pa, converged


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
    za_mu = np.zeros((config.nperiod, nz))
    Pa = np.zeros((config.nperiod, nz, nz))

    za_mu_current = zf_mu.copy() if config.za_mu_warmstart is None else config.za_mu_warmstart.copy()
    converge_count = 0

    for t in range(config.nperiod):
        H = config.Hz_dic[t]
        Y = config.Y_dic[t]
        sigma_obs = config.sigma_obs_dic[t]
        err_var = config.sigma2_rep + sigma_obs**2
        err_var_inv = 1 / err_var

        if t == 0:
            # za_mu[-1] = config.zprior_mus
            # Pa[-1] = config.zprior_covariance
            zf_mu[t] = config.zprior_mus
            Pf[t] = np.diag(config.zprior_sigma2s)
        else:
            zf_mu[t], Pf[t] = augmented_forecast_model(za_mu[t-1], Pa[t-1], kappa_xout, kappa_xin, config.forecast_noise, config.nbasis, config.nbc, config.nxout,  config.nr, config.kappa_r, config.rprior["mu"])
            # zf_mu[t], Pf[t] = slow_augmented_forecast_model(za_mu[t-1], Pa[t-1], config.F_aug, Q_aug)


        za_mu[t], Pa[t], converged = iterative_analysis_update(zf_mu[t], za_mu_current[t], Y, err_var, err_var_inv, Pf[t], H, config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc, config.nr)
        
        if converged:
            converge_count += 1
                

    print(f"            {converge_count} convergences in {config.nperiod} MXKF time steps")
    
    return abs_inputs(zf_mu=zf_mu,
                         Pf=Pf,
                         za_mu=za_mu,
                         Pa=Pa,
                         kappa_xout=kappa_xout,
                         kappa_xin=kappa_xin,
                         nperiod=config.nperiod,
                         Y_dic=config.Y_dic,
                         Hz_dic=config.Hz_dic,
                         nbasis=config.nbasis,
                         nbc=config.nbc,
                         nxout=config.nxout,
                         xprior=config.xprior,
                         bcprior=config.bcprior,
                         rprior=config.rprior,
                         nr=config.nr
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
    Pf = np.asarray(config.Pf)
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
        Pf_tp1 = Pf[t + 1]

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
        Pf_L, _, _ = _cholesky_with_repair(Pf_tp1)
        Pf_factor = (Pf_L, True)

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