import re
from pathlib import Path
from warnings import warn

import numpy as np
import xarray as xr
import arviz as az
from scipy import stats
from scipy.linalg import cholesky, solve_triangular, solve
from dataclasses import dataclass

from inversion_methods.manipulation.matrix_identities import kalman_gain_woodbury
from inversion_methods.manipulation.lognormal_transformations import state_vector_mu_transform, build_Wb

from openghg_inversions import utils, convert
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename



@dataclass
class amxkf_inputs:
    Y_dic: dict
    sigma_obs_dic: dict
    Ytime_dic: dict
    Hz_dic: dict
    siteindicator_dic: dict
    nbasis: int
    zprior_mus: np.ndarray
    zprior_covariance: np.ndarray
    forecast_noise: np.ndarray
    nperiod: int
    nbasis: int
    sigma_rep: float
    kappa_x: float
    xprior: dict
    bcprior: dict
    rprior: dict
    nbc: int | None = None


@dataclass
class abs_inputs:
    zf_mu: np.ndarray
    Pf: np.ndarray
    za_mu: np.ndarray
    Pa: np.ndarray
    kappa_x: float
    nperiod: int
    Y_dic: dict
    Hz_dic: dict
    nbasis: int
    nbc: int
    xprior: dict
    bcprior: dict
    rprior: dict



def augmented_forecast_model(z_mu, P_aug, kappa_x, Q_aug, nbasis, nbc):

    """
    This is a vectorised implementation of the forecast model for the augmented MXKF.

    The code implements the following calculations:

    z_t = F z_{t-1} + omega_z

    Where

    z = [[x], [x_bc], [x_r]]

    F = [[M, 0, B], [0, I, 0], [0, 0, I]]

    M = kappa_x I

    B = ones(nbasis) * (1 - kappa_x) 

    omega_z = [[omega_x], [omega_bc], [omega_r]]

    """

    ix = slice(0, nbasis)
    ir = -1

    one = np.ones(nbasis)

    # --- mean ---
    x = z_mu[ix]
    xr = z_mu[ir]

    zf_mu = np.empty_like(z_mu)
    zf_mu[ix] = kappa_x * x + (1 - kappa_x) * xr
    zf_mu[ir] = xr

    # --- covariance ---
    P_xx = P_aug[ix, ix]
    P_xr = P_aug[ix, ir]
    P_rx = P_aug[ir, ix]
    P_rr = P_aug[ir, ir]

    if nbc:

        ibc = slice(nbasis, nbasis + nbc)

        zf_mu[ibc] = z_mu[ibc]

        P_xbc = P_aug[ix, ibc]
        P_bcbc = P_aug[ibc, ibc]
        P_bcr = P_aug[ibc, ir]
        P_rbc = P_aug[ir, ibc]

    # P_xx
    P_xx_new = kappa_x**2 * P_xx

    alpha = kappa_x * (1 - kappa_x)
    P_xx_new += alpha * (P_xr[:, None] + P_rx[None, :])

    beta = (1 - kappa_x)**2 * P_rr
    P_xx_new += beta * np.outer(one, one)

    # P_xr
    P_xr_new = kappa_x * P_xr + (1 - kappa_x) * P_rr * one

    # P_rr
    P_rr_new = P_rr.copy()

    # add Q
    P_xx_new[np.diag_indices(nbasis)] += Q_aug[ix]
    P_rr_new += Q_aug[ir]

    # assemble
    Pf_aug = np.zeros_like(P_aug)
    
    Pf_aug[ix, ix] = P_xx_new
    Pf_aug[ix, ir] = P_xr_new
    Pf_aug[ir, ix] = P_xr_new
    Pf_aug[ir, ir] = P_rr_new


    if nbc:

        # # P_xbc
        # P_xbc_new = kappa_x * P_xbc
        # P_bcx_new = P_xbc_new.T

        # # P_bcbc
        # P_bcbc_new = P_bcbc.copy()
        
        # # P_bcr
        # P_bcr_new = P_bcr
        # P_rbc_new = P_rbc

        # # Add Q
        # P_bcbc_new[np.diag_indices(nbc)] += Q_aug[ibc]

        # Pf_aug[ibc, ix] = P_bcx_new
        # Pf_aug[ibc, ibc] = P_bcbc_new
        # Pf_aug[ix, ibc] = P_xbc_new
        # Pf_aug[ibc, ir] = P_bcr_new
        # Pf_aug[ir, ibc] = P_rbc_new


        # P_xbc
        P_xbc_new = kappa_x * P_xbc + (1 - kappa_x) * P_rbc[None, :]
        P_bcx_new = P_xbc_new.T

        # P_bcbc
        P_bcbc_new = P_bcbc.copy()
        
        # P_bcr
        P_bcr_new = P_bcr
        P_rbc_new = P_rbc

        # Add Q
        P_bcbc_new[np.diag_indices(nbc)] += Q_aug[ibc]

        Pf_aug[ibc, ix] = P_bcx_new
        Pf_aug[ibc, ibc] = P_bcbc_new
        Pf_aug[ix, ibc] = P_xbc_new
        Pf_aug[ibc, ir] = P_bcr_new
        Pf_aug[ir, ibc] = P_rbc_new


    return zf_mu, Pf_aug



def augmented_forecast_innovations(y, sigma_obs, sigma2_rep, zf, Pf_aug, H_aug, H_hat_aug):
    
    """
    Calculates the augmented Kalman gain and the augmented forecast innovations. Utilises a vectorised inverse of the measurement covariance to avoid inefficient sparse matrix propagation.
    
    """

    r_inv = 1 / (sigma2_rep + sigma_obs**2)

    K_aug = kalman_gain_woodbury(Pf_aug, H_hat_aug, r_inv)

    d = y - H_aug @ zf

    # print(f"observed: {y.mean()}, modelled: {(H_aug @ zf).mean()}")

    return K_aug, d



def augmented_analysis_update(zf_mu, Pf_aug, K_aug, H_hat_aug, d, sigma2_rep, sigma_obs):

    """
    The analysis update for the augmented MXKF. Utilises vectorised R.
    """

    update = K_aug @ d

    za_mu = zf_mu + update

    r = sigma2_rep + sigma_obs**2

    KH = K_aug @ H_hat_aug
    I_KH = np.eye(Pf_aug.shape[0]) - KH

    Pa = I_KH @ Pf_aug @ I_KH.T

    K_r = K_aug * r
    Pa += K_r @ K_aug.T

    return za_mu, Pa



def augmented_mxkf(config: amxkf_inputs) -> abs_inputs:
        

    nr = 1 # currently only 1 global relaxation term
    nx = config.nbasis
    
    if config.nbc:
        nz = nx + config.nbc + nr
    else:
        nz = nx + nr

    zf_mu = np.zeros((config.nperiod, nz))
    Pf = np.zeros((config.nperiod, nz, nz))
    za_mu = np.zeros((config.nperiod, nz))
    Pa = np.zeros((config.nperiod, nz, nz))

    for t in range(config.nperiod):
        H = config.Hz_dic[t]
        Y = config.Y_dic[t]
        sigma_obs = config.sigma_obs_dic[t]

        ny = len(Y)

        if t == 0:
            za_mu[-1] = config.zprior_mus
            Pa[-1] = config.zprior_covariance

        zf_mu[t], Pf[t] = augmented_forecast_model(za_mu[t-1], Pa[t-1], config.kappa_x, config.forecast_noise, config.nbasis, config.nbc)

        zf = state_vector_mu_transform(zf_mu[t], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)

        Wb = build_Wb(zf, config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)
        Wo_inv = np.eye(ny)
        H_hat = Wo_inv @ H @ Wb

        K, d = augmented_forecast_innovations(Y, sigma_obs, config.sigma_rep**2, zf, Pf[t], H, H_hat)
        
        za_mu[t], Pa[t] = augmented_analysis_update(zf_mu[t], Pf[t], K, H_hat, d, config.sigma_rep**2, sigma_obs)

        # if not np.allclose(Pa[t], Pa[t].T, atol=1e-12):

        #     print(f"ahhh Pa not symmetric")


        # eigvals = np.linalg.eigvalsh(Pa[t])  # for symmetric matrices
        # is_psd = np.all(eigvals >= -1e-10)

        # if not is_psd:

        #     print(f"ahhh Pa not PSD")

    return abs_inputs(zf_mu=zf_mu,
                         Pf=Pf,
                         za_mu=za_mu,
                         Pa=Pa,
                         kappa_x=config.kappa_x,
                         nperiod=config.nperiod,
                         Y_dic=config.Y_dic,
                         Hz_dic=config.Hz_dic,
                         nbasis=config.nbasis,
                         nbc=config.nbc,
                         xprior=config.xprior,
                         bcprior=config.bcprior,
                         rprior=config.rprior
                         )



# def conditional_samples(mean_t, cov_t, ix, ibc, ir):
#     # print(f"Pa_t: {cov_t}, Pa_diag: {np.diag(cov_t)}")
#     mu_x = mean_t[ix]
#     mu_r = mean_t[ir]

#     Sigma_xx = cov_t[ix, ix]
#     Sigma_xr = np.atleast_1d(cov_t[ix, ir])
#     Sigma_rr = cov_t[ir, ir]
    
#     # if Sigma_rr <= 0:
#     #     raise ValueError("Sigma_rr must be positive")

#     Sigma_rr = max(Sigma_rr, 1e-12)

#     if ibc is not None:
        
#         mu_bc = mean_t[ibc]

#         mu_joint = np.concatenate([mu_x, mu_bc])
        
#         Sigma_bcbc = cov_t[ibc, ibc]
#         Sigma_bcx = cov_t[ibc, ix]
#         Sigma_bcr = np.atleast_1d(cov_t[ibc, ir])

#         Sigma_xbc = cov_t[ix, ibc]

#         Sigma_joint = np.block([[Sigma_xx,  Sigma_xbc], 
#                                 [Sigma_bcx, Sigma_bcbc]])
        
#         Sigma_joint_r = np.concatenate([Sigma_xr, Sigma_bcr])

#     else:

#         mu_joint = mu_x

#         Sigma_joint = Sigma_xx

#         Sigma_joint_r = Sigma_xr

#     Sigma_joint_r = np.atleast_1d(Sigma_joint_r)

#     n = len(mu_joint)

#     r_t = np.random.normal(mu_r, np.sqrt(Sigma_rr))                 #  Samples from relaxation term

#     inv_sqrt = 1.0 / np.sqrt(Sigma_rr)
#     v = Sigma_joint_r * inv_sqrt

#     cond_mean = mu_joint + v * (r_t - mu_r) * inv_sqrt
#     cond_cov  = Sigma_joint - np.outer(v, v)

#     # Enforce symmetry
#     eigvals, eigvecs = np.linalg.eigh(cond_cov)
#     eigvals = np.maximum(eigvals, 1e-12)
#     cond_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

#     # enforce symmetry and add some jitter
#     cond_cov = 0.5 * (cond_cov + cond_cov.T)
#     cond_cov += 1e-10 * np.eye(n)

#     # min_eig = np.min(np.linalg.eigvalsh(cond_cov))
#     # if min_eig < -1e-8:
#     #     print("WARNING: cond_cov is not PSD:", min_eig)


#     # try:
#     #     L = cholesky(cond_cov, lower=True)
#     # except np.linalg.LinAlgError:
#     #     cond_cov += 1e-10 * np.eye(n)
#     #     L = cholesky(cond_cov, lower=True)
        
#     L = cholesky(cond_cov, lower=True)

#     sample = cond_mean + L @ np.random.randn(n)

#     return np.concatenate([sample, np.array([r_t])])


# def conditional_samples(mean_t, cov_t, ix, ibc, ir):
#     """
#     Draw a conditional sample from the joint Gaussian using
#     a projection-based method that avoids forming the conditional covariance.

#     This version is numerically stable and avoids Cholesky failures
#     on ill-conditioned conditional covariances.
#     """

#     mu_x = mean_t[ix]
#     mu_r = mean_t[ir]

#     Sigma_xx = cov_t[ix, ix]
#     Sigma_xr = np.atleast_1d(cov_t[ix, ir])
#     Sigma_rr = max(cov_t[ir, ir], 1e-12)

#     # --- build joint state (x [+ bc]) ---
#     if ibc is not None:
#         mu_bc = mean_t[ibc]
#         mu_joint = np.concatenate([mu_x, mu_bc])

#         Sigma_bcbc = cov_t[ibc, ibc]
#         Sigma_bcx  = cov_t[ibc, ix]
#         Sigma_bcr  = np.atleast_1d(cov_t[ibc, ir])
#         Sigma_xbc  = cov_t[ix, ibc]

#         Sigma_joint = np.block([
#             [Sigma_xx,  Sigma_xbc],
#             [Sigma_bcx, Sigma_bcbc]
#         ])

#         Sigma_joint_r = np.concatenate([Sigma_xr, Sigma_bcr])

#     else:
#         mu_joint = mu_x
#         Sigma_joint = Sigma_xx
#         Sigma_joint_r = Sigma_xr

#     Sigma_joint_r = np.asarray(Sigma_joint_r).reshape(-1)
#     n = len(mu_joint)

#     # --- sample r ---
#     r_t = np.random.normal(mu_r, np.sqrt(Sigma_rr))

#     # --- sample from joint marginal (unconditional) ---
#     # stabilise joint covariance (much safer than cond_cov)
#     Sigma_joint = 0.5 * (Sigma_joint + Sigma_joint.T)

#     try:
#         L = cholesky(Sigma_joint + 1e-10 * np.eye(n), lower=True)

#     except np.linalg.LinAlgError:
#         eigvals, eigvecs = np.linalg.eigh(Sigma_joint)
#         eigvals = np.maximum(eigvals, 1e-12)
#         Sigma_joint = eigvecs @ np.diag(eigvals) @ eigvecs.T

#         try:
#             L = cholesky(Sigma_joint + 1e-10 * np.eye(n), lower=True)
        
#         except np.linalg.LinAlgError:
#             warn("Panic stations as we have a Cholesky failure in backward sampler (not PSD). Thankfully we have a backup...")

#             U, s, _ = np.linalg.svd(Sigma_joint)
#             s = np.maximum(s, 0)   # ensure non-negative

#             L = U @ np.diag(np.sqrt(s))


#     x_unc = mu_joint + L @ np.random.randn(n)

#     # --- project to conditional ---
#     gain = Sigma_joint_r / Sigma_rr

#     # correction term
#     correction = (x_unc - mu_joint) @ Sigma_joint_r - (r_t - mu_r)

#     x_cond = x_unc - gain * correction

#     return np.concatenate([x_cond, np.array([r_t])])



def conditional_samples(mean_t, cov_t, ix, ibc, ir):
    """
    Draw a conditional sample from a joint Gaussian using a projection method.

    This version:
      - uses a mathematically correct projection
      - avoids explicit conditional covariance
      - is robust to ill-conditioning
    """

    # --- extract means ---
    mu_x = mean_t[ix]
    mu_r = mean_t[ir]

    # --- extract covariances (robust shapes) ---
    Sigma_xx = cov_t[ix, ix]
    Sigma_xr = np.asarray(cov_t[ix, ir]).reshape(-1)
    Sigma_rr = cov_t[ir, ir]

    if Sigma_rr <= 0:
        raise ValueError("Sigma_rr must be positive")

    # --- build joint state (x [+ bc]) ---
    if ibc is not None:
        mu_bc = mean_t[ibc]

        mu_joint = np.concatenate([mu_x, mu_bc])

        Sigma_bcbc = cov_t[ibc, ibc]
        Sigma_bcx  = cov_t[ibc, ix]
        Sigma_xbc  = cov_t[ix, ibc]
        Sigma_bcr  = np.asarray(cov_t[ibc, ir]).reshape(-1)

        Sigma_joint = np.block([
            [Sigma_xx,  Sigma_xbc],
            [Sigma_bcx, Sigma_bcbc]
        ])

        Sigma_joint_r = np.concatenate([Sigma_xr, Sigma_bcr])

    else:
        mu_joint = mu_x
        Sigma_joint = Sigma_xx
        Sigma_joint_r = Sigma_xr

    # --- enforce symmetry early ---
    Sigma_joint = 0.5 * (Sigma_joint + Sigma_joint.T)

    n = len(mu_joint)

    # --- build FULL joint covariance including r ---
    Sigma_full = np.block([
        [Sigma_joint,                Sigma_joint_r[:, None]],
        [Sigma_joint_r[None, :],     np.array([[Sigma_rr]])]
    ])

    mu_full = np.concatenate([mu_joint, [mu_r]])

    # symmetrise
    Sigma_full = 0.5 * (Sigma_full + Sigma_full.T)

    # --- sample full joint ---
    try:
        L = cholesky(Sigma_full + 1e-10 * np.eye(n + 1), lower=True)

    except np.linalg.LinAlgError:
        # eigenvalue repair
        eigvals, eigvecs = np.linalg.eigh(Sigma_full)
        eigvals = np.maximum(eigvals, 1e-12)
        Sigma_full = eigvecs @ np.diag(eigvals) @ eigvecs.T

        try:
            L = cholesky(Sigma_full + 1e-10 * np.eye(n + 1), lower=True)

        except np.linalg.LinAlgError:
            warn("Cholesky failed — using SVD fallback")

            U, s, _ = np.linalg.svd(Sigma_full)
            s = np.maximum(s, 0)

            L = U @ np.diag(np.sqrt(s))

    z_unc = mu_full + L @ np.random.randn(n + 1)

    x_unc = z_unc[:-1]
    r_unc = z_unc[-1]

    # --- independently sample the desired r_t ---
    r_t = np.random.normal(mu_r, np.sqrt(Sigma_rr))

    # --- correct projection (KEY FIX) ---
    gain = Sigma_joint_r / Sigma_rr

    x_cond = x_unc + gain * (r_t - r_unc)

    # --- return combined state ---
    return np.concatenate([x_cond, np.array([r_t])])



def augmented_backward_sampler(config=abs_inputs):
    
    """
    Mixed FFBS sampler:
      - x_r (random walk) sampled directly from marginal MVN
      - x sampled conditionally
    No explicit F propagation to avoid inefficiencies of sparse matrices.
    """

    ix = slice(0, config.nbasis)
    ir = -1

    if config.nbc:
        ibc = slice(config.nbasis, config.nbasis + config.nbc)
    else:
        ibc = None

    ones_x = np.ones(config.nbasis)
    z_mu = np.zeros_like(config.za_mu)
    z = np.zeros_like(z_mu)

    z_mu[-1] = conditional_samples(config.za_mu[-1], config.Pa[-1], ix, ibc, ir)
    z[-1] = state_vector_mu_transform(z_mu[-1], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)

    # Wb = np.diag(z[-1])
    # Wo_inv = np.eye(len(config.Y_dic[config.nperiod-1]))
    # H_hat = Wo_inv @ config.Hz_dic[config.nperiod-1] @ Wb

    # state_residuals = config.Y_dic[config.nperiod-1] - (H_hat @ z_mu[-1])
    state_residuals = config.Y_dic[config.nperiod-1] - (config.Hz_dic[config.nperiod-1] @ z[-1])

    for t in reversed(range(config.nperiod - 1)):
        
        Pa = config.Pa[t]

        Pa_xx = Pa[ix, ix]
        Pa_xr = Pa[ix, ir]
        Pa_rx = Pa_xr.T
        Pa_rr = Pa[ir, ir]

        if config.nbc:

            Pa_bcx = Pa[ibc, ix]
            Pa_xbc = Pa_bcx.T
            Pa_bcr = Pa[ibc, ir]
            Pa_rbc = Pa_bcr.T
            Pa_bcbc = Pa[ibc, ibc]

        C = np.zeros_like(Pa)

        C[ix, ix] = config.kappa_x * Pa_xx + (1 - config.kappa_x) * np.outer(Pa_xr, ones_x)
        C[ix, ir] = Pa_xr
        C[ir, ix] = config.kappa_x * Pa_rx + (1 - config.kappa_x) * Pa_rr * ones_x
        C[ir, ir] = Pa_rr

        if config.nbc:

            C[ibc, ibc] = Pa_bcbc
            C[ibc, ix] = config.kappa_x * Pa_bcx
            C[ibc, ir] = Pa_bcr
            C[ix, ibc] = Pa_xbc
            C[ir, ibc] = Pa_rbc

        A = solve(config.Pf[t+1].T, C.T).T

        mean_t = config.za_mu[t] + A @ (z_mu[t+1] - config.zf_mu[t+1])
        cov_t  = config.Pa[t] - A @ config.Pf[t+1] @ A.T
        # cov_t = 0.5 * (cov_t + cov_t.T)

        z_mu[t] = conditional_samples(mean_t, cov_t, ix, ibc, ir)

        z[t] = state_vector_mu_transform(z_mu[t], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)

        # Wb = build_Wb(z[t], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)
        # Wo_inv = np.eye(len(config.Y_dic[t]))
        # H_hat = Wo_inv @ config.Hz_dic[t] @ Wb

        # resid = config.Y_dic[t] - (H_hat @ z_mu[t])

        resid = config.Y_dic[t] - (config.Hz_dic[t] @ z[t])
        state_residuals = np.concatenate([resid, state_residuals])

    return z_mu, z, state_residuals



def build_Pz(rprior_sigma2, bcprior_sigma2, sigma2_qx, sigma2_qr, kappa_x, nbasis, nbc):


    ix = slice(0, nbasis)
    ir = -1

    if nbc:
        ibc = slice(nbasis, nbasis + nbc)
        nparam = nbasis + nbc +1
        Pbc = np.eye(nbc) * bcprior_sigma2
    else:
        nparam = nbasis + 1

    Pz_prior = np.zeros((nparam, nparam))

    L = np.ones((nbasis, 1))
    Ix = np.eye(nbasis)

    Pd = (sigma2_qx * Ix + sigma2_qr * L @ L.T) / (1 - kappa_x**2)

    Pz_prior[ix, ix] = Pd + rprior_sigma2 * L @ L.T
    Pz_prior[ix, ir] = (rprior_sigma2 * L).ravel()
    Pz_prior[ir, ix] = (rprior_sigma2 * L.T).ravel()
    Pz_prior[ir, ir] = rprior_sigma2

    if nbc:
        Pz_prior[ibc, ibc] = Pbc
        Pz_prior[ibc, ir] = np.zeros(nbc)
        Pz_prior[ir, ibc] = np.zeros(nbc)
        Pz_prior[ibc, ix] = np.zeros((nbc, nbasis))
        Pz_prior[ix, ibc] = np.zeros((nbasis, nbc))

    return 0.5 * (Pz_prior + Pz_prior.T)