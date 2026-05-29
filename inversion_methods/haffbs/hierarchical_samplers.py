import re
from pathlib import Path

import numpy as np


def sigma2_rep_log_posterior(s_current, r2, sigma_obs, alpha_prior, beta_prior):

    """
    Calculates the log posterior for the representation error variance. Gaussiamn likelihood, inverse-gamma prior.

    The Gaussian likelihood is given by:

    p(y|x,sigma2_rep) ∝ product_i[ (sigma2_rep + sigma2_obs_i)**{-0.5} exp(- (y_i - H_i x)**2 / 2(sigma2_rep + sigma2_obs_i))]

    By finding the log likelihood, instabilities arrising from tiny likelihood product values are avoided.

    s_current = log(sigma2_rep)

    r2_i = (y_i - H_i x)**2

    """

    sigma2_rep = np.exp(s_current)
    
    err = sigma2_rep + sigma_obs**2

    if np.any(err <= 0):

        print(f"If you think about it really deeply, why can't we have a negative variance? sigma^2_rep = {sigma2_rep}")

    if np.isnan(err).any():

        print(f"If you think about it really deeply, why can't we have an NaN variance? sigma^2_rep = {sigma2_rep}")

    loglik = -0.5 * np.sum(np.log(err) + r2/err)

    logprior = -(alpha_prior + 1)*s_current - beta_prior / sigma2_rep

    return loglik + logprior + s_current



def sample_sigma2_rep(sigma2_rep_current, r2, sigma_obs, alpha_prior, beta_prior, w=1.0, m=100):

    """
    Slice sampler generating samples from the posterior of sigma2_rep. Slice sampler required as conjugacy broken as sigma2_rep appears as a component of the sum
    of the likelihood denominator - no closed form.
    """

    s_current = np.log(sigma2_rep_current)

    logp = lambda s: sigma2_rep_log_posterior(s, r2, sigma_obs, alpha_prior, beta_prior)

    logy = logp(s_current) - np.random.exponential(1)

    # initial bracket
    u = np.random.rand()
    L = s_current - w * u
    R = L + w

    # step out
    j = int(np.floor(m * np.random.rand()))
    k = (m - 1) - j

    while j > 0 and logp(L) > logy:
        L -= w
        j -= 1

    while k > 0 and logp(R) > logy:
        R += w
        k -= 1

    # shrinkage
    while True:
        s_new = np.random.uniform(L, R)

        if logp(s_new) > logy:
            return np.exp(s_new)
        elif s_new < s_current:
            L = s_new
        else:
            R = s_new



def sample_sigma2_qx(zmusample, kappa_x, alpha_prior, beta_prior, nbasis):

    """
    Conjugate update to generate samples from the inverse-gamma posterior of the forecast model noise variance for the emission fluxes x.


    x_bc and x_r are treated as fixed currently and so this sampler only works for sigma2_qx.
    """

    xr_prev = zmusample[:-1, -1][:, None]
    x_pred = kappa_x * zmusample[:-1, :nbasis] + (1 - kappa_x) * xr_prev
    
    x_innov = zmusample[1:,:nbasis] - x_pred
    shape = alpha_prior + 0.5 * x_innov.size
    scale = beta_prior + 0.5 * np.sum(x_innov**2)

    sigma2_qx_sample = 1 / np.random.gamma(shape, 1/scale)

    return sigma2_qx_sample


def kappa_max(nperiod, c=2):

    """
    Returns the maximum value of kappa_x given the number of periods and a constant c. This is required to ensure stationarity of the AR(1) forecast model.

    nperiod: number of periods in the inversion.

    c: the minimum number of e-folding decays the system must exhibit over the inversion period.
    """

    if c > nperiod:
        raise ValueError("c must be less than nperiod to ensure kappa_max is positive.")

    return 1 - c/nperiod



def kappa_log_posterior(kappa_current, kappa_max, dev_curr, dev_prev, sigma2_qx, kappa_aprior, kappa_bprior):

    """
    Generates a log-posterior value for a sample of kappa_x given samples of x and x_r.

    Compares current deviations of x from x_r to previous deviations of x from x_r. By doing this we can redefine the forecast model as an AR(1).

    We have a standard Gaussian likelihood from the residuals and a beta prior.

    """

    if kappa_current <= 0 or kappa_current >= kappa_max:
        raise ValueError("Nahh bro")
    
    else:

        e = dev_curr - kappa_current * dev_prev

        ss = np.sum(e**2)
        
        loglik = -0.5 * ss / sigma2_qx

        logprior = (kappa_aprior-1)*np.log(kappa_current) + (kappa_bprior-1)*np.log(1-kappa_current)
        
        return loglik + logprior
    


def sample_kappa(zmusample, sigma2_qx, kappa_current, kappa_max, kappa_aprior, kappa_bprior, nbasis, w=0.05, m=100):

    """
    Slice sampler generating samples from kappa_x given x, x_r, and sigma2_qx.

    """

    xmusample = zmusample[:,:nbasis]
    rmusample = zmusample[:,-1:]

    xmu_dev = xmusample - rmusample

    dev_prev = xmu_dev[:-1]
    dev_curr = xmu_dev[1:]

    logy = kappa_log_posterior(kappa_current, kappa_max, dev_curr, dev_prev, sigma2_qx, kappa_aprior, kappa_bprior) + np.log(np.random.rand())

    u = np.random.rand()
    L = kappa_current - w*u
    R = L + w
    
    L = max(L, 0.0)
    R = min(R, kappa_max)
    
    j = int(np.floor(m*np.random.rand()))
    k = (m-1) - j
    
    while j > 0 and L > 0.0 and kappa_log_posterior(L, kappa_max, dev_curr, dev_prev, sigma2_qx, kappa_aprior, kappa_bprior) > logy:
        L = max(L - w, 0.0)
        j -= 1
        
    while k > 0 and R < kappa_max and kappa_log_posterior(R, kappa_max, dev_curr, dev_prev, sigma2_qx, kappa_aprior, kappa_bprior) > logy:
        R = min(R + w, kappa_max)
        k -= 1
    
    # Step 3: shrinkage
    while True:
        kappa_new = np.random.uniform(L, R)
        if kappa_log_posterior(kappa_new, kappa_max, dev_curr, dev_prev, sigma2_qx, kappa_aprior, kappa_bprior) >= logy:
            return kappa_new
        
        if kappa_new < kappa_current:
            L = kappa_new
        else:
            R = kappa_new



