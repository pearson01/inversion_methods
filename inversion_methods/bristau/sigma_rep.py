import numpy as np


def sigma_rep_scheme_select(config):

    # bool is a subclass of int, so exclude it explicitly.
    sigma_rep_is_fixed = (isinstance(config.sigma_rep, (int, float, np.integer, np.floating),) and not isinstance(config.sigma_rep, (bool, np.bool_)))

    if sigma_rep_is_fixed:
        sigma_rep_scheme = "fixed additive"
        fixed_sigma2_rep = float(config.sigma_rep)**2

        if not np.isfinite(fixed_sigma2_rep):
            raise ValueError("Fixed config.sigma_rep must be finite.")

        if fixed_sigma2_rep < 0:
            raise ValueError("Fixed config.sigma_rep must be non-negative.")

    elif isinstance(config.sigma_rep, str):
        # Normalise case and repeated whitespace.
        sigma_rep_scheme = " ".join(config.sigma_rep.lower().split())
        fixed_sigma2_rep = None

    else:
        raise TypeError("config.sigma_rep must be a numeric fixed value or 'global additive'. Further options in the pipeline.")

    valid_schemes = {"fixed additive", "global additive"}

    if sigma_rep_scheme not in valid_schemes:
        raise ValueError(f"Unknown config.sigma_rep value: {config.sigma_rep!r}. Expected one of {sorted(valid_schemes)} or a numeric value.")

    return sigma_rep_scheme, fixed_sigma2_rep


def update_sigma2_rep(state_residuals,
                   sigma_obs,
                   sigma2_rep_aprior, 
                   sigma2_rep_bprior, 
                   sigma2_rep_max, 
                   sigma_rep_scheme,
                   sigma2_rep_current,
                   fixed_sigma2_rep, 
                   ):
    
    if sigma_rep_scheme == "fixed additive":

        sigma2_rep_sample = fixed_sigma2_rep

    elif sigma_rep_scheme == "global additive":

        sigma2_rep_sample = sample_sigma2_rep(sigma2_rep_current, state_residuals**2, sigma_obs, sigma2_rep_aprior, sigma2_rep_bprior, sigma2_rep_max)

    return sigma2_rep_sample


def sigma2_rep_log_posterior(s_current, r2, sigma_obs, alpha_prior, beta_prior, sigma2_rep_max):
    """
    Log posterior for sigma2_rep under a scaled Beta prior.

    Prior:
        sigma2_rep / sigma2_rep_max ~ Beta(alpha_prior, beta_prior)

    Parameters
    ----------
    s_current : float
        Current value of log(sigma2_rep).
    r2 : array-like
        Squared residuals.
    sigma_obs : float or array-like
        Observation standard deviation.
    alpha_prior : float
        Alpha shape parameter of the Beta prior.
    beta_prior : float
        Beta shape parameter of the Beta prior.
    sigma2_rep_max : float
        Upper bound for sigma2_rep.
    include_prior_constant : bool
        Whether to include the normalising constant of the scaled Beta prior.
        Usually unnecessary for MCMC if constants cancel.

    Returns
    -------
    float
        Log posterior density in terms of s_current.
    """

    sigma2_rep = np.exp(s_current)

    # Scaled beta support: sigma2_rep must be in (0, sigma2_rep_max)
    if sigma2_rep <= 0 or sigma2_rep >= sigma2_rep_max:
        return -np.inf

    err = sigma2_rep + sigma_obs**2

    if np.any(err <= 0):
        print(
            "If you think about it really deeply, why can't we have a "
            f"negative variance? sigma^2_rep = {sigma2_rep}"
        )
        return -np.inf

    if np.isnan(err).any():
        print(
            "If you think about it really deeply, why can't we have an "
            f"NaN variance? sigma^2_rep = {sigma2_rep}"
        )
        return -np.inf

    # Gaussian log likelihood, dropping constants
    loglik = -0.5 * np.sum(np.log(err) + r2 / err)

    # Rescale sigma2_rep to the unit interval
    z = sigma2_rep / sigma2_rep_max

    # Beta prior on z = sigma2_rep / sigma2_rep_max
    # p(sigma2_rep) = BetaPDF(z; alpha, beta) / sigma2_rep_max
    logprior = (
        (alpha_prior - 1) * np.log(z)
        + (beta_prior - 1) * np.log1p(-z)
    )

    # Jacobian for sigma2_rep = exp(s_current)
    return loglik + logprior + s_current



def sample_sigma2_rep(sigma2_rep_current, r2, sigma_obs, alpha_prior, beta_prior, sigma2_rep_max, w=1.0, m=100):

    """
    Slice sampler generating samples from the posterior of sigma2_rep. Slice sampler required as conjugacy broken as sigma2_rep appears as a component of the sum
    of the likelihood denominator - no closed form.
    """

    s_current = np.log(sigma2_rep_current)
    s_upper = np.log(sigma2_rep_max)

    if not np.isfinite(s_current):
        raise ValueError("Current sigma2_rep has non-finite log posterior")


    logp = lambda s: sigma2_rep_log_posterior(s, r2, sigma_obs, alpha_prior, beta_prior, sigma2_rep_max)

    logy = logp(s_current) - np.random.exponential(1)

    # initial bracket
    u = np.random.rand()
    L = s_current - w * u
    R = L + w

    R = min(R, s_upper)

    # step out
    j = int(np.floor(m * np.random.rand()))
    k = (m - 1) - j

    while j > 0 and logp(L) > logy:
        L -= w
        j -= 1

    while k > 0 and R < s_upper and logp(R) > logy:
        R += w
        R = min(R, s_upper)
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
