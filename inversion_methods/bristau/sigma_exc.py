"""

Excess model-data mismatch variance, sigma2_exc.

The likelihood variance for each observation is split as

    sigma2_y = sigma_obs**2 + sigma2_exc

where sigma_obs is the stated observation uncertainty (instrument
repeatability plus sub-averaging-period variability, from mf_error) and
sigma2_exc is the excess variance the residuals require beyond it. It is
expected to be dominated by forward-model error (transport, representation,
aggregation and boundary-condition error in the modelled mole fractions), but
also absorbs any measurement error not captured by sigma_obs, so the name
makes no claim about which source dominates.

"""

import numpy as np


def sigma_exc_scheme_select(config):

    # None means no excess error: equivalent to a fixed value of 0.
    if config.sigma_exc is None:
        return "fixed additive", 0.0

    # bool is a subclass of int, so exclude it explicitly.
    sigma_exc_is_fixed = (isinstance(config.sigma_exc, (int, float, np.integer, np.floating),) and not isinstance(config.sigma_exc, (bool, np.bool_)))

    if sigma_exc_is_fixed:
        sigma_exc_scheme = "fixed additive"
        fixed_sigma2_exc = float(config.sigma_exc)**2

        if not np.isfinite(fixed_sigma2_exc):
            raise ValueError("Fixed config.sigma_exc must be finite.")

        if fixed_sigma2_exc < 0:
            raise ValueError("Fixed config.sigma_exc must be non-negative.")

    elif isinstance(config.sigma_exc, str):
        # Normalise case and repeated whitespace.
        sigma_exc_scheme = " ".join(config.sigma_exc.lower().split())
        fixed_sigma2_exc = None

    else:
        raise TypeError("config.sigma_exc must be None, a numeric fixed value or 'global additive'. Further options in the pipeline.")

    valid_schemes = {"fixed additive", "global additive"}

    if sigma_exc_scheme not in valid_schemes:
        raise ValueError(f"Unknown config.sigma_exc value: {config.sigma_exc!r}. Expected one of {sorted(valid_schemes)} or a numeric value.")

    return sigma_exc_scheme, fixed_sigma2_exc


def initialise_sigma2_exc(sigma_exc_scheme, fixed_sigma2_exc, initial_sigma2_exc):
    if sigma_exc_scheme == "fixed additive":
        return float(fixed_sigma2_exc)
    return float(initial_sigma2_exc)


def update_sigma2_exc(state_residuals,
                   sigma_obs,
                   sigma2_exc_aprior,
                   sigma2_exc_bprior,
                   sigma2_exc_max,
                   sigma_exc_scheme,
                   sigma2_exc_current,
                   fixed_sigma2_exc,
                   tau,
                   prev_index_flat,
                   gap_flat,
                   rng=None,
                   ):

    if sigma_exc_scheme == "fixed additive":

        sigma2_exc_sample = fixed_sigma2_exc

    elif sigma_exc_scheme == "global additive":

        sigma2_exc_sample = sample_sigma2_exc(sigma2_exc_current, state_residuals, sigma_obs, sigma2_exc_aprior, sigma2_exc_bprior, sigma2_exc_max, tau, prev_index_flat, gap_flat, rng=rng)

    return sigma2_exc_sample


def sigma2_exc_log_posterior(s_current, r, sigma_obs, alpha_prior, beta_prior, sigma2_exc_max, tau, prev_index_flat, gap_flat):
    """
    Log posterior for sigma2_exc under a scaled Beta prior.

    Likelihood:
        residuals ~ N(0, sigma_obs**2 + sigma2_exc) marginally, with the
        AR(1)/OU same-site correlation phi(gap) = exp(-gap / tau) used by
        tau_resid.whiten_observations(). Factorising the joint density over
        each observation's preceding same-site residual, with standardised
        residuals u = r / sqrt(err_var):

            u_i | u_prev ~ N(phi_i * u_prev, 1 - phi_i**2)

        The log(1 - phi_i**2) term doesn't depend on sigma2_exc, so it is
        dropped. tau <= 0 sets phi = 0 everywhere, recovering the iid
        likelihood exactly.

    Prior:
        sigma2_exc / sigma2_exc_max ~ Beta(alpha_prior, beta_prior)

    Parameters
    ----------
    s_current : float
        Current value of log(sigma2_exc).
    r : array-like
        Residuals (signed, not squared), in the flat order of prev_index_flat.
    sigma_obs : float or array-like
        Stated observation uncertainty (standard deviation).
    alpha_prior : float
        Alpha shape parameter of the Beta prior.
    beta_prior : float
        Beta shape parameter of the Beta prior.
    sigma2_exc_max : float
        Upper bound for sigma2_exc.
    tau : float
        Current residual correlation length (hours). tau <= 0 means iid.
    prev_index_flat : np.ndarray of int
        Flat index of each observation's preceding same-site observation,
        or -1 if none (from tau_resid.prepare_tau_resid_indexing).
    gap_flat : np.ndarray
        Time gap in hours to that preceding observation.

    Returns
    -------
    float
        Log posterior density in terms of s_current.
    """

    sigma2_exc = np.exp(s_current)

    # Scaled beta support: sigma2_exc must be in (0, sigma2_exc_max)
    if sigma2_exc <= 0 or sigma2_exc >= sigma2_exc_max:
        return -np.inf

    err = sigma2_exc + sigma_obs**2

    if np.any(err <= 0):
        print(
            "If you think about it really deeply, why can't we have a "
            f"negative variance? sigma^2_exc = {sigma2_exc}"
        )
        return -np.inf

    if np.isnan(err).any():
        print(
            "If you think about it really deeply, why can't we have an "
            f"NaN variance? sigma^2_exc = {sigma2_exc}"
        )
        return -np.inf

    err = np.broadcast_to(err, np.shape(r))
    u = r / np.sqrt(err)

    if tau is None or tau <= 0:
        innovation = u
        variance = 1.0
    else:
        has_prev = prev_index_flat >= 0
        phi = np.where(has_prev, np.exp(-gap_flat / tau), 0.0)
        # Match whiten_observations(), which flushes these to zero.
        phi[phi < np.finfo(float).eps] = 0.0
        u_prev = np.where(has_prev, u[np.maximum(prev_index_flat, 0)], 0.0)
        innovation = u - phi * u_prev
        variance = 1.0 - phi**2

    # Gaussian log likelihood, dropping constants
    loglik = -0.5 * np.sum(np.log(err) + innovation**2 / variance)

    # Rescale sigma2_exc to the unit interval
    z = sigma2_exc / sigma2_exc_max

    # Beta prior on z = sigma2_exc / sigma2_exc_max
    # p(sigma2_exc) = BetaPDF(z; alpha, beta) / sigma2_exc_max
    logprior = (alpha_prior - 1) * np.log(z) + (beta_prior - 1) * np.log1p(-z)

    # Jacobian for sigma2_exc = exp(s_current)
    return loglik + logprior + s_current



def sample_sigma2_exc(sigma2_exc_current, r, sigma_obs, alpha_prior, beta_prior, sigma2_exc_max, tau, prev_index_flat, gap_flat, w=1.0, m=100, rng=None):

    """
    Slice sampler generating samples from the posterior of sigma2_exc. Slice sampler required as conjugacy broken as sigma2_exc appears as a component of the sum
    of the likelihood denominator - no closed form.

    r must be the signed residuals, not their squares: the AR(1) innovation
    depends on the sign of the preceding same-site residual. See
    sigma2_exc_log_posterior for tau, prev_index_flat and gap_flat.

    rng : numpy.random.Generator, optional
        Source of randomness for this draw. If not supplied (the default),
        this falls back to plain `numpy.random`, exactly as it always has --
        so passing nothing here is a complete no-op change. Pass an
        explicit, seeded Generator (e.g. `numpy.random.default_rng(seed)`)
        to make this draw -- and therefore the whole chain that keeps
        calling it -- reproducible.
    """

    if rng is None:
        rng = np.random

    s_current = np.log(sigma2_exc_current)
    s_upper = np.log(sigma2_exc_max)

    if not np.isfinite(s_current):
        raise ValueError("Current sigma2_exc has non-finite log posterior")


    logp = lambda s: sigma2_exc_log_posterior(s, r, sigma_obs, alpha_prior, beta_prior, sigma2_exc_max, tau, prev_index_flat, gap_flat)

    logy = logp(s_current) - rng.exponential(1)

    # initial bracket
    u = rng.uniform()
    L = s_current - w * u
    R = L + w

    R = min(R, s_upper)

    # step out
    j = int(np.floor(m * rng.uniform()))
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
        s_new = rng.uniform(L, R)

        if logp(s_new) > logy:
            return np.exp(s_new)
        elif s_new < s_current:
            L = s_new
        else:
            R = s_new
