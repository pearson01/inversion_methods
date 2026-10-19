import numpy as np
from scipy.stats import invgamma


def kappa_x_scheme_select(config):

    # bool is a subclass of int, so exclude it explicitly.
    kappa_x_is_fixed = (isinstance(config.kappa_x, (int, float, np.integer, np.floating),) and not isinstance(config.kappa_x, (bool, np.bool_)))

    if kappa_x_is_fixed:
        kappa_x_scheme = "fixed"
        fixed_kappa_x = float(config.kappa_x)

        if not np.isfinite(fixed_kappa_x):
            raise ValueError("Fixed config.kappa_x must be finite.")

        if fixed_kappa_x < 0:
            raise ValueError("Fixed config.kappa_x must be non-negative.")

    elif isinstance(config.kappa_x, str):
        # Normalise case and repeated whitespace.
        kappa_x_scheme = " ".join(config.kappa_x.lower().split())
        fixed_kappa_x = None

    else:
        raise TypeError("config.kappa_x must be a numeric fixed value or one of 'global', or 'inner outer'.")

    valid_schemes = {"fixed", "global", "inner outer"}

    if kappa_x_scheme not in valid_schemes:
        raise ValueError(f"Unknown config.kappa_x value: {config.kappa_x!r}. Expected one of {sorted(valid_schemes)} or a numeric value.")
    
    if kappa_x_scheme == "inner outer" and config.nxout == 0:
        raise ValueError(f"Attempting to infer inner and outer kappa terms but only 1 relaxation term exists. kappa_x_scheme: {kappa_x_scheme}, nxout: {config.nxout}")

    return kappa_x_scheme, fixed_kappa_x


def kappa_x_trace_params(kappa_x_scheme):

    if kappa_x_scheme in {"fixed", "global"}:
        kappa_x_trace_labels = ["global"]
        n_kappa_x_parameters = 1

    elif kappa_x_scheme == "inner outer":
        kappa_x_trace_labels = ["outer", "inner",]
        n_kappa_x_parameters = 2

    return kappa_x_trace_labels, n_kappa_x_parameters


def initialise_kappa_x_vector(kappa_x_scheme, fixed_kappa_x, initial_kappa_x, n_kappa_x_parameters):

    if kappa_x_scheme == "fixed":
        kappa_x_global_current = fixed_kappa_x

        kappa_x_vector_current = np.full(n_kappa_x_parameters, kappa_x_global_current, dtype=float,)

    elif kappa_x_scheme == "global":
        kappa_x_global_current = initial_kappa_x

        kappa_x_vector_current = np.full(n_kappa_x_parameters, kappa_x_global_current, dtype=float,)

    elif kappa_x_scheme == "inner outer":

        kappa_x_global_current = initial_kappa_x

        kappa_x_vector_current = np.full(n_kappa_x_parameters, kappa_x_global_current, dtype=float,)

    return kappa_x_vector_current


def update_kappa_x(zmusample,
                   zmusample_out,
                   zmusample_in,
                   sigma2_qx_bf_current,
                   kappa_x_aprior,
                   kappa_x_bprior,
                   nxout,
                   nxin,
                   kappa_x_max,
                   kappa_x_scheme,
                   kappa_x_vector_current,
                   fixed_kappa_x,
                   rng=None,
                   ):


    if kappa_x_scheme == "fixed":

        kappa_x_sample = np.full_like(kappa_x_vector_current, fixed_kappa_x)

    elif kappa_x_scheme == "global" and nxout == 0:

        kappa_x_sample = sample_kappa(zmusample, sigma2_qx_bf_current, kappa_x_vector_current, kappa_x_max, kappa_x_aprior, kappa_x_bprior, nxin, rng=rng)

    elif kappa_x_scheme == "inner outer" and nxout > 0:

        kappa_x_sample = np.zeros_like(kappa_x_vector_current)

        kappa_x_sample[0] = sample_kappa(zmusample_out, sigma2_qx_bf_current[:nxout], kappa_x_vector_current[0], kappa_x_max, kappa_x_aprior, kappa_x_bprior, nxout, rng=rng)
        kappa_x_sample[1] = sample_kappa(zmusample_in, sigma2_qx_bf_current[nxout:], kappa_x_vector_current[1], kappa_x_max, kappa_x_aprior, kappa_x_bprior, nxin, rng=rng)

    else:

        raise ValueError(f"No kappa_x update today >:(")

    return kappa_x_sample


def kappa_max(nperiod, c=2):

    """
    Returns the maximum value of kappa_x given the number of periods and a constant c. This is required to ensure stationarity of the AR(1) forecast model.

    nperiod: number of periods in the inversion.

    c: the minimum number of e-folding decays the system must exhibit over the inversion period.
    """

    if c > nperiod:
        raise ValueError("c must be less than nperiod to ensure kappa_max is positive.")

    return 1 - c/nperiod


def kappa_log_posterior(
    kappa_current,
    kappa_max,
    dev_curr,
    dev_prev,
    rmusample,
    sigma2_qx_bf_current,
    kappa_aprior,
    kappa_bprior
):
    """
    Log-posterior for a scalar kappa_x with basis-specific sigma2_qx.

    Parameters
    ----------
    kappa_current : float
        Proposed scalar kappa value.
    kappa_max : float
        Upper bound for kappa.
    dev_curr : ndarray, shape (T-1, nbasis)
        Current deviations x_t - r_t
    dev_prev : ndarray, shape (T-1, nbasis)
        Previous deviations x_{t-1} - r_{t-1}
    rmusample : ndarray, shape (T, 1) or (T,)
        Relaxing mean trajectory
    sigma2_qx : float or ndarray, shape (nbasis,)
        Process noise variance(s) for x basis functions
    kappa_aprior, kappa_bprior : float
        Beta prior parameters

    Returns
    -------
    float
        Log posterior up to an additive constant.
    """

    if kappa_current <= 0 or kappa_current >= kappa_max:
        return -np.inf

    # ensure sigma2_qx is a vector
    sigma2_qx_bf_current = np.asarray(sigma2_qx_bf_current, dtype=float)

    # residuals:
    # e_t = dev_t - kappa * dev_{t-1} - (r_{t-1} - r_t)
    drift = (rmusample[:-1] - rmusample[1:])  # shape (T-1, 1) or (T-1,)
    if drift.ndim == 1:
        drift = drift[:, None]

    e = dev_curr - kappa_current * dev_prev - drift

    # weighted sum of squares:
    # sum_j sum_t e[t,j]^2 / sigma2_qx[j]
    inv_sigma2 = 1.0 / sigma2_qx_bf_current
    weighted_ss = np.einsum("tj,j->", e**2, inv_sigma2)

    loglik = -0.5 * weighted_ss

    # Beta prior on kappa over (0,1).
    # If kappa_max < 1 and you want a proper Beta on (0, kappa_max),
    # see note below.
    logprior = ((kappa_aprior - 1) * np.log(kappa_current) + (kappa_bprior - 1) * np.log(1 - kappa_current))

    return loglik + logprior


def sample_kappa(
    zmusample,
    sigma2_qx_bf_current,
    kappa_current,
    kappa_max,
    kappa_aprior,
    kappa_bprior,
    nbasis,
    w=0.05,
    m=100,
    rng=None,
):
    """
    Slice sampler for a scalar kappa_x given x, r, and basis-specific sigma2_qx.

    rng : numpy.random.Generator, optional
        Source of randomness for this draw. Defaults to plain
        `numpy.random` (today's exact behaviour) when not supplied; pass a
        seeded Generator for a reproducible chain.
    """
    if rng is None:
        rng = np.random


    xmusample = zmusample[:, :nbasis]
    rmusample = zmusample[:, -1:]   # keep as (T,1) for broadcasting

    xmu_dev = xmusample - rmusample

    dev_prev = xmu_dev[:-1]
    dev_curr = xmu_dev[1:]

    def log_density(kappa):
        return kappa_log_posterior(kappa, kappa_max, dev_curr, dev_prev, rmusample, sigma2_qx_bf_current, kappa_aprior, kappa_bprior)

    return slice_sample_kappa(log_density, kappa_current, kappa_max, w=w, m=m, rng=rng)


def slice_sample_kappa(log_density, kappa_current, kappa_max, w=0.05, m=100, rng=None):
    """
    Stepping-out slice sampler for a scalar kappa on (0, kappa_max).

    log_density : callable
        Unnormalised log density of kappa. Shared by the conditional
        (sample_kappa) and collapsed (sample_kappa_marginal) kappa updates.
    """
    if rng is None:
        rng = np.random

    logy = log_density(kappa_current) + np.log(rng.uniform())

    u = rng.uniform()
    L = kappa_current - w * u
    R = L + w

    L = max(L, 0.0)
    R = min(R, kappa_max)

    j = int(np.floor(m * rng.uniform()))
    k = (m - 1) - j

    while j > 0 and L > 0.0 and log_density(L) > logy:
        L = max(L - w, 0.0)
        j -= 1

    while k > 0 and R < kappa_max and log_density(R) > logy:
        R = min(R + w, kappa_max)
        k -= 1

    while True:
        kappa_new = rng.uniform(L, R)

        if log_density(kappa_new) >= logy:
            return kappa_new

        if kappa_new < kappa_current:
            L = kappa_new
        else:
            R = kappa_new


# ----------------------------------------------------------------------
# Collapsed kappa_x update (sigma2_qx integrated out)
# ----------------------------------------------------------------------
#
# sigma2_qx has a conjugate (truncated) inverse-gamma prior, so it can be
# integrated out of p(kappa_x, sigma2_qx | x, r) analytically. Drawing
# kappa_x from this marginal and then sigma2_qx | kappa_x (the existing
# update_sigma2_qx) gives an exact joint draw of (kappa_x, sigma2_qx).


def innovation_moments(xsample, rsample):
    """
    Sufficient statistics of the x process-model innovations as a function of kappa.

        e_t(kappa) = x_t - kappa * x_{t-1} - (1 - kappa) * r_{t-1}
                   = a_t - kappa * b_t,   a_t = x_t - r_{t-1},   b_t = x_{t-1} - r_{t-1}

    so the summed squared innovations are a quadratic in kappa:

        S(kappa) = saa - 2 * kappa * sab + kappa**2 * sbb

    xsample : ndarray, shape (T, ncols)
    rsample : ndarray, shape (T,)

    Returns (saa, sab, sbb, n) where n is the number of innovations.
    """
    r_prev = rsample[:-1, None]

    a = xsample[1:] - r_prev
    b = xsample[:-1] - r_prev

    return np.sum(a * a), np.sum(a * b), np.sum(b * b), a.size


def kappa_log_marginal(
    kappa_current,
    kappa_max,
    group_moments,
    sigma2_qx_aprior,
    sigma2_qx_bprior,
    sigma2_qx_max,
    kappa_aprior,
    kappa_bprior,
):
    """
    Log-posterior of a scalar kappa_x with sigma2_qx integrated out.

    Each entry of group_moments (from innovation_moments) belongs to one
    sigma2_qx group sharing this kappa. For a group with n innovations and
    an IG(alpha, beta) prior truncated to (0, sigma2_qx_max):

        log p(x | kappa) = -(alpha + n/2) * log(beta + S(kappa)/2)
                           + log F_IG(sigma2_qx_max; alpha + n/2, beta + S(kappa)/2)
                           + const

    The truncation term depends on kappa so is kept.
    """
    if kappa_current <= 0 or kappa_current >= kappa_max:
        return -np.inf

    loglik = 0.0

    for saa, sab, sbb, n in group_moments:
        S = saa - 2.0 * kappa_current * sab + kappa_current**2 * sbb

        shape = sigma2_qx_aprior + 0.5 * n
        scale = sigma2_qx_bprior + 0.5 * S

        loglik += -shape * np.log(scale) + invgamma.logcdf(sigma2_qx_max, a=shape, scale=scale)

    logprior = ((kappa_aprior - 1) * np.log(kappa_current) + (kappa_bprior - 1) * np.log(1 - kappa_current))

    return loglik + logprior


def sample_kappa_marginal(
    group_moments,
    kappa_current,
    kappa_max,
    kappa_aprior,
    kappa_bprior,
    sigma2_qx_aprior,
    sigma2_qx_bprior,
    sigma2_qx_max,
    rng=None,
):
    """
    Slice sampler for a scalar kappa_x from its marginal posterior (sigma2_qx integrated out).
    """

    def log_density(kappa):
        return kappa_log_marginal(kappa, kappa_max, group_moments, sigma2_qx_aprior, sigma2_qx_bprior, sigma2_qx_max, kappa_aprior, kappa_bprior)

    return slice_sample_kappa(log_density, kappa_current, kappa_max, rng=rng)


def update_kappa_x_marginal(zmusample,
                            zmusample_out,
                            zmusample_in,
                            nbasis,
                            nxout,
                            nxin,
                            kappa_x_aprior,
                            kappa_x_bprior,
                            kappa_x_max,
                            kappa_x_scheme,
                            kappa_x_vector_current,
                            sigma2_qx_aprior,
                            sigma2_qx_bprior,
                            sigma2_qx_max,
                            sigma_qx_scheme,
                            inner_group_id,
                            ningroup,
                            rng=None,
                            ):
    """
    Collapsed counterpart of update_kappa_x: samples kappa_x with sigma2_qx
    integrated out. Must be followed by update_sigma2_qx using the returned
    kappa_x so that the pair is a joint draw from p(kappa_x, sigma2_qx | x, r).

    Each kappa_x is paired with the sigma2_qx groups it drives:

        kappa_x scheme   sigma_qx scheme         groups per kappa_x
        global           global                  all basis functions
        inner outer      inner outer             outer | inner
        inner outer      inner outer country     outer | each inner country group
    """

    if kappa_x_scheme == "global" and nxout == 0:

        if sigma_qx_scheme != "global":
            raise ValueError(f"Collapsed kappa_x update with global kappa_x requires global sigma_qx, not {sigma_qx_scheme!r}.")

        moments = [innovation_moments(zmusample[:, :nbasis], zmusample[:, -1])]

        kappa_x_sample = np.full_like(kappa_x_vector_current, sample_kappa_marginal(moments, kappa_x_vector_current[0], kappa_x_max, kappa_x_aprior, kappa_x_bprior, sigma2_qx_aprior, sigma2_qx_bprior, sigma2_qx_max, rng=rng))

    elif kappa_x_scheme == "inner outer" and nxout > 0:

        xout, rout = zmusample_out[:, :nxout], zmusample_out[:, -1]
        xin, rin = zmusample_in[:, :nxin], zmusample_in[:, -1]

        outer_moments = [innovation_moments(xout, rout)]

        if sigma_qx_scheme == "inner outer":
            inner_moments = [innovation_moments(xin, rin)]

        elif sigma_qx_scheme == "inner outer country":
            inner_moments = [innovation_moments(xin[:, inner_group_id == g], rin) for g in range(ningroup)]

        else:
            raise ValueError(f"Collapsed kappa_x update with inner outer kappa_x requires 'inner outer' or 'inner outer country' sigma_qx, not {sigma_qx_scheme!r}.")

        kappa_x_sample = np.zeros_like(kappa_x_vector_current)

        kappa_x_sample[0] = sample_kappa_marginal(outer_moments, kappa_x_vector_current[0], kappa_x_max, kappa_x_aprior, kappa_x_bprior, sigma2_qx_aprior, sigma2_qx_bprior, sigma2_qx_max, rng=rng)
        kappa_x_sample[1] = sample_kappa_marginal(inner_moments, kappa_x_vector_current[1], kappa_x_max, kappa_x_aprior, kappa_x_bprior, sigma2_qx_aprior, sigma2_qx_bprior, sigma2_qx_max, rng=rng)

    else:

        raise ValueError(f"No collapsed kappa_x update for kappa_x scheme {kappa_x_scheme!r} with nxout={nxout}.")

    return kappa_x_sample
