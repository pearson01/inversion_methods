import numpy as np


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
                   ):
    

    if kappa_x_scheme == "fixed":

        kappa_x_sample = np.full_like(kappa_x_vector_current, fixed_kappa_x)

    elif kappa_x_scheme == "global" and nxout == 0:

        kappa_x_sample = sample_kappa(zmusample, sigma2_qx_bf_current, kappa_x_vector_current, kappa_x_max, kappa_x_aprior, kappa_x_bprior, nxin)

    elif kappa_x_scheme == "inner outer" and nxout > 0:

        kappa_x_sample = np.zeros_like(kappa_x_vector_current)

        kappa_x_sample[0] = sample_kappa(zmusample_out, sigma2_qx_bf_current[:nxout], kappa_x_vector_current[0], kappa_x_max, kappa_x_aprior, kappa_x_bprior, nxout)
        kappa_x_sample[1] = sample_kappa(zmusample_in, sigma2_qx_bf_current[nxout:], kappa_x_vector_current[1], kappa_x_max, kappa_x_aprior, kappa_x_bprior, nxin)

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
    m=100
):
    """
    Slice sampler for a scalar kappa_x given x, r, and basis-specific sigma2_qx.
    """

    xmusample = zmusample[:, :nbasis]
    rmusample = zmusample[:, -1:]   # keep as (T,1) for broadcasting

    xmu_dev = xmusample - rmusample

    dev_prev = xmu_dev[:-1]
    dev_curr = xmu_dev[1:]

    logy = kappa_log_posterior(kappa_current, kappa_max, dev_curr, dev_prev, rmusample, sigma2_qx_bf_current, kappa_aprior, kappa_bprior) + np.log(np.random.rand())

    u = np.random.rand()
    L = kappa_current - w * u
    R = L + w

    L = max(L, 0.0)
    R = min(R, kappa_max)

    j = int(np.floor(m * np.random.rand()))
    k = (m - 1) - j

    while (
        j > 0 and
        L > 0.0 and
        kappa_log_posterior(
            L, kappa_max,
            dev_curr, dev_prev, rmusample,
            sigma2_qx_bf_current, kappa_aprior, kappa_bprior
        ) > logy
    ):
        L = max(L - w, 0.0)
        j -= 1

    while (
        k > 0 and
        R < kappa_max and
        kappa_log_posterior(
            R, kappa_max,
            dev_curr, dev_prev, rmusample,
            sigma2_qx_bf_current, kappa_aprior, kappa_bprior
        ) > logy
    ):
        R = min(R + w, kappa_max)
        k -= 1

    while True:
        kappa_new = np.random.uniform(L, R)

        if (
            kappa_log_posterior(
                kappa_new, kappa_max,
                dev_curr, dev_prev, rmusample,
                sigma2_qx_bf_current, kappa_aprior, kappa_bprior
            ) >= logy
        ):
            return kappa_new

        if kappa_new < kappa_current:
            L = kappa_new
        else:
            R = kappa_new

