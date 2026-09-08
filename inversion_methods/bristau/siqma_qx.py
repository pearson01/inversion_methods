import numpy as np

from scipy.stats import invgamma
from inversion_methods.bristau.data_bristau import inner_basis_country_groups


def sigma_qx_groups(bfds, cntryds, nxout, nbasis, sigma_qx, min_group_size=5):
    """
    Assign each basis function to a group based on the sigma_qx requirements of the inversion. 
    If sigma_qx is divided into countries, the function returns the country-based 
    group using a land country mask. Outer basis functions are intentionally left out of this grouping, 
    since they represent large aggregated INTEM regions rather than individual countries.

    If sigma_qx is not divided into countries, the function will check to see if inner outer groups are
    needed.

    Parameters
    ----------
    bfds : xr.DataArray
        Basis function definition field (1-indexed, as used elsewhere).
    cntryds : xr.Dataset
        Country definition dataset with a "country" grid on the same
        lat/lon as bfds.
    nxout : int
        Number of outer basis functions (excluded from grouping).
    nbasis : int
        Total number of basis functions.
    sigma_qx : str | float
        Either a fixed float or if a hyperparameter, sigma_qx will describe requirements
        "global", "inner outer", or "inner outer countries"
    min_group_size : int
        Countries with fewer than this many inner basis functions are
        merged into a single "other" group.

    Returns
    -------
    group_id : ndarray, shape (nxin,)
        Group index (0..ngroup-1) for each inner basis function, where
        nxin = nbasis - nxout.
    ngroup : int
        Number of distinct groups.
    """
    
    if type(sigma_qx) in [int, float]:

        return None, None
    
    elif type(sigma_qx) is str:

        if sigma_qx.lower().strip() == "inner outer":

            if nxout == 0 or nxout == None:

                raise ValueError(f"sigma_qx wants inner and outer groups, but nxout set to {nxout}. Check how many outer groups in fp_basis_case.")
            
            else:

                return None, None
            
        elif "country" in sigma_qx.lower().strip():

            return inner_basis_country_groups(bfds, cntryds, nxout, nbasis, min_group_size)
        
        else:

            raise ValueError(f"Give your head a wobble. sigma_qx is '{sigma_qx}', which makes no sense to me or you.")



def sigma_qx_scheme_select(config, nxin):

    # bool is a subclass of int, so exclude it explicitly.
    sigma_qx_is_fixed = (isinstance(config.sigma_qx, (int, float, np.integer, np.floating),) and not isinstance(config.sigma_qx, (bool, np.bool_)))

    if sigma_qx_is_fixed:
        sigma_qx_scheme = "fixed"
        fixed_sigma2_qx = float(config.sigma_qx)**2

        if not np.isfinite(fixed_sigma2_qx):
            raise ValueError("Fixed config.sigma_qx must be finite.")

        if fixed_sigma2_qx < 0:
            raise ValueError("Fixed config.sigma_qx must be non-negative.")

    elif isinstance(config.sigma_qx, str):
        # Normalise case and repeated whitespace.
        sigma_qx_scheme = " ".join(config.sigma_qx.lower().split())
        fixed_sigma2_qx = None

    else:
        raise TypeError("config.sigma_qx must be a numeric fixed value or one of 'global', 'inner outer', or 'inner outer country'.")

    valid_schemes = {"fixed", "global", "inner outer", "inner outer country",}

    if sigma_qx_scheme not in valid_schemes:
        raise ValueError(f"Unknown config.sigma_qx value: {config.sigma_qx!r}. Expected one of {sorted(valid_schemes)} or a numeric value.")

    if sigma_qx_scheme == "inner outer country":
        if getattr(config, "inner_group_id", None) is None:
            raise ValueError("config.inner_group_id is required when config.sigma_qx == 'inner outer country'")

        if getattr(config, "ningroup", None) is None:
            raise ValueError("config.ningroup is required when config.sigma_qx == 'inner outer country'.")

        if (not isinstance(config.ningroup, (int, np.integer)) or isinstance(config.ningroup, (bool, np.bool_))):
            raise TypeError("config.ningroup must be an integer when config.sigma_qx == 'inner outer country'.")

        ningroup = int(config.ningroup)

        if ningroup <= 0:
            raise ValueError("config.ningroup must be greater than zero when config.sigma_qx == 'inner outer country'; received {ningroup}.")

        inner_group_id = np.asarray(config.inner_group_id, dtype=int,)

        if inner_group_id.shape != (nxin,):
            raise ValueError(f"config.inner_group_id must have one entry per inner basis function. Expected shape ({nxin},), but received {inner_group_id.shape}.")

        expected_groups = np.arange(ningroup)
        actual_groups = np.unique(inner_group_id)

        if not np.array_equal(actual_groups, expected_groups):
            raise ValueError(f"config.inner_group_id must contain contiguous group IDs 0..{ningroup - 1}. Found {actual_groups}.")

    else:
        ningroup = None
        inner_group_id = None

    return sigma_qx_scheme, fixed_sigma2_qx, ningroup, inner_group_id


def initialise_sigma2_qx_vector(sigma_qx_scheme, fixed_sigma2_qx, initial_sigma2_qx, nbasis, nxout, nxin, ningroup, inner_group_id):

    if sigma_qx_scheme == "fixed":
        sigma2_qx_global_current = fixed_sigma2_qx

        sigma2_qx_bf_current = np.full(nbasis, sigma2_qx_global_current, dtype=float,)

    elif sigma_qx_scheme == "global":
        sigma2_qx_global_current = initial_sigma2_qx

        sigma2_qx_bf_current = np.full(nbasis, sigma2_qx_global_current, dtype=float,)

    elif sigma_qx_scheme == "inner outer":
        sigma2_qxout_current = initial_sigma2_qx
        sigma2_qxin_current = initial_sigma2_qx

        sigma2_qx_bf_current = np.concatenate(
            (np.full(nxout, sigma2_qxout_current, dtype=float,),np.full(nxin, sigma2_qxin_current, dtype=float,),)
            )

    elif sigma_qx_scheme == "inner outer country":
        sigma2_qxout_current = initial_sigma2_qx

        sigma2_qxin_group_current = np.full(ningroup, initial_sigma2_qx, dtype=float,)

        sigma2_qxin_per_bf = (sigma2_qxin_group_current[inner_group_id])

        sigma2_qx_bf_current = np.concatenate((np.full(nxout, sigma2_qxout_current, dtype=float,),sigma2_qxin_per_bf,))

    return sigma2_qx_bf_current


def sigma_qx_trace_params(sigma_qx_scheme, ningroup):

    if sigma_qx_scheme in {"fixed", "global"}:
        sigma2_qx_trace_labels = ["global"]
        n_sigma2_qx_parameters = 1

    elif sigma_qx_scheme == "inner outer":
        sigma2_qx_trace_labels = ["outer", "inner",]
        n_sigma2_qx_parameters = 2

    elif sigma_qx_scheme == "inner outer country":
        sigma2_qx_trace_labels = (["outer"] + [f"inner_group_{group_id}" for group_id in range(ningroup)])
        n_sigma2_qx_parameters = 1 + ningroup

    return sigma2_qx_trace_labels, n_sigma2_qx_parameters
    

def build_group_id_coordinate(nxout, nbasis, inner_group_id):
    """
    Build the "group_id" output coordinate labelling each basis function's
    sigma_qx country group (offset by 1, with 0 reserved for outer basis
    functions), or all zeros when inner_group_id isn't set.

    `inner_group_id` is only populated by sigma_qx_groups() when
    config.sigma_qx == "inner outer country" -- for every other scheme
    (including plain "inner outer") it's None regardless of nxout, in which
    case there's no meaningful country grouping to label.
    """
    if inner_group_id is None:
        return np.zeros(nbasis)

    if nxout > 0:
        return np.concatenate((np.zeros(nxout), inner_group_id + 1))

    return inner_group_id


def update_sigma2_qx(zmusample,
                     zmusample_out,
                     zmusample_in,   
                     sigma2_qx_aprior, 
                     sigma2_qx_bprior, 
                     nbasis, 
                     nxout, 
                     nxin, 
                     sigma2_qx_max, 
                     sigma_qx_scheme, 
                     sigma2_qx_bf_current, 
                     fixed_sigma2_qx, 
                     inner_group_id, 
                     ningroup,
                     kappa_x_vector_current,
                     n_kappa_x_parameters,
                     ):
    

    kappa_xin_current = kappa_x_vector_current[-1]
    
    if n_kappa_x_parameters == 2:

        kappa_xout_current = kappa_x_vector_current[0]

        if sigma_qx_scheme == "global":

            if not np.isclose(kappa_xout_current, kappa_xin_current,):
                raise ValueError("Global sigma2_qx sampling currently requires kappa_xout_current and kappa_xin_current to be equal.")
            
    elif n_kappa_x_parameters == 1:
        # A single fixed/global kappa_x applies equally to outer and inner
        # basis functions, so sigma_qx sampling (which may still be split
        # "inner outer"/"inner outer country") uses the same value for both.
        kappa_xout_current = kappa_xin_current

    else:
        raise ValueError("Currently kappa_x only setup for 1 or 2 parameters.")


    if sigma_qx_scheme == "fixed":
        # No sampling. The configured value is applied everywhere.
        sigma2_qx_global_current = fixed_sigma2_qx

        sigma2_qx_bf_current.fill(sigma2_qx_global_current)

        sigma2_qx_sample = (sigma2_qx_global_current)

    elif sigma_qx_scheme == "global":

        inner_deviations = (zmusample[:, nxout:nbasis,] - zmusample[:, -1, np.newaxis])
        zero_reference = np.zeros((zmusample.shape[0], 1,), dtype=float,)

        if nxout > 0:
            outer_deviations = (zmusample[:, :nxout] - zmusample[:, -2, np.newaxis])
            zmusample_global = np.column_stack((outer_deviations, inner_deviations, zero_reference,))
        else:
            zmusample_global = np.column_stack((inner_deviations, zero_reference,))

        sigma2_qx_global_current = sample_sigma2_qx(zmusample_global, kappa_xin_current, sigma2_qx_aprior, sigma2_qx_bprior, nbasis, sigma2_qx_max,)

        sigma2_qx_bf_current.fill(sigma2_qx_global_current)

        sigma2_qx_sample = (sigma2_qx_global_current)

    elif sigma_qx_scheme == "inner outer":
        sigma2_qxout_current = sample_sigma2_qx(zmusample_out, kappa_xout_current, sigma2_qx_aprior, sigma2_qx_bprior, nxout, sigma2_qx_max,)

        sigma2_qxin_current = sample_sigma2_qx(zmusample_in, kappa_xin_current, sigma2_qx_aprior, sigma2_qx_bprior, nxin, sigma2_qx_max,)

        sigma2_qx_bf_current[:nxout] = (sigma2_qxout_current)

        sigma2_qx_bf_current[nxout:] = (sigma2_qxin_current)

        sigma2_qx_sample = np.array([sigma2_qxout_current, sigma2_qxin_current,], dtype=float,)

    elif sigma_qx_scheme == "inner outer country":
        sigma2_qxout_current = sample_sigma2_qx(zmusample_out, kappa_xout_current, sigma2_qx_aprior, sigma2_qx_bprior, nxout, sigma2_qx_max,)

        (sigma2_qxin_per_bf, sigma2_qxin_group_current,) = sample_sigma2_qx_grouped(zmusample_in,
                                                                                    kappa_xin_current,
                                                                                    sigma2_qx_aprior,
                                                                                    sigma2_qx_bprior,
                                                                                    inner_group_id,
                                                                                    ningroup,
                                                                                    sigma2_qx_max,)

        sigma2_qxin_per_bf = np.asarray(sigma2_qxin_per_bf, dtype=float,)

        sigma2_qxin_group_current = np.asarray(sigma2_qxin_group_current, dtype=float,)

        sigma2_qx_bf_current[:nxout] = (sigma2_qxout_current)

        sigma2_qx_bf_current[nxout:] = (sigma2_qxin_per_bf)
        
        sigma2_qx_sample = np.concatenate((np.atleast_1d(sigma2_qxout_current), sigma2_qxin_group_current))

    return sigma2_qx_sample, sigma2_qx_bf_current



def sample_sigma2_qx(zmusample, kappa_x, alpha_prior, beta_prior, nbasis, sigma2_qx_max=0.5):

    """
    Conjugate update to generate samples from the truncated inverse-gamma posterior of the forecast model noise variance for the emission fluxes x.


    x_bc and x_r are treated as fixed currently and so this sampler only works for sigma2_qx.
    """

    xr_prev = zmusample[:-1, -1][:, None]
    x_pred = kappa_x * zmusample[:-1, :nbasis] + (1 - kappa_x) * xr_prev
    
    x_innov = zmusample[1:,:nbasis] - x_pred
    shape = alpha_prior + 0.5 * x_innov.size
    scale = beta_prior + 0.5 * np.sum(x_innov**2)

    F_max = invgamma.cdf(sigma2_qx_max, a=shape, scale=scale)
    sigma2_qx_sample = invgamma.ppf(np.random.uniform(0, F_max), a=shape, scale=scale)

    return float(sigma2_qx_sample)


def sample_sigma2_qx_grouped(zmusample_in, kappa_xin, alpha_prior, beta_prior, group_id, ngroup, sigma2_qx_max=0.5):

    """
    Group-wise conjugate update for sigma2_qxin.

    Each group (e.g. country) gets its own truncated inverse-gamma sample,
    computed only from the innovations of the inner basis functions
    belonging to that group. Groups are independent conditional on the
    shared kappa_xin.

    Parameters
    ----------
    zmusample_in : ndarray, shape (T, nxin + 1)
        Inner basis function samples with r_in in the final column,
        same layout as consumed by `sample_sigma2_qx`.
    group_id : ndarray, shape (nxin,)
        Group index (0..ngroup-1) for each inner basis function.

    Returns
    -------
    sigma2_qxin_per_bf : ndarray, shape (nxin,)
        Sampled sigma2_qx broadcast back out to each inner basis function.
    sigma2_qxin_group : ndarray, shape (ngroup,)
        Sampled sigma2_qx for each group.
    """
    sigma2_qxin_group = np.zeros(ngroup)

    for g in range(ngroup):
        cols = np.where(group_id == g)[0]
        sub = np.column_stack((zmusample_in[:, cols], zmusample_in[:, -1]))
        sigma2_qxin_group[g] = sample_sigma2_qx(sub, kappa_xin, alpha_prior, beta_prior, len(cols), sigma2_qx_max)

    sigma2_qxin_per_bf = sigma2_qxin_group[group_id]

    return sigma2_qxin_per_bf, sigma2_qxin_group