import re
from pathlib import Path

import numpy as np
import xarray as xr
import arviz as az
from pandas import date_range, to_datetime
from dataclasses import dataclass

from inversion_methods.bristau.sigma_rep import update_sigma2_rep, sigma_rep_scheme_select
from inversion_methods.bristau.siqma_qx import update_sigma2_qx, sigma_qx_scheme_select, initialise_sigma2_qx_vector, sigma_qx_trace_params, build_group_id_coordinate
from inversion_methods.bristau.kappa_x import update_kappa_x, kappa_x_scheme_select, initialise_kappa_x_vector, kappa_x_trace_params, kappa_max
from inversion_methods.bristau.bristau_filter_sampler import iterative_augmented_mxkf, augmented_backward_sampler, amxkf_inputs
from inversion_methods.bristau.tau_resid import tau_scheme_select, tau_trace_params, initialise_tau, update_tau_resid, prepare_tau_resid_indexing, whiten_observations, prepare_tau_sampler_inputs, sample_tau

from openghg_inversions import utils, convert
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename


@dataclass
class MessyInput:
    start_date: str
    end_date: str
    Hx: np.ndarray
    Y: np.ndarray
    Ytime: np.ndarray
    sigma_obs: np.ndarray
    siteindicator: np.ndarray
    nxout: int
    Hbc: np.ndarray | None = None
    use_bc: bool = False


@dataclass
class InversionInput:
    Y_dic: dict
    sigma_obs_dic: dict
    Ytime_dic: dict
    Hz_dic: dict
    Hx_dic: dict
    Hbc_dic: dict
    siteindicator_dic: dict
    nperiod: int
    nbc: int | None
    nxout: int = 6
    nr: int = 2
    nbasis: int | None = None
    xprior: dict | None = None
    bcprior: dict | None = None
    rprior: dict | None = None
    sigma2_rep_prior: dict | None = None
    sigma2_qx_prior: dict | None = None
    sigma_qbc: float | None = None
    sigma_qr: float | None = None
    kappa_x_prior: dict | None = None
    iterations: int | None = 2500
    inner_group_id: np.ndarray | None = None
    ningroup: int | None = None
    sigma_rep: str | float | None = 'global additive'
    sigma_rep_max: float | None = 100.0
    sigma_qx: str | float = "inner outer"
    sigma_qx_max: float | None = 0.5
    kappa_x: str | float | None = 0.0
    kappa_x_minfold: int | None = 1
    kappa_bc: None = None
    kappa_r: float | None = 0.0
    tau_resid: str | float | None = 0.0
    tau_resid_prior: dict | None = None
    tau_resid_max: float | None = 200


def bristau_monthly_dictionaries(config: MessyInput) -> InversionInput:

    """
    Manipulates inversion data into dictionaries separated by month.
    """

    Y_dic = {}
    sigma_obs_dic = {}
    Ytime_dic = {}
    Hz_dic = {}
    Hx_dic = {}
    siteindicator_dic = {}
    allmonth = date_range(config.start_date, config.end_date, freq="MS")[:-1]
    Ymonth = to_datetime(config.Ytime).to_period("M")
    nperiod = len(allmonth)
    HxT = config.Hx.T

    if config.use_bc:
        nbc = config.Hbc.shape[0]
        # bc_count = np.arange(0, nbc, nperiod)
        Hbc_dic = {}
        HbcT = config.Hbc.T
    else:
        nbc = None
        Hbc_dic = None

    if config.nxout > 0:
        nr = 2
    else:
        nr = 1

    for period, timestamp in enumerate(allmonth):
        mnth = timestamp.month
        yr = timestamp.year
        mnthloc = np.where(np.logical_and(Ymonth.month == mnth, Ymonth.year == yr))[0]
        Y_dic[period] = config.Y[mnthloc]
        sigma_obs_dic[period] = config.sigma_obs[mnthloc]
        Ytime_dic[period] = config.Ytime[mnthloc]
        Hx_dic[period] = HxT[mnthloc, :]
        siteindicator_dic[period] = config.siteindicator[mnthloc]
        if config.use_bc:
            Hbc_dic[period] = HbcT[mnthloc, :]
            Hz_dic[period] = np.hstack((Hx_dic[period], Hbc_dic[period]))
        else:
            Hz_dic[period] = Hx_dic[period]
        
        Hr = np.zeros((len(Y_dic[period]), nr))
        Hz_dic[period] = np.hstack((Hz_dic[period], Hr))


    return InversionInput(Y_dic=Y_dic, 
                        sigma_obs_dic=sigma_obs_dic,
                        Ytime_dic=Ytime_dic, 
                        Hz_dic=Hz_dic,
                        Hx_dic=Hx_dic,
                        Hbc_dic=Hbc_dic,
                        siteindicator_dic=siteindicator_dic,
                        nperiod=nperiod,
                        nbc=nbc,
                        nxout=config.nxout,
                        nr=nr) 


def prior_parser(prior):

    if prior["pdf"] in ["normal", "lognormal"]:

        return prior["mu"], prior["sigma"]
    
    elif prior["pdf"] in ["beta", "inv-gamma", "inverse gamma"]:

        return prior["shape"], prior["scale"]

    else:

        raise ValueError(f"Your choice of prior is rather whack, check that ini file for {prior}")


def augmented_ffbs_mxkf_gibbs_double_slice(config: InversionInput):
    """
    Run the augmented FFBS/MXKF Gibbs sampler.

    Supported config.sigma_qx values
    --------------------------------
    "global"
        Sample one sigma2_qx shared by every basis function.

    "inner outer"
        Sample one sigma2_qx for all outer basis functions and one
        sigma2_qx for all inner basis functions.

    "inner outer country"
        Sample one sigma2_qx for all outer basis functions and group-wise
        sigma2_qx values for the inner basis functions.

    float or int
        Use config.sigma_qx as a fixed sigma2_qx value for every basis
        function. No sigma2_qx sampling is performed.

    Returns
    -------
    xtrace : ndarray
        Posterior samples for basis-function states. Shape:
        (retained_iterations, nperiod, nbasis).

    bctrace : ndarray or None
        Posterior samples for boundary-condition states. Shape:
        (retained_iterations, nperiod, nbc). Returns None when nbc is 0.

    rtrace : ndarray
        Posterior samples for reference states. Shape:
        (retained_iterations, nperiod, 2).

    var_rep_trace : ndarray
        Trace of sigma2_rep. Shape:
        (retained_iterations,).

    var_qx_trace : ndarray
        Trace of the sampled or fixed sigma2_qx parameters.

        Shapes by mode:

        fixed or global
            (retained_iterations, 1)

        inner outer
            (retained_iterations, 2), containing outer and inner values.

        inner outer country
            (retained_iterations, 1 + ningroup), containing the outer
            value followed by one value for each inner country group.

    kappaouttrace : ndarray
        Trace of outer kappa. Shape:
        (retained_iterations,).

    kappaintrace : ndarray
        Trace of inner kappa. Shape:
        (retained_iterations,).

    tau_trace : ndarray
        Trace of the AR(1)/OU same-site residual-correlation length (hours).
        Shape: (retained_iterations, 1). Fixed at config.tau_resid (0.0 by
        default, disabling the feature) when config.tau_resid is numeric.

    tau_trace_labels : list[str]
        Labels for the columns of tau_trace.
    """

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------

    if config.iterations > 1000:
        burn = 500
    else:
        burn = int(0.2 * config.iterations)

    nxout = config.nxout
    nxin = config.nbasis - nxout

    sigma_rep_scheme, fixed_sigma2_rep = sigma_rep_scheme_select(config)
    sigma_qx_scheme, fixed_sigma2_qx, ningroup, inner_group_id = sigma_qx_scheme_select(config, nxin)
    kappa_x_scheme, fixed_kappa_x = kappa_x_scheme_select(config)
    tau_scheme, fixed_tau = tau_scheme_select(config)

    sigma_obs = np.concatenate([config.sigma_obs_dic[t] for t in range(config.nperiod)])

    # ------------------------------------------------------------------
    # AR(1)/OU same-site residual-correlation setup
    #
    # This depends only on the fixed Y/Hz/Ytime/siteindicator data, not on
    # any Gibbs-sampled hyperparameter, so it is computed once here rather
    # than once per iteration. With tau_scheme == "fixed" and
    # config.tau_resid == 0.0 (the default), whiten_observations() below is
    # a no-op and reproduces the previous behaviour exactly.
    # ------------------------------------------------------------------

    (obs_prev_Y_dic, obs_prev_H_dic, obs_gap_dic, obs_has_prev_dic,
     obs_gap_flat, obs_prev_index_flat) = prepare_tau_resid_indexing(
        config.Y_dic, config.Hz_dic, config.Ytime_dic, config.siteindicator_dic, config.nperiod,
    )

    if tau_scheme != "fixed":
        tau_aprior, tau_bprior = prior_parser(config.tau_resid_prior)
        tau_max = config.tau_resid_max
    else:
        tau_aprior = None
        tau_bprior = None
        tau_max = None

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    xprior_mu, xprior_sigma = prior_parser(config.xprior)
    rprior_mu, rprior_sigma = prior_parser(config.rprior)

    if sigma_rep_scheme != "fixed additive":
        sigma2_rep_aprior, sigma2_rep_bprior = prior_parser(config.sigma2_rep_prior)
        sigma2_rep_max=config.sigma_rep_max**2
    else:
        sigma2_rep_aprior = None
        sigma2_rep_bprior = None
        sigma2_rep_max = None

    if sigma_qx_scheme != "fixed":
        sigma2_qx_aprior, sigma2_qx_bprior = prior_parser(config.sigma2_qx_prior)
        sigma2_qx_max=config.sigma_qx_max**2
    else:
        sigma2_qx_aprior = None
        sigma2_qx_bprior = None
        sigma2_qx_max = None


    if kappa_x_scheme != "fixed":
        kappa_x_aprior, kappa_x_bprior = prior_parser(config.kappa_x_prior)
        kappa_x_max = kappa_max(config.nperiod, config.kappa_x_minfold)
    else:
        kappa_x_aprior = None
        kappa_x_bprior = None
        kappa_x_max = None

    rprior_sigma2 = rprior_sigma**2
    rprior_sigma2s = np.full(2, rprior_sigma2, dtype=float,)

    xprior_mus = np.full(config.nbasis, xprior_mu, dtype=float,)

    rprior_mus = np.full(config.nr, rprior_mu, dtype=float,)

    # ------------------------------------------------------------------
    # Boundary-condition setup
    # ------------------------------------------------------------------

    if config.nbc:
        bcprior_mu, bcprior_sigma = prior_parser(config.bcprior)

        bcprior_mus = np.full(config.nbc, bcprior_mu, dtype=float,)

        bcprior_sigma2s = np.full(config.nbc, bcprior_sigma**2, dtype=float,)

        xprior_mus = np.concatenate((xprior_mus, bcprior_mus,))

        sigma2_qbcs = np.full(config.nbc, config.sigma_qbc**2, dtype=float,)

        bctrace = np.zeros((config.iterations, config.nperiod, config.nbc,), dtype=float,)

    else:
        bcprior_sigma2s = None
        sigma2_qbcs = None
        bctrace = None

    zprior_mus = np.concatenate((xprior_mus, rprior_mus,))

    # ------------------------------------------------------------------
    # Output traces
    # ------------------------------------------------------------------

    xtrace = np.zeros((config.iterations, config.nperiod, config.nbasis,), dtype=float,)

    rtrace = np.zeros((config.iterations, config.nperiod, config.nr,), dtype=float,)

    var_rep_trace = np.zeros(config.iterations, dtype=float,)

    sigma2_qx_trace_labels, n_sigma2_qx_parameters = sigma_qx_trace_params(sigma_qx_scheme, ningroup)

    sigma2_qx_trace = np.zeros((config.iterations, n_sigma2_qx_parameters,), dtype=float,)

    kappa_x_trace_labels, n_kappa_x_parameters = kappa_x_trace_params(kappa_x_scheme)

    kappa_x_trace = np.zeros((config.iterations, n_kappa_x_parameters), dtype=float,)

    tau_trace_labels, n_tau_parameters = tau_trace_params(tau_scheme)

    tau_trace = np.zeros((config.iterations, n_tau_parameters), dtype=float,)

    za_mu_prev = None

    # ------------------------------------------------------------------
    # Initial values
    # ------------------------------------------------------------------

    sigma2_rep_current = 400.0
    initial_sigma2_qx = 0.02

    sigma2_qx_bf_current = initialise_sigma2_qx_vector(sigma_qx_scheme, fixed_sigma2_qx, initial_sigma2_qx, config.nbasis, config.nxout, nxin, ningroup, inner_group_id)

    sigma2_qr = config.sigma_qr**2

    sigma2_qrs = np.full(config.nr, sigma2_qr, dtype=float,)

    initial_kappa_x = 0.5

    kappa_x_vector_current = initialise_kappa_x_vector(kappa_x_scheme, fixed_kappa_x, initial_kappa_x, n_kappa_x_parameters)

    initial_tau = 6.0

    tau_current = initialise_tau(tau_scheme, fixed_tau, initial_tau)

    # ------------------------------------------------------------------
    # Gibbs iterations
    # ------------------------------------------------------------------

    for i in range(config.iterations):

        print(
            f"Iteration: {i}, sigma2_rep: {sigma2_rep_current}, sigma2_qx min/max: ({sigma2_qx_bf_current.min()}, {sigma2_qx_bf_current.max()}), kappa_x: ({kappa_x_vector_current}), tau_resid: {tau_current}", flush=True,)

        # --------------------------------------------------------------
        # Construct initial-state prior variances
        # --------------------------------------------------------------

        sigma2_qxin_bf = sigma2_qx_bf_current[nxout:]

        xinprior_sigma2s = (sigma2_qxin_bf + sigma2_qr) / (1.0 - kappa_x_vector_current[-1]**2)
        xinprior_sigma2s = np.minimum(xinprior_sigma2s, xprior_sigma**2,)

        if nxout > 0:
            sigma2_qxout_bf = sigma2_qx_bf_current[:nxout]

            xoutprior_sigma2s = (sigma2_qxout_bf + sigma2_qr) / (1.0 - kappa_x_vector_current[0]**2)
            xoutprior_sigma2s = np.minimum(xoutprior_sigma2s, xprior_sigma**2,)
            xprior_sigma2s = np.concatenate((xoutprior_sigma2s, xinprior_sigma2s,))
            
        else:
            xprior_sigma2s = xinprior_sigma2s


        # --------------------------------------------------------------
        # Construct forecast covariance
        # --------------------------------------------------------------

        sigma2_qxs = sigma2_qx_bf_current.copy()

        if config.nbc:
            xprior_sigma2s = np.concatenate((xprior_sigma2s, bcprior_sigma2s,))

            sigma2_qxs = np.concatenate((sigma2_qxs, sigma2_qbcs,))

        zprior_sigma2s = np.concatenate((xprior_sigma2s, rprior_sigma2s,))

        # zprior_covariance = np.diag(zprior_sigma2s)

        sigma2_qzs = np.concatenate((sigma2_qxs, sigma2_qrs,))

        # F_aug = augmented_forecast_jacobian(kappa_xout_current, kappa_xin_current, config.nbasis, config.nbc, nxout,)

        # --------------------------------------------------------------
        # Forward filter
        # --------------------------------------------------------------

        # AR(1)/OU residual-correlation whitening. tau_current only changes
        # once per iteration (below), and Y_dic/Hz_dic never change, so this
        # is the only place this needs recomputing. A no-op (identical
        # Y/Hz, err_var == sigma2_rep + sigma_obs**2) whenever tau_current
        # is 0, i.e. whenever this feature isn't in use.
        Y_dic_whitened, Hz_dic_whitened, err_var_dic = whiten_observations(
            config.Y_dic, config.Hz_dic, config.sigma_obs_dic,
            obs_prev_Y_dic, obs_prev_H_dic, obs_gap_dic, obs_has_prev_dic,
            sigma2_rep_current, tau_current, config.nperiod,
        )

        filter_inputs = amxkf_inputs(
            Y_dic=Y_dic_whitened,
            sigma_obs_dic=config.sigma_obs_dic,
            Ytime_dic=config.Ytime_dic,
            Hz_dic=Hz_dic_whitened,
            siteindicator_dic=config.siteindicator_dic,
            nbasis=config.nbasis,
            zprior_mus=zprior_mus,
            zprior_sigma2s=zprior_sigma2s,
            forecast_noise=sigma2_qzs,
            nperiod=config.nperiod,
            sigma2_rep=sigma2_rep_current,
            kappa_x_vector=kappa_x_vector_current,
            xprior=config.xprior,
            bcprior=config.bcprior,
            rprior=config.rprior,
            nbc=config.nbc,
            nr=config.nr,
            nxout=nxout,
            za_mu_warmstart=za_mu_prev,
            err_var_dic=err_var_dic,
            Y_dic_raw=config.Y_dic,
            Hz_dic_raw=config.Hz_dic,
        )

        sampler_inputs = iterative_augmented_mxkf(filter_inputs)

        za_mu_prev = sampler_inputs.za_mu

        # --------------------------------------------------------------
        # Backward sampling
        # --------------------------------------------------------------

        (zmusample, zsample, state_residuals) = (augmented_backward_sampler(sampler_inputs))

        print(f"z sample: median: {np.median(zsample)}, mean: {np.mean(zsample)}, std: {np.std(zsample)}", flush=True,)

        xtrace[i] = zsample[:, :config.nbasis,]

        if config.nbc:
            bctrace[i] = zsample[:, config.nbasis:-config.nr,]

        rtrace[i] = zsample[:, -config.nr:,]

        # --------------------------------------------------------------
        # Update sigma2_rep
        # --------------------------------------------------------------
 
        sigma2_rep_current = update_sigma2_rep(state_residuals, sigma_obs, sigma2_rep_aprior, sigma2_rep_bprior, sigma2_rep_max, sigma_rep_scheme, sigma2_rep_current, fixed_sigma2_rep)

        var_rep_trace[i] = sigma2_rep_current

        # --------------------------------------------------------------
        # Update tau (AR(1)/OU same-site residual-correlation length)
        # --------------------------------------------------------------
        
        tau_current = update_tau_resid(state_residuals, sigma_obs, tau_aprior, tau_bprior, tau_max, tau_scheme, tau_current, fixed_tau, sigma2_rep_current, obs_prev_index_flat, obs_gap_flat)

        tau_trace[i] = tau_current

        # --------------------------------------------------------------
        # Optional state split
        # --------------------------------------------------------------

        if nxout > 0:
            zmusample_out = np.column_stack((zmusample[:, :nxout], zmusample[:,-2],))
            zmusample_in = np.column_stack((zmusample[:, nxout:config.nbasis,], zmusample[:,-1],))
        else:
            zmusample_out = None
            zmusample_in = None


        # --------------------------------------------------------------
        # Update sigma2_qx
        # --------------------------------------------------------------

        (sigma2_qx_trace[i], sigma2_qx_bf_current) = (
            update_sigma2_qx(zmusample, 
                             zmusample_out, 
                             zmusample_in, 
                             sigma2_qx_aprior, 
                             sigma2_qx_bprior, 
                             config.nbasis, 
                             nxout, 
                             nxin, 
                             sigma2_qx_max, 
                             sigma_qx_scheme, 
                             sigma2_qx_bf_current, 
                             fixed_sigma2_qx, 
                             inner_group_id, 
                             ningroup,
                             kappa_x_vector_current,
                             n_kappa_x_parameters,) 
                             )

        # --------------------------------------------------------------
        # Update kappa
        # --------------------------------------------------------------

        kappa_x_vector_current = update_kappa_x(zmusample, 
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
                                          fixed_kappa_x,)

        kappa_x_trace[i] = kappa_x_vector_current

    # ------------------------------------------------------------------
    # Remove burn-in and return
    # ------------------------------------------------------------------

    if config.nbc:
        retained_bctrace = bctrace[burn:]
    else:
        retained_bctrace = None

    return (
        xtrace[burn:],
        retained_bctrace,
        rtrace[burn:],
        var_rep_trace[burn:],
        sigma2_qx_trace[burn:],
        kappa_x_trace[burn:],
        sigma2_qx_trace_labels,
        kappa_x_trace_labels,
        tau_trace[burn:],
        tau_trace_labels,
    )


@dataclass
class PostProcessInput:
    xtrace: np.ndarray
    rtrace: np.ndarray
    var_rep_trace: np.ndarray
    sigma2_qx_trace: np.ndarray
    sigma2_qx_trace_labels: list[str]
    kappa_x_trace: np.ndarray
    kappa_x_trace_labels: list[str]
    xprior: dict
    rprior: dict
    Hx_dic: dict
    Y_dic: dict
    sigma_obs_dic: dict
    Ytime_dic: dict
    siteindicator_dic: dict
    domain: str
    species: str
    sites: list[str]
    start_date: str
    end_date: str
    outputname: str
    outputpath: str
    emissions_name: list[str]
    fp_data: dict
    country_file: str
    nbasis: int
    nperiod: int
    nxout: int
    nr: int
    inner_group_id: np.ndarray
    country_unit_prefix: str | None
    nbc: int | None = None
    bctrace: np.ndarray | None = None
    bcprior: dict | None = None
    Hbc_dic: dict | None = None
    ningroup: int | None = None
    tau_trace: np.ndarray | None = None
    tau_trace_labels: list[str] | None = None


def bristau_postprocessouts(config: PostProcessInput) -> xr.Dataset:
    r"""Takes the output from inferpymc function, along with some other input
    information, calculates statistics on them and places it all in a dataset.
    Also calculates statistics on posterior emissions for the countries in
    the inversion domain and saves all in netcdf.

    Note that the uncertainties are defined by the highest posterior
    density (HPD) region and NOT percentiles (as the tdMCMC code).
    The HPD region is defined, for probability content (1-a), as:
        1) P(x \in R | y) = (1-a)
        2) for x1 \in R and x2 \notin R, P(x1|y)>=P(x2|y)

    Args:
      xouts:
        MCMC chain for emissions scaling factors for each basis function.
      Hx:
        Transpose of the sensitivity matrix to map emissions to measurement.
        This is the same as what is given from fp_data[site].H.values, where
        fp_data is the output from e.g. footprint_data_merge, but where it
        has been stacked for all sites.
      Y:
        Measurement vector containing all measurements
      error:
        Measurement error vector, containg a value for each element of Y.
      Ytime:
        Time stamp of measurements as used by the inversion.
      siteindicator:
        Numerical indicator of which site the measurements belong to,
        same length at Y.
      domain:
        Inversion spatial domain.
      species:
        Species of interest
      sites:
        List of sites in inversion
      start_date:
        Start time of inversion "YYYY-mm-dd"
      end_date:
        End time of inversion "YYYY-mm-dd"
      outputname:
        Unique identifier for output/run name.
      outputpath:
        Path to where output should be saved.
      country_unit_prefix:
        A prefix for scaling the country emissions. Current options are:
        'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
        To add additional options add to acrg_convert.prefix
        Default is none and no scaling will be applied (output in g).
      emissions_name:
        List with "source" values as used when adding emissions data to the OpenGHG object store.
      bcouts:
        MCMC chain for boundary condition scaling factors.
      Hbc:
        Same as Hx but for boundary conditions
      obs_repeatability:
        Instrument error
      obs_variability:
        Error from resampling observations
      fp_data:
        Output from footprints_data_merge + sensitivies
      country_file:
        Path of country definition file
      use_bc:
        When True, use and infer boundary conditions
      x_freq:
        The period over which the emissions scalings are estimated. Set to "monthly"
        to estimate per calendar month; set to a number of days,
        as e.g. "30D" for 30 days; or set to None to estimate to have one
        scaling for the whole inversion period
      x_correlation:
        The exponential time constant representing the time at which the covariance 
        between period paramters is equal to 1/e. Units reflect the period chosen in x_freq.
    Returns:
        xarray dataset containing results from inversion

    """
    print("Post-processing analytical output")

    # Get parameters for output file
    Hx = np.vstack(list(config.Hx_dic.values()))
    Y = np.concatenate(list(config.Y_dic.values()))
    sigma_obs = np.concatenate(list(config.sigma_obs_dic.values()))
    Ytime = np.concatenate(list(config.Ytime_dic.values()))
    siteindicator = np.concatenate(list(config.siteindicator_dic.values()))
    Ymod_dic = {}

    ny = len(Y)
    nui = np.arange(2)
    nmeasure = np.arange(ny)
        
    if config.nbc:
        Hbc = np.vstack(list(config.Hbc_dic.values()))
        YaprioriBC = np.sum(Hbc, axis=1)
        Yapriori = np.sum(Hx, axis=1) + np.sum(Hbc, axis=1)
        Ymodbc_dic = {}
        bcouts_median = np.median(config.bctrace, axis=0)
        bcouts_68 = np.zeros((config.nperiod, len(nui), config.nbc))
        bcouts_95 = np.zeros((config.nperiod, len(nui), config.nbc))
    else:
        Yapriori = np.sum(Hx, axis=1)        

    sitenum = np.arange(len(config.sites))

    lon = config.fp_data[config.sites[0]].lon.values
    lat = config.fp_data[config.sites[0]].lat.values
    site_lat = np.zeros(len(config.sites))
    site_lon = np.zeros(len(config.sites))
    for si, site in enumerate(config.sites):
        site_lat[si] = config.fp_data[site].release_lat.values[0]
        site_lon[si] = config.fp_data[site].release_lon.values[0]
    bfds = config.fp_data[".basis"]

    # Calculate mean  and mode posterior scale map and flux field
    scalemap = []

    xouts_median = np.median(config.xtrace, axis=0)
    xouts_68 = np.zeros((config.nperiod, len(nui), config.nbasis))
    xouts_95 = np.zeros((config.nperiod, len(nui), config.nbasis))

    for period in np.arange(config.nperiod):

        Ymod_dic[period] = config.Hx_dic[period] @ xouts_median[period]

        if config.nbc:

            Ymodbc_dic[period] = config.Hbc_dic[period] @ bcouts_median[period]

            Ymod_dic[period] += Ymodbc_dic[period]

            for bc in np.arange(config.nbc):

                bcouts_68[period, :, bc] = az.hdi(config.bctrace[:, period, bc], 0.68)
                bcouts_95[period, :, bc] = az.hdi(config.bctrace[:, period, bc], 0.95)
        
            
        scalemap_single = np.zeros_like(bfds.values, dtype=float)

        for basis in np.arange(config.nbasis):

            xouts_68[period, :, basis] = az.hdi(config.xtrace[:, period, basis], 0.68)
            xouts_95[period, :, basis] = az.hdi(config.xtrace[:, period, basis], 0.95)
            scalemap_single[bfds.values == (basis + 1)] = xouts_median[period, basis]

        scalemap.append(scalemap_single)
    
    scalemap = np.stack(scalemap, axis=-1)

    emds = config.fp_data[".flux"][config.emissions_name[0]]
    flux_array_all = emds.data.flux.values

    # HACK: assume that smallest flux dim is time, then re-order flux so that
    # time is the last coordinate
    flux_dim_shape = flux_array_all.shape
    flux_dim_positions = range(len(flux_dim_shape))
    smallest_dim_position = min(list(zip(flux_dim_positions, flux_dim_shape)), key=(lambda x: x[1]))[0]

    flux_array_all = np.moveaxis(flux_array_all, smallest_dim_position, -1)
    # end HACK

    allmonth = date_range(config.start_date, config.end_date, freq="MS")[:-1]

    apriori_flux = np.zeros((*flux_array_all.shape[:2], config.nperiod))
    if flux_array_all.shape[2] == 1:
        print("\nAssuming flux prior is annual and using it for every period.")
        apriori_flux[:, :, :] = flux_array_all[:, :, [0]]
    else:
        print("\nAssuming flux prior is a calendar-month climatology.")
        for period, timestamp in enumerate(allmonth):
            apriori_flux[:, :, period] = flux_array_all[:, :, timestamp.month - 1]

    flux = np.zeros_like(scalemap)
    for period in np.arange(config.nperiod):
        flux[:, :, period] = scalemap[:,:,period] * apriori_flux[:,:,period]

    # Basis functions to save
    bfarray = bfds.values - 1

    # Calculate country totals
    area = utils.areagrid(lat, lon)

    c_object = utils.get_country(config.domain, country_file=config.country_file)
    cntryds = xr.Dataset(
        {"country": (["lat", "lon"], c_object.country), "name": (["ncountries"], c_object.name)},
        coords={"lat": (c_object.lat), "lon": (c_object.lon)},
    )
    cntrynames = cntryds.name.values
    cntrygrid = cntryds.country.values

    molarmass = convert.molar_mass(config.species)
    unit_factor = convert.prefix(config.country_unit_prefix)

    cntrymedian = np.zeros((len(cntrynames), config.nperiod))
    cntry68 = np.zeros((len(cntrynames), len(nui), config.nperiod))
    cntry95 = np.zeros((len(cntrynames), len(nui), config.nperiod))
    cntryprior = np.zeros((len(cntrynames), config.nperiod))

    if config.country_unit_prefix is None:
        config.country_unit_prefix = ""
    country_units = config.country_unit_prefix + "g"

    obs_units = str(config.fp_data[".units"])

    steps= len(config.var_rep_trace)

    for period in np.arange(config.nperiod):
        apriori_flux_period = apriori_flux[:,:,period]
        for ci, cntry in enumerate(cntrynames):
            cntrytottrace = np.zeros(steps)
            cntrytotprior = 0
            for bf in range(int(np.max(bfarray)) + 1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray == bf)

                weight_bf = (
                np.sum(area[bothinds].ravel() * apriori_flux_period[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                / unit_factor
                )

                cntrytottrace += weight_bf * config.xtrace[:, period, bf]
                cntrytotprior += weight_bf
            
            cntrymedian[ci, period] = np.median(cntrytottrace, axis=0)
            cntry68[ci, :, period] = az.hdi(cntrytottrace, 0.68)
            cntry95[ci, :, period] = az.hdi(cntrytottrace, 0.95)

            cntryprior[ci, period] = cntrytotprior
            
    Ymod = np.concatenate(list(Ymod_dic.values()))

    all_group_id = build_group_id_coordinate(config.nxout, config.nbasis, config.inner_group_id)


    # Make output netcdf file
    data_vars = {
        "Yobs": (["nmeasure"], Y),
        "sigma_obs": (["nmeasure"], sigma_obs),
        "Ytime": (["nmeasure"], Ytime),
        "Yapriori": (["nmeasure"], Yapriori),
        "Ymod": (["nmeasure"], Ymod),
        "xtrace": (["stepnum", "period", "bf"], config.xtrace),
        "xouts_median": (["period", "bf"], xouts_median),
        "xouts_68": (["period", "nUI", "bf"], xouts_68),
        "xouts_95": (["period", "nUI", "bf"], xouts_95),
        "rtrace": (["stepnum", "period", "r"], config.rtrace),
        "var_rep_trace": (["stepnum"], config.var_rep_trace),
        "sigma2_qx_trace": (["stepnum", "gqx"], config.sigma2_qx_trace),
        "kappa_x_trace": (["stepnum", "gk"], config.kappa_x_trace),
        "siteindicator": (["nmeasure"], siteindicator),
        "sitenames": (["nsite"], config.sites),
        "sitelons": (["nsite"], site_lon),
        "sitelats": (["nsite"], site_lat),
        "fluxapriori": (["lat", "lon", "period"], apriori_flux),
        "basisfunctions": (["lat", "lon"], bfarray),
        "countrydefinition": (["lat", "lon"], cntrygrid),
        "xsensitivity": (["nmeasure", "bf"], Hx),
        "flux": (["lat", "lon", "period"], flux),
        "scaling": (["lat", "lon", "period"], scalemap),
        "countrymedian": (["countrynames", "period"], cntrymedian),
        "countryapriori": (["countrynames", "period"], cntryprior),
        "country68": (["countrynames", "nUI", "period"], cntry68),
        "country95": (["countrynames", "nUI", "period"], cntry95)
    }

    stepnum = np.arange(steps)
    numbf = np.arange(config.nbasis)
    numr = np.arange(config.nr)

    coords = {
        "stepnum": ("stepnum", stepnum),
        "numbf": ("numbf", numbf),
        "group_id": ("numbf", all_group_id),
        "numr": ("numr", numr),
        "qxgroup": ("gqx", config.sigma2_qx_trace_labels),
        "kgroup": ("gk", config.kappa_x_trace_labels),
        "nmeasure": ("nmeasure", nmeasure),
        "nUI": ("nUI", nui),
        "nsite": ("nsite", sitenum),
        "lat": ("lat", lat),
        "lon": ("lon", lon),
        "countrynames": ("countrynames", cntrynames),
        "periodstart": ("period", date_range(to_datetime(config.start_date), to_datetime(config.end_date), freq="MS")[:-1]),
        }

    if config.tau_trace is not None:
        data_vars["tau_trace"] = (["stepnum", "gtau"], config.tau_trace)
        coords["taugroup"] = ("gtau", config.tau_trace_labels)

    if config.nbc:

        Ymodbc = np.concatenate(list(Ymodbc_dic.values()))
        numbc = np.arange(config.nbc)

        coords["numbc"] = (["bc"], numbc)
        data_vars.update({
            "YaprioriBC": (["nmeasure"], YaprioriBC),
            "YmodBC": (["nmeasure"], Ymodbc),
            "bctrace": (["stepnum", "period", "bc"], config.bctrace),
            "bcouts_median": (["period", "bc"], bcouts_median),
            "bcouts_68": (["period", "nUI", "bc"], bcouts_68),
            "bcouts_95": (["period", "nUI", "bc"], bcouts_95),
            "bcsensitivity": (["nmeasure", "bc"], Hbc),
        })

    outds = xr.Dataset(data_vars, coords=coords)

    outds.flux.attrs["units"] = "mol/m2/s"
    outds.fluxapriori.attrs["units"] = "mol/m2/s"
    outds.Yobs.attrs["units"] = obs_units + " " + "mol/mol"
    outds.sigma_obs.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yapriori.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod.attrs["units"] = obs_units + " " + "mol/mol"
    outds.countrymedian.attrs["units"] = country_units
    outds.countryapriori.attrs["units"] = country_units

    outds.Yobs.attrs["longname"] = "observations"
    outds.sigma_obs.attrs["longname"] = "measurement error"
    outds.Ytime.attrs["longname"] = "time of measurements"
    outds.Yapriori.attrs["longname"] = "a priori simulated measurements"
    outds.Ymod.attrs["longname"] = "Analytical posterior simulated measurements"
    outds.siteindicator.attrs["longname"] = "index of site of measurement corresponding to sitenames"
    outds.sitenames.attrs["longname"] = "site names"
    outds.sitelons.attrs["longname"] = "site longitudes corresponding to site names"
    outds.sitelats.attrs["longname"] = "site latitudes corresponding to site names"
    outds.fluxapriori.attrs["longname"] = "mean a priori flux over period"
    outds.flux.attrs["longname"] = "Mean posterior flux over period"
    outds.scaling.attrs["longname"] = f"Mean scaling factor field over period"
    outds.basisfunctions.attrs["longname"] = "basis function field"
    outds.countrymedian.attrs["longname"] = "median of ocean and country totals"
    outds.countryapriori.attrs["longname"] = "prior mean of ocean and country totals"
    outds.countrydefinition.attrs["longname"] = "grid definition of countries"
    outds.xsensitivity.attrs["longname"] = "emissions sensitivity timeseries"
    outds.rtrace.attrs["longname"] = "state relaxation term trace"
    outds.var_rep_trace.attrs["longname"] = "representation error variance trace"
    outds.sigma2_qx_trace.attrs["longname"] = "trace for the state forecast model error variance of each bf group"
    outds.kappa_x_trace.attrs["longname"] = "trace for the state persistance terms"

    if config.tau_trace is not None:
        outds.tau_trace.attrs["longname"] = "trace for the AR(1)/OU same-site residual-correlation length"
        outds.tau_trace.attrs["units"] = "hours"

    if config.nbc:
        outds.YmodBC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.YaprioriBC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.bcsensitivity.attrs["units"] = obs_units + " " + "mol/mol"

        outds.YaprioriBC.attrs["longname"] = "a priori simulated boundary conditions"
        outds.YmodBC.attrs["longname"] = "mean of posterior simulated boundary conditions"
        outds.bcsensitivity.attrs["longname"] = "boundary conditions sensitivity timeseries"
        
    outds.attrs["Start date"] = config.start_date
    outds.attrs["End date"] = config.end_date

    # variables with variable length data types shouldn't be compressed
    # e.g. object ("O") or unicode ("U") type
    do_not_compress = []
    dtype_pat = re.compile(r"[<>=]?[UO]")  # regex for Unicode and Object dtypes
    for dv in outds.data_vars:
        if dtype_pat.match(outds[dv].data.dtype.str):
            do_not_compress.append(dv)

    # setting compression levels for data vars in outds
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in outds.data_vars if var not in do_not_compress}

    output_filename = define_output_filename(config.outputpath, config.species, config.domain, config.outputname, config.start_date, ext=".nc")
    Path(config.outputpath).mkdir(parents=True, exist_ok=True)
    outds.to_netcdf(output_filename, encoding=encoding, mode="w")

    return outds