import re
from pathlib import Path

import numpy as np
import xarray as xr
import arviz as az
from pandas import date_range, to_datetime
from dataclasses import dataclass


from inversion_methods.haffbs.affbs import augmented_mxkf, augmented_backward_sampler, amxkf_inputs, build_Pz
from inversion_methods.haffbs.hierarchical_samplers import sample_sigma2_rep, sample_sigma2_qx, sample_kappa, kappa_max

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
    nbasis: int
    xprior: dict
    bcprior: dict
    rprior: dict
    sigma2_rep_prior: dict
    sigma2_qx_prior: dict
    sigma_qbc: float
    sigma_qr: float    
    kappa_x_prior: dict
    iterations: int
    Hbc: np.ndarray | None = None


@dataclass
class InversionInput:
    Y_dic: dict
    sigma_obs_dic: dict
    Ytime_dic: dict
    Hz_dic: dict
    Hx_dic: dict
    Hbc_dic: dict
    siteindicator_dic: dict
    nbasis: int
    nperiod: int
    xprior: dict
    bcprior: dict
    rprior: dict
    sigma2_rep_prior: dict
    sigma2_qx_prior: dict
    sigma_qbc: float
    sigma_qr: float
    kappa_x_prior: dict
    iterations: int
    nbc: int | None = None



def haffbs_monthly_dictionaries(config: MessyInput) -> InversionInput:

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

    if config.bcprior["pdf"]:
        nbc = config.Hbc.shape[0]
        # bc_count = np.arange(0, nbc, nperiod)
        Hbc_dic = {}

    else:
        nbc = None
        Hbc_dic = None

    for period in range(nperiod):
        mnth = allmonth[period].month
        yr = allmonth[period].year
        mnthloc = np.where(np.logical_and(Ymonth.month == mnth, Ymonth.year == yr))[0]
        Y_dic[period] = config.Y[mnthloc]
        sigma_obs_dic[period] = config.sigma_obs[mnthloc]
        Ytime_dic[period] = config.Ytime[mnthloc]
        Hx_dic[period] = config.Hx.T[mnthloc, :]
        siteindicator_dic[period] = config.siteindicator[mnthloc]
        if nbc:
            # Hbc_dic[period] = config.Hbc.T[np.ix_(mnthloc, bc_count)]
            Hbc_dic[period] = config.Hbc.T[mnthloc, :]
            Hz_dic[period] = np.hstack((Hx_dic[period], Hbc_dic[period]))
            # bc_count += 1
        else:
            Hz_dic[period] = Hx_dic[period]
        
        Hr = np.zeros((len(Y_dic[period]), 1))
        Hz_dic[period] = np.hstack((Hz_dic[period], Hr))


    return InversionInput(Y_dic=Y_dic, 
                        sigma_obs_dic=sigma_obs_dic,
                        Ytime_dic=Ytime_dic, 
                        Hz_dic=Hz_dic,
                        Hx_dic=Hx_dic,
                        Hbc_dic=Hbc_dic,
                        siteindicator_dic=siteindicator_dic,
                        nbasis=config.nbasis, 
                        nperiod=nperiod, 
                        xprior=config.xprior,
                        bcprior=config.bcprior,
                        rprior=config.rprior,
                        sigma2_rep_prior=config.sigma2_rep_prior,
                        sigma2_qx_prior=config.sigma2_qx_prior,
                        kappa_x_prior=config.kappa_x_prior,
                        sigma_qbc=config.sigma_qbc,
                        sigma_qr=config.sigma_qr,
                        iterations=config.iterations,
                        nbc=nbc,
                        )


def prior_parser(prior):

    if prior["pdf"] in ["normal", "lognormal"]:

        return prior["mu"], prior["sigma"]
    
    elif prior["pdf"] in ["beta", "inv-gamma", "inverse gamma"]:

        return prior["shape"], prior["scale"]

    else:

        raise ValueError(f"Your choice of prior is rather whack, check that ini file for {prior}")



def augmented_ffbs_mxkf_gibbs_double_slice(config: InversionInput):

    burn = int(0.2 * config.iterations)

    xprior_mu, xprior_sigma = prior_parser(config.xprior)

    rprior_mu, rprior_sigma = prior_parser(config.rprior)

    sigma2_rep_aprior, sigma2_rep_bprior = prior_parser(config.sigma2_rep_prior)
    
    sigma2_qx_aprior, sigma2_qx_bprior = prior_parser(config.sigma2_qx_prior)

    kappa_x_aprior, kappa_x_bprior = prior_parser(config.kappa_x_prior)

    rprior_sigma2 = rprior_sigma**2
    rprior_sigma2s = np.ones(1) * rprior_sigma2

    xprior_mus = np.ones(config.nbasis) * xprior_mu
    rprior_mus = np.ones(1) * rprior_mu

    if config.nbc:
        bcprior_mu, bcprior_sigma = prior_parser(config.bcprior)
        bcprior_mus = np.ones(config.nbc) * bcprior_mu
        bcprior_sigma2 = bcprior_sigma**2
        bcprior_sigma2s = np.ones(config.nbc) * bcprior_sigma2
        xprior_mus = np.concatenate((xprior_mus, bcprior_mus))     

        sigma2_qbc = config.sigma_qbc**2
        sigma2_qbcs = np.ones(config.nbc) * sigma2_qbc

        bctrace = np.zeros((config.iterations, config.nperiod, config.nbc))
    
    else:

        bctrace = None
   

    zprior_mus = np.concatenate((xprior_mus, rprior_mus))

    sigma_obs = np.concatenate(list(config.sigma_obs_dic.values()))

    # Initial guesses
    sigma2_rep_current = 1.01
    
    sigma2_qx_current = 0.2

    sigma2_qr = config.sigma_qr**2
    sigma2_qrs = np.ones(1) * sigma2_qr

    kappa_x_current = 0.5
    kappa_x_max = kappa_max(config.nperiod, 1)

    xtrace = np.zeros((config.iterations, config.nperiod,  config.nbasis))
    rtrace = np.zeros((config.iterations, config.nperiod, 1))
    var_rep_trace = np.zeros(config.iterations)
    var_qx_trace = np.zeros(config.iterations)
    kappatrace = np.zeros(config.iterations)


    for i in range(config.iterations):

        # if i > 0 and i % 100 == 0:
        #     print(f"Gibbs iteration {i}", flush=True)
        
        # xprior_sigma2 = (sigma2_qx_current / (1-kappa_x_current**2))
        # xprior_sigma2s = np.ones(config.nbasis) * xprior_sigma2
        # xprior_sigma2s = np.ones(config.nbasis) * xprior_sigma**2
        
        sigma2_qxs = np.ones(config.nbasis) * sigma2_qx_current

        if config.nbc:
            # xprior_sigma2s = np.concatenate((xprior_sigma2s, bcprior_sigma2s))
            sigma2_qxs = np.concatenate((sigma2_qxs, sigma2_qbcs))

        
        # zprior_sigma2s = np.concatenate((xprior_sigma2s, rprior_sigma2s))
        # zprior_covariance = np.diag(zprior_sigma2s)
        
        zprior_covariance = build_Pz(rprior_sigma2, bcprior_sigma2, sigma2_qx_current, sigma2_qr, kappa_x_current, config.nbasis, config.nbc)

        sigma2_qzs = np.concatenate((sigma2_qxs, sigma2_qrs))   # forecast noise all centred on zero

        filter_inputs = amxkf_inputs(Y_dic=config.Y_dic,
                                     sigma_obs_dic=config.sigma_obs_dic,
                                     Ytime_dic=config.Ytime_dic,
                                     Hz_dic=config.Hz_dic,
                                     siteindicator_dic=config.siteindicator_dic,
                                     nbasis=config.nbasis,
                                     zprior_mus=zprior_mus,
                                     zprior_covariance=zprior_covariance,
                                     forecast_noise=sigma2_qzs,
                                     nperiod=config.nperiod,
                                     sigma_rep=sigma2_rep_current**0.5,
                                     kappa_x=kappa_x_current,
                                     xprior=config.xprior,
                                     bcprior=config.bcprior,
                                     rprior=config.rprior,
                                     nbc=config.nbc)


        sampler_inputs = augmented_mxkf(filter_inputs)

        # zmusample, zsample = augmented_sampler(za_mu, Pa, zf_mu, Pf, F, nt)
        zmusample, zsample, state_residuals = augmented_backward_sampler(sampler_inputs)




        # # --- DIAGNOSTICS START (temporary, remove after debug) ---
        # try:
        #     print("---DIAG--- iter", i, flush=True)
        #     # sigma_obs summary (global)
        #     print("sigma_obs min/med/max:", np.nanmin(sigma_obs), np.nanmedian(sigma_obs), np.nanmax(sigma_obs), flush=True)

        #     # zmusample / zsample summaries (pre/post-transform)
        #     if zmusample is not None:
        #         zmus_x = zmusample[:, :config.nbasis].ravel()
        #         print("zmusample bf percentiles (1/50/99):", np.percentile(zmus_x, [1,50,99]), flush=True)
        #     if zsample is not None:
        #         zsample_x = zsample[:, :config.nbasis].ravel()
        #         print("zsample bf percentiles (1/50/99):", np.percentile(zsample_x, [1,50,99]), flush=True)

        #     # state residuals
        #     if state_residuals is not None and len(state_residuals) > 0:
        #         print("state_residuals percentiles (1/50/99):", np.percentile(state_residuals, [1,50,99]), flush=True)
        #         print("state_residuals any NaN/inf:", np.isnan(state_residuals).any(), np.isinf(state_residuals).any(), flush=True)

        #     # Pf conditioning (if available on sampler_inputs)
        #     if hasattr(sampler_inputs, "Pf") and sampler_inputs.Pf is not None:
        #         try:
        #             eigs = np.linalg.eigvalsh(sampler_inputs.Pf[-1])
        #             print("Pf[-1] eig min/max:", eigs[0], eigs[-1], flush=True)
        #         except Exception as _e:
        #             print("Pf eigvals error:", _e, flush=True)

        #     # hyperparameters
        #     print("hyperparams sigma2_qx_current, kappa_x_current:", sigma2_qx_current, kappa_x_current, flush=True)

        #     # quick derived prior-variance check (if you use AR(1) form)
        #     try:
        #         denom = (1.0 - kappa_x_current**2)
        #         print("derived xprior_sigma2 (if AR1):", sigma2_qx_current / denom if denom > 0 else np.inf, flush=True)
        #     except Exception as _e:
        #         print("derived xprior_sigma2 error:", _e, flush=True)

        # except Exception as e:
        #     print("DIAGNOSTIC ERROR:", e, flush=True)
        # # --- DIAGNOSTICS END ---



        xtrace[i] = zsample[:,:config.nbasis]
        
        if config.nbc:
            bctrace[i] = zsample[:,config.nbasis:-1]

        rtrace[i] = zsample[:,-1:]

        print(f"Iteration:{i}, Current sigma2_rep: {sigma2_rep_current}", flush=True)

        sigma2_rep_current = sample_sigma2_rep(sigma2_rep_current, state_residuals**2, sigma_obs, sigma2_rep_aprior, sigma2_rep_bprior)
        
        var_rep_trace[i] = sigma2_rep_current

        sigma2_qx_current = sample_sigma2_qx(zmusample, kappa_x_current, sigma2_qx_aprior, sigma2_qx_bprior, config.nbasis)

        var_qx_trace[i] = (np.exp(sigma2_qx_current) - 1) * np.exp(sigma2_qx_current)
        # var_qx_trace[i] = omega_sig2_current

        kappa_x_current = sample_kappa(zmusample, sigma2_qx_current, kappa_x_current, kappa_x_max, kappa_x_aprior, kappa_x_bprior, config.nbasis)

        kappatrace[i] = kappa_x_current

    return xtrace[burn:], bctrace[burn:], rtrace[burn:], var_rep_trace[burn:], var_qx_trace[burn:], kappatrace[burn:]


@dataclass
class PostProcessInput:
    xtrace: np.ndarray
    rtrace: np.ndarray
    var_rep_trace: np.ndarray
    var_qx_trace: np.ndarray
    kappatrace: np.ndarray
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
    emissions_name: str
    fp_data: dict
    country_file: str
    nbasis: int
    nperiod: int
    country_unit_prefix: str | None
    nbc: int | None = None        
    bctrace: np.ndarray | None = None
    bcprior: dict | None = None
    Hbc_dic: dict | None = None



def haffbs_postprocessouts(config: PostProcessInput) -> xr.Dataset:
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
        
            
        scalemap_single = np.zeros_like(bfds.values)

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

    if flux_array_all.shape[2] == 1:
        print("\nAssuming flux prior is annual and extracting first index of flux array.")
        apriori_flux = flux_array_all[:, :, 0]
    else:
        print("\nAssuming flux prior is monthly.")
        print(f"Extracting weighted average flux prior from {config.start_date} to {config.end_date}")
        allmonths = date_range(config.start_date, config.end_date).month[:-1].values
        allmonths -= 1  # to align with zero indexed array

        apriori_flux = np.zeros_like(flux_array_all[:, :, 0])

        # calculate the weighted average flux across the whole inversion period
        for m in np.unique(allmonths):
            apriori_flux += flux_array_all[:, :, m] * np.sum(allmonths == m) / len(allmonths)

    flux = np.zeros_like(scalemap)
    for period in np.arange(config.nperiod):
        flux[:, :, period] = scalemap[:, :, period] * apriori_flux

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

    steps = len(config.var_rep_trace)

    for period in np.arange(config.nperiod):
        
        for ci, cntry in enumerate(cntrynames):
            cntrytottrace = np.zeros(steps)
            cntrytotprior = 0
            for bf in range(int(np.max(bfarray)) + 1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray == bf)
                cntrytottrace += (
                    np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    * config.xtrace[:, period, bf]
                    / unit_factor
                )
                cntrytotprior += (
                    np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    / unit_factor
                )
            cntrymedian[ci, period] = np.median(cntrytottrace)

            cntry68[ci, :, period] = az.hdi(cntrytottrace, 0.68)
            cntry95[ci, :, period] = az.hdi(cntrytottrace, 0.95)
            cntryprior[ci, period] = cntrytotprior
            
    Ymod = np.concatenate(list(Ymod_dic.values()))

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
        "var_qx_trace": (["stepnum"], config.var_qx_trace),
        "kappatrace": (["stepnum"], config.kappatrace),
        "siteindicator": (["nmeasure"], siteindicator),
        "sitenames": (["nsite"], config.sites),
        "sitelons": (["nsite"], site_lon),
        "sitelats": (["nsite"], site_lat),
        "fluxapriori": (["lat", "lon"], apriori_flux),
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
    numr = np.arange(1)

    coords = {
        "stepnum": (["steps"], stepnum),
        "numbf": (["bf"], numbf),
        "numr": (["r"], numr),
        "measurenum": (["nmeasure"], nmeasure),
        "UInum": (["nUI"], nui),
        "nsites": (["nsite"], sitenum),
        "lat": (["lat"], lat),
        "lon": (["lon"], lon),
        "countrynames": (["countrynames"], cntrynames),
        "periodstart": (["period"], date_range(to_datetime(config.start_date), to_datetime(config.end_date), freq="MS")[:-1])
    }

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
    outds.var_qx_trace.attrs["longname"] = "state forecast model error variance trace"
    outds.kappatrace.attrs["longname"] = "state persistance term trace"


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