import re
from pathlib import Path

import numpy as np
import xarray as xr
import arviz as az
from pandas import date_range, to_datetime
from dataclasses import dataclass
from scipy.linalg import cholesky

from inversion_methods.haffbs.affbs import slow_augmented_forecast_model, augmented_forecast_innovations, augmented_analysis_update, augmented_forecast_jacobian
from inversion_methods.manipulation.lognormal_transformations import state_vector_mu_transform, build_Wb, state_percentiles

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
    sigma_rep: dict
    sigma_qx: dict
    sigma_qbc: float
    sigma_qr: float    
    kappa_x: dict
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
    nbasis: int
    nperiod: int
    xprior: dict
    bcprior: dict
    rprior: dict
    sigma_rep: dict
    sigma_qx: dict
    sigma_qbc: float
    sigma_qr: float
    kappa_x: dict
    nbc: int | None = None



def amxkf_monthly_dictionaries(config: MessyInput) -> InversionInput:

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

    if config.use_bc:
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
        if config.use_bc:
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
                        sigma_rep=config.sigma_rep,
                        sigma_qx=config.sigma_qx,
                        kappa_x=config.kappa_x,
                        sigma_qbc=config.sigma_qbc,
                        sigma_qr=config.sigma_qr,
                        nbc=nbc,
                        )



def prior_parser(prior):

    if prior["pdf"] in ["normal", "lognormal"]:

        return prior["mu"], prior["sigma"]

    else:

        raise ValueError(f"Your choice of prior is rather whack, check that ini file for {prior}")




def augmented_mxkf(config: InversionInput):
        

    nr = 1 # currently only 1 global relaxation term
    nx = config.nbasis
    
    xprior_mus = np.ones(nx) * config.xprior["mu"]
    xprior_sigma2s = np.ones(nx) * config.xprior["sigma"]**2

    sigma2_qxs = np.ones(nx) * config.sigma_qx**2

    if config.nbc:
        nz = nx + config.nbc + nr
        bcprior_mus = np.ones(config.nbc) * config.bcprior["mu"]
        bcprior_sigma2s = np.ones(config.nbc) * config.bcprior["sigma"]**2
        xprior_mus = np.concatenate((xprior_mus, bcprior_mus))
        xprior_sigma2s = np.concatenate((xprior_sigma2s, bcprior_sigma2s))

        sigma2_qbcs = np.ones(config.nbc) * config.sigma_qbc**2
        sigma2_qxs = np.concatenate((sigma2_qxs, sigma2_qbcs))

    else:
        nz = nx + nr

    rprior_mus = np.ones(1) * config.rprior["mu"]
    rprior_sigma2s = np.ones(1) * config.rprior["sigma"]**2

    sigma2_qrs = np.ones(1) * config.sigma_qr**2
    sigma2_qxs = np.concatenate((sigma2_qxs, sigma2_qrs))

    zprior_mus = np.concatenate((xprior_mus, rprior_mus))
    zprior_covariance = np.diag(np.concatenate((xprior_sigma2s, rprior_sigma2s)))

    za_mu = np.zeros((config.nperiod, nz))
    za = np.zeros((config.nperiod, nz))
    Pa = np.zeros((config.nperiod, nz, nz))

    Q_aug = np.diag(sigma2_qxs)

    F_aug = augmented_forecast_jacobian(config.kappa_x, config.nbasis, config.nbc)

    for t in range(config.nperiod):
        H = config.Hz_dic[t]
        Y = config.Y_dic[t]
        sigma_obs = config.sigma_obs_dic[t]

        ny = len(Y)

        if t == 0:
            za_mu[-1] = zprior_mus
            Pa[-1] = zprior_covariance

        # zf_mu[t], Pf[t] = augmented_forecast_model(za_mu[t-1], Pa[t-1], config.kappa_x, config.forecast_noise, config.nbasis, config.nbc)
        zf_mu, Pf = slow_augmented_forecast_model(za_mu[t-1], Pa[t-1], F_aug, Q_aug)

        zf = state_vector_mu_transform(zf_mu, config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)

        Wb = build_Wb(zf, config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)
        Wo_inv = np.eye(ny)
        H_hat = Wo_inv @ H @ Wb

        K, d = augmented_forecast_innovations(Y, sigma_obs, config.sigma_rep**2, zf, Pf, H, H_hat)
        
        za_mu[t], Pa[t] = augmented_analysis_update(zf_mu, Pf, K, H_hat, d, config.sigma_rep**2, sigma_obs)

        za[t] = state_vector_mu_transform(za_mu[t], config.xprior, config.bcprior, config.rprior, config.nbasis, config.nbc)


    return za, za_mu, Pa


@dataclass
class PostProcessInput:
    z: np.ndarray
    z_mu: np.ndarray
    Pz: np.ndarray
    sigma_rep: np.ndarray
    sigma_qx: np.ndarray
    kappa: np.ndarray
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
    country_unit_prefix: str | None
    nbc: int | None = None        
    bcouts: np.ndarray | None = None
    bcprior: dict | None = None
    Hbc_dic: dict | None = None


def outs_trace(
        xouts_mu: np.ndarray, 
        xouts_covariance: np.ndarray, 
        xprior: dict, 
        nbasis: int, 
        nperiod: int,
        nsamples: int = 15000,
        ):
    
    xtrace = np.zeros((nsamples, nperiod, nbasis))

    for period in np.arange(nperiod):

        L = cholesky(xouts_covariance[period], lower=True)

        normal_samples = np.random.randn(nsamples,nbasis)

        if xprior["pdf"] == "normal":
            
            xtrace[:, period] = xouts_mu[period] + normal_samples @ L.T

        elif xprior["pdf"] == "lognormal":
            
            xtrace[:, period] = np.exp(xouts_mu[period] + normal_samples @ L.T)
        
        else:
            
            raise ValueError("Xprior must be lognormal or normal.")

    return xtrace, nsamples



def amxkf_postprocessouts(config: PostProcessInput) -> xr.Dataset:
    """Takes the output from inferpymc function, along with some other input
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
        
    if config.nbc is not None and config.nbc > 0:
        Hbc = np.vstack(list(config.Hbc_dic.values()))
        YaprioriBC = np.sum(Hbc, axis=1)
        Yapriori = np.sum(Hx, axis=1) + np.sum(Hbc, axis=1)
        Ymodbc_dic = {}
        bcouts_mu = config.z_mu[:, config.nbasis:config.nbasis+config.nbc]
        bcouts_median = config.z[:, config.nbasis:config.nbasis+config.nbc]
        bcouts_68 = np.zeros((config.nperiod, len(nui), config.nbc))
        bcouts_95 = np.zeros((config.nperiod, len(nui), config.nbc))
    else:
        Yapriori = np.sum(Hx, axis=1)
        bcouts_mu = None
        bcouts_median = None
        bcouts_68 = None 
        bcouts_95 = None       

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

    xouts_mu = config.z_mu[:, :config.nbasis]
    xouts_median = config.z[:, :config.nbasis]
    routs_mu = config.z_mu[:, -1]
    routs_median = config.z[:, -1]

    xouts_68 = np.zeros((config.nperiod, len(nui), config.nbasis))
    xouts_95 = np.zeros((config.nperiod, len(nui), config.nbasis))
    routs_68 = np.zeros((config.nperiod, len(nui), 1))
    routs_95 = np.zeros((config.nperiod, len(nui), 1))

    for period in np.arange(config.nperiod):

        state_sigmas = np.diag(config.Pz[period])**0.5

        xouts_68[period], xouts_95[period] = state_percentiles(xouts_mu[period], state_sigmas[:config.nbasis], config.xprior, config.nbasis)
        routs_68[period], routs_95[period] = state_percentiles(routs_mu[period], state_sigmas[-1], config.rprior, 1)

        Ymod_dic[period] = config.Hx_dic[period] @ xouts_median[period]

        if config.nbc is not None and config.nbc > 0:

            bcouts_68[period], bcouts_95[period] = state_percentiles(bcouts_mu[period], state_sigmas[config.nbasis:config.nbasis+config.nbc], config.bcprior, config.nbc)

            Ymodbc_dic[period] = config.Hbc_dic[period] @ bcouts_median[period]

            Ymod_dic[period] += Ymodbc_dic[period]

            
        scalemap_single = np.zeros_like(bfds.values)

        for basis in np.arange(config.nbasis):

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
    cntrymean = np.zeros((len(cntrynames), config.nperiod))
    cntry68 = np.zeros((len(cntrynames), len(nui), config.nperiod))
    cntry95 = np.zeros((len(cntrynames), len(nui), config.nperiod))
    cntryprior = np.zeros((len(cntrynames), config.nperiod))

    if config.country_unit_prefix is None:
        config.country_unit_prefix = ""
    country_units = config.country_unit_prefix + "g"

    obs_units = str(config.fp_data[".units"])

    xtrace, steps = outs_trace(xouts_mu=xouts_mu, xouts_covariance=config.Pz[:, :config.nbasis, :config.nbasis], xprior=config.xprior, nbasis=config.nbasis, nperiod=config.nperiod)

    for period in np.arange(config.nperiod):
        
        for ci, cntry in enumerate(cntrynames):
            cntrytottrace = np.zeros(steps)
            cntrytotprior = 0
            for bf in range(int(np.max(bfarray)) + 1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray == bf)
                cntrytottrace += (
                    np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    * xtrace[:, period, bf]
                    / unit_factor
                )
                cntrytotprior += (
                    np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    / unit_factor
                )
            cntrymedian[ci, period] = np.median(cntrytottrace)
            cntrymean[ci, period] = np.mean(cntrytottrace)
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
        "xouts_median": (["period", "bf"], xouts_median),
        "xouts_mu": (["period", "bf"], xouts_mu),
        "xouts_68": (["period", "nUI", "bf"], xouts_68),
        "xouts_95": (["period", "nUI", "bf"], xouts_95),
        "routs_median": (["period"], routs_median),
        "routs_mu": (["period"], routs_mu),
        "routs_68": (["period", "nUI", "r"], routs_68),
        "routs_95": (["period", "nUI", "r"], routs_95),
        "sigma_rep": (["sig"], np.array([config.sigma_rep])),
        "sigma_qx": (["sig"], np.array([config.sigma_qx])),
        "kappa": (["sig"], np.array([config.kappa])),
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
        "countrymean": (["countrynames", "period"], cntrymean),
        "countryapriori": (["countrynames", "period"], cntryprior),
        "country68": (["countrynames", "nUI", "period"], cntry68),
        "country95": (["countrynames", "nUI", "period"], cntry95)
    }

    stepnum = np.arange(steps)
    numbf = np.arange(config.nbasis)
    numr = np.arange(1)
    numsig = np.arange(1)

    coords = {
        "stepnum": (["steps"], stepnum),
        "numbf": (["bf"], numbf),
        "numr": (["r"], numr),
        "numsig": (["sig"], numsig),
        "measurenum": (["nmeasure"], nmeasure),
        "UInum": (["nUI"], nui),
        "nsites": (["nsite"], sitenum),
        "lat": (["lat"], lat),
        "lon": (["lon"], lon),
        "countrynames": (["countrynames"], cntrynames),
        "periodstart": (["period"], date_range(to_datetime(config.start_date), to_datetime(config.end_date), freq="MS")[:-1])
    }

    if config.nbc is not None and config.nbc > 0:

        Ymodbc = np.concatenate(list(Ymodbc_dic.values()))
        numbc = np.arange(config.nbc)

        coords["numbc"] = (["bc"], numbc)
        data_vars.update({
            "YaprioriBC": (["nmeasure"], YaprioriBC),
            "YmodBC": (["nmeasure"], Ymodbc),
            "bcouts_median": (["period", "bc"], bcouts_median),
            "bcouts_mu": (["period", "bc"], bcouts_mu),
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
    outds.routs_median.attrs["longname"] = "state relaxation term median"
    outds.sigma_rep.attrs["longname"] = "representation error sigma"
    outds.sigma_qx.attrs["longname"] = "state forecast model error sigma"
    outds.kappa.attrs["longname"] = "state persistance term"


    if config.nbc is not None and config.nbc > 0:
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