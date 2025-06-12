import re
from pathlib import Path

import numpy as np
import xarray as xr
from pandas import date_range, to_datetime
from dataclasses import dataclass

from inversion_methods.manipulation import woodbury
from openghg_inversions import utils, convert
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename


@dataclass
class InversionInput:
    start_date: str
    end_date: str
    Hx: np.ndarray
    Y: np.ndarray
    Ytime: np.ndarray
    error: np.ndarray
    siteindicator: np.ndarray
    use_bc: bool = False
    Hbc: np.ndarray | None = None
    nbasis: int
    xprior: dict
    bcprior: dict
    x_covariance: np.ndarray
    bc_covariance: np.ndarray | None = None


@dataclass
class InversionIntermediate:
    Y_dic: dict
    Yerr_dic: dict
    H_dic: dict
    Hbc_dic: dict | None = None
    nbasis: int
    nperiod: int | None = None
    use_bc: bool = False
    xprior: dict
    bcprior: dict
    x_covariance: np.ndarray
    bc_covariance: np.ndarray | None = None


def mxkf_monthly_dictionaries(config: InversionInput) -> InversionIntermediate:

    """
    Manipulates inversion data into dictionaries separated by month.
    """

    Y_dic = {}
    Yerr_dic = {}
    Ytime_dic = {}
    H_dic = {}
    siteindicator_dic = {}
    allmonth = date_range(config.start_date, config.end_date, freq="MS")[:-1]
    Ymonth = to_datetime(config.Ytime).to_period("M")
    nperiod = len(allmonth)

    if config.use_bc is True:
        Hbc_dic = {}

    else:
        Hbc_dic = None

    bc_count = 0
    for period in range(nperiod):
        mnth = allmonth[period].month
        yr = allmonth[period].year
        mnthloc = np.where(np.logical_and(Ymonth.month == mnth, Ymonth.year == yr))[0]
        Y_dic[period] = config.Y[mnthloc]
        Yerr_dic[period] = config.error[mnthloc]
        Ytime_dic[period] = config.Ytime[mnthloc]
        H_dic[period] = config.Hx.T[mnthloc, :]
        siteindicator_dic[period] = config.siteindicator[mnthloc]
        if config.use_bc:
            Hbc_dic[period] = config.Hbc.T[mnthloc, bc_count:bc_count+4]
            bc_count += 4

    return InversionIntermediate(Y_dic, 
                                 Yerr_dic, 
                                 H_dic, 
                                 Hbc_dic, 
                                 config.nbasis, 
                                 nperiod, 
                                 config.use_bc,
                                 config.xprior,
                                 config.bcprior,
                                 config.x_covariance,
                                 config.bc_covariance,
                                 )




def MX_Kalman_Filter(config: InversionIntermediate):

    if config.use_bc is True:
        nparam = config.nbasis + 4
        P_prior = np.zeros((nparam, nparam))
        P_prior[:config.nbasis, :config.basis] = config.x_covariance
        P_prior[config.nbasis:, config.nbasis:] = config.bc_covariance

    else:
        nparam = config.nbasis
        P_prior = config.x_covariance

    Ymod_dic = {}

    xprior_mu = config.xprior["mu"]
    xprior_ln_median = np.exp(xprior_mu)

    bcprior_mu = config.bcprior["mu"]
    bcprior_ln_median = np.exp(bcprior_mu)

    xb = np.ones(config.nbasis) * xprior_ln_median
    bcb = np.ones(4) * bcprior_ln_median

    periods = np.arange(config.nperiod)
    Q = np.eye(nparam) * 0

    xouts_median = np.zeros((nparam, config.nperiod))
    xouts_stdev = np.zeros((nparam, config.nperiod))
    xouts_mu = np.zeros((nparam, config.nperiod))
    xouts_sigma = np.zeros((nparam, config.nperiod))
    xouts_mean_kalman = np.zeros((nparam, config.nperiod))
    xouts_mode_kalman = np.zeros((nparam, config.nperiod))
    xouts_68 = np.zeros((nparam, 2, config.nperiod))
    xouts_95 = np.zeros((nparam, 2, config.nperiod))

    for t in periods:
        if config.use_bc:
            H = np.hstack((config.H_dic[t], config.Hbc_dic[t]))
            if t == 0:
                xb = np.append(xb, bcb)
        else:
            H = config.H_dic[t]

        Y = config.Y_dic[t]
        Yerr = config.Yerr_dic[t]
        nm = len(Yerr)
        R = np.diag(Yerr**2)
        R_inv = np.diag(1/(Yerr**2))

        # Wb = np.diag(1/xb)
        Wb = np.diag(xb)
        Wo_inv = np.eye(nm)
        H_hat = Wo_inv @ H @ Wb
        

        if t == 0:
            Pf = P_prior + Q

        K = Pf @ H_hat.T @ woodbury(R_inv, H_hat, Pf, H_hat.T)
        # K = Pf @ H_hat.T @ np.linalg.inv(H_hat @ Pf @ H_hat.T + R)
        xa_mu = np.log(xb) + K @ (Y - H @ xb)
        xa = np.exp(xa_mu)

        Pa = (np.eye(nparam) - K @ H_hat) @ Pf

        print(f"{t} - Condition of R_inv: {round(np.linalg.cond(R_inv),2)}, H_hat: {round(np.linalg.cond(H_hat),2)}, Pf: {round(np.linalg.cond(Pa),2)}, Pf_inv + HR_invH.T: {round(np.linalg.cond(np.linalg.inv(Pf) +H_hat.T@R_inv@H_hat),2)}")

        # Pf = Pa + Q
        # xb = xa
    
        Pf = P_prior + Q

        if config.use_bc:
            xb = np.append(np.ones(config.nbasis)*xprior_ln_median, np.ones(4)*bcprior_ln_median)
        else:
            xb = np.ones(config.nbasis)*xprior_ln_median
    
        xouts_median[:,t] = xa
        xouts_sigma[:,t] = np.diag(Pa)**0.5
        xouts_mu[:,t] = xa_mu
        xouts_stdev[:,t] = ((np.exp(xouts_sigma[:,t]**2) - 1) * np.exp(2*xouts_mu[:,t] + xouts_sigma[:,t]**2))**0.5
        xouts_mean_kalman[:,t] = np.exp(xouts_mu[:,t] + 0.5*xouts_sigma[:,t]**2)
        # for basis in np.arange(nparam):
            # xouts_68[basis, 0, t] = lognorm.ppf(0.16, xouts_sigma[basis, t], scale=np.exp(xouts_mu[basis, t]))
            # xouts_68[basis, 1, t] = lognorm.ppf(0.34, xouts_sigma[basis, t], scale=np.exp(xouts_mu[basis, t]))
        
        
        xouts_68[:, 0, t] = np.exp(xouts_mu[:,t] - xouts_sigma[:,t])
        xouts_68[:, 1, t] = np.exp(xouts_mu[:,t] + xouts_sigma[:,t])

        xouts_95[:, 0, t] = np.exp(xouts_mu[:,t] - 2*xouts_sigma[:,t])
        xouts_95[:, 1, t] = np.exp(xouts_mu[:,t] + 2*xouts_sigma[:,t])
        

        xouts_mode_kalman[:,t] = np.exp(xouts_mu[:,t] - xouts_sigma[:,t]**2)

        Ymod_dic[t] = H @ xouts_mean_kalman[:,t]

    return xouts_mean_kalman, xouts_68, xouts_95, Ymod_dic, nparam


@dataclass
class PostProcessInput:
    xouts: np.ndarray
    xouts_68: np.ndarray
    xouts_95: np.ndarray
    H_dic: dict
    Y_dic: dict
    Ymod_dic: dict
    Yerr_dic: dict
    Ytime_dic: dict
    siteindicator_dic: dict
    domain: str
    species: str
    sites: list[str]
    start_date: str
    end_date: str
    outputname: str
    outputpath: str
    country_unit_prefix: str | None
    emissions_name: str
    # bcouts,
    # bcouts_covariance: np.ndarray | None = None,
    # YBC_mod: np.ndarray | None = None,
    # YBC_mod_covariance: np.ndarray | None = None,
    # Hbc: np.ndarray | None = None,
    # obs_repeatability: np.ndarray | None = None,
    # obs_variability: np.ndarray | None = None,
    fp_data: dict
    country_file: str
    use_bc: bool = False
    nbasis: int
    nperiod: int


def mxkf_postprocessouts(config: PostProcessInput) -> xr.Dataset:
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
    Hx = np.vstack(list(config.H_dic.values()))
    Y = np.hstack(list(config.Y_dic.values()))
    Yerr = np.hstack(list(config.Yerr_dic.values()))
    Ymod = np.hstack(list(config.Ymod_dic.values()))
    Ytime = np.hstack(list(config.Ytime_dic.values()))
    siteindicator = np.hstack(list(config.siteindicator_dic.values()))

    nx = Hx.shape[1]
    ny = len(Y)

    # if use_bc:
    #     nbc = Hbc.shape[0]
    #     nBC = np.arange(nbc)

    nui = np.arange(2)
    nmeasure = np.arange(ny)
    nparam = np.arange(nx)
        
    # if use_bc:
    #     YaprioriBC = np.sum(Hbc, axis=0)
    #     Yapriori = np.sum(Hx.T, axis=1) + np.sum(Hbc.T, axis=1)
    # else:
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

    for period in np.arange(config.nperiod):
    
        scalemap_single = np.zeros_like(bfds.values)

        for basis in np.arange(config.nbasis):
            scalemap_single[bfds.values == (basis + 1)] = config.xouts[basis, period]

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

    cntrymean = np.zeros((len(cntrynames), config.nperiod))
    # cntry68 = np.zeros((len(cntrynames), len(nui), nperiod))
    # cntry95 = np.zeros((len(cntrynames), len(nui), nperiod))
    # cntrysd = np.zeros((len(cntrynames), nperiod))
    cntryprior = np.zeros((len(cntrynames), config.nperiod))

    if config.country_unit_prefix is None:
        config.country_unit_prefix = ""
    country_units = config.country_unit_prefix + "g"

    obs_units = str(config.fp_data[".units"])

    for period in np.arange(config.nperiod):

        for ci, cntry in enumerate(cntrynames):
            cntrytot = 0
            cntrytotprior = 0
            for bf in range(int(np.max(bfarray)) + 1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray == bf)
                cntrytot += (
                    np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    * config.xouts[bf, period]
                    / unit_factor
                )
                cntrytotprior += (
                    np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    / unit_factor
                )
            
            cntrymean[ci, period] = cntrytot
            # cntrysd[ci, period] = np.std(cntrytottrace)
            # cntry68[ci, :, period] = pm.stats.hdi(cntrytottrace.values, 0.68)
            # cntry95[ci, :, period] = pm.stats.hdi(cntrytottrace.values, 0.95)
            cntryprior[ci, period] = cntrytotprior


    # Make output netcdf file
    data_vars = {
        "Yobs": (["nmeasure"], Y),
        "Yerror": (["nmeasure"], Yerr),
        "Ytime": (["nmeasure"], Ytime),
        "Yapriori": (["nmeasure"], Yapriori),
        "Ymod": (["nmeasure"], Ymod),
        "xouts": (["nparam", "period"], config.xouts),
        "xouts_68": (["nparam", "nUI", "period"], config.xouts_68),
        "xouts_95": (["nparam", "nUI", "period"], config.xouts_95),
        "siteindicator": (["nmeasure"], siteindicator),
        "sitenames": (["nsite"], config.sites),
        "sitelons": (["nsite"], site_lon),
        "sitelats": (["nsite"], site_lat),
        "fluxapriori": (["lat", "lon"], apriori_flux),
        "basisfunctions": (["lat", "lon"], bfarray),
        "countrydefinition": (["lat", "lon"], cntrygrid),
        "xsensitivity": (["nmeasure", "nparam"], Hx),
        "flux": (["lat", "lon", "period"], flux),
        "scaling": (["lat", "lon", "period"], scalemap),
        "countrymean": (["countrynames", "period"], cntrymean),
        "countryapriori": (["countrynames", "period"], cntryprior),
    }


    coords = {
        "paramnum": (["nparam"], nparam),
        "measurenum": (["nmeasure"], nmeasure),
        "UInum": (["nUI"], nui),
        "nsites": (["nsite"], sitenum),
        "lat": (["lat"], lat),
        "lon": (["lon"], lon),
        "countrynames": (["countrynames"], cntrynames),
        "periodstart": (["period"], date_range(to_datetime(config.start_date), to_datetime(config.end_date), freq="MS")[:-1])
    }

    # if use_bc:
    #     data_vars.update({
    #         "YaprioriBC": (["nmeasure"], YaprioriBC),
    #         "YmodBC": (["nmeasure"], YBC_mod),
    #         "YmodBC_covariance": (["nmeasure", "nmeasure"], YBC_mod_covariance),
    #         "bcouts": (["nBC"], bcouts),
    #         "bcsensitivity": (["nmeasure", "nBC"], Hbc.T),
    #     })
    #     coords["numBC"] = (["nBC"], nBC)

    outds = xr.Dataset(data_vars, coords=coords)

    outds.flux.attrs["units"] = "mol/m2/s"
    outds.fluxapriori.attrs["units"] = "mol/m2/s"
    outds.Yobs.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yerror.attrs["units"] = obs_units + " " + "mol/mol"
    # outds.Yerror_repeatability.attrs["units"] = obs_units + " " + "mol/mol"
    # outds.Yerror_variability.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yapriori.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod.attrs["units"] = obs_units + " " + "mol/mol"
    outds.countrymean.attrs["units"] = country_units
    outds.countryapriori.attrs["units"] = country_units

    outds.Yobs.attrs["longname"] = "observations"
    outds.Yerror.attrs["longname"] = "measurement error"
    outds.Ytime.attrs["longname"] = "time of measurements"
    outds.Yapriori.attrs["longname"] = "a priori simulated measurements"
    outds.Ymod.attrs["longname"] = "Analytical posterior simulated measurements"
    outds.siteindicator.attrs["longname"] = "index of site of measurement corresponding to sitenames"
    outds.sitenames.attrs["longname"] = "site names"
    outds.sitelons.attrs["longname"] = "site longitudes corresponding to site names"
    outds.sitelats.attrs["longname"] = "site latitudes corresponding to site names"
    outds.fluxapriori.attrs["longname"] = "mean a priori flux over period"
    outds.flux.attrs["longname"] = "posterior flux over period"
    outds.scaling.attrs["longname"] = "scaling factor field over period"
    outds.basisfunctions.attrs["longname"] = "basis function field"
    outds.countrymean.attrs["longname"] = "mean of ocean and country totals"
    outds.countryapriori.attrs["longname"] = "prior mean of ocean and country totals"
    outds.countrydefinition.attrs["longname"] = "grid definition of countries"
    outds.xsensitivity.attrs["longname"] = "emissions sensitivity timeseries"

    # if use_bc:
    #     outds.YmodBC.attrs["units"] = obs_units + " " + "mol/mol"
    #     outds.YaprioriBC.attrs["units"] = obs_units + " " + "mol/mol"
    #     outds.bcsensitivity.attrs["units"] = obs_units + " " + "mol/mol"

    #     outds.YaprioriBC.attrs["longname"] = "a priori simulated boundary conditions"
    #     outds.YmodBC.attrs["longname"] = "mean of posterior simulated boundary conditions"
    #     outds.bcsensitivity.attrs["longname"] = "boundary conditions sensitivity timeseries"

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