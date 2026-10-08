import numpy as np
import xarray as xr
import openghg_inversions.hbmcmc.inversionsetup as setup
from openghg_inversions import utils
from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.inversion_data import data_processing_surface_notracer
from openghg_inversions.filters import filtering
from inversion_methods.manipulation.lognormal_transformations import update_log_normal_prior, covariance_lognormal_transform
from dataclasses import dataclass


@dataclass
class DataConfig:
    species: str
    sites: list[str]
    domain: str
    averaging_period: list[str]
    start_date: str
    end_date: str
    obs_data_level: str
    bc_store: str
    obs_store: str
    footprint_store: str
    emissions_store: str
    outputname: str
    nbasis: int
    xprior: dict
    bcprior: dict
    rprior: dict
    platform: list[str | None] | str | None = None
    met_model: list | None = None
    fp_model: str | None = None
    fp_height: list[str] | None = None
    fp_species: str | None = None
    emissions_name: list[str] | None = None
    inlet: list[str] | None = None
    instrument: list[str] | None = None
    calibration_scale: str | None = None
    use_bc: bool = True
    bc_input: str | None = None
    averaging_error: bool = True
    save_merged_data: bool = False
    merged_data_dir: str | None = None
    merged_data_name: str | None = None
    basis_algorithm: str = "weighted"
    fp_basis_case: str | None = None
    bc_basis_case: str = "NESW"
    basis_directory: str | None = None
    bc_basis_directory: str | None = None
    fix_basis_outer_regions: bool = False
    basis_output_path: str | None = None
    bc_freq: str | None = None
    filters: dict | list | None = None



def extract_observation_data(config: DataConfig):
    return data_processing_surface_notracer(
        species=config.species,
        sites=config.sites,
        domain=config.domain,
        averaging_period=config.averaging_period,
        start_date=config.start_date,
        end_date=config.end_date,
        obs_data_level=config.obs_data_level,
        platform=config.platform,
        met_model=config.met_model,
        fp_model=config.fp_model,
        fp_height=config.fp_height,
        fp_species=config.fp_species,
        emissions_name=config.emissions_name,
        inlet=config.inlet,
        instrument=config.instrument,
        calibration_scale=config.calibration_scale,
        use_bc=config.use_bc,
        bc_input=config.bc_input,
        bc_store=config.bc_store,
        obs_store=config.obs_store,
        footprint_store=config.footprint_store,
        emissions_store=config.emissions_store,
        averagingerror=config.averaging_error,
        save_merged_data=config.save_merged_data,
        merged_data_name=config.merged_data_name,
        merged_data_dir=config.merged_data_dir,
        output_name=config.outputname
    )


def build_basis_functions(fp_all, config: DataConfig):
    return basis_functions_wrapper(
        basis_algorithm=config.basis_algorithm,
        nbasis=config.nbasis,
        fp_basis_case=config.fp_basis_case,
        bc_basis_case=config.bc_basis_case,
        basis_directory=config.basis_directory,
        bc_basis_directory=config.bc_basis_directory,
        fp_all=fp_all,
        use_bc=config.use_bc,
        species=config.species,
        domain=config.domain,
        start_date=config.start_date,
        fix_outer_regions=config.fix_basis_outer_regions,
        emissions_name=config.emissions_name,
        outputname=config.outputname,
        output_path=config.basis_output_path
    )


def finite_time_mask(ds, var_names):
    """
    Boolean mask over `time` that is True where every listed variable is finite
    (not NaN or +/-inf) across all of its non-time dimensions. Also requires
    mf_error > 0.

    Returns the mask and a dict of the number of time steps failing each check.
    """
    keep = np.ones(ds.sizes["time"], dtype=bool)
    n_bad = {}

    for var in var_names:
        if var not in ds.data_vars:
            continue
        da = ds[var]
        other_dims = [d for d in da.dims if d != "time"]
        finite = np.isfinite(da)
        if other_dims:
            finite = finite.all(dim=other_dims)
        finite = np.asarray(finite.transpose("time").values, dtype=bool)
        n_bad[var] = int((~finite).sum())
        keep &= finite

    if "mf_error" in ds.data_vars:
        positive = np.asarray(ds["mf_error"].values > 0, dtype=bool)
        n_bad["mf_error<=0"] = int((~positive & np.isfinite(ds["mf_error"].values)).sum())
        keep &= positive

    return keep, n_bad


def build_obs_vectors(fp_data, sites):
    """
    Drop time steps with non-finite H, H_bc, mf or mf_error (or mf_error <= 0) from
    each site, writing the filtered dataset back into fp_data so that H_bc extracted
    later stays aligned. Sites left with no valid time steps are removed.

    Returns Hx, Y, Ytime, error, siteindicator and the list of retained sites.
    """
    error = np.zeros(0)
    Y = np.zeros(0)
    siteindicator = np.zeros(0)
    Ytime = None
    Hx = None
    kept_sites = []

    for site in sites:

        keep, n_bad = finite_time_mask(fp_data[site], ["H", "H_bc", "mf", "mf_error"])
        n_drop = int((~keep).sum())

        if n_drop > 0:
            details = ", ".join(f"{var}: {n}" for var, n in n_bad.items() if n > 0)
            print(f"{site}: dropping {n_drop} of {keep.size} time steps with invalid values ({details})", flush=True)

        fp_data[site] = fp_data[site].isel(time=np.flatnonzero(keep))

        if fp_data[site].sizes["time"] == 0:
            print(f"WARNING: {site} has no valid time steps remaining; removing it from the inversion.", flush=True)
            fp_data.pop(site)
            continue

        si = len(kept_sites)
        kept_sites.append(site)

        error = np.concatenate((error, fp_data[site].mf_error.values))

        Y = np.concatenate((Y, fp_data[site].mf.values))
        siteindicator = np.concatenate((siteindicator, np.ones_like(fp_data[site].mf.values) * si))

        if si == 0:
            Ytime = fp_data[site].time.values
        else:
            Ytime = np.concatenate((Ytime, fp_data[site].time.values))

        Hx = fp_data[site].H.values if si == 0 else np.hstack((Hx, fp_data[site].H.values))

    if not kept_sites:
        raise ValueError("No sites have any valid (finite) observations/sensitivities remaining.")

    return Hx, Y, Ytime, error, siteindicator, kept_sites


def build_boundary_conditions(fp_data, sites, config: DataConfig):

    if not config.use_bc:
        return None
    
    else:
        for si, site in enumerate(sites):

            Hmbc = fp_data[site].H_bc.values

            Hbc = Hmbc if si == 0 else np.hstack((Hbc, Hmbc))

        return Hbc


def check_finite_inputs(arrays):
    """Raise if any of the named arrays (None entries skipped) contain non-finite values."""
    for name, arr in arrays.items():
        if arr is None:
            continue
        n_bad = int(np.size(arr) - np.isfinite(arr).sum())
        if n_bad > 0:
            raise ValueError(f"{name} contains {n_bad} non-finite values after filtering.")


def apply_filters(fp_all, sites, filters):
    # Same approach as RHIME preparation: apply openghg_inversions filters,
    # then drop any site left with no observations.
    if not filters:
        return fp_all, sites

    try:
        fp_all = filtering(fp_all, filters)
    except ValueError:
        for site in sites:
            fp_all[site] = fp_all[site].compute()
        fp_all = filtering(fp_all, filters)

    empty = [site for site in sites if fp_all[site].sizes.get("time", 0) == 0]
    for site in empty:
        del fp_all[site]
    if empty:
        print(f"WARNING: no observations left after filtering for sites: {empty}", flush=True)

    return fp_all, [site for site in sites if site not in empty]


def extract_data(config: DataConfig):
    # sites with no obs/footprints are dropped by data_processing_surface_notracer
    fp_all, sites, _, _, _, _ = extract_observation_data(config)
    fp_all, sites = apply_filters(fp_all, sites, config.filters)
    fp_data = build_basis_functions(fp_all, config)

    for site in sites:
        fp_data[site].attrs["Domain"] = config.domain

    Hx, Y, Ytime, error, siteindicator, sites = build_obs_vectors(fp_data, sites)
    nbasis = Hx.shape[0]

    dropped = [site for site in config.sites if site.upper() not in sites]
    if dropped:
        print(f"WARNING: sites dropped from the inversion: {dropped}", flush=True)

    update_log_normal_prior(config.xprior)
    update_log_normal_prior(config.rprior)

    if config.use_bc:
        update_log_normal_prior(config.bcprior)
        Hbc = build_boundary_conditions(fp_data, sites, config)
    else:
        Hbc = None

    check_finite_inputs({"Hx": Hx, "Y": Y, "error": error, "Hbc": Hbc})

    return Hx, Y, Ytime, error, siteindicator, nbasis, config.xprior, config.bcprior, Hbc, fp_data, sites


def inner_basis_country_groups(bfds, cntryds, nxout, nbasis, min_group_size=5):
    """
    Assign basis functions to country-based groups using a land country mask.

    If nxout is None or 0, the entire domain is grouped by country.
    Otherwise only basis functions nxout..nbasis-1 are grouped, with
    the first nxout basis functions assumed to be outer regions.

    Parameters
    ----------
    bfds : xr.DataArray
        Basis function definition field (1-indexed, as used elsewhere).
    cntryds : xr.Dataset
        Country definition dataset with a "country" grid on the same
        lat/lon as bfds.
    nxout : int or None
        Number of outer basis functions. If None or 0, all basis
        functions are grouped.
    nbasis : int
        Total number of basis functions.
    min_group_size : int
        Countries with fewer than this many basis functions are merged
        into a single "other" group.

    Returns
    -------
    group_id : ndarray
        Group index (0..ngroup-1) for each grouped basis function.
    ngroup : int
        Number of distinct groups.
    """
    bfarray = bfds.values - 1
    cntrygrid = cntryds.country.values

    # Full-domain mode
    if nxout is None or nxout == 0:
        start_bf = 0
        nbf = nbasis
    else:
        start_bf = nxout
        nbf = nbasis - nxout

    bf_country = np.full(nbf, -1, dtype=int)

    for k in range(nbf):
        bf = start_bf + k
        mask = (bfarray == bf)

        countries_here = cntrygrid[mask]
        if countries_here.size:
            vals, counts = np.unique(countries_here, return_counts=True)
            bf_country[k] = vals[np.argmax(counts)]

    # Merge small countries (and unassigned basis functions) into "other"
    unique_countries, counts = np.unique(bf_country, return_counts=True)
    print(f"counts: {counts}, min_group: {min_group_size}", flush=True)
    small = unique_countries[counts < min_group_size]

    group_id = bf_country.copy()
    group_id[np.isin(group_id, small)] = -1

    unique_groups, group_counts = np.unique(group_id, return_counts=True)

    remap = {g: idx for idx, g in enumerate(unique_groups)}
    group_id = np.array([remap[g] for g in group_id])

    for g, n in zip(remap.values(), group_counts):
        print(f"Group {g}: {n} BFs", flush=True)

    return group_id, len(unique_groups)


def build_cntryds(domain, country_file):

    c_object = utils.get_country(domain, country_file=country_file)
    cntryds = xr.Dataset(
        {"country": (["lat", "lon"], c_object.country), "name": (["ncountries"], c_object.name)},
        coords={"lat": (c_object.lat), "lon": (c_object.lon)},
    )

    return cntryds
