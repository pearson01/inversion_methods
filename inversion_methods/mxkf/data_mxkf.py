import numpy as np
import openghg_inversions.hbmcmc.inversionsetup as setup
from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.inversion_data import data_processing_surface_notracer
from inversion_methods.manipulation import update_log_normal_prior, covariance_lognormal_transform, spatial_covariance
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
    bc_store: str
    obs_store: str
    footprint_store: str
    emissions_store: str
    averaging_error: bool = True
    save_merged_data: bool = False
    merged_data_dir: str | None = None
    merged_data_name: str | None = None
    outputname: str
    basis_algorithm: str = "weighted"
    nbasis: int
    fp_basis_case: str | None = None
    bc_basis_case: str = "NESW"
    basis_directory: str | None = None
    bc_basis_directory: str | None = None
    fix_basis_outer_regions: bool = False
    basis_output_path: str | None = None
    xprior: dict
    bcprior: dict
    bc_freq: str | None = None
    spatial_decay: float | int | None = None



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


def build_obs_vectors(fp_data, sites):

    error = np.zeros(0)
    obs_repeatability = np.zeros(0)
    obs_variability = np.zeros(0)
    Hx = np.zeros((0, fp_data[sites[0]].H.shape[1]))
    Y = np.zeros(0)
    siteindicator = np.zeros(0)

    for si, site in enumerate(sites):
        drop_vars = [v for v in ["H", "H_bc", "mf", "mf_error", "mf_variability", "mf_repeatability"]
                     if v in fp_data[site].data_vars]
        fp_data[site] = fp_data[site].dropna("time", subset=drop_vars)

        error = np.concatenate((error, fp_data[site].mf_error.values))
        obs_repeatability = np.concatenate((obs_repeatability, fp_data[site].mf_repeatability.values))
        obs_variability = np.concatenate((obs_variability, fp_data[site].mf_variability.values))
        Y = np.concatenate((Y, fp_data[site].mf.values))
        siteindicator = np.concatenate((siteindicator, np.ones_like(fp_data[site].mf.values) * si))

        if si == 0:
            Ytime = fp_data[site].time.values
        else:
            Ytime = np.concatenate((Ytime, fp_data[site].time.values))

        Hx = np.vstack((Hx, fp_data[site].H.values))

    return Hx, Y, Ytime, error, siteindicator, obs_repeatability, obs_variability


def build_boundary_conditions(fp_data, sites, config: DataConfig):

    if not config.use_bc:
        return None, None, None

    Hbc = np.zeros((0, fp_data[sites[0]].H_bc.shape[1]))

    for si, site in enumerate(sites):
        if config.bc_freq == "monthly":
            Hmbc = setup.monthly_bcs(config.start_date, config.end_date, site, fp_data)
        elif config.bc_freq is None:
            Hmbc = fp_data[site].H_bc.values
        else:
            Hmbc = setup.create_bc_sensitivity(config.start_date, config.end_date, site, fp_data, config.bc_freq)

        Hbc = np.vstack((Hbc, Hmbc))

    bc_covariance = np.eye(4) * config.bcprior["sigma"]**2
    return Hbc, Hbc.shape[0], bc_covariance


def build_x_covariance(fp_data, nbasis, config: DataConfig):

    if isinstance(config.spatial_decay, (int, float)) and config.xprior["pdf"] == "lognormal":
        lognormal_cov = np.eye(nbasis) * np.exp(config.xprior["sigma"]**2 - 1) * \
                        np.exp(2 * config.xprior["mu"] + config.xprior["sigma"]**2)
        x_cov = spatial_covariance(
            cov_space=lognormal_cov,
            sigma_space=config.xprior["sigma"],
            spatial_decay=config.spatial_decay,
            nbasis=nbasis,
            fp_data=fp_data
        )
        return covariance_lognormal_transform(x_cov, config.xprior["mu"], config.xprior["sigma"])
    else:
        return np.eye(nbasis) * config.xprior["sigma"]**2
    

def extract_data(config: DataConfig):
    fp_all, _, _, _, _, _ = extract_observation_data(config)
    fp_data = build_basis_functions(fp_all, config)

    for site in config.sites:
        fp_data[site].attrs["Domain"] = config.domain

    Hx, Y, Ytime, error, siteindicator, obs_rep, obs_var = build_obs_vectors(fp_data, config.sites)
    nbasis = Hx.shape[0]

    update_log_normal_prior(config.xprior)
    update_log_normal_prior(config.bcprior)

    Hbc, nbc, bc_cov = build_boundary_conditions(fp_data, config.sites, config)
    x_cov = build_x_covariance(fp_data, nbasis, config)

    return Hx, Y, Ytime, error, siteindicator, nbasis, config.xprior, x_cov, bc_cov, config.bcprior, Hbc, nbc, bc_cov, fp_data

