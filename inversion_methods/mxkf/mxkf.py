
import time
from dataclasses import dataclass

from inversion_methods.mxkf import DataConfig, InversionInput, InversionIntermediate, PostProcessInput
from inversion_methods.mxkf import mxkf_monthly_dictionaries, mx_kalmanfilter, mxkf_postprocessouts, extract_data


@dataclass
class InversionParameters:
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
    outputpath: str
    basis_algorithm: str = "weighted"
    nbasis: int
    fp_basis_case: str | None = None
    bc_basis_case: str = "NESW"
    basis_directory: str | None = None
    bc_basis_directory: str | None = None
    fix_basis_outer_regions: bool = False
    country_file: str
    country_unit_prefix: str = "T"
    basis_output_path: str | None = None
    xprior: dict
    bcprior: dict
    bc_freq: str | None = None
    spatial_decay: float | int | None = None


def mxkf_function(config: InversionParameters):

    start_data = time.time()

    data_config = DataConfig(config.species, 
                             config.sites,
                             config.domain,
                             config.averaging_period,
                             config.start_date,
                             config.end_date,
                             config.obs_data_level,
                             config.platform,
                             config.met_model,
                             config.fp_model,
                             config.fp_height,
                             config.fp_species,
                             config.emissions_name,
                             config.inlet,
                             config.instrument,
                             config.calibration_scale,
                             config.use_bc,
                             config.bc_input,
                             config.bc_store,
                             config.obs_store,
                             config.footprint_store,
                             config.emissions_store,
                             config.averaging_error,
                             config.save_merged_data,
                             config.merged_data_dir,
                             config.merged_data_name,
                             config.outputname,
                             config.basis_algorithm,
                             config.nbasis,
                             config.fp_basis_case,
                             config.bc_basis_case,
                             config.basis_directory,
                             config.bc_basis_directory,
                             config.fix_basis_outer_regions,
                             config.basis_output_path,
                             config.xprior,
                             config.bcprior,
                             config.bc_freq,
                             config.spatial_decay,
                            )

    (Hx, 
    Y, 
    Ytime, 
    error, 
    siteindicator, 
    nbasis, 
    xprior, 
    x_covariance, 
    bc_covariance, 
    bcprior, 
    Hbc, 
    nbc, 
    bc_covariance, 
    fp_data
    ) = extract_data(data_config)
                    
    inversion_input = InversionInput(config.start_date,
                                     config.end_date,
                                     Hx,
                                     Y,
                                     Ytime,
                                     error,
                                     siteindicator,
                                     config.use_bc,
                                     Hbc,
                                     nbasis,
                                     xprior,
                                     bcprior,
                                     x_covariance,
                                     bc_covariance
                                     )

    inversion_intermediate = mxkf_monthly_dictionaries(inversion_input)

    end_data = time.time()

    print(f"Data extraction and preparation complete. Time taken = {end_data-start_data:.4f} seconds")

    start_kalman = time.time()

    (xouts, 
    xouts_68, 
    xouts_95, 
    Ymod_dic,
    nparam,
    ) = mx_kalmanfilter(inversion_intermediate)
    
    end_kalman = time.time()

    print(f"MXKF Complete. Time taken = {end_kalman-start_kalman:.4f} seconds")


    start_post = time.time()

    post_process_input = PostProcessInput(xouts,
                                          xouts_68,
                                          xouts_95,
                                          inversion_intermediate.H_dic,
                                          inversion_intermediate.Y_dic,
                                          Ymod_dic,
                                          inversion_intermediate.Yerr_dic,
                                          inversion_intermediate.Ytime_dic,
                                          inversion_intermediate.siteindicator_dic,
                                          config.domain,
                                          config.species,
                                          config.sites,
                                          config.start_date,
                                          config.end_date,
                                          config.outputname,
                                          config.outputpath,
                                          config.country_unit_prefix,
                                          config.emissions_name,
                                          # bcouts,
                                          # bcouts_covariance,
                                          # YBC_mod,
                                          # YBC_mod_covariance,
                                          # Hbc,
                                          # obs_repeatability,
                                          # obs_variability,
                                          fp_data,
                                          config.country_file,
                                          config.use_bc,
                                          nbasis,
                                          inversion_intermediate.nperiod
                                          )

    outsds = mxkf_postprocessouts(post_process_input)

    end_post = time.time()

    print(f"Post processing Complete. Time taken = {end_post-start_post:.4f} seconds")

