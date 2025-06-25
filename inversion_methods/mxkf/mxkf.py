
import time
from dataclasses import dataclass

from inversion_methods.mxkf.data_mxkf import DataConfig, extract_data
from inversion_methods.mxkf.inversion_mxkf import mxkf_monthly_dictionaries, mx_kalmanfilter, mxkf_postprocessouts, InversionInput, PostProcessInput


@dataclass
class InversionParameters:
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
    outputpath: str
    nbasis: int
    country_file: str
    xprior: dict
    bcprior: dict
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
    fixed_model_error: float | int | None = None
    save_merged_data: bool = False
    merged_data_dir: str | None = None
    merged_data_name: str | None = None
    basis_algorithm: str = "weighted"
    fp_basis_case: str | None = None
    bc_basis_case: str = "NESW"
    basis_directory: str | None = None
    bc_basis_directory: str | None = None
    fix_basis_outer_regions: bool = False
    country_unit_prefix: str = "T"
    basis_output_path: str | None = None
    bc_freq: str | None = None
    spatial_decay: float | int | None = None


def mxkf_function(config: InversionParameters):
    
    start_data = time.time()

    data_config = DataConfig(species=config.species, 
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
                             averaging_error=config.averaging_error,
                             save_merged_data=config.save_merged_data,
                             merged_data_dir=config.merged_data_dir,
                             merged_data_name=config.merged_data_name,
                             outputname=config.outputname,
                             basis_algorithm=config.basis_algorithm,
                             nbasis=config.nbasis,
                             fp_basis_case=config.fp_basis_case,
                             bc_basis_case=config.bc_basis_case,
                             basis_directory=config.basis_directory,
                             bc_basis_directory=config.bc_basis_directory,
                             fix_basis_outer_regions=config.fix_basis_outer_regions,
                             basis_output_path=config.basis_output_path,
                             xprior=config.xprior,
                             bcprior=config.bcprior,
                             bc_freq=config.bc_freq,
                             spatial_decay=config.spatial_decay,
                            )

    (
    Hx, 
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
    
    inversion_input = InversionInput(start_date=config.start_date,
                                     end_date=config.end_date,
                                     Hx=Hx,
                                     Y=Y,
                                     Ytime=Ytime,
                                     error=error,
                                     siteindicator=siteindicator,
                                     use_bc=config.use_bc,
                                     Hbc=Hbc,
                                     nbasis=nbasis,
                                     xprior=xprior,
                                     bcprior=bcprior,
                                     x_covariance=x_covariance,
                                     bc_covariance=bc_covariance,
                                     fixed_model_error = config.fixed_model_error
                                     )

    inversion_intermediate = mxkf_monthly_dictionaries(inversion_input)

    end_data = time.time()

    print(f"Data extraction and preparation complete. Time taken = {end_data-start_data:.4f} seconds")

    start_kalman = time.time()

    (xouts_mean,
     xouts_median,
     xouts_mode,
     xouts_stdev, 
     xouts_68, 
     xouts_95, 
     Ymod_dic,
     nparam,
     fixed_model_error
     ) = mx_kalmanfilter(inversion_intermediate)
    
    end_kalman = time.time()

    print(f"MXKF Complete. Time taken = {end_kalman-start_kalman:.4f} seconds")


    start_post = time.time()

    post_process_input = PostProcessInput(xouts_mean=xouts_mean,
                                          xouts_median=xouts_median,
                                          xouts_mode=xouts_mode,
                                          xouts_stdev=xouts_stdev,
                                          xouts_68=xouts_68,
                                          xouts_95=xouts_95,
                                          H_dic=inversion_intermediate.H_dic,
                                          Y_dic=inversion_intermediate.Y_dic,
                                          Ymod_dic=Ymod_dic,
                                          Yerr_dic=inversion_intermediate.Yerr_dic,
                                          Ytime_dic=inversion_intermediate.Ytime_dic,
                                          siteindicator_dic=inversion_intermediate.siteindicator_dic,
                                          domain=config.domain,
                                          species=config.species,
                                          sites=config.sites,
                                          start_date=config.start_date,
                                          end_date=config.end_date,
                                          outputname=config.outputname,
                                          outputpath=config.outputpath,
                                          country_unit_prefix=config.country_unit_prefix,
                                          emissions_name=config.emissions_name,
                                          # bcouts,
                                          # bcouts_covariance,
                                          # YBC_mod,
                                          # YBC_mod_covariance,
                                          # Hbc,
                                          # obs_repeatability,
                                          # obs_variability,
                                          fp_data=fp_data,
                                          country_file=config.country_file,
                                          use_bc=config.use_bc,
                                          nbasis=nbasis,
                                          nperiod=inversion_intermediate.nperiod,
                                          fixed_model_error=fixed_model_error
                                          )

    outsds = mxkf_postprocessouts(post_process_input)

    end_post = time.time()

    print(f"Post processing Complete. Time taken = {end_post-start_post:.4f} seconds")

