import time
from dataclasses import dataclass, replace

from inversion_methods.bristau.siqma_qx import sigma_qx_groups
from inversion_methods.bristau.data_bristau import DataConfig, extract_data, build_cntryds
from inversion_methods.bristau.inversion_bristau import bristau_monthly_dictionaries, augmented_ffbs_mxkf_gibbs_double_slice, bristau_postprocessouts, MessyInput, PostProcessInput


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
    rprior: dict
    sigma2_rep_prior: dict
    sigma2_qx_prior: dict
    sigma_qbc: float
    sigma_qr: float    
    kappa_x_prior: dict         
    iterations: int = 2500
    nxout: int = 6
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
    basis_algorithm: str | None = None
    fp_basis_case: str | None = None
    bc_basis_case: str = "NESW"
    basis_directory: str | None = None
    bc_basis_directory: str | None = None
    fix_basis_outer_regions: bool = False
    country_unit_prefix: str = "T"
    basis_output_path: str | None = None
    bc_freq: str | None = None
    sigma_rep: str | float | None = 'global additive'
    sigma_rep_max: float | None = 100.0
    sigma_qx: str | float = "inner outer"
    sigma_qx_max: float | None = 0.5
    kappa_x: str | float | None = 0.0
    kappa_x_minfold: int | None = 1
    kappa_bc: None = None
    kappa_r: float | None = 0.0
    min_group_size: int | None = 15
    tau_resid: str | float | None = 0.0
    tau_resid_prior: dict | None = None
    tau_resid_max: float | None = 200


def bristau_function(config: InversionParameters):
    
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
                             rprior=config.rprior,
                             bc_freq=config.bc_freq,
                            )
    (
    Hx, 
    Y, 
    Ytime, 
    sigma_obs, 
    siteindicator, 
    nbasis, 
    xprior, 
    bcprior, 
    Hbc, 
    fp_data
    ) = extract_data(data_config)

    cntryds = build_cntryds(config.domain, config.country_file)
    
    inner_group_id, ningroup = sigma_qx_groups(fp_data[".basis"], cntryds, config.nxout, nbasis, config.sigma_qx, config.min_group_size)

    messy_input = MessyInput(start_date=config.start_date,
                                     end_date=config.end_date,
                                     Hx=Hx,
                                     Y=Y,
                                     Ytime=Ytime,
                                     sigma_obs=sigma_obs,                                     
                                     siteindicator=siteindicator,
                                     Hbc=Hbc,
                                     nxout=config.nxout,
                                     use_bc = config.use_bc,
                                     )

    inversion_input = bristau_monthly_dictionaries(messy_input)

    inversion_input = replace(
        inversion_input,
        nbasis=nbasis,
        xprior=xprior,
        bcprior=bcprior,
        rprior=config.rprior,
        sigma2_rep_prior=config.sigma2_rep_prior,
        sigma2_qx_prior=config.sigma2_qx_prior,
        sigma_qbc=config.sigma_qbc,
        sigma_qr=config.sigma_qr,
        kappa_x_prior=config.kappa_x_prior,
        iterations=config.iterations,
        inner_group_id=inner_group_id,
        ningroup=ningroup,
        sigma_rep=config.sigma_rep,
        sigma_rep_max=config.sigma_rep_max,
        sigma_qx=config.sigma_qx,
        sigma_qx_max=config.sigma_qx_max,
        kappa_x=config.kappa_x,
        kappa_x_minfold=config.kappa_x_minfold,
        kappa_bc=config.kappa_bc,
        kappa_r=config.kappa_r,
        nxout=config.nxout,
        tau_resid=config.tau_resid,
        tau_resid_prior=config.tau_resid_prior,
        tau_resid_max=config.tau_resid_max,
        )

    end_data = time.time()

    if ningroup is not None:
        print(f"Data extraction and preparation complete. {ningroup} inner groups. Time taken = {end_data-start_data:.4f} seconds", flush=True)
    else:
        print(f"Data extraction and preparation complete. No country groups. Time taken = {end_data-start_data:.4f} seconds", flush=True)

    start_haffbs = time.time()

        
    (xtrace,
    bctrace,
    rtrace,
    var_rep_trace,
    sigma2_qx_trace,
    kappa_x_trace,
    sigma2_qx_trace_labels,
    kappa_x_trace_labels,
    tau_trace,
    tau_trace_labels,
    ) = augmented_ffbs_mxkf_gibbs_double_slice(inversion_input)
    

    end_haffbs = time.time()

    print(f"Sampling Complete. Time taken = {end_haffbs-start_haffbs:.4f} seconds")

    start_post = time.time()

    post_process_input = PostProcessInput(xtrace=xtrace,
                                          bctrace=bctrace,
                                          rtrace=rtrace,
                                          var_rep_trace=var_rep_trace,
                                          sigma2_qx_trace=sigma2_qx_trace,
                                          sigma2_qx_trace_labels=sigma2_qx_trace_labels,
                                          kappa_x_trace=kappa_x_trace,
                                          kappa_x_trace_labels=kappa_x_trace_labels,
                                          xprior=inversion_input.xprior,
                                          bcprior=inversion_input.bcprior,
                                          rprior=inversion_input.rprior,
                                          Hx_dic=inversion_input.Hx_dic,
                                          Hbc_dic=inversion_input.Hbc_dic,
                                          Y_dic=inversion_input.Y_dic,
                                          sigma_obs_dic=inversion_input.sigma_obs_dic,
                                          Ytime_dic=inversion_input.Ytime_dic,
                                          siteindicator_dic=inversion_input.siteindicator_dic,
                                          domain=config.domain,
                                          species=config.species,
                                          sites=config.sites,
                                          start_date=config.start_date,
                                          end_date=config.end_date,
                                          outputname=config.outputname,
                                          outputpath=config.outputpath,
                                          emissions_name=config.emissions_name,
                                          fp_data=fp_data,
                                          country_file=config.country_file,
                                          nbasis=nbasis,
                                          nxout=config.nxout,
                                          nr=inversion_input.nr,
                                          nperiod=inversion_input.nperiod,
                                          nbc=inversion_input.nbc,
                                          country_unit_prefix=config.country_unit_prefix,
                                          ningroup=ningroup,
                                          inner_group_id=inner_group_id,
                                          tau_trace=tau_trace,
                                          tau_trace_labels=tau_trace_labels,
                                          )


    outsds = bristau_postprocessouts(post_process_input)

    end_post = time.time()

    print(f"Post processing Complete. Time taken = {end_post-start_post:.4f} seconds")

