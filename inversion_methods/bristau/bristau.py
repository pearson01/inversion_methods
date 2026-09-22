import time
from dataclasses import dataclass, replace

from inversion_methods.bristau.siqma_qx import sigma_qx_groups
from inversion_methods.bristau.data_bristau import DataConfig, extract_data, build_cntryds
from inversion_methods.bristau.inversion_bristau import bristau_monthly_dictionaries, bristau_postprocessouts, MessyInput, PostProcessInput
from inversion_methods.bristau.multichain import run_chains


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

    # --- Multi-chain convergence checking (see multichain.py) -----------
    # nchain (int): Number of independent copies of the Gibbs sampler to run.
    #               1 (the default) just runs the sampler once. Set to e.g. 4 to also get a
    #               convergence check: all nchain chains are run in parallel and compared against each
    #               other, but only one chain's output is actually kept and
    #               saved -- the rest are used only to check agreement, then
    #               discarded.
    # chain_seed (int / optional): Fixes every random draw the sampler makes
    #               (the emissions state and every hyperparameter alike), so
    #               the exact same run -- or, with nchain>1, the exact same
    #               set of chains -- can be reproduced later. Leave as None
    #               (the default) for a fresh, genuinely random run every
    #               time. Applies whether nchain is 1 or greater than 1.
    # rhat_threshold (float): How strict the convergence check is. Chains are
    #               judged to disagree if their worst "R-hat" value (a
    #               standard MCMC convergence statistic; 1.0 is perfect
    #               agreement) is at or above this. 1.01 is a commonly used,
    #               fairly strict, default.
    # require_convergence (bool): What to do if the chains fail the check.
    #               False (the default) prints a loud warning but still
    #               completes the run and writes output -- useful while you
    #               are still finding out how often/how badly this happens.
    #               True instead stops the run with an error and produces no
    #               output at all if the chains disagree.
    nchain: int = 1
    chain_seed: int | None = None
    rhat_threshold: float = 1.01
    require_convergence: bool = False


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


    # run_chains() runs the Gibbs sampler once (config.nchain=1, the
    # default) or as several independent chains that get compared against
    # each other as a convergence check (config.nchain>1) -- see
    # multichain.py for the full explanation. Either way, `gibbs_output`
    # below is exactly the tuple the sampler itself produces, so nothing
    # past this point needs to know or care which case happened.
    # `convergence_report` is None unless config.nchain>1.
    gibbs_output, convergence_report = run_chains(
        inversion_input,
        nchain=config.nchain,
        chain_seed=config.chain_seed,
        rhat_threshold=config.rhat_threshold,
        require_convergence=config.require_convergence,
    )

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
    ) = gibbs_output


    end_haffbs = time.time()

    print(f"Sampling Complete. Time taken = {end_haffbs-start_haffbs:.4f} seconds.")

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
                                          convergence_report=convergence_report,
                                          )


    outsds = bristau_postprocessouts(post_process_input)

    end_post = time.time()

    print(f"Post processing Complete. Time taken = {end_post-start_post:.4f} seconds")

