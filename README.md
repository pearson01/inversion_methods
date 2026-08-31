## `bristau`

`bristau` (in [inversion_methods/bristau](inversion_methods/bristau)) is a "top-down" emissions estimation
method: it works out how much of a gas (e.g. methane) is likely being emitted from different
regions, based on measurements of its concentration in the atmosphere at a handful of monitoring
sites, plus a model of how emissions travel through the atmosphere to reach those sites.

Example ini file in the inversion_methods/templates folder.

### The problem it solves

We don't observe emissions directly - we observe concentrations in the air. `bristau` uses
Bayesian statistics to work backwards from the observed concentrations to the most likely
emissions, while also being honest about how uncertain that estimate is. It reports not just a
single "best guess" for emissions, but a full range of plausible values.

### How it works, in plain terms

1. **Gather the data.** For a chosen gas, set of monitoring sites, and date range, `bristau` pulls
   in observed concentrations, "footprints" (a model of how sensitive each measurement is to
   emissions from each part of the map), boundary conditions (concentration entering the domain
   from outside it), and a prior "best guess" for emissions, all from the OpenGHG data store.
2. **Simplify the map.** Rather than solving for emissions at every grid cell, the region is
   divided into a manageable number of "basis functions" - patches of the map (and, further out,
   larger aggregated regions) that are each assigned a single scaling factor on the prior
   emissions.
3. **Split into monthly chunks.** The data are divided up by calendar month so that emissions can
   evolve over time rather than being estimated as one single fixed value for the whole period.
4. **Run the filter.** A Kalman-filter-style algorithm steps through the months in order,
   updating its belief about emissions (and boundary conditions) as each new month of
   observations arrives, then a second pass samples plausible full time histories consistent with
   all the data (this "forward filter, backward sample" pattern is where part of the method's
   `ffbs` naming comes from). Because gas concentrations respond in a skewed, non-negative way
   (you can't have negative emissions), the filter is "augmented" to handle this non-Gaussian
   behaviour rather than assuming everything follows a simple bell curve.
5. **Learn the uncertainty settings too.** Things like how noisy the measurements are, and how
   much emissions are allowed to drift from month to month, aren't fixed in advance - they are
   estimated from the data alongside the emissions themselves, using a Gibbs sampler that
   repeatedly cycles through updating each unknown quantity in turn.
6. **Summarise the results.** The many samples produced by the sampler are combined into
   posterior estimates (with uncertainty ranges) for emissions per basis function, per country,
   and for boundary conditions, and saved to a NetCDF output file.

### Running it

`bristau` is run from the command line via
[run_bristau.py](inversion_methods/bristau/run_bristau.py), using a configuration file that
specifies the species, sites, domain, date range, data stores, and prior settings:

```bash
python -m inversion_methods.bristau.run_bristau <start_date> <end_date> -c path/to/config.ini
```

`start_date`/`end_date` and other settings can also be overridden on the command line; any value
not supplied there must be present in the configuration file.

### Technical summary

The core sampler is `augmented_ffbs_mxkf_gibbs_double_slice` in
[inversion_bristau.py](inversion_methods/bristau/inversion_bristau.py), a Gibbs sampler that
alternates between an augmented Kalman filter/smoother sweep over the state and conjugate/slice
updates of the model hyperparameters, for `config.iterations` iterations with the first 20% (fixed
`burn = 0.2 * iterations`) discarded as burn-in.

**State space and transformation.** For each calendar-month time step `t`, the augmented state is

```
z_t = [x_out_t, x_in_t, x_bc_t, r_out_t, r_in_t]
```

where `x_out`/`x_in` are the outer/inner basis-function scaling factors (`nxout` outer, `nbasis -
nxout` inner), `x_bc` are optional boundary-condition scaling factors, and `r_out`/`r_in` are
scalar "reference" states that the basis coefficients relax toward (see below). Because emissions
scalings must stay non-negative and are treated as log-normal rather than Gaussian, the filter
works in a transformed ("optimisation") space via `state_vector_mu_transform` / `build_Wb`
([lognormal_transformations.py](inversion_methods/manipulation/lognormal_transformations.py)),
which is what makes this a non-Gaussian ("augmented") Kalman filter rather than a standard linear
one.

**Forecast (transition) model.** `augmented_forecast_model`
([bristau_filter_sampler.py](inversion_methods/bristau/bristau_filter_sampler.py)) is a linear,
AR(1)-style relaxation:

```
x_out(t+1) = kappa_out * x_out(t) + (1 - kappa_out) * r_out(t)
x_in(t+1)  = kappa_in  * x_in(t)  + (1 - kappa_in)  * r_in(t)
r(t+1)     = kappa_r   * r(t)     + (1 - kappa_r)   * rprior
```

`x_bc` (if present) is persistent (identity transition). `kappa_out`/`kappa_in` (jointly
`kappa_x`) control how strongly each month's scaling factors are pulled back toward the reference
state (i.e. mean reversion / temporal smoothness), and `kappa_r` (fixed, default 0) controls
persistence of the reference state itself. Process noise is added via a diagonal or full
covariance `Q_aug`, whose diagonal entries are the per-parameter process-noise variances
`sigma2_qx` (basis functions), `sigma_qbc` (boundary conditions) and `sigma_qr` (reference states).

**Forward filter.** `iterative_augmented_mxkf` propagates the forecast mean/covariance forward
through each monthly `Hz_dic`/`Y_dic` pair and computes the analysis update. Because the
log-normal state transformation makes the observation operator nonlinear in the untransformed
state, `iterative_analysis_update` performs an inner Gauss-Newton loop (with optional Anderson
acceleration and backtracking, `max_inner_iters=5`, `tol=1e-2`) to relinearise around the current
analysis estimate until it is consistent with the forecast to within tolerance. The Kalman gain
and analysis covariance are computed with a Woodbury-identity formulation
(`kalman_gain_woodbury_from_cholesky`, `covariance_from_woodbury_factor` in
[matrix_identities.py](inversion_methods/manipulation/matrix_identities.py)) that factorises an
`nz x nz` system (`nz` being the augmented state dimension) instead of the `ny x ny` innovation
covariance, so that the update cost scales with the (small) state dimension rather than the
(larger) number of observations per month, which is typically much greater than `nz`.

**Backward sampling.** `augmented_backward_sampler` performs the "BS" half of FFBS: it draws one
full posterior trajectory of `z_t` for every month, conditioned on all months of data, by sampling
backwards from the last filtered estimate using the standard Rauch-Tung-Striebel smoothing
recursion applied to the Gaussian (transformed-space) approximation. `xtrace`, `bctrace`, `rtrace`
store, respectively, the basis-function, boundary-condition and reference-state samples for every
retained iteration.

**Hyperparameter updates (Gibbs steps).** After each FFBS sweep, three groups of hyperparameters
are resampled conditional on the newly sampled state trajectory (residuals between consecutive
months, `state_residuals`):

- `update_sigma2_rep` ([sigma_rep.py](inversion_methods/bristau/sigma_rep.py)) - additional
  ("representation") observation-error variance, on top of the reported instrument/model
  measurement error `sigma_obs`. Under the `"global additive"` scheme it is drawn via a
  Metropolis/slice update (`sigma2_rep_log_posterior`) targeting a scaled-Beta prior on
  `sigma2_rep / sigma2_rep_max`, bounded above by `sigma_rep_max**2`; under `"fixed additive"` it
  is held at the user-supplied value.
- `update_sigma2_qx` ([siqma_qx.py](inversion_methods/bristau/siqma_qx.py)) - the state
  process-noise variance(s) controlling how much basis-function scalings are allowed to drift
  month to month. Depending on `sigma_qx` this can be a single global value, separate
  inner/outer values, or separate values per country group (`sigma_qx_groups`, which assigns each
  inner basis function to a country using a land-country mask, merging countries with fewer than
  `min_group_size` basis functions into an "other" group), each capped at `sigma_qx_max**2`.
- `update_kappa_x` ([kappa_x.py](inversion_methods/bristau/kappa_x.py)) - the persistence
  parameter(s) `kappa_out`/`kappa_in` (or a single global value), sampled subject to an upper
  bound `kappa_max(nperiod, kappa_x_minfold)` that prevents excessive persistence over short
  inversion windows.

All three updates accept `"fixed"` values (skipping sampling entirely) or hyperparameter priors
(`sigma2_rep_prior`, `sigma2_qx_prior`, `kappa_x_prior`, each parsed by `prior_parser` into
shape/scale-style parameters) when sampling is enabled.

**Post-processing.** `haffbs_postprocessouts` combines the retained trace arrays into posterior
summaries (median/mean and highest posterior density credible intervals, not simple percentiles)
per basis function, aggregated to per-country emissions totals (scaled by `country_unit_prefix`),
and boundary-condition scalings, and writes them to a NetCDF file via
[hbmcmc_output](../openghg_inversions).
