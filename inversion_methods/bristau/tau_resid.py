"""
AR(1)/OU treatment of same-site, temporally-correlated observation residuals.

`bristau` assimilates a whole calendar month as one joint Kalman update,
still assuming a diagonal observation-error covariance across every
observation in it. In reality, transport-model error at a given site is
correlated over a timescale of hours to a few days (see e.g. Brunner et al.
2012), so nearby-in-time observations at the same site are not independent
evidence the way a diagonal R assumes.

This module adds an optional, temporally-invariant correlation-length
hyperparameter `tau` (in hours), modelling same-site residual correlation as
phi(gap) = exp(-gap / tau) (a continuous-time AR(1)/Ornstein-Uhlenbeck
process). Because an AR(1)/OU process has a tridiagonal precision matrix,
removing the correlation is a cheap, one-shot, vectorised GLS whitening
transform applied directly to the fixed (Y, Hz) data -- it never touches the
evolving state, so it can be recomputed once per Gibbs iteration (after
`tau` and `sigma2_rep` are resampled) with no changes needed to the
Kalman-gain/Woodbury machinery in `bristau_filter_sampler.py` or
`matrix_identities.py`.

`tau_resid = 0.0` (the default) disables this entirely: phi is 0
everywhere, whiten_observations() returns Y/Hz unchanged and err_var exactly
as computed before this module existed.
"""

import numpy as np


def tau_scheme_select(config):
    """
    Select the tau (residual-correlation-length) scheme from
    config.tau_resid, mirroring kappa_x_scheme_select / sigma_rep_scheme_select.
    """
    tau_is_fixed = (
        isinstance(config.tau_resid, (int, float, np.integer, np.floating))
        and not isinstance(config.tau_resid, (bool, np.bool_))
    )

    if tau_is_fixed:
        scheme = "fixed"
        fixed_tau = float(config.tau_resid)

        if not np.isfinite(fixed_tau):
            raise ValueError("Fixed config.tau_resid must be finite.")

        if fixed_tau < 0:
            raise ValueError("Fixed config.tau_resid must be non-negative.")

    elif isinstance(config.tau_resid, str):
        scheme = " ".join(config.tau_resid.lower().split())
        fixed_tau = None

    else:
        raise TypeError("config.tau_resid must be a numeric fixed value (hours) or 'global'.")

    valid_schemes = {"fixed", "global"}

    if scheme not in valid_schemes:
        raise ValueError(f"Unknown config.tau_resid value: {config.tau_resid!r}. Expected one of {sorted(valid_schemes)} or a numeric value.")

    return scheme, fixed_tau


def tau_trace_params(scheme):
    return ["global"], 1


def initialise_tau(scheme, fixed_tau, initial_tau):
    if scheme == "fixed":
        return float(fixed_tau)
    return float(initial_tau)


def _gap_hours(t_new, t_old):
    if isinstance(t_new, np.datetime64):
        return (t_new - t_old) / np.timedelta64(1, "h")
    return float(t_new - t_old)


def prepare_tau_resid_indexing(Y_dic, Hz_dic, Ytime_dic, siteindicator_dic, nperiod):
    """
    One-time precomputation of same-site temporal adjacency.

    This depends only on the fixed, raw Y/Hz/Ytime/siteindicator data, not on
    any Gibbs-sampled hyperparameter, so it only needs to be computed once
    per inversion run rather than once per iteration.

    Returns
    -------
    prev_Y_dic, prev_H_dic : dict[int, np.ndarray]
        Per period: the raw Y value / Hz row of the preceding same-site
        observation (which may sit in an earlier period). Zero where no
        preceding observation exists (that observation is the first of its
        site in the whole run).
    gap_dic : dict[int, np.ndarray]
        Per period: the time gap in hours to that preceding observation.
    has_prev_dic : dict[int, np.ndarray of bool]
        Per period: whether a preceding same-site observation exists.
    gap_flat : np.ndarray
        Per-observation gap, in the flat order
        np.concatenate([Y_dic[t] for t in range(nperiod)]) -- the same order
        `state_residuals` is built in by `augmented_backward_sampler`.
    prev_index_flat : np.ndarray of int
        For each observation (same flat order as above), the flat index of
        its preceding same-site observation, or -1 if none.
    """
    prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic = {}, {}, {}, {}
    gap_flat_parts, prev_index_flat_parts = [], []

    last_time, last_Y, last_H, last_flat_index = {}, {}, {}, {}
    flat_offset = 0

    for t in range(nperiod):
        Y = Y_dic[t]
        H = Hz_dic[t]
        times = Ytime_dic[t]
        sites = siteindicator_dic[t]
        n = len(Y)

        prev_Y = np.zeros(n, dtype=float)
        prev_H = np.zeros_like(H, dtype=float)
        gap = np.zeros(n, dtype=float)
        has_prev = np.zeros(n, dtype=bool)
        prev_index = np.full(n, -1, dtype=int)

        # Process in time order so "preceding observation" is correct even
        # if a period's rows aren't already time-sorted; results are written
        # back at each row's original position.
        order = np.argsort(times, kind="stable")

        for idx in order:
            site = sites[idx]

            if site in last_time:
                prev_Y[idx] = last_Y[site]
                prev_H[idx] = last_H[site]
                gap[idx] = _gap_hours(times[idx], last_time[site])
                has_prev[idx] = True
                prev_index[idx] = last_flat_index[site]

            last_time[site] = times[idx]
            last_Y[site] = Y[idx]
            last_H[site] = H[idx]
            last_flat_index[site] = flat_offset + idx

        prev_Y_dic[t] = prev_Y
        prev_H_dic[t] = prev_H
        gap_dic[t] = gap
        has_prev_dic[t] = has_prev
        gap_flat_parts.append(gap)
        prev_index_flat_parts.append(prev_index)
        flat_offset += n

    gap_flat = np.concatenate(gap_flat_parts)
    prev_index_flat = np.concatenate(prev_index_flat_parts)

    return prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic, gap_flat, prev_index_flat


def whiten_observations(Y_dic, Hz_dic, sigma_obs_dic, prev_Y_dic, prev_H_dic, gap_dic, has_prev_dic, sigma2_rep, tau, nperiod):
    """
    One-shot GLS whitening transform for the AR(1)/OU residual-correlation
    model, phi(gap) = exp(-gap / tau).

    Only ever uses fixed, precomputed data (Y_dic, Hz_dic, and the same-site
    adjacency from prepare_ar1_indexing) -- never the evolving state -- so it
    is cheap to recompute every Gibbs iteration as new tau/sigma2_rep values
    are drawn, and its output (Y, Hz, err_var) can be fed directly into the
    existing, unmodified diagonal-R Kalman-gain machinery.

    tau <= 0 (including the disabled default) returns Y_dic/Hz_dic unchanged
    and err_var computed exactly as before this module existed.
    """
    Y_out, Hz_out, err_var_out = {}, {}, {}

    for t in range(nperiod):
        sigma_obs = sigma_obs_dic[t]
        err_var = sigma2_rep + sigma_obs**2

        if tau is None or tau <= 0:
            Y_out[t] = Y_dic[t]
            Hz_out[t] = Hz_dic[t]
            err_var_out[t] = err_var
            continue

        has_prev = has_prev_dic[t]
        phi = np.where(has_prev, np.exp(-gap_dic[t] / tau), 0.0)

        Y_out[t] = Y_dic[t] - phi * prev_Y_dic[t]
        Hz_out[t] = Hz_dic[t] - phi[:, None] * prev_H_dic[t]
        err_var_out[t] = np.where(has_prev, err_var * (1.0 - phi**2), err_var)

    return Y_out, Hz_out, err_var_out


def prepare_tau_sampler_inputs(state_residuals, sigma_obs, sigma2_rep, prev_index_flat):
    """
    Build the standardised-residual arrays consumed by sample_tau(), from the
    raw (unwhitened) residuals of the just-completed FFBS sweep.
    """
    err_var = sigma2_rep + sigma_obs**2
    standardised = state_residuals / np.sqrt(err_var)

    has_prev = prev_index_flat >= 0
    prev_standardised = np.zeros_like(standardised)
    prev_standardised[has_prev] = standardised[prev_index_flat[has_prev]]

    return standardised, prev_standardised, has_prev


def update_tau_resid(state_residuals, sigma_obs, tau_aprior, tau_bprior, tau_max, tau_scheme, tau_current, fixed_tau, sigma2_rep_current, obs_prev_index_flat, obs_gap_flat):
        
        
    if tau_scheme == "fixed":
            tau_current = fixed_tau
    else:
        standardised, prev_standardised, has_prev = prepare_tau_sampler_inputs(
            state_residuals, sigma_obs, sigma2_rep_current, obs_prev_index_flat,
        )
        tau_current = sample_tau(
            tau_current, tau_max, standardised, prev_standardised, has_prev,
            obs_gap_flat, tau_aprior, tau_bprior,
            )
    return tau_current


def tau_log_posterior(log_tau, tau_max, standardised, prev_standardised, has_prev, gap, alpha_prior, beta_prior):
    """
    Log posterior for tau under a scaled-Beta prior on tau / tau_max,
    mirroring sigma_rep.sigma2_rep_log_posterior's slice-sampler-friendly
    log-transformed form.
    """
    tau = np.exp(log_tau)

    if tau <= 0 or tau >= tau_max:
        return -np.inf

    phi = np.where(has_prev, np.exp(-gap / tau), 0.0)
    variance = np.where(has_prev, np.maximum(1.0 - phi**2, 1e-12), 1.0)
    innovation = standardised - phi * prev_standardised

    # Rows with no preceding same-site observation contribute a constant
    # N(0, 1) term that doesn't depend on tau, so they're excluded rather
    # than included as a no-op.
    loglik = -0.5 * np.sum(np.where(has_prev, np.log(variance) + innovation**2 / variance, 0.0))

    z = tau / tau_max
    logprior = (alpha_prior - 1) * np.log(z) + (beta_prior - 1) * np.log1p(-z)

    return loglik + logprior + log_tau  # Jacobian for tau = exp(log_tau)


def sample_tau(tau_current, tau_max, standardised, prev_standardised, has_prev, gap, alpha_prior, beta_prior, w=1.0, m=100):
    """
    Slice sampler for tau, directly analogous to sigma_rep.sample_sigma2_rep.
    """
    s_current = np.log(tau_current)
    s_upper = np.log(tau_max)

    if not np.isfinite(s_current):
        raise ValueError("Current tau has non-finite log posterior")

    logp = lambda s: tau_log_posterior(s, tau_max, standardised, prev_standardised, has_prev, gap, alpha_prior, beta_prior)

    logy = logp(s_current) - np.random.exponential(1)

    u = np.random.rand()
    L = s_current - w * u
    R = L + w
    R = min(R, s_upper)

    j = int(np.floor(m * np.random.rand()))
    k = (m - 1) - j

    while j > 0 and logp(L) > logy:
        L -= w
        j -= 1

    while k > 0 and R < s_upper and logp(R) > logy:
        R += w
        R = min(R, s_upper)
        k -= 1

    while True:
        s_new = np.random.uniform(L, R)

        if logp(s_new) > logy:
            return np.exp(s_new)
        elif s_new < s_current:
            L = s_new
        else:
            R = s_new
