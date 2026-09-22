"""
Run several independent copies of the BRISTAU Gibbs sampler ("chains") and
check that they agree with each other, as a convergence diagnostic.

Why bother running more than one chain?
----------------------------------------
A single MCMC chain can *look* perfectly healthy -- a nice, stable-looking
trace with no obvious trend -- while actually only having explored one small
region of the true posterior distribution. There is nothing you can see in
a single trace that rules this out. The standard fix is to run several
chains from independent random starting points and check that they all end
up describing the same distribution. If they do, that is good evidence the
sampler has converged; if they don't, at least one of them (maybe all of
them) cannot be trusted. See Gelman & Rubin (1992) and Vehtari et al. (2021)
for the statistical background to the "R-hat" statistic used below.

Why only keep one chain's output?
------------------------------------
If the chains agree, they are statistically interchangeable -- any one of
them is an equally valid sample from the posterior. There is therefore no
benefit to keeping all of them, only a large and unnecessary storage cost
(the full emissions trace, "xtrace", can be a sizeable array). This module
keeps one designated "keeper" chain in full and only ever pulls small
hyperparameter summaries out of the others, before discarding them.

How this stays fast
----------------------
Because the chains are completely independent of each other (same input,
different random numbers), they can run at the same time rather than one
after another. This module farms the extra chains out to separate
operating-system processes using Python's ProcessPoolExecutor. As long as
the machine has at least `nchain` free CPU cores, the wall-clock time for
this step stays close to the time for a single chain, not `nchain` times
that. If your job/environment doesn't have that many cores available,
Python and the operating system will still get the right answer, just more
slowly (chains queue up rather than truly running in parallel).
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import arviz as az

from inversion_methods.bristau.inversion_bristau import augmented_ffbs_mxkf_gibbs_multi_slice


# The names of the traces we compare across chains, and where to find each
# one in the tuple returned by augmented_ffbs_mxkf_gibbs_multi_slice.
#
#   index 0: xtrace           -- the full emissions state (kept only for the keeper chain)
#   index 3: var_rep_trace    -- sigma2_rep (observation "representation error" variance)
#   index 4: sigma2_qx_trace  -- flux forecast noise variance(s)
#   index 5: kappa_x_trace    -- flux persistence parameter(s)
#   index 8: tau_trace        -- same-site residual correlation length(s)
#
# See augmented_ffbs_mxkf_gibbs_multi_slice's own docstring in
# inversion_bristau.py for the full description of every element.


def _summarise_chain_output(gibbs_output):
    """
    Pull out just the small pieces of a finished Gibbs sampler run that are
    needed to check whether chains agree with each other, and throw away
    the large per-basis-function emissions trace (xtrace).

    We keep:
      - The hyperparameter traces (sigma2_rep, sigma2_qx, kappa_x, tau).
        These are the values most likely to mix slowly, since each one is
        shared across many basis functions/observations and is sampled
        conditional on everything else in the model. Note that if any of
        these was configured as a *fixed* value rather than sampled (see
        the ini file's HYPERPARAMETER_OPTIONS), its trace here is just a
        constant -- there is nothing to check for that one, and
        _check_convergence() below is written to quietly skip it rather
        than treat "constant" as "failed to converge".
      - One cheap, always-available stand-in for the emissions state
        itself: the plain (unweighted) average scaling factor across all
        basis functions, for each time period. This is NOT the same as a
        proper flux-weighted national/domain total (that needs area and
        prior-flux information this function doesn't have -- see
        bristau_postprocessouts for that calculation), but it is a useful,
        free-to-compute check that the emissions state itself is mixing
        well, not just the hyperparameters.
    """
    (xtrace, _bctrace, _rtrace, var_rep_trace, sigma2_qx_trace,
     kappa_x_trace, _sigma2_qx_labels, _kappa_x_labels, tau_trace,
     _tau_labels) = gibbs_output

    return {
        "sigma2_rep": var_rep_trace,
        "sigma2_qx": sigma2_qx_trace,
        "kappa_x": kappa_x_trace,
        "tau_resid": tau_trace,
        "mean_x": xtrace.mean(axis=-1),  # shape (retained_iterations, nperiod)
    }


def _run_one_chain(inversion_input, seed_sequence, keep_full_output):
    """
    Run a single, independent chain. This function is what actually
    executes inside each separate worker process spawned by run_chains().

    Parameters
    ----------
    inversion_input : InversionInput
        Exactly the same sampler input used by every chain -- only the
        random numbers differ between chains, not the data or priors.
    seed_sequence : numpy.random.SeedSequence
        A seed unique to this chain (see run_chains() for how these are
        generated). This is turned into an explicit numpy.random.Generator
        below and passed all the way through the sampler, so this chain's
        entire sequence of random draws -- hyperparameters and the
        emissions state alike -- depends only on this seed and nothing
        else (not on process start-up order, not on any global state).
        That is what makes each chain independent of the others, and what
        makes the same seed reproduce the same chain again later.
    keep_full_output : bool
        True for exactly one chain (the "keeper", see run_chains()). That
        chain returns its full set of traces. Every other chain only needs
        to feed into the convergence check, so it summarises its own
        output (see _summarise_chain_output above) and throws the large
        traces away *before* sending anything back to the main process --
        this keeps the amount of data copied between processes small.

    Returns
    -------
    Either the full Gibbs sampler output tuple (see
    augmented_ffbs_mxkf_gibbs_multi_slice's docstring), or the small
    summary dictionary produced by _summarise_chain_output.
    """
    # Every random draw in the sampler -- every slice-sampled
    # hyperparameter and the backward-sampled emissions state -- accepts
    # an explicit Generator and uses it instead of any shared/global
    # random state whenever one is given (see augmented_ffbs_mxkf_gibbs_
    # multi_slice's own docstring). Building that Generator directly from
    # this chain's own seed is what makes this chain's whole random
    # sequence independent of every other chain's, and reproducible on
    # its own if you rerun it with the same seed later.
    rng = np.random.default_rng(seed_sequence)

    gibbs_output = augmented_ffbs_mxkf_gibbs_multi_slice(inversion_input, rng=rng)

    if keep_full_output:
        return gibbs_output

    return _summarise_chain_output(gibbs_output)


def _stack_summaries(summaries):
    """
    Line up a list of per-chain summary dictionaries (one dictionary per
    chain, all with the same keys) into arrays shaped
    (nchain, retained_iterations, ...) -- the layout arviz expects for
    computing R-hat and effective sample size (ESS).
    """
    keys = summaries[0].keys()
    return {key: np.stack([summary[key] for summary in summaries]) for key in keys}


def _check_convergence(stacked_summaries, rhat_threshold):
    """
    Compute R-hat and ESS for every monitored parameter, across all
    chains, and decide whether the chains agree well enough to trust.

    What R-hat means
    ------------------
    R-hat compares how much a parameter varies *within* each chain to how
    much it varies *between* chains. If the chains have converged to the
    same posterior distribution, these two should be similar and R-hat
    should sit close to 1.0. A value clearly above the threshold (commonly
    1.01, see Vehtari et al. 2021) is a warning sign that the chains
    disagree -- usually meaning the sampler has not converged, and the
    chain you keep may not be a trustworthy sample from the true
    posterior.

    What ESS means
    -----------------
    MCMC draws are typically autocorrelated (each draw depends on the one
    before it), so your N retained iterations are usually worth *fewer*
    than N independent samples. ESS estimates how many independent samples
    your draws are actually equivalent to. A low ESS means your reported
    uncertainty intervals may be narrower than they should be, even if
    R-hat looks fine.

    A note on fixed (non-sampled) hyperparameters
    ------------------------------------------------
    If a hyperparameter was configured as a fixed numeric value rather
    than something the sampler infers (see the ini file's
    HYPERPARAMETER_OPTIONS section), its trace is just the same constant
    number repeated -- there is no variation to compare, and R-hat for it
    comes back as "not a number" (NaN). That is not a failure, it just
    means there was nothing to check for that particular parameter, so
    those NaNs are quietly ignored below rather than counted against
    convergence.
    """
    idata = az.from_dict(posterior=stacked_summaries)
    rhat = az.rhat(idata)
    ess = az.ess(idata)

    report = {"rhat_threshold": rhat_threshold, "parameters": {}}
    finite_rhat_values = []

    for name in stacked_summaries:
        rhat_values = np.atleast_1d(rhat[name].values).astype(float)
        ess_values = np.atleast_1d(ess[name].values).astype(float)

        report["parameters"][name] = {
            "rhat": rhat_values.tolist(),
            "ess": ess_values.tolist(),
        }
        finite_rhat_values.extend(rhat_values[np.isfinite(rhat_values)].tolist())

    if finite_rhat_values:
        report["worst_rhat"] = float(np.max(finite_rhat_values))
        report["converged"] = bool(report["worst_rhat"] < rhat_threshold)
        report["any_parameter_checked"] = True
    else:
        # Every monitored parameter was fixed in this run, so there was
        # nothing stochastic to compare between chains. This is not a
        # sign of a broken sampler -- it just means this particular
        # configuration left nothing here to check.
        report["worst_rhat"] = float("nan")
        report["converged"] = True
        report["any_parameter_checked"] = False

    return report


def _report_convergence(report, require_convergence):
    """
    Print a clear pass/fail summary of the convergence check, and either
    raise an error (require_convergence=True) or just warn loudly
    (require_convergence=False, the default) if it failed.
    """
    if report["converged"]:
        print(
            f"[nchain convergence check] PASSED -- all {report['nchain']} chains agree "
            f"(worst R-hat = {report['worst_rhat']:.4f}, threshold = {report['rhat_threshold']}).",
            flush=True,
        )
        return

    banner = "!" * 78
    message_lines = [
        banner,
        "CONVERGENCE CHECK FAILED",
        f"The {report['nchain']} independent chains do not agree with each other",
        f"(worst R-hat = {report['worst_rhat']:.4f}, threshold = {report['rhat_threshold']}).",
        "This usually means the sampler has not converged, and the chain that",
        "was kept may not be a trustworthy sample from the true posterior.",
        "",
        "Per-parameter R-hat values (close to 1.0 is good; 'nan' means that",
        "parameter was fixed rather than sampled in this run, so it was not checked):",
    ]
    for name, values in report["parameters"].items():
        message_lines.append(f"    {name}: {values['rhat']}")
    message_lines.append(banner)
    message = "\n".join(message_lines)

    if require_convergence:
        raise RuntimeError(message)

    print(message, flush=True)


def run_chains(
    inversion_input,
    nchain=1,
    chain_seed=None,
    rhat_threshold=1.01,
    require_convergence=False,
    keep_chain_index=0,
):
    """
    Run the Gibbs sampler once (nchain=1, the default) or as several fully
    independent chains (nchain > 1), and, in the latter case, check that
    they agree with each other before returning just one of them. See the
    module docstring above for the reasoning behind all of this.

    Parameters
    ----------
    inversion_input : InversionInput
        The sampler input, identical for every chain.
    nchain : int, default 1
        Number of independent chains to run. nchain=1 skips the
        multi-process/convergence-check machinery below entirely -- the
        sampler is just called directly, exactly as this function's
        single-chain predecessor always did.
    chain_seed : int or None, default None
        Optional seed controlling every random draw the sampler makes
        (both here and for nchain=1). Leave as None for a genuinely random
        run every time -- the default, and the same as before this option
        existed. Supply an integer to make the run fully reproducible:
        the same chain_seed always regenerates the same chain(s) later,
        which is useful for reproducing a specific result or a specific
        convergence check. With nchain>1, chain_seed is the single
        top-level seed that per_chain_seeds below are independently
        derived from -- it does not, by itself, pick any one chain's
        seed directly.
    rhat_threshold : float, default 1.01
        R-hat values at or above this are treated as "the chains
        disagree". 1.01 is a commonly used, fairly strict, default
        (Vehtari et al. 2021).
    require_convergence : bool, default False
        If True, a failed convergence check raises a RuntimeError and no
        result is returned. If False (the default), a failed check only
        prints a loud warning and still returns the kept chain's result
        and the convergence report -- this is deliberately the default so
        that you can run this a few times, look at how often (and by how
        much) real configurations fail the check, and only turn on the
        strict/blocking behaviour once you trust it.
    keep_chain_index : int, default 0
        Which of the nchain chains to keep and return in full. This is
        picked *before* looking at any results, since if the chains agree,
        any one of them is an equally valid choice -- picking a chain
        based on how it looks after the fact would bias the result.

    Returns
    -------
    gibbs_output : tuple
        Exactly what augmented_ffbs_mxkf_gibbs_multi_slice would have
        returned on its own -- everything downstream of this function
        (post-processing, output file writing) is unaffected by nchain.
    convergence_report : dict or None
        None when nchain == 1 (there is nothing to compare against).
        Otherwise a dictionary with a per-parameter R-hat/ESS breakdown
        and an overall "converged" True/False flag -- see
        _check_convergence for the full description.
    """
    if nchain == 1:
        # No seed given: behave exactly as this function's single-chain
        # predecessor always did (an ordinary, non-reproducible run).
        if chain_seed is None:
            return augmented_ffbs_mxkf_gibbs_multi_slice(inversion_input), None
        # A seed was given, even though there's only one chain -- honour
        # it anyway, so "give me a reproducible run" works the same way
        # whether or not you're also asking for a convergence check.
        rng = np.random.default_rng(chain_seed)
        return augmented_ffbs_mxkf_gibbs_multi_slice(inversion_input, rng=rng), None

    if nchain < 1:
        raise ValueError(f"nchain must be a positive integer, got {nchain}.")

    if not (0 <= keep_chain_index < nchain):
        raise ValueError(f"keep_chain_index must be between 0 and nchain-1, got {keep_chain_index} for nchain={nchain}.")

    # SeedSequence.spawn() hands out `nchain` seeds that are guaranteed to
    # be statistically independent of each other and of chain_seed itself
    # -- this is what makes "the chains agree" a meaningful statement:
    # if chains secretly shared random draws, agreement between them
    # would prove nothing.
    per_chain_seeds = np.random.SeedSequence(chain_seed).spawn(nchain)

    # "spawn" starts each worker as a brand new Python process, rather
    # than copying ("forking") the current one. Each chain's random draws
    # come from its own explicit Generator (built from per_chain_seeds
    # above, see _run_one_chain), so independence between chains no longer
    # depends on this choice -- but "spawn" is still the safer default: a
    # forked child also inherits copies of open files, locks, and anything
    # else live in the parent process at the moment of the fork, which can
    # cause its own (unrelated, harder-to-debug) problems.
    spawn_context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=nchain, mp_context=spawn_context) as pool:
        futures = [
            pool.submit(_run_one_chain, inversion_input, seed, chain_index == keep_chain_index)
            for chain_index, seed in enumerate(per_chain_seeds)
        ]
        # .result() blocks until that particular chain is done; since all
        # nchain chains were already submitted above and are running
        # concurrently, this loop finishes as soon as the slowest chain
        # does, not after nchain chains' worth of sequential waiting.
        chain_results = [future.result() for future in futures]

    kept_result = chain_results[keep_chain_index]

    # Every chain needs to be in "summary" form (small hyperparameter
    # traces only) before the convergence check -- the keeper chain is
    # still holding its full output at this point, so it gets summarised
    # here, in the main process, right before checking.
    summaries = [
        _summarise_chain_output(result) if chain_index == keep_chain_index else result
        for chain_index, result in enumerate(chain_results)
    ]

    convergence_report = _check_convergence(_stack_summaries(summaries), rhat_threshold)
    convergence_report["nchain"] = nchain

    _report_convergence(convergence_report, require_convergence)

    return kept_result, convergence_report
