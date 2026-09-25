"""
Tests for inversion_methods.bristau.multichain: running several independent
copies of the Gibbs sampler and checking they agree with each other (a
convergence check), while only keeping one chain's full output.
"""

import numpy as np
import pytest

from inversion_methods.bristau.inversion_bristau import InversionInput
from inversion_methods.bristau.multichain import (
    run_chains,
    _run_one_chain,
    _summarise_chain_output,
    _stack_summaries,
    _check_convergence,
)


def _make_small_inversion_input(iterations=40, seed=0):
    """
    A tiny, fully-synthetic InversionInput -- fast enough to run several
    times per test, but real enough to exercise the whole Gibbs loop.

    sigma_rep is left as the (sampled) 'global additive' scheme so at
    least one hyperparameter genuinely varies across iterations, in
    addition to the emissions state itself; sigma_qx and kappa_x are
    fixed, purely to keep this toy problem simple and fast.
    """
    # nr is left at 2 (rather than the 1 that would normally pair with
    # nxout=0) because inversion_bristau.py currently hardcodes the
    # reference-state prior variance to length 2 regardless of nr -- an
    # unrelated, pre-existing quirk, not something this test is about.
    nperiod, nbasis, nxout, nr, ny_per_period = 3, 4, 0, 2, 6
    rng = np.random.default_rng(seed)

    Y_dic, sigma_obs_dic, Ytime_dic, Hz_dic, Hx_dic, siteindicator_dic = {}, {}, {}, {}, {}, {}
    for t in range(nperiod):
        Hx = rng.uniform(0.1, 1.0, size=(ny_per_period, nbasis))
        Hr = np.zeros((ny_per_period, nr))
        Hz_dic[t] = np.hstack([Hx, Hr])
        Hx_dic[t] = Hx
        true_x = rng.uniform(0.8, 1.2, size=nbasis)
        Y_dic[t] = Hx @ true_x + rng.normal(scale=0.1, size=ny_per_period)
        sigma_obs_dic[t] = np.full(ny_per_period, 0.1)
        Ytime_dic[t] = np.arange(ny_per_period)
        siteindicator_dic[t] = np.zeros(ny_per_period)

    xprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}
    rprior = {"pdf": "lognormal", "mu": 0.0, "sigma": 0.5}

    return InversionInput(
        Y_dic=Y_dic, sigma_obs_dic=sigma_obs_dic, Ytime_dic=Ytime_dic, Hz_dic=Hz_dic,
        Hx_dic=Hx_dic, Hbc_dic={}, siteindicator_dic=siteindicator_dic,
        nperiod=nperiod, nbc=0, nxout=nxout, nr=nr, nbasis=nbasis,
        xprior=xprior, bcprior=None, rprior=rprior,
        sigma2_rep_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        sigma2_qx_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        sigma_qbc=0.0, sigma_qr=0.1, kappa_x_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
        iterations=iterations, inner_group_id=None, ningroup=None,
        sigma_rep="global additive", sigma_rep_max=100.0,
        sigma_qx=0.02, sigma_qx_max=0.5,
        kappa_x=0.3, kappa_x_minfold=1, kappa_bc=None, kappa_r=0.0,
    )


# ---------------------------------------------------------------------
# nchain=1: this feature must be a complete no-op unless explicitly used
# ---------------------------------------------------------------------

def test_nchain_1_bypasses_multichain_machinery_entirely(monkeypatch):
    """
    nchain=1 (the default) must be a pure pass-through to the sampler: no
    new processes, no random-number reseeding, no convergence report.

    This is checked with a stub sampler rather than by comparing two real
    runs' random output, because augmented_backward_sampler draws from its
    own internal, unseeded random generator (see
    bristau_filter_sampler.py) -- so two real calls are never expected to
    produce identical numbers, with or without this feature. What we can
    (and should) confirm is that run_chains(nchain=1) calls the sampler
    exactly once, with the input unchanged, and hands its result straight
    back with no modification.
    """
    import inversion_methods.bristau.multichain as multichain

    sentinel_result = object()
    calls = []

    def fake_sampler(inversion_input):
        calls.append(inversion_input)
        return sentinel_result

    monkeypatch.setattr(multichain, "augmented_ffbs_mxkf_gibbs_multi_slice", fake_sampler)

    config = _make_small_inversion_input()
    result, report = run_chains(config, nchain=1)

    assert result is sentinel_result
    assert report is None
    assert calls == [config]


def test_run_chains_rejects_bad_nchain_or_keep_index():
    config = _make_small_inversion_input()

    with pytest.raises(ValueError):
        run_chains(config, nchain=0)

    with pytest.raises(ValueError):
        run_chains(config, nchain=2, keep_chain_index=2)


# ---------------------------------------------------------------------
# Chain independence: the whole convergence check only means something if
# each chain's random draws are genuinely independent of the others.
# ---------------------------------------------------------------------

def test_run_one_chain_with_different_seeds_gives_different_draws():
    config = _make_small_inversion_input()
    seeds = np.random.SeedSequence(0).spawn(2)

    result_a = _run_one_chain(config, seeds[0], keep_full_output=True)
    result_b = _run_one_chain(config, seeds[1], keep_full_output=True)

    xtrace_a, xtrace_b = result_a[0], result_b[0]
    assert not np.array_equal(xtrace_a, xtrace_b)


def test_run_one_chain_only_prints_for_the_keeper_chain(capsys):
    """
    Progress output (the periodic 'Iteration: ...'/'z sample: ...' lines
    and the MXKF convergence-count line) must only appear for the chain
    being kept -- otherwise running nchain chains at once would interleave
    nchain copies of the same progress output.
    """
    config = _make_small_inversion_input(iterations=5)
    seed = np.random.SeedSequence(3).spawn(1)[0]

    _run_one_chain(config, seed, keep_full_output=False)
    assert "Iteration:" not in capsys.readouterr().out

    _run_one_chain(config, seed, keep_full_output=True)
    assert "Iteration:" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Summary/stacking helpers, tested directly (no need for real subprocesses)
# ---------------------------------------------------------------------

def test_summarise_and_stack_round_trip():
    config = _make_small_inversion_input()
    seed = np.random.SeedSequence(1).spawn(1)[0]
    full_result = _run_one_chain(config, seed, keep_full_output=True)

    summary = _summarise_chain_output(full_result)
    assert set(summary.keys()) == {"sigma2_rep", "sigma2_qx", "kappa_x", "tau_resid", "mean_x"}
    assert summary["mean_x"].shape == (full_result[0].shape[0], full_result[0].shape[1])

    stacked = _stack_summaries([summary, summary])
    for array in stacked.values():
        assert array.shape[0] == 2  # the new leading "chain" axis
        np.testing.assert_array_equal(array[0], array[1])


def test_check_convergence_flags_clearly_disagreeing_chains():
    """Two chains centred on very different values must be flagged as not
    having converged."""
    chain_a = {"toy_param": np.random.default_rng(0).normal(loc=0.0, size=(50,))}
    chain_b = {"toy_param": np.random.default_rng(1).normal(loc=10.0, size=(50,))}

    report = _check_convergence(_stack_summaries([chain_a, chain_b]), rhat_threshold=1.01)

    assert report["converged"] is False
    assert report["any_parameter_checked"] is True
    assert report["worst_rhat"] > 1.01


def test_check_convergence_passes_agreeing_chains():
    """Several chains drawing from the same distribution should be judged
    as having converged."""
    rng = np.random.default_rng(0)
    summaries = [{"toy_param": rng.normal(loc=0.0, scale=1.0, size=(500,))} for _ in range(4)]

    report = _check_convergence(_stack_summaries(summaries), rhat_threshold=1.05)

    assert report["converged"] is True
    assert report["any_parameter_checked"] is True


def test_check_convergence_treats_all_fixed_parameters_as_vacuously_converged():
    """If every monitored parameter was fixed (constant) rather than
    sampled, there is nothing to disagree about -- this must not be
    mistaken for a failed check."""
    chain_a = {"fixed_param": np.full(30, 7.0)}
    chain_b = {"fixed_param": np.full(30, 7.0)}

    report = _check_convergence(_stack_summaries([chain_a, chain_b]), rhat_threshold=1.01)

    assert report["converged"] is True
    assert report["any_parameter_checked"] is False
    assert np.isnan(report["worst_rhat"])


# ---------------------------------------------------------------------
# End-to-end: real multi-process chains via run_chains()
# ---------------------------------------------------------------------

def test_run_chains_two_chains_end_to_end():
    config = _make_small_inversion_input(iterations=40)

    result, report = run_chains(config, nchain=2, rhat_threshold=1.01, require_convergence=False)

    n_retained = config.iterations - int(0.2 * config.iterations)
    xtrace = result[0]
    assert xtrace.shape == (n_retained, config.nperiod, config.nbasis)

    assert report["nchain"] == 2
    assert set(report["parameters"].keys()) == {"sigma2_rep", "sigma2_qx", "kappa_x", "tau_resid", "mean_x"}
    assert isinstance(report["converged"], bool)


# ---------------------------------------------------------------------
# Reproducibility: chain_seed must make a run (or a set of chains) exactly
# repeatable, since augmented_backward_sampler otherwise seeds its own
# generator freshly (and unreproducibly) on every call.
# ---------------------------------------------------------------------

def test_run_chains_nchain_1_with_seed_is_reproducible():
    config = _make_small_inversion_input(iterations=20)

    result_a, report_a = run_chains(config, nchain=1, chain_seed=7)
    result_b, report_b = run_chains(config, nchain=1, chain_seed=7)

    assert report_a is None and report_b is None
    for array_a, array_b in zip(result_a, result_b):
        if isinstance(array_a, np.ndarray):
            np.testing.assert_array_equal(array_a, array_b)
        else:
            assert array_a == array_b


def test_run_chains_multi_chain_with_seed_is_reproducible():
    """
    Running the same nchain=2, chain_seed=... configuration twice must
    produce the same kept chain and the same convergence report -- not
    just "two chains that happen to agree with each other", but the exact
    same numbers both times.
    """
    config = _make_small_inversion_input(iterations=20)

    result_a, report_a = run_chains(config, nchain=2, chain_seed=11)
    result_b, report_b = run_chains(config, nchain=2, chain_seed=11)

    for array_a, array_b in zip(result_a, result_b):
        if isinstance(array_a, np.ndarray):
            np.testing.assert_array_equal(array_a, array_b)
        else:
            assert array_a == array_b

    assert report_a["worst_rhat"] == report_b["worst_rhat"]


def test_run_chains_different_keep_chain_index_gives_different_chains():
    """
    With the same chain_seed and nchain, asking for chain 0 vs chain 1
    should return genuinely different chains (proving each index really is
    an independent chain, not accidentally the same seed reused).
    """
    config = _make_small_inversion_input(iterations=20)

    result_chain_0, _ = run_chains(config, nchain=2, chain_seed=5, keep_chain_index=0)
    result_chain_1, _ = run_chains(config, nchain=2, chain_seed=5, keep_chain_index=1)

    assert not np.array_equal(result_chain_0[0], result_chain_1[0])


def test_run_chains_require_convergence_raises_on_failure():
    """
    rhat_threshold=0.0 can never be satisfied (R-hat is always positive),
    so this deterministically exercises the "failed" branch without
    depending on whether this particular toy problem happens to converge.
    """
    config = _make_small_inversion_input(iterations=20)

    with pytest.raises(RuntimeError, match="CONVERGENCE CHECK FAILED"):
        run_chains(config, nchain=2, rhat_threshold=0.0, require_convergence=True)


def test_run_chains_default_does_not_raise_on_failure(capsys):
    """
    The default (require_convergence=False) must only warn loudly, not
    raise, so a user can see how often runs fail to converge without every
    failure aborting the pipeline.
    """
    config = _make_small_inversion_input(iterations=20)

    result, report = run_chains(config, nchain=2, rhat_threshold=0.0, require_convergence=False)

    assert report["converged"] is False
    assert "CONVERGENCE CHECK FAILED" in capsys.readouterr().out

    n_retained = config.iterations - int(0.2 * config.iterations)
    assert result[0].shape[0] == n_retained
