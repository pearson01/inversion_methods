"""
Tests for the joint (kappa_x, sigma2_qx) update: kappa_x drawn from its
marginal with sigma2_qx integrated out (kappa_x.update_kappa_x_marginal),
followed by the existing conjugate sigma2_qx | kappa_x draw.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import invgamma

from inversion_methods.bristau.kappa_x import (
    innovation_moments,
    kappa_log_marginal,
    sample_kappa,
    sample_kappa_marginal,
    update_kappa_x_marginal,
)
from inversion_methods.bristau.siqma_qx import sample_sigma2_qx

from test_integration_gibbs import _make_inversion_input
from inversion_methods.bristau.inversion_bristau import augmented_ffbs_mxkf_gibbs_multi_slice


def _synthetic_relaxing_samples(true_kappa, n_steps, n_series, sigma_process=0.3, seed=1):
    """
    zmusample-style trajectories [x..., r] with x_t = kappa*x_{t-1} + (1-kappa)*r_{t-1} + noise
    and a non-constant reference trajectory r (so the r_{t-1} term matters).
    """
    rng = np.random.default_rng(seed)
    r = 1.0 + np.cumsum(rng.normal(scale=0.1, size=n_steps))
    x = np.zeros((n_steps, n_series))
    x[0] = r[0] + rng.normal(size=n_series)
    for t in range(1, n_steps):
        x[t] = true_kappa * x[t - 1] + (1 - true_kappa) * r[t - 1] + rng.normal(scale=sigma_process, size=n_series)
    return np.column_stack((x, r))


# ---------------------------------------------------------------------------
# innovation_moments / kappa_log_marginal
# ---------------------------------------------------------------------------


def test_innovation_moments_match_sigma2_qx_innovations():
    """S(kappa) from the moments equals the sum of squared innovations used by sample_sigma2_qx."""
    z = _synthetic_relaxing_samples(0.6, 20, 3)
    saa, sab, sbb, n = innovation_moments(z[:, :3], z[:, -1])

    for kappa in [0.1, 0.5, 0.9]:
        x_pred = kappa * z[:-1, :3] + (1 - kappa) * z[:-1, -1][:, None]
        direct = np.sum((z[1:, :3] - x_pred) ** 2)
        assert saa - 2 * kappa * sab + kappa**2 * sbb == pytest.approx(direct)

    assert n == 19 * 3


@pytest.mark.parametrize("sigma2_qx_max", [np.inf, 0.15])
def test_kappa_log_marginal_matches_numerical_integration(sigma2_qx_max):
    """
    Differences in the collapsed log density between two kappa values must
    match log of the numerically integrated int p(x | kappa, s2) p(s2) ds2
    over the truncated inverse-gamma prior, including when truncation binds.
    """
    z = _synthetic_relaxing_samples(0.6, 6, 2, sigma_process=0.3, seed=4)
    moments = [innovation_moments(z[:, :2], z[:, -1])]
    saa, sab, sbb, n = moments[0]
    alpha, beta = 2.0, 0.1

    def log_integrated(kappa):
        S = saa - 2 * kappa * sab + kappa**2 * sbb
        integrand = lambda s2: s2 ** (-n / 2) * np.exp(-S / (2 * s2)) * invgamma.pdf(s2, a=alpha, scale=beta)
        upper = min(sigma2_qx_max, 50.0)
        return np.log(quad(integrand, 0, upper, points=[S / n], limit=200)[0])

    def log_marginal(kappa):
        # Uniform kappa prior so only the likelihood term differs.
        return kappa_log_marginal(kappa, 0.99, moments, alpha, beta, sigma2_qx_max, 1.0, 1.0)

    for k1, k2 in [(0.2, 0.6), (0.4, 0.9)]:
        assert log_marginal(k1) - log_marginal(k2) == pytest.approx(log_integrated(k1) - log_integrated(k2), abs=1e-6)


def test_kappa_log_marginal_rejects_out_of_bounds():
    moments = [(1.0, 0.5, 1.0, 10)]
    assert kappa_log_marginal(0.0, 0.9, moments, 2.0, 0.1, 1.0, 2.0, 2.0) == -np.inf
    assert kappa_log_marginal(0.9, 0.9, moments, 2.0, 0.1, 1.0, 2.0, 2.0) == -np.inf


# ---------------------------------------------------------------------------
# Joint sampler targets the same posterior as the conditional Gibbs sampler
# ---------------------------------------------------------------------------


def test_joint_update_matches_conditional_gibbs_posterior():
    """
    For fixed x, r: alternating (kappa | sigma2, sigma2 | kappa) and the joint
    (kappa marginal, then sigma2 | kappa) both target p(kappa, sigma2 | x, r),
    so their posterior means must agree.
    """
    z = _synthetic_relaxing_samples(0.6, 8, 2, sigma_process=0.3, seed=2)
    nbasis = 2
    alpha, beta, s2max, kmax = 2.0, 0.1, 5.0, 0.99
    moments = [innovation_moments(z[:, :nbasis], z[:, -1])]
    niter = 4000

    rng = np.random.default_rng(0)
    kappa, sigma2 = 0.5, 0.1
    conditional = np.zeros((niter, 2))
    for i in range(niter):
        sigma2 = sample_sigma2_qx(z, kappa, alpha, beta, nbasis, s2max, rng=rng)
        kappa = sample_kappa(z, np.full(nbasis, sigma2), kappa, kmax, 1.0, 1.0, nbasis, rng=rng)
        conditional[i] = kappa, sigma2

    rng = np.random.default_rng(1)
    kappa = 0.5
    joint = np.zeros((niter, 2))
    for i in range(niter):
        kappa = sample_kappa_marginal(moments, kappa, kmax, 1.0, 1.0, alpha, beta, s2max, rng=rng)
        sigma2 = sample_sigma2_qx(z, kappa, alpha, beta, nbasis, s2max, rng=rng)
        joint[i] = kappa, sigma2

    burn = 500
    np.testing.assert_allclose(joint[burn:].mean(axis=0), conditional[burn:].mean(axis=0), rtol=0.05)


def test_sample_kappa_marginal_recovers_true_persistence():
    true_kappa = 0.75
    z = _synthetic_relaxing_samples(true_kappa, 300, 4, sigma_process=0.05, seed=11)
    moments = [innovation_moments(z[:, :4], z[:, -1])]

    rng = np.random.default_rng(0)
    draws = np.array([sample_kappa_marginal(moments, 0.5, 0.995, 1.0, 1.0, 2.0, 0.1, 10.0, rng=rng) for _ in range(50)])

    assert np.all((draws > 0) & (draws < 0.995))
    assert abs(np.mean(draws) - true_kappa) < 0.1


# ---------------------------------------------------------------------------
# update_kappa_x_marginal dispatch
# ---------------------------------------------------------------------------


def _split(z, nxout, nbasis):
    """Mimic the outer/inner state split in the Gibbs loop for [x_out, x_in, r_out, r_in]."""
    z_out = np.column_stack((z[:, :nxout], z[:, -2]))
    z_in = np.column_stack((z[:, nxout:nbasis], z[:, -1]))
    return z_out, z_in


def _outer_inner_samples(nxout=2, nxin=4, n_steps=30):
    outer = _synthetic_relaxing_samples(0.4, n_steps, nxout, seed=5)
    inner = _synthetic_relaxing_samples(0.8, n_steps, nxin, seed=6)
    return np.column_stack((outer[:, :nxout], inner[:, :nxin], outer[:, -1], inner[:, -1]))


@pytest.mark.parametrize("sigma_qx_scheme", ["inner outer", "inner outer country"])
def test_update_kappa_x_marginal_inner_outer(sigma_qx_scheme):
    nxout, nxin = 2, 4
    z = _outer_inner_samples(nxout, nxin)
    z_out, z_in = _split(z, nxout, nxout + nxin)
    inner_group_id = np.array([0, 0, 1, 1])

    sample = update_kappa_x_marginal(z, z_out, z_in, nxout + nxin, nxout, nxin,
                                     1.1, 1.1, 0.95, "inner outer", np.array([0.5, 0.5]),
                                     2.0, 0.1, 10.0, sigma_qx_scheme, inner_group_id, 2,
                                     rng=np.random.default_rng(0))

    assert sample.shape == (2,)
    assert np.all((sample > 0) & (sample < 0.95))


def test_update_kappa_x_marginal_global():
    z = _synthetic_relaxing_samples(0.6, 30, 3)
    sample = update_kappa_x_marginal(z, None, None, 3, 0, 3,
                                     1.1, 1.1, 0.95, "global", np.array([0.5]),
                                     2.0, 0.1, 10.0, "global", None, None,
                                     rng=np.random.default_rng(0))

    assert sample.shape == (1,)
    assert 0 < sample[0] < 0.95


def test_update_kappa_x_marginal_rejects_unsupported_combination():
    nxout, nxin = 2, 4
    z = _outer_inner_samples(nxout, nxin)
    z_out, z_in = _split(z, nxout, nxout + nxin)

    with pytest.raises(ValueError):
        update_kappa_x_marginal(z, z_out, z_in, nxout + nxin, nxout, nxin,
                                1.1, 1.1, 0.95, "inner outer", np.array([0.5, 0.5]),
                                2.0, 0.1, 10.0, "global", None, None)


# ---------------------------------------------------------------------------
# End-to-end Gibbs loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint", [True, False])
def test_gibbs_sampler_kappa_and_sigma_qx_inner_outer(joint):
    config = _make_inversion_input(
        kappa_x="inner outer", nxout=2, nbc=0, sigma_qx="inner outer",
        sigma2_exc_prior={"pdf": "beta", "shape": 2.0, "scale": 2.0},
    )
    config.joint_kappa_sigma_qx = joint
    outputs = augmented_ffbs_mxkf_gibbs_multi_slice(config, rng=np.random.default_rng(0), verbose=False)
    sigma2_qx_trace, kappa_x_trace = outputs[4], outputs[5]

    assert kappa_x_trace.shape[1] == 2
    assert sigma2_qx_trace.shape[1] == 2
    assert np.all(np.isfinite(kappa_x_trace)) and np.all(np.isfinite(sigma2_qx_trace))
    assert len(np.unique(kappa_x_trace[:, 1])) > 1
