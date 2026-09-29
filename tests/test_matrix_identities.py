"""
Unit tests for inversion_methods.manipulation.matrix_identities.

These check the Woodbury-identity/Cholesky-based Kalman gain and covariance
shortcuts against naive, textbook dense-linear-algebra implementations of
the same quantities. This is exactly the kind of invariant that would catch
a subtle indexing/assignment bug in the fast path (the same category of bug
as the kappa_x `==`/`=` typo) without needing to know the fast path's
internals.
"""

import numpy as np
import pytest
from scipy.linalg import cholesky

from inversion_methods.manipulation.matrix_identities import (
    woodbury,
    Ainv_B,
    kalman_gain_woodbury,
    kalman_gain_woodbury_from_gram,
    covariance_from_woodbury_factor,
    precision_from_cholesky,
    kalman_gain_information_from_gram,
    covariance_from_information_factor,
    _cholesky_with_repair,
    _sample_gaussian_cholesky,
    _symmetrize_square,
)


def _random_spd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def _naive_kalman_gain(Pf, H, r_diag):
    S = H @ Pf @ H.T + np.diag(r_diag)
    return Pf @ H.T @ np.linalg.inv(S)


def test_woodbury_matches_direct_inverse():
    n, k = 6, 2
    A = _random_spd(n, seed=1)
    U = np.random.default_rng(2).normal(size=(n, k))
    C = _random_spd(k, seed=3)
    V = U.T

    naive = np.linalg.inv(A + U @ C @ V)
    fast = woodbury(np.linalg.inv(A), U, C, V)
    np.testing.assert_allclose(fast, naive, rtol=1e-8, atol=1e-10)


def test_ainv_b_matches_solve():
    A = _random_spd(5, seed=4)
    B = np.random.default_rng(5).normal(size=(5, 3))
    np.testing.assert_allclose(Ainv_B(A, B), np.linalg.solve(A, B), rtol=1e-8, atol=1e-10)


def test_kalman_gain_woodbury_matches_naive_formula():
    nz, ny = 5, 20
    Pf = _random_spd(nz, seed=6)
    H = np.random.default_rng(7).normal(size=(ny, nz))
    r_diag = np.random.default_rng(8).uniform(0.5, 2.0, size=ny)

    K_naive = _naive_kalman_gain(Pf, H, r_diag)
    K_fast = kalman_gain_woodbury(Pf, H, 1.0 / r_diag)
    np.testing.assert_allclose(K_fast, K_naive, rtol=1e-6, atol=1e-8)


def test_kalman_gain_woodbury_from_gram_matches_naive_and_updates_covariance():
    """
    kalman_gain_woodbury_from_gram is a further reduction of
    kalman_gain_woodbury for callers (the MXKF's Gauss-Newton loop) that
    have already reduced the (ny, nz) observation operator down to a fixed,
    reusable (nz, nz) Gram matrix G = H.T @ diag(r_inv) @ H and vector
    h = H.T @ diag(r_inv) @ d, and only need to apply the gain to d for
    varying, per-relinearisation Jacobian weights wb (H_hat = H @ diag(wb)).
    With wb = ones (no reweighting), it should reduce exactly to applying
    the plain Kalman gain from kalman_gain_woodbury to d.
    """
    nz, ny = 4, 15
    Pf = _random_spd(nz, seed=9)
    H = np.random.default_rng(10).normal(size=(ny, nz))
    r_diag = np.random.default_rng(11).uniform(0.5, 2.0, size=ny)
    d = np.random.default_rng(12).normal(size=ny)

    L = cholesky(Pf, lower=True)
    r_inv = 1.0 / r_diag
    G = H.T @ (r_inv[:, None] * H)
    h = H.T @ (r_inv * d)
    wb = np.ones(nz)

    Kd_fast, S_factor = kalman_gain_woodbury_from_gram(L, G, wb, h)

    K_naive = _naive_kalman_gain(Pf, H, r_diag)
    np.testing.assert_allclose(Kd_fast, K_naive @ d, rtol=1e-6, atol=1e-8)

    Pa_fast = covariance_from_woodbury_factor(L, S_factor)
    Pa_naive = (np.eye(nz) - K_naive @ H) @ Pf
    np.testing.assert_allclose(Pa_fast, Pa_naive, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(Pa_fast, Pa_fast.T)
    assert np.all(np.linalg.eigvalsh(Pa_fast) > -1e-8)


def test_kalman_gain_woodbury_from_gram_handles_nontrivial_jacobian_weight():
    """
    With a non-trivial wb, kalman_gain_woodbury_from_gram should match the
    plain Kalman gain applied against the reweighted operator
    H_hat = H @ diag(wb) -- this is the actual usage pattern inside
    iterative_analysis_update, where wb comes from build_Wb's lognormal
    Jacobian and changes at every Gauss-Newton relinearisation without G/h
    (both built from the fixed, un-reweighted H) ever being recomputed.
    """
    nz, ny = 4, 15
    Pf = _random_spd(nz, seed=15)
    H = np.random.default_rng(16).normal(size=(ny, nz))
    r_diag = np.random.default_rng(17).uniform(0.5, 2.0, size=ny)
    wb = np.random.default_rng(18).uniform(0.3, 2.5, size=nz)
    d = np.random.default_rng(19).normal(size=ny)

    L = cholesky(Pf, lower=True)
    r_inv = 1.0 / r_diag
    G = H.T @ (r_inv[:, None] * H)  # not yet weighted by wb
    h = H.T @ (r_inv * d)  # not yet weighted by wb

    Kd_fast, _S_factor = kalman_gain_woodbury_from_gram(L, G, wb, h)

    H_hat = H * wb[None, :]  # H_hat = H @ diag(wb)
    K_naive = _naive_kalman_gain(Pf, H_hat, r_diag)
    np.testing.assert_allclose(Kd_fast, K_naive @ d, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_information_form_matches_woodbury_form(seed):
    """
    kalman_gain_information_from_gram computes the same K @ d and Pa as
    kalman_gain_woodbury_from_gram, via Pa = (Pf^-1 + diag(wb) G diag(wb))^-1
    rather than the Woodbury system S = I + L.T diag(wb) G diag(wb) L.
    """
    nz, ny = 12, 40
    rng = np.random.default_rng(seed)
    Pf = _random_spd(nz, seed=seed)
    H = rng.normal(size=(ny, nz))
    r_inv = 1.0 / rng.uniform(0.5, 2.0, size=ny)
    wb = rng.uniform(0.3, 2.5, size=nz)
    d = rng.normal(size=ny)

    L = cholesky(Pf, lower=True)
    G = H.T @ (r_inv[:, None] * H)
    h = H.T @ (r_inv * d)

    Kd_wood, S_factor = kalman_gain_woodbury_from_gram(L, G, wb, h)
    Pa_wood = covariance_from_woodbury_factor(L, S_factor)

    Kd_info, A_factor = kalman_gain_information_from_gram(precision_from_cholesky(L), G, wb, h)
    Pa_info = covariance_from_information_factor(A_factor)

    np.testing.assert_allclose(Kd_info, Kd_wood, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Pa_info, Pa_wood, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(Pa_info, Pa_info.T)


def test_information_form_matches_naive_kalman_filter():
    nz, ny = 5, 20
    rng = np.random.default_rng(31)
    Pf = _random_spd(nz, seed=31)
    H = rng.normal(size=(ny, nz))
    r_diag = rng.uniform(0.5, 2.0, size=ny)
    wb = rng.uniform(0.3, 2.5, size=nz)
    d = rng.normal(size=ny)

    L = cholesky(Pf, lower=True)
    G = H.T @ ((1.0 / r_diag)[:, None] * H)
    h = H.T @ ((1.0 / r_diag) * d)

    Kd, A_factor = kalman_gain_information_from_gram(precision_from_cholesky(L), G, wb, h)
    Pa = covariance_from_information_factor(A_factor)

    H_hat = H * wb[None, :]
    K_naive = _naive_kalman_gain(Pf, H_hat, r_diag)
    np.testing.assert_allclose(Kd, K_naive @ d, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(Pa, (np.eye(nz) - K_naive @ H_hat) @ Pf, rtol=1e-5, atol=1e-7)


def test_cholesky_with_repair_recovers_valid_factor_for_borderline_matrix():
    n = 6
    spd = _random_spd(n, seed=12)
    eigvals, eigvecs = np.linalg.eigh(spd)
    eigvals[0] = -1e-10  # nudge just barely non-PD
    almost_spd = (eigvecs * eigvals) @ eigvecs.T

    L, repaired, jitter = _cholesky_with_repair(almost_spd)
    np.testing.assert_allclose(L @ L.T, repaired, rtol=1e-6, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(repaired) > -1e-9)


def test_cholesky_with_repair_rejects_non_finite_input():
    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        _cholesky_with_repair(bad)


def test_sample_gaussian_cholesky_matches_target_moments():
    mean = np.array([1.0, -2.0])
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    rng = np.random.default_rng(13)

    samples = np.array([_sample_gaussian_cholesky(mean, covariance, rng) for _ in range(20000)])
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.05)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=0.1)


def test_symmetrize_square_averages_asymmetric_matrix():
    M = np.array([[1.0, 2.0], [0.0, 1.0]])
    np.testing.assert_allclose(_symmetrize_square(M), np.array([[1.0, 1.0], [1.0, 1.0]]))
