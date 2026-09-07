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
    kalman_gain_woodbury_from_cholesky,
    covariance_from_woodbury_factor,
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


def test_kalman_gain_woodbury_from_cholesky_matches_naive_and_updates_covariance():
    nz, ny = 4, 15
    Pf = _random_spd(nz, seed=9)
    H = np.random.default_rng(10).normal(size=(ny, nz))
    r_diag = np.random.default_rng(11).uniform(0.5, 2.0, size=ny)

    L = cholesky(Pf, lower=True)
    K_fast, S_factor = kalman_gain_woodbury_from_cholesky(L, H, 1.0 / r_diag)
    K_naive = _naive_kalman_gain(Pf, H, r_diag)
    np.testing.assert_allclose(K_fast, K_naive, rtol=1e-6, atol=1e-8)

    Pa_fast = covariance_from_woodbury_factor(L, S_factor)
    Pa_naive = (np.eye(nz) - K_naive @ H) @ Pf
    np.testing.assert_allclose(Pa_fast, Pa_naive, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(Pa_fast, Pa_fast.T)
    assert np.all(np.linalg.eigvalsh(Pa_fast) > -1e-8)


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
