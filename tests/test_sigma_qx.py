"""
Unit tests for inversion_methods.bristau.siqma_qx and the country-grouping
helper it depends on in inversion_methods.bristau.data_bristau.

`test_update_sigma2_qx_inner_outer_returns_flat_array` is a regression test
for a real bug: `sample_sigma2_qx` returns a 1-element array rather than a
scalar, so building `np.array([out, in])` for the "inner outer" scheme
produces a (2, 1) array instead of the (2,) shape the Gibbs sampler's trace
array expects, crashing `sigma2_qx_trace[i] = sample` with a broadcast
error. Both example configs in codey_things/ini_files/bristau/ set
`sigma_qx = 'inner outer'`, so this is not a hypothetical configuration.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from inversion_methods.bristau.siqma_qx import (
    sigma_qx_scheme_select,
    sigma_qx_trace_params,
    initialise_sigma2_qx_vector,
    update_sigma2_qx,
    sample_sigma2_qx,
    sample_sigma2_qx_grouped,
)
from inversion_methods.bristau.data_bristau import inner_basis_country_groups


def _zmusample_with_reference(n_steps, n_series, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.1, size=(n_steps, n_series))
    r = np.zeros((n_steps, 1))
    return np.hstack([x, r])


# ---------------------------------------------------------------------------
# sigma_qx_scheme_select
# ---------------------------------------------------------------------------


def test_scheme_select_fixed():
    config = SimpleNamespace(sigma_qx=0.2)
    scheme, fixed, ningroup, inner_group_id = sigma_qx_scheme_select(config, nxin=5)
    assert scheme == "fixed"
    assert fixed == pytest.approx(0.04)
    assert ningroup is None
    assert inner_group_id is None


def test_scheme_select_squares_negative_input_rather_than_rejecting():
    """
    sigma_qx is a standard deviation, so it's squared before the
    negativity check runs -- meaning that check can never trigger, and a
    negative input is silently treated the same as its magnitude.
    """
    config = SimpleNamespace(sigma_qx=-2.0)
    scheme, fixed, _, _ = sigma_qx_scheme_select(config, nxin=5)
    assert scheme == "fixed"
    assert fixed == pytest.approx(4.0)


def test_scheme_select_rejects_non_finite():
    config = SimpleNamespace(sigma_qx=float("inf"))
    with pytest.raises(ValueError):
        sigma_qx_scheme_select(config, nxin=5)


def test_scheme_select_rejects_unknown_string():
    config = SimpleNamespace(sigma_qx="banana")
    with pytest.raises(ValueError):
        sigma_qx_scheme_select(config, nxin=5)


def test_scheme_select_country_requires_group_info():
    config = SimpleNamespace(sigma_qx="inner outer country")
    with pytest.raises(ValueError):
        sigma_qx_scheme_select(config, nxin=5)


def test_scheme_select_country_validates_group_ids():
    config = SimpleNamespace(
        sigma_qx="inner outer country",
        inner_group_id=np.array([0, 0, 1, 1, 5]),  # not contiguous 0..ningroup-1
        ningroup=2,
    )
    with pytest.raises(ValueError):
        sigma_qx_scheme_select(config, nxin=5)


def test_scheme_select_country_accepts_valid_groups():
    config = SimpleNamespace(
        sigma_qx="Inner Outer Country",
        inner_group_id=np.array([0, 0, 1, 1, 2]),
        ningroup=3,
    )
    scheme, fixed, ningroup, inner_group_id = sigma_qx_scheme_select(config, nxin=5)
    assert scheme == "inner outer country"
    assert ningroup == 3
    np.testing.assert_array_equal(inner_group_id, [0, 0, 1, 1, 2])


# ---------------------------------------------------------------------------
# sigma_qx_trace_params / initialise_sigma2_qx_vector
# ---------------------------------------------------------------------------


def test_trace_params_fixed_and_global():
    assert sigma_qx_trace_params("fixed", ningroup=None) == (["global"], 1)
    assert sigma_qx_trace_params("global", ningroup=None) == (["global"], 1)


def test_trace_params_inner_outer():
    assert sigma_qx_trace_params("inner outer", ningroup=None) == (["outer", "inner"], 2)


def test_trace_params_inner_outer_country():
    labels, n = sigma_qx_trace_params("inner outer country", ningroup=2)
    assert labels == ["outer", "inner_group_0", "inner_group_1"]
    assert n == 3


def test_initialise_fixed_and_global_fill_all_basis_functions():
    vec = initialise_sigma2_qx_vector("fixed", 0.02, 0.05, nbasis=6, nxout=2, nxin=4, ningroup=None, inner_group_id=None)
    np.testing.assert_allclose(vec, np.full(6, 0.02))


def test_initialise_inner_outer_country_broadcasts_group_values():
    inner_group_id = np.array([0, 0, 1, 1])
    vec = initialise_sigma2_qx_vector(
        "inner outer country", None, 0.03, nbasis=6, nxout=2, nxin=4, ningroup=2, inner_group_id=inner_group_id,
    )
    assert vec.shape == (6,)
    np.testing.assert_allclose(vec, np.full(6, 0.03))


# ---------------------------------------------------------------------------
# update_sigma2_qx
# ---------------------------------------------------------------------------


def test_update_sigma2_qx_fixed_fills_and_returns_fixed_value():
    current = np.full(5, 0.5)
    sample, updated = update_sigma2_qx(
        zmusample=None, zmusample_out=None, zmusample_in=None,
        sigma2_qx_aprior=None, sigma2_qx_bprior=None, nbasis=5, nxout=0, nxin=5,
        sigma2_qx_max=None, sigma_qx_scheme="fixed", sigma2_qx_bf_current=current,
        fixed_sigma2_qx=0.09, inner_group_id=None, ningroup=None,
        kappa_x_vector_current=np.array([0.5]), n_kappa_x_parameters=1,
    )
    assert sample == pytest.approx(0.09)
    np.testing.assert_allclose(updated, np.full(5, 0.09))


def test_update_sigma2_qx_global_requires_matching_inner_outer_kappa():
    zmusample = _zmusample_with_reference(30, 4, seed=1)
    with pytest.raises(ValueError):
        update_sigma2_qx(
            zmusample=zmusample, zmusample_out=None, zmusample_in=None,
            sigma2_qx_aprior=2.0, sigma2_qx_bprior=2.0, nbasis=4, nxout=0, nxin=4,
            sigma2_qx_max=0.5, sigma_qx_scheme="global", sigma2_qx_bf_current=np.full(4, 0.02),
            fixed_sigma2_qx=None, inner_group_id=None, ningroup=None,
            kappa_x_vector_current=np.array([0.2, 0.8]),  # deliberately mismatched
            n_kappa_x_parameters=2,
        )


def test_update_sigma2_qx_inner_outer_works_with_single_fixed_kappa():
    """
    Regression test: kappa_xout_current was only ever assigned inside the
    n_kappa_x_parameters == 2 branch. With kappa_x fixed/global
    (n_kappa_x_parameters == 1) and sigma_qx="inner outer" (or "inner outer
    country"), the "inner outer" branch below referenced kappa_xout_current
    unconditionally, raising UnboundLocalError. A single fixed/global kappa
    should transparently serve as both the outer and inner value.
    """
    nxout, nxin = 2, 3
    zmusample_out = _zmusample_with_reference(50, nxout, seed=2)
    zmusample_in = _zmusample_with_reference(50, nxin, seed=3)
    current = np.full(nxout + nxin, 0.02)

    sample, updated = update_sigma2_qx(
        zmusample=None, zmusample_out=zmusample_out, zmusample_in=zmusample_in,
        sigma2_qx_aprior=2.0, sigma2_qx_bprior=2.0, nbasis=nxout + nxin, nxout=nxout, nxin=nxin,
        sigma2_qx_max=0.5, sigma_qx_scheme="inner outer", sigma2_qx_bf_current=current,
        fixed_sigma2_qx=None, inner_group_id=None, ningroup=None,
        kappa_x_vector_current=np.array([0.3]), n_kappa_x_parameters=1,
    )

    assert sample.shape == (2,)
    assert np.all(sample > 0) and np.all(sample <= 0.5)


def test_update_sigma2_qx_inner_outer_returns_flat_array_matching_trace_shape():
    """
    Regression test: update_sigma2_qx's "inner outer" branch must return a
    (2,) array -- that's what `sigma2_qx_trace[i] = sample` expects. If
    `sample_sigma2_qx` still returns a 1-element array instead of a scalar,
    this comes back as (2, 1) and the assignment in
    augmented_ffbs_mxkf_gibbs_double_slice raises a broadcast ValueError.
    """
    nxout, nxin = 2, 3
    zmusample_out = _zmusample_with_reference(50, nxout, seed=2)
    zmusample_in = _zmusample_with_reference(50, nxin, seed=3)
    current = np.full(nxout + nxin, 0.02)

    sample, updated = update_sigma2_qx(
        zmusample=None, zmusample_out=zmusample_out, zmusample_in=zmusample_in,
        sigma2_qx_aprior=2.0, sigma2_qx_bprior=2.0, nbasis=nxout + nxin, nxout=nxout, nxin=nxin,
        sigma2_qx_max=0.5, sigma_qx_scheme="inner outer", sigma2_qx_bf_current=current,
        fixed_sigma2_qx=None, inner_group_id=None, ningroup=None,
        kappa_x_vector_current=np.array([0.5, 0.5]), n_kappa_x_parameters=2,
    )

    assert sample.shape == (2,)
    assert np.all(sample > 0) and np.all(sample <= 0.5)
    np.testing.assert_allclose(updated[:nxout], sample[0])
    np.testing.assert_allclose(updated[nxout:], sample[1])


def test_update_sigma2_qx_inner_outer_country_returns_flat_array_matching_trace_shape():
    """
    Regression test: update_sigma2_qx's "inner outer country" branch builds
    its outer contribution from `sample_sigma2_qx`, which returns a plain
    scalar. `np.concatenate((scalar, group_array))` raises "zero-dimensional
    arrays cannot be concatenated" -- this crashed a real production run
    after `sample_sigma2_qx` was changed to return a scalar (fixing a
    different shape bug in the "inner outer" branch) without updating this
    branch to match.
    """
    nxout, nxin = 2, 4
    ningroup = 2
    inner_group_id = np.array([0, 0, 1, 1])
    zmusample_out = _zmusample_with_reference(50, nxout, seed=6)
    zmusample_in = _zmusample_with_reference(50, nxin, seed=7)
    current = np.full(nxout + nxin, 0.02)

    sample, updated = update_sigma2_qx(
        zmusample=None, zmusample_out=zmusample_out, zmusample_in=zmusample_in,
        sigma2_qx_aprior=2.0, sigma2_qx_bprior=2.0, nbasis=nxout + nxin, nxout=nxout, nxin=nxin,
        sigma2_qx_max=0.5, sigma_qx_scheme="inner outer country", sigma2_qx_bf_current=current,
        fixed_sigma2_qx=None, inner_group_id=inner_group_id, ningroup=ningroup,
        kappa_x_vector_current=np.array([0.5, 0.5]), n_kappa_x_parameters=2,
    )

    assert sample.shape == (1 + ningroup,)  # outer + one value per inner group
    assert np.all(sample > 0) and np.all(sample <= 0.5)
    np.testing.assert_allclose(updated[:nxout], sample[0])


def test_sample_sigma2_qx_returns_a_bounded_scalar():
    zmusample = _zmusample_with_reference(100, 5, seed=4)
    draws = [
        sample_sigma2_qx(zmusample, kappa_x=0.5, alpha_prior=2.0, beta_prior=0.01, nbasis=5, sigma2_qx_max=0.3)
        for _ in range(50)
    ]
    assert all(isinstance(d, float) for d in draws)
    assert all(0 < d <= 0.3 for d in draws)


def test_sample_sigma2_qx_grouped_broadcasts_and_matches_group_values():
    group_id = np.array([0, 0, 1, 1, 1])
    zmusample_in = _zmusample_with_reference(60, len(group_id), seed=5)

    per_bf, per_group = sample_sigma2_qx_grouped(
        zmusample_in, kappa_xin=0.5, alpha_prior=2.0, beta_prior=0.01,
        group_id=group_id, ngroup=2, sigma2_qx_max=0.5,
    )

    assert per_group.shape == (2,)
    np.testing.assert_allclose(per_bf, per_group[group_id])


# ---------------------------------------------------------------------------
# inner_basis_country_groups
# ---------------------------------------------------------------------------


def test_inner_basis_country_groups_merges_small_countries():
    # 4 basis functions (1-indexed values 1..4); basis fn 4 is the only one
    # assigned to country 1, so with min_group_size=2 it should be merged
    # into an "other" group rather than kept as its own group.
    bfds = xr.DataArray(np.array([[1, 1, 2], [2, 3, 3], [4, 4, 4]]), dims=["lat", "lon"])
    cntryds = xr.Dataset({"country": (["lat", "lon"], np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]))})

    group_id, ngroup = inner_basis_country_groups(bfds, cntryds, nxout=0, nbasis=4, min_group_size=2)

    assert ngroup == 2  # country 0 (3 basis fns) + merged "other" (basis fn 4)
    assert group_id.shape == (4,)
    assert group_id[0] == group_id[1] == group_id[2]
    assert group_id[3] != group_id[0]


def test_inner_basis_country_groups_respects_nxout_offset():
    # 5 basis functions total; the first 2 are "outer" and excluded from grouping.
    bfds = xr.DataArray(np.array([[1, 2, 3], [3, 4, 5]]), dims=["lat", "lon"])
    cntryds = xr.Dataset({"country": (["lat", "lon"], np.array([[0, 0, 1], [1, 2, 2]]))})

    group_id, ngroup = inner_basis_country_groups(bfds, cntryds, nxout=2, nbasis=5, min_group_size=1)

    # Only inner basis functions 3, 4, 5 (nxin=3) should be grouped.
    assert group_id.shape == (3,)
