"""Unit tests for post-processing helpers in inversion_methods.bristau.inversion_bristau."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import arviz as az

from inversion_methods.bristau.inversion_bristau import build_group_id_coordinate, select_period_flux, country_totals


def test_build_group_id_coordinate_all_zeros_when_no_country_grouping():
    """
    Regression test: inner_group_id is only populated by sigma_qx_groups()
    when sigma_qx == "inner outer country" -- for every other scheme
    (including plain "inner outer" with nxout > 0) it's None regardless of
    nxout, and the old inline code assumed nxout > 0 implied inner_group_id
    was set, crashing on `None + 1`.
    """
    result = build_group_id_coordinate(nxout=2, nbasis=6, inner_group_id=None)
    np.testing.assert_array_equal(result, np.zeros(6))


def test_build_group_id_coordinate_all_zeros_when_no_country_grouping_and_no_outer():
    result = build_group_id_coordinate(nxout=0, nbasis=4, inner_group_id=None)
    np.testing.assert_array_equal(result, np.zeros(4))


def test_build_group_id_coordinate_offsets_by_one_with_outer_basis_functions():
    inner_group_id = np.array([0, 0, 1, 1])
    result = build_group_id_coordinate(nxout=2, nbasis=6, inner_group_id=inner_group_id)
    np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2])


def test_build_group_id_coordinate_passes_through_when_no_outer_basis_functions():
    inner_group_id = np.array([0, 0, 1, 1])
    result = build_group_id_coordinate(nxout=0, nbasis=4, inner_group_id=inner_group_id)
    np.testing.assert_array_equal(result, inner_group_id)


def _flux(times):
    """Flux on a 2x3 grid where every cell holds the time step's index."""
    values = np.broadcast_to(np.arange(len(times))[:, None, None], (len(times), 2, 3)).astype(float)
    return xr.DataArray(values, dims=("time", "lat", "lon"),
                        coords={"time": pd.to_datetime(times), "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]})


def test_select_period_flux_matches_multiyear_monthly_flux_by_date():
    """
    Regression test: a 10 year monthly flux used to be indexed by calendar month,
    so every year reused the first year's 12 time steps.
    """
    times = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    periods = pd.date_range("2016-01-01", "2026-01-01", freq="MS")[:-1]
    result = select_period_flux(_flux(times), periods)
    assert result.shape == (2, 3, 120)
    np.testing.assert_array_equal(result[0, 0, :], np.arange(120))


def test_select_period_flux_forward_fills_annual_flux():
    times = pd.to_datetime(["2016-01-01", "2017-01-01"])
    periods = pd.date_range("2016-01-01", "2018-01-01", freq="MS")[:-1]
    result = select_period_flux(_flux(times), periods)
    np.testing.assert_array_equal(result[0, 0, :], np.repeat([0, 1], 12))


def test_select_period_flux_offset_run_start():
    times = pd.date_range("2016-01-01", "2018-12-01", freq="MS")
    periods = pd.date_range("2017-03-01", "2017-06-01", freq="MS")[:-1]
    result = select_period_flux(_flux(times), periods)
    np.testing.assert_array_equal(result[0, 0, :], [14, 15, 16])


def test_select_period_flux_single_time_step_used_for_every_period():
    periods = pd.date_range("2020-01-01", "2021-01-01", freq="MS")[:-1]
    result = select_period_flux(_flux(["2018-01-01"]), periods)
    np.testing.assert_array_equal(result[0, 0, :], np.zeros(12))


def test_select_period_flux_climatology_when_flux_does_not_cover_start():
    times = pd.date_range("2018-01-01", "2018-12-01", freq="MS")
    periods = pd.date_range("2016-11-01", "2017-03-01", freq="MS")[:-1]
    result = select_period_flux(_flux(times), periods)
    np.testing.assert_array_equal(result[0, 0, :], [10, 11, 0, 1])


def test_select_period_flux_raises_when_flux_does_not_cover_start():
    times = pd.date_range("2018-01-01", "2019-12-01", freq="MS")
    periods = pd.date_range("2016-01-01", "2016-03-01", freq="MS")[:-1]
    with pytest.raises(ValueError):
        select_period_flux(_flux(times), periods)


def test_country_totals_matches_per_cell_loop():
    """The vectorised country totals must match the original per (country, basis) loop."""
    rng = np.random.default_rng(0)
    nlat, nlon, nperiod, nbasis, ncountry, steps = 6, 7, 3, 5, 4, 200
    xtrace = rng.lognormal(size=(steps, nperiod, nbasis))
    apriori_flux = rng.random((nlat, nlon, nperiod))
    bfarray = rng.integers(0, nbasis, (nlat, nlon))
    cntrygrid = rng.integers(-1, ncountry, (nlat, nlon))
    area = rng.random((nlat, nlon)) * 1e8
    molarmass, unit_factor = 16.04, 1e12

    result = country_totals(xtrace, apriori_flux, bfarray, cntrygrid, ncountry, area, molarmass, unit_factor)

    expected_median = np.zeros((ncountry, nperiod))
    expected_68 = np.zeros((ncountry, 2, nperiod))
    expected_95 = np.zeros((ncountry, 2, nperiod))
    expected_prior = np.zeros((ncountry, nperiod))
    for period in range(nperiod):
        for ci in range(ncountry):
            trace = np.zeros(steps)
            prior = 0
            for bf in range(nbasis):
                both = (cntrygrid == ci) & (bfarray == bf)
                w = np.sum(area[both] * apriori_flux[:, :, period][both] * 3600 * 24 * 365 * molarmass) / unit_factor
                trace += w * xtrace[:, period, bf]
                prior += w
            expected_median[ci, period] = np.median(trace)
            expected_68[ci, :, period] = az.hdi(trace, 0.68)
            expected_95[ci, :, period] = az.hdi(trace, 0.95)
            expected_prior[ci, period] = prior

    for got, want in zip(result, (expected_median, expected_68, expected_95, expected_prior)):
        np.testing.assert_allclose(got, want, rtol=1e-10)
