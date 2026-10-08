"""Unit tests for the observation/sensitivity filtering in inversion_methods.bristau.data_bristau."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import inversion_methods.bristau.data_bristau as data_bristau
from inversion_methods.bristau.data_bristau import (
    finite_time_mask,
    build_obs_vectors,
    build_boundary_conditions,
    check_finite_inputs,
)

NTIME = 6
NREGION = 3
NBC = 4


def make_site(H_bad=None, Hbc_bad=None, mf_nan=None, err_value=None, rep_nan=False, var_nan=False, offset=0.0):
    """Synthetic per-site dataset shaped like fp_data[site] (H: region x time, H_bc: bc_region x time)."""
    time = pd.date_range("2020-01-01", periods=NTIME, freq="4h")
    H = np.ones((NREGION, NTIME)) + offset
    Hbc = np.ones((NBC, NTIME)) + offset
    mf = 1900.0 + np.arange(NTIME) + offset
    err = np.ones(NTIME)

    if H_bad is not None:
        H[1, H_bad] = np.inf
    if Hbc_bad is not None:
        Hbc[0, Hbc_bad] = -np.inf
    if mf_nan is not None:
        mf[mf_nan] = np.nan
    if err_value is not None:
        idx, value = err_value
        err[idx] = value

    return xr.Dataset(
        {
            "H": (("region", "time"), H),
            "H_bc": (("bc_region", "time"), Hbc),
            "mf": ("time", mf),
            "mf_error": ("time", err),
            "mf_repeatability": ("time", np.full(NTIME, np.nan) if rep_nan else np.ones(NTIME)),
            "mf_variability": ("time", np.full(NTIME, np.nan) if var_nan else np.ones(NTIME)),
        },
        coords={"time": time},
    )


# ---------------------------------------------------------------------------
# finite_time_mask
# ---------------------------------------------------------------------------

def test_mask_all_valid():
    keep, n_bad = finite_time_mask(make_site(), ["H", "H_bc", "mf", "mf_error"])
    assert keep.all()
    assert all(n == 0 for n in n_bad.values())


@pytest.mark.parametrize(
    "kwargs, var",
    [
        ({"H_bad": 2}, "H"),
        ({"Hbc_bad": 2}, "H_bc"),
        ({"mf_nan": 2}, "mf"),
        ({"err_value": (2, np.inf)}, "mf_error"),
        ({"err_value": (2, np.nan)}, "mf_error"),
    ],
)
def test_mask_flags_nan_and_inf(kwargs, var):
    keep, n_bad = finite_time_mask(make_site(**kwargs), ["H", "H_bc", "mf", "mf_error"])
    assert keep.tolist() == [True, True, False, True, True, True]
    assert n_bad[var] == 1


def test_mask_reduces_over_non_time_dims():
    # a single inf in one region of H invalidates the whole time step
    ds = make_site()
    ds["H"][0, 3] = np.inf
    keep, n_bad = finite_time_mask(ds, ["H"])
    assert keep.tolist() == [True, True, True, False, True, True]
    assert n_bad["H"] == 1


def test_mask_handles_time_first_dimension_order():
    ds = make_site(H_bad=4)
    ds["H"] = ds["H"].transpose("time", "region")
    keep, _ = finite_time_mask(ds, ["H"])
    assert keep.tolist() == [True, True, True, True, False, True]


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_mask_flags_non_positive_error(value):
    keep, n_bad = finite_time_mask(make_site(err_value=(1, value)), ["mf_error"])
    assert keep.tolist() == [True, False, True, True, True, True]
    assert n_bad["mf_error<=0"] == 1
    assert n_bad["mf_error"] == 0


def test_mask_nan_error_not_double_counted_as_non_positive():
    _, n_bad = finite_time_mask(make_site(err_value=(1, np.nan)), ["mf_error"])
    assert n_bad["mf_error"] == 1
    assert n_bad["mf_error<=0"] == 0


def test_mask_ignores_absent_variables():
    ds = make_site(H_bad=0).drop_vars("H_bc")
    keep, n_bad = finite_time_mask(ds, ["H", "H_bc", "mf", "mf_error"])
    assert "H_bc" not in n_bad
    assert keep.sum() == NTIME - 1


def test_mask_ignores_repeatability_and_variability():
    keep, _ = finite_time_mask(make_site(rep_nan=True, var_nan=True), ["H", "H_bc", "mf", "mf_error"])
    assert keep.all()


# ---------------------------------------------------------------------------
# build_obs_vectors / build_boundary_conditions
# ---------------------------------------------------------------------------

def test_build_obs_vectors_drops_invalid_rows_and_keeps_alignment(capsys):
    fp_data = {
        "AAA": make_site(H_bad=0, Hbc_bad=2),
        "BBB": make_site(mf_nan=1, err_value=(4, 0.0), offset=10.0),
        "CCC": make_site(),
    }
    Hx, Y, Ytime, error, siteindicator, sites = build_obs_vectors(fp_data, ["AAA", "BBB", "CCC"])

    assert sites == ["AAA", "BBB", "CCC"]
    assert Hx.shape == (NREGION, 4 + 4 + 6)
    assert Y.shape == Ytime.shape == error.shape == siteindicator.shape == (14,)
    assert np.isfinite(Hx).all() and np.isfinite(Y).all() and np.isfinite(error).all()
    assert (error > 0).all()
    np.testing.assert_array_equal(siteindicator, [0] * 4 + [1] * 4 + [2] * 6)

    # filtered datasets are written back so H_bc extracted later matches Hx
    assert fp_data["AAA"].sizes["time"] == 4
    Hbc = build_boundary_conditions(fp_data, sites, SimpleNamespace(use_bc=True))
    assert Hbc.shape == (NBC, Hx.shape[1])
    assert np.isfinite(Hbc).all()

    out = capsys.readouterr().out
    assert "AAA: dropping 2 of 6" in out
    assert "BBB: dropping 2 of 6" in out
    assert "CCC" not in out


def test_build_obs_vectors_keeps_rows_with_nan_repeatability():
    fp_data = {"AAA": make_site(rep_nan=True, var_nan=True)}
    _, Y, _, _, _, sites = build_obs_vectors(fp_data, ["AAA"])
    assert sites == ["AAA"]
    assert Y.size == NTIME


def test_build_obs_vectors_removes_empty_site_and_renumbers(capsys):
    fp_data = {
        "AAA": make_site(),
        "BBB": make_site(H_bad=slice(None)),
        "CCC": make_site(offset=5.0),
    }
    Hx, Y, _, _, siteindicator, sites = build_obs_vectors(fp_data, ["AAA", "BBB", "CCC"])

    assert sites == ["AAA", "CCC"]
    assert "BBB" not in fp_data
    np.testing.assert_array_equal(siteindicator, [0] * NTIME + [1] * NTIME)
    np.testing.assert_allclose(Y[NTIME:], make_site(offset=5.0).mf.values)
    assert "WARNING: BBB has no valid time steps remaining" in capsys.readouterr().out


def test_build_obs_vectors_first_site_empty():
    fp_data = {"AAA": make_site(mf_nan=slice(None)), "BBB": make_site()}
    Hx, Y, Ytime, _, siteindicator, sites = build_obs_vectors(fp_data, ["AAA", "BBB"])
    assert sites == ["BBB"]
    assert Hx.shape == (NREGION, NTIME)
    assert Ytime.shape == (NTIME,)
    assert (siteindicator == 0).all()


def test_build_obs_vectors_raises_when_no_sites_remain():
    fp_data = {"AAA": make_site(H_bad=slice(None)), "BBB": make_site(err_value=(slice(None), 0.0))}
    with pytest.raises(ValueError, match="No sites"):
        build_obs_vectors(fp_data, ["AAA", "BBB"])


def test_build_boundary_conditions_disabled():
    assert build_boundary_conditions({"AAA": make_site()}, ["AAA"], SimpleNamespace(use_bc=False)) is None


# ---------------------------------------------------------------------------
# check_finite_inputs
# ---------------------------------------------------------------------------

def test_check_finite_inputs_passes_and_skips_none():
    check_finite_inputs({"Hx": np.ones((2, 3)), "Y": np.ones(3), "Hbc": None})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_finite_inputs_raises_with_name_and_count(bad):
    arr = np.ones((2, 3))
    arr[0, 1] = bad
    arr[1, 2] = bad
    with pytest.raises(ValueError, match=r"Hbc contains 2 non-finite"):
        check_finite_inputs({"Hx": np.ones(3), "Hbc": arr})


# ---------------------------------------------------------------------------
# extract_data (openghg loading and basis functions stubbed out)
# ---------------------------------------------------------------------------

def _stub_extract(monkeypatch, fp_all, returned_sites):
    monkeypatch.setattr(
        data_bristau, "extract_observation_data",
        lambda config: (fp_all, returned_sites, None, None, None, None),
    )
    monkeypatch.setattr(data_bristau, "build_basis_functions", lambda fp, config: fp)
    monkeypatch.setattr(data_bristau, "update_log_normal_prior", lambda prior: None)


def _config(sites, use_bc=True):
    return SimpleNamespace(sites=sites, domain="EUROPE", use_bc=use_bc, xprior={}, rprior={}, bcprior={}, filters=None)


def test_extract_data_uses_sites_returned_by_data_processing(monkeypatch, capsys):
    # BBB was dropped upstream (no obs/footprints) so it is absent from fp_all
    fp_all = {"AAA": make_site(), "CCC": make_site(H_bad=3)}
    _stub_extract(monkeypatch, fp_all, ["AAA", "CCC"])

    out = data_bristau.extract_data(_config(["AAA", "BBB", "CCC"]))
    Hx, Y, Ytime, error, siteindicator, nbasis, _, _, Hbc, fp_data, sites = out

    assert sites == ["AAA", "CCC"]
    assert nbasis == NREGION
    assert Hx.shape[1] == Hbc.shape[1] == Y.size == 2 * NTIME - 1
    assert fp_data["AAA"].attrs["Domain"] == "EUROPE"
    assert "sites dropped from the inversion: ['BBB']" in capsys.readouterr().out


def test_extract_data_without_bc(monkeypatch):
    _stub_extract(monkeypatch, {"AAA": make_site(Hbc_bad=0)}, ["AAA"])
    out = data_bristau.extract_data(_config(["AAA"], use_bc=False))
    Hx, Hbc, sites = out[0], out[8], out[10]
    assert Hbc is None
    assert sites == ["AAA"]
    # with use_bc False an inf only in H_bc still drops the row, as H_bc is present in the dataset
    assert Hx.shape[1] == NTIME - 1
