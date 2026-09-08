"""Unit tests for post-processing helpers in inversion_methods.bristau.inversion_bristau."""

import numpy as np
import pytest

from inversion_methods.bristau.inversion_bristau import build_group_id_coordinate


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
