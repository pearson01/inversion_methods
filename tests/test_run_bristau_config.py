"""ini parsing of kappa_x_minfold: numbers, 'annual' and None."""

import re
from pathlib import Path

import pytest

from inversion_methods.bristau.run_bristau import ini_extract_param, convert_to_type
from inversion_methods.bristau.kappa_x import kappa_x_upper_bound

TEMPLATE = Path(__file__).resolve().parents[1] / "inversion_methods" / "templates" / "bristau_example.ini"


def _parse_with_minfold(tmp_path, ini_value):
    text = TEMPLATE.read_text()
    text, n = re.subn(r"(?m)^kappa_x_minfold\s*=.*$", f"kappa_x_minfold = {ini_value}", text)
    assert n == 1
    path = tmp_path / "test.ini"
    path.write_text(text)
    return convert_to_type(ini_extract_param(str(path), print_param=False))


@pytest.mark.parametrize("ini_value, expected", [
    ("10", 10),
    ("1", 1),
    ("'annual'", "annual"),
    ('"annual"', "annual"),
    ("annual", "annual"),
    ("None", None),
])
def test_kappa_x_minfold_ini_values(tmp_path, ini_value, expected):
    params = _parse_with_minfold(tmp_path, ini_value)
    assert params.kappa_x_minfold == expected
    assert type(params.kappa_x_minfold) is type(expected)


@pytest.mark.parametrize("ini_value, nperiod, bound", [
    ("10", 120, 1 - 10 / 120),
    ("'annual'", 12, 11 / 12),
    ("'annual'", 120, 11 / 12),
    ("None", 12, 1.0),
])
def test_kappa_x_minfold_ini_to_bound(tmp_path, ini_value, nperiod, bound):
    params = _parse_with_minfold(tmp_path, ini_value)
    assert kappa_x_upper_bound(params.kappa_x_minfold, nperiod) == pytest.approx(bound)
