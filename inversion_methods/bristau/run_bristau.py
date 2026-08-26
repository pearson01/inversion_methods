import json
import argparse
import types


from typing import Union, get_args, get_origin, get_type_hints
from pathlib import Path
from dataclasses import fields, MISSING

import openghg_inversions.hbmcmc.hbmcmc_output as output
from openghg_inversions.config import config
from inversion_methods.bristau.bristau import InversionParameters, bristau_function



def ini_extract_param(
    config_file: str, print_param: bool | None = True, **command_line
):

    expected_param = ["species",
                      "sites",
                      "averaging_period",
                      "domain",
                      "start_date",
                      "end_date",
                      "outputpath",
                      "outputname",
                      "obs_data_level",
                      "bc_store",
                      "obs_store",
                      "footprint_store",
                      "emissions_store",
                      "nbasis",
                      "country_file",
                      "xprior",
                      "bcprior",
                      "rprior",
                      "sigma2_rep_prior",
                      "sigma2_qx_prior",
                      "kappa_x_prior",
                      "sigma_qbc",
                      "sigma_qr",
                      "iterations"
                      ]
    
    
    # If an expected parameter has been passed from the command line,
    # this does not need to be within the config file
    for key, value in command_line.items():
        if key in expected_param and value is not None:
            expected_param.remove(key)

    param = config.extract_params(
        config_file, expected_param=expected_param
    )

    # Command line values added to param (or superceed inputs from the config
    # file)
    for key, value in command_line.items():
        if value is not None:
            param[key] = value

    # If configuration file does not include values for the
    # required parameters - produce an error
    for ep in expected_param:
        if ep not in param or not param[ep]:
            raise ValueError(f"Required parameter '{ep}' has not been defined")

    if print_param:
        print("\nInput parameters: ")
        for key, value in param.items():
            print(f"{key} = {value}")

    return param


def convert_bool(value) -> bool:
    """Convert common Boolean representations safely."""
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalised = value.strip().lower()

        if normalised in {"true", "1", "yes", "y", "on"}:
            return True

        if normalised in {"false", "0", "no", "n", "off"}:
            return False

    raise ValueError(
        f"Cannot convert {value!r} to bool. "
        "Use true/false, yes/no, on/off, or 1/0."
    )


def convert_to_type(param_dict: dict) -> InversionParameters:
    """
    Convert scalar configuration values to the types expected by
    InversionParameters.

    Missing fields are omitted so that dataclass defaults are used.
    Values already converted by extract_params, such as lists and
    dictionaries, are preserved.

    Special behaviour for nxout:
        - Missing nxout uses the dataclass default.
        - nxout=None or nxout=null is converted to 0.
        - Numeric nxout values are converted to int.
    """
    type_hints = get_type_hints(InversionParameters)
    kwargs = {}
    missing_required = []

    for field in fields(InversionParameters):

        # Do not pass missing values to the constructor.
        # This allows dataclass defaults such as nxout=6 to be used.
        if field.name not in param_dict:
            has_default = (
                field.default is not MISSING
                or field.default_factory is not MISSING
            )

            if not has_default:
                missing_required.append(field.name)

            continue

        raw_value = param_dict[field.name]
        expected_type = type_hints[field.name]
        union_types = get_args(expected_type)
        origin = get_origin(expected_type)

        allows_none = (
            origin in (Union, types.UnionType)
            and type(None) in union_types
        )

        # Handle an actual Python None value.
        if raw_value is None:
            if field.name == "nxout":
                value = 0
            elif allows_none or expected_type is type(None):
                value = None
            else:
                raise ValueError(
                    f"Parameter {field.name!r} does not allow None."
                )

        # Convert explicit INI None/null strings.
        elif (
            isinstance(raw_value, str)
            and raw_value.strip().lower() in {"none", "null"}
        ):
            if field.name == "nxout":
                value = 0
            elif allows_none or expected_type is type(None):
                value = None
            else:
                raise ValueError(
                    f"Parameter {field.name!r} does not allow None."
                )

        # nxout must be converted to an integer.
        elif field.name == "nxout":
            value = int(raw_value)

        # Non-union scalar types.
        elif expected_type is int:
            value = int(raw_value)

        elif expected_type is float:
            value = float(raw_value)

        elif expected_type is bool:
            value = convert_bool(raw_value)

        # Optional integer, for example int | None.
        elif int in union_types and str not in union_types:
            value = int(raw_value)

        # Optional float, for example float | None.
        elif float in union_types and str not in union_types:
            value = float(raw_value)

        # Mixed str | float unions.
        elif str in union_types and float in union_types:
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
            else:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    value = raw_value

        # Lists, dictionaries, strings and other values are assumed to
        # have already been handled correctly by extract_params().
        else:
            value = raw_value

        kwargs[field.name] = value

    if missing_required:
        names = ", ".join(missing_required)
        raise ValueError(
            f"Missing required parameter(s): {names}"
        )

    return InversionParameters(**kwargs)


if __name__ == "__main__":

    mxkf_inv_path = Path(__file__).parents[2]

    parser = argparse.ArgumentParser(description="Running HAFFBS script")
    parser.add_argument("start", help="Start date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument("end", help="End date sting of the format YYYY-MM-DD", nargs="?")
    parser.add_argument(
        "-c", "--config", help="Name (including path) of configuration file"
    )
    parser.add_argument(
    "--kwargs",
    type=json.loads,
    help='Pass keyword arguments to mcmc function. Format: \'{"key1": "val1", "key2": "val2"}\'.',
    )
    parser.add_argument(
        "--output-path",
        help="Path to write ini file and results to.",
    )

    args = parser.parse_args()

    config_file = Path(args.config)
    command_line_args = {}
    if args.start:
        command_line_args["start_date"] = args.start
    if args.end:
        command_line_args["end_date"] = args.end
    if args.output_path:
        command_line_args["outputpath"] = args.output_path

    if args.kwargs:
        command_line_args.update(args.kwargs)

    if not config_file.exists():
        raise ValueError(
            "Configuration file cannot be found.\n"
            f"Please check path and filename are correct: {config_file}"
        )


    param = ini_extract_param(config_file, **command_line_args)

    output.copy_config_file(config_file, param=param, **command_line_args)

    param = convert_to_type(param)

    bristau_function(param)
