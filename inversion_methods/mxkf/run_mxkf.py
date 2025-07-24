import json
import argparse


from typing import get_type_hints
from pathlib import Path
from dataclasses import fields

import openghg_inversions.hbmcmc.hbmcmc_output as output
from openghg_inversions.config import config
from inversion_methods.mxkf.mxkf import InversionParameters, mxkf_function


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
                      "bcprior"
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



def convert_to_type(param_dict: dict) -> InversionParameters:

    type_hints = get_type_hints(InversionParameters)
    kwargs = {}

    for f in fields(InversionParameters):

        if f.name in param_dict:
            
            raw_value = param_dict[f.name]
            expected_type = type_hints[f.name]
             # Convert the string to the appropriate type
            if expected_type == int:
                value = int(raw_value)
            elif expected_type == float:
                value = float(raw_value)
            elif expected_type == bool:
                value = bool(raw_value)
            else:
                value = raw_value  # assume string or other already-valid type
        
        else:
            
            value = None

        
        kwargs[f.name] = value

    return InversionParameters(**kwargs)



if __name__ == "__main__":

    mxkf_inv_path = Path(__file__).parents[2]

    parser = argparse.ArgumentParser(description="Running Bayesian MXKF script")
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

    mxkf_function(param)
