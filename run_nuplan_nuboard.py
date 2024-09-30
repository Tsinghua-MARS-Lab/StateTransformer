import os
import argparse
from typing import Dict
from pathlib import Path

import hydra

from nuplan.planning.script.run_nuboard import main as nuplan_run_nuboard


# default nuplan_devkit_path, if ${NUPLAN_DEVKIT_PATH} is not set
DEFAULT_NUPLAN_DEVKIT_PATH = "/root/workspace/nuplan-devkit"
if os.environ.get("NUPLAN_DEVKIT_PATH") is not None:
    DEFAULT_NUPLAN_DEVKIT_PATH = os.environ["NUPLAN_DEVKIT_PATH"]
# default nuplan_data_root
DEFAULT_NUPLAN_DATA_ROOT = "/lpai/volumes/lpai-autopilot-root-muses/wenxin/str_datasets/nuplan-v1.1_val/data/cache/val"
# default nuplan_map_root
DEFAULT_NUPLAN_MAP_ROOT = (
    "/lpai/volumes/lpai-autopilot-root-muses/wenxin/str_datasets/nuplan-maps/maps"
)
# config path in nuplan_devkit
NUPLAN_HYDRA_CONFIG_PATH = "nuplan/planning/script/config/nuboard"


def key_value_pairs(input_string: str) -> Dict[str, str]:
    try:
        result = {}
        pairs = input_string.split(";")
        for pair in pairs:
            key, value = pair.split("=")
            result[key] = value
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Wrong format. Please use as 'key1=value1,key2=value2'"
        )


def arg_parse():
    parser = argparse.ArgumentParser(description="Run simulation with nuplan.")

    parser.add_argument(
        "--nuplan_devkit_path",
        type=str,
        default=DEFAULT_NUPLAN_DEVKIT_PATH,
        help="Path to nuplan devkit",
    )
    parser.add_argument(
        "--nuplan_data_path",
        type=str,
        default=DEFAULT_NUPLAN_DATA_ROOT,
        help="Path to nuplan data root",
    )
    parser.add_argument(
        "--nuplan_map_path",
        type=str,
        default=DEFAULT_NUPLAN_MAP_ROOT,
        help="Path to nuplan data root",
    )
    parser.add_argument(
        "--hydra-overrides",
        type=key_value_pairs,
        default={},
        help='key-value pairs, splitted by comma, i.e. "key1=value1,key2=value2"',
    )

    args = parser.parse_args()
    return args


def parse_hydra_config(args):
    # settings
    nuplan_devkit_path = Path(args.nuplan_devkit_path)
    config_abs_path = nuplan_devkit_path / NUPLAN_HYDRA_CONFIG_PATH
    current_path = Path.cwd()

    # Location of path with all simulation configs
    CONFIG_PATH = os.path.relpath(config_abs_path, current_path)
    CONFIG_NAME = "default_nuboard"

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    overrides_dict = {
        "scenario_builder": "nuplan",  # use nuplan mini database
        "scenario_builder.data_root": args.nuplan_data_path,
        "scenario_builder.map_root": args.nuplan_map_path,
        "scenario_builder.sensor_root": "null",
    }

    overrides_dict.update(args.hydra_overrides)
    # Compose the configuration
    overrides = [f"{k}={v}" for k, v in overrides_dict.items()]
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=overrides)

    return cfg


def main(args):
    cfg = parse_hydra_config(args)
    nuplan_run_nuboard(cfg)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
