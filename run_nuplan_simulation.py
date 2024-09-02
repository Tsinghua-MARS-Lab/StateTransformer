import os
import argparse
from typing import Dict
from pathlib import Path

import hydra

from nuplan.planning.script.run_simulation import (
    run_simulation as nuplan_run_simulation,
)
from nuplan.planning.script.builders.planner_builder import build_planners

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

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
NUPLAN_HYDRA_CONFIG_PATH = "nuplan/planning/script/config/simulation"


def key_value_pairs(input_string: str) -> Dict[str, str]:
    try:
        result = {}
        pairs = input_string.split(",")
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
        "--model-path",
        type=str,
        help="Path to HF model",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable or disable debug mode"
    )
    parser.add_argument(
        "--number_of_gpus_allocated_per_simulation",
        type=float,
        default=0.0,
        help="Number (or fractional, e.g., 0.25) of GPUs available for single simulation",
    )
    parser.add_argument(
        "--number_of_cpus_allocated_per_simulation",
        type=int,
        default=1,
        help="number of CPU threads that are used for simulation",
    )
    parser.add_argument(
        "--hydra-overrides",
        type=key_value_pairs,
        default={},
        help='key-value pairs, splitted by comma, i.e. "key1=value1,key2=value2"',
    )
    parser.add_argument(
        "--extrapolate",
        action='store_true',  # 如果提供则为 True，否则为 False
        default=False,
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default="closed_loop_nonreactive_agents",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="str_closed_planner",
    )
    parser.add_argument(
        "--always_emergency_brake",
        action='store_true',
        default=False,
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
    CONFIG_NAME = "default_simulation"

    # Select the planner and simulation challenge
    PLANNER = args.planner 
    # PLANNER = "str_closed_planner"  # [simple_planner, ml_planner]
    # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    CHALLENGE = args.challenge
    SPLIT = "val14_split"

    # Name of the experiment
    EXPERIMENT = "simulation_simple_experiment"

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    overrides_dict = {
        "worker.debug_mode": args.debug,
        "group": "testing_log",
        "experiment_name": EXPERIMENT,
        "planner": PLANNER,
        "planner.str_closed_planner.always_emergency_brake": args.always_emergency_brake,
        "+simulation": CHALLENGE,
        "scenario_builder": "nuplan",  # use nuplan mini database
        "scenario_builder.data_root": args.nuplan_data_path,
        "scenario_builder.map_root": args.nuplan_map_path,
        "scenario_builder.sensor_root": "null",
        "scenario_filter": SPLIT,  # initially select all scenarios in the database
        "number_of_gpus_allocated_per_simulation": args.number_of_gpus_allocated_per_simulation,
        "number_of_cpus_allocated_per_simulation": args.number_of_cpus_allocated_per_simulation,
        "hydra.searchpath": '["pkg://tuplan_garage.planning.script.config.common","pkg://tuplan_garage.planning.script.config.simulation","pkg://nuplan.planning.script.config.common","pkg://nuplan.planning.script.experiments"]',
    }
    if args.model_path:
        overrides_dict.update(
            {"planner.str_closed_planner.str_generator.model_path": args.model_path}
        )
    if args.extrapolate:
        overrides_dict.update(
            {"planner.str_closed_planner.str_generator.extrapolate": args.extrapolate}
        )
    overrides_dict.update(args.hydra_overrides)
    # Compose the configuration
    overrides = [f"{k}={v}" for k, v in overrides_dict.items()]
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=overrides)

    return cfg


def main(args):
    cfg = parse_hydra_config(args)
    planners = build_planners(planner_cfg=cfg.planner, scenario=None)
    nuplan_run_simulation(cfg, planners)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
