import os
import argparse
from typing import List
from pathlib import Path

import hydra
from omegaconf import DictConfig

from nuplan.planning.script.run_simulation import (
    run_simulation as nuplan_run_simulation,
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.script.builders.planner_builder import build_planners


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
    PLANNER = "str_closed_planner"  # [simple_planner, ml_planner]
    # PLANNER = "simple_planner"  # [simple_planner, ml_planner]
    CHALLENGE = "closed_loop_nonreactive_agents"  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    SPLIT = "val14_split"

    # Name of the experiment
    EXPERIMENT = "simulation_simple_experiment"

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    overrides = [
        f"worker.debug_mode={args.debug}",
        f"group=testing_log",
        f"experiment_name={EXPERIMENT}",
        f"planner={PLANNER}",
        f"+simulation={CHALLENGE}",
        f"scenario_builder=nuplan",  # use nuplan mini database
        f"scenario_builder.data_root={args.nuplan_data_path}",
        f"scenario_builder.map_root={args.nuplan_map_path}",
        f"scenario_builder.sensor_root=null",
        f"scenario_filter={SPLIT}",  # initially select all scenarios in the database
        f"hydra.searchpath=['pkg://tuplan_garage.planning.script.config.common', 'pkg://tuplan_garage.planning.script.config.simulation', 'pkg://nuplan.planning.script.config.common', 'pkg://nuplan.planning.script.experiments']",
    ]
    if args.model_path:
        overrides.append(
            f"planner.str_closed_planner.str_generator.model_path={args.model_path}"
        )

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=overrides)

    return cfg


def main(args):
    cfg = parse_hydra_config(args)
    planners = build_planners(planner_cfg=cfg.planner, scenario=None)
    nuplan_run_simulation(cfg, planners)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
