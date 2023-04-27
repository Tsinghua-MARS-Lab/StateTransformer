import os
import tempfile
import hydra
from dataclasses import dataclass
from planner import ControlTFPlanner
from pathlib import Path
from nuplan.planning.script.run_simulation import run_simulation as main_simulation


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str

def construct_simulation_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = os.path.join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = os.path.join(base_config_path, 'config', 'simulation')
    experiment_dir = os.path.join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)

# Location of paths with all simulation configs
BASE_CONFIG_PATH = "../script"
simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

# Create a temporary directory to store the simulation artifacts
SAVE_DIR = tempfile.mkdtemp()

# Select simulation parameters
EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]
DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
    'scenario_filter=one_continuous_log',  # simulate only one log
    "scenario_filter.log_names=['2021.07.16.20.45.29_veh-35_01095_01486']",
    'scenario_filter.limit_total_scenarios=2',  # use 2 total scenarios
]

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=simulation_hydra_paths.config_path)

# Compose the configuration
cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=[
    f'group={SAVE_DIR}',
    f'experiment_name=planner_tutorial',
    f'job_name=planner_tutorial',
    'experiment=${experiment_name}/${job_name}/${experiment_time}',
    'worker=sequential',
    f'ego_controller={EGO_CONTROLLER}',
    f'observation={OBSERVATION}',
    f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
    'output_dir=${group}/${experiment}',
    *DATASET_PARAMS,
])


os.environ["NUPLAN_MAPS_ROOT"]="/home/shiduozhang/nuplan/dataset/maps/"
planner = ControlTFPlanner(horizon_seconds=10.0, sampling_time=0.1, acceleration=[5, 5])

# Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
main_simulation(cfg, planner)

# Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
results_dir = list(list(list(Path(SAVE_DIR).iterdir())[0].iterdir())[0].iterdir())[0]  # get the child dir 2 levels in
simulation_file = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard']