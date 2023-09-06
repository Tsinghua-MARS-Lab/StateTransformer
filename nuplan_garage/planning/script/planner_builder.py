from typing import List, Optional, Type, cast

from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
# customized model builder
from nuplan_garage.planning.simulation.planner.control_tf_planner.control_tf_planner import ControlTFPlanner
from transformer4planning.models.model import build_models
from transformer4planning.utils import ModelArguments
from transformers import (HfArgumentParser)
import os

model = None
planner_counter = 0

def _build_planner(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> AbstractPlanner:
    """
    Instantiate planner
    :param planner_cfg: config of a planner
    :param scenario: scenario
    :return AbstractPlanner
    """
    global model, planner_counter

    config = planner_cfg.copy()
    if is_target_type(planner_cfg, MLPlanner):
        # Build model and feature builders needed to run an ML model in simulation
        torch_module_wrapper = build_torch_module_wrapper(planner_cfg.model_config)
        model = LightningModuleWrapper.load_from_checkpoint(
            planner_cfg.checkpoint_path, model=torch_module_wrapper
        ).model

        # Remove config elements that are redundant to MLPlanner
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)

        planner: AbstractPlanner = instantiate(config, model=model)
    elif is_target_type(planner_cfg, ControlTFPlanner):

        # planner: AbstractPlanner = instantiate(config)
        # print('testing without initializing model')
        if True: # planner_counter % 200 == 0:  # model is None:  # for small models
        # if model is None:  # for large models
            parser = HfArgumentParser((ModelArguments))
            model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
            model_args.model_name = 'pretrain-gpt-small'
            files_in_dir = os.listdir(config.checkpoint_path)
            config_file_path = None
            for file_in_dir in files_in_dir:
                """
                Load the config file from the checkpoint path if exists
                """
                if file_in_dir.endswith('.json') and 'config' in file_in_dir:
                    config_file_path = os.path.join(config.checkpoint_path, file_in_dir)
                    break
            if config_file_path is not None:
                model_args, = parser.parse_json_file(config_file_path, allow_extra_keys=True)
            print('debug model args: ', model_args.model_name, model_args, config.checkpoint_path)
            model_args.model_pretrain_name_or_path = config.checkpoint_path
            new_model = build_models(model_args=model_args)
            print('model built')
            # use cpu only for ray distributed simulations
            # import torch
            # if torch.cuda.is_available():
            #     new_model.to('cuda')
            model = new_model
        else:
            new_model = model
        print("control transformer planner initialized ", planner_counter)
        planner: AbstractPlanner = instantiate(config, model=new_model, scenario=scenario)
        planner_counter += 1
    else:
        planner_cls: Type[AbstractPlanner] = _locate(config._target_)

        if planner_cls.requires_scenario:
            assert scenario is not None, (
                "Scenario was not provided to build the planner. " f"Planner {config} can not be build!"
            )
            planner = cast(AbstractPlanner, instantiate(config, scenario=scenario))
        else:
            planner = cast(AbstractPlanner, instantiate(config))

    return planner


def build_planners(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> List[AbstractPlanner]:
    """
    Instantiate multiple planners by calling build_planner
    :param planners_cfg: planners config
    :param scenario: scenario
    :return planners: List of AbstractPlanners
    """
    return [_build_planner(planner, scenario) for planner in planner_cfg.values()]
