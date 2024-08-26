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
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_hybrid_planner import PDMHybridPlanner
# from transformer4planning.models.model import build_models
from transformer4planning.models.backbone.str_base import build_models
from transformer4planning.utils.args import ModelArguments
from transformers import (HfArgumentParser)
import os
import json

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
    elif is_target_type(planner_cfg, PDMHybridPlanner):
        # planner: AbstractPlanner = instantiate(config)
        if model is None:
        # if True: # planner_counter % 200 == 0:  # model is None:  # for small models
            # initialize model args by default values
            parser = HfArgumentParser((ModelArguments))
            # load model args from config.json
            config_path = os.path.join(planner_cfg.checkpoint_path, 'config.json')
            # with open(config_path) as f:
            #     loaded_model_args = json.load(f)
            # print('loaded model args: ', loaded_model_args)
            # # update model args
            # for key, value in loaded_model_args.items():
            #     if hasattr(model_args, key):
            #         setattr(model_args, key, value)
            #     else:
            #         print('WARNING key not found in model args: ', key)
            if not os.path.exists(config_path):
                print('WARNING config.json not found in checkpoint path, using default model args ', config_path)
                model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
            else:
                model_args, = parser.parse_json_file(config_path, allow_extra_keys=True)
                model_args.model_pretrain_name_or_path = planner_cfg.checkpoint_path
                model_args.model_name = model_args.model_name.replace('scratch', 'pretrained')
            print('debug model args: ', model_args, model_args.model_name, planner_cfg.checkpoint_path)
            new_model = build_models(model_args=model_args)
            print('model built')
            # use cpu only for ray distributed simulations
            # import torch
            # if torch.cuda.is_available():
            #     new_model.to('cuda')
            model = new_model.to('cpu')
        else:
            new_model = model
        # print("STR planner initialized ", planner_counter)
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