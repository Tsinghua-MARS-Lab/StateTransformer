from typing import Dict, List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import (
    AbstractObjective,
)
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import (
    extract_scenario_type_weight,
)
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from transformer4planning.models.decoder.base import mean_circular_error

class L1Objective(AbstractObjective):
    """
    Objective for imitating the expert behavior via an L1-Loss function.
    """

    def __init__(
        self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = "l1_objective"
        self._weight = weight
        self._loss_function = torch.nn.modules.loss.L1Loss(reduction="none")
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(
        self,
        predictions: FeaturesType,
        targets: TargetsType,
        scenarios: ScenarioListType,
    ) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])

        scenario_weights = extract_scenario_type_weight(
            scenarios,
            self._scenario_type_loss_weighting,
            device=predicted_trajectory.data.device,
        )
        # data shape: bsz seq_len 3
        batch_size = predicted_trajectory.data.shape[0]
        assert predicted_trajectory.data.shape == targets_trajectory.data.shape

        xy_data = predicted_trajectory.data[:,:,:2]
        xy_targets = targets_trajectory.data[:,:,:2]
        yaw_data = predicted_trajectory.data[:,:,2]
        yaw_targets = targets_trajectory.data[:,:,2]
        
        loss = self._loss_function(
            xy_data.view(batch_size, -1),
            xy_targets.view(batch_size, -1),
        )
        
        # calculate the mean circular error
        mean_circular_error_value = mean_circular_error(
            yaw_data,
            xy_targets,
        )
        loss += mean_circular_error_value

        return self._weight * torch.mean(loss * scenario_weights[..., None])
