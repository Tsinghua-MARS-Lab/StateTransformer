from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)

from nuplan_simulation.str_trajectory_generator import STRTrajectoryGenerator


class STRClosedPlanner(PDMClosedPlanner):

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        str_generator: STRTrajectoryGenerator,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
    ):
        """
        Constructor for PDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """
        super(STRClosedPlanner, self).__init__(
            trajectory_sampling,
            proposal_sampling,
            idm_policies,
            lateral_offsets,
            map_radius,
        )
        self._str_generator = str_generator

    def initialize(self, initialization: PlannerInitialization) -> None:
        super().initialize(initialization)
        self._str_generator.initialize(initialization)

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """

        ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )
        str_pred_states = self._str_generator.predict_states(current_input)

        # 3.1 transform str_pred_states to proposal array
        str_proposal_array = ego_states_to_state_array(str_pred_states)
        # 3.2 Concat proposals array
        concatenated_proposals_array = np.concatenate(
            (proposals_array, str_proposal_array[None, ...]), axis=0
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            concatenated_proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )

        # 6.b Otherwise, extend and output best proposal
        if trajectory is None:
            max_score_idx = np.argmax(proposal_scores)

            if max_score_idx == len(proposal_scores) - 1:
                # str trajectory is the best option
                trajectory = InterpolatedTrajectory(str_pred_states)
            else:
                # regular pdm trajectory is the best option
                trajectory = self._generator.generate_trajectory(max_score_idx)

        return trajectory
