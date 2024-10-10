from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
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
    state_array_to_ego_states,
)

from nuplan_simulation.str_trajectory_generator import STRTrajectoryGenerator
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path,
)
import time
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
import gc

from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import (
    PDMGenerator,
)

class STRClosedPlanner(PDMClosedPlanner):

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        str_generator: STRTrajectoryGenerator,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        str_idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        str_lateral_offsets: Optional[List[float]],
        map_radius: float,
        debug_candidate_trajectories: bool,
        always_emergency_brake: bool,
        conservative_factor: float,
        comfort_weight: float,
        initstable_time: int,
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
        self.debug_candidate_trajectories = debug_candidate_trajectories
        self.always_emergency_brake = always_emergency_brake
        self._str_idm_policies = str_idm_policies
        self._str_lateral_offsets = str_lateral_offsets
        self._str_proposal_manager: Optional[PDMProposalManager] = None
        self._strpdm_generator = PDMGenerator(trajectory_sampling, proposal_sampling)

        self._scorer.conservative_factor = conservative_factor
        self._scorer.comfort_weight = comfort_weight
        self.initstable_time = initstable_time
        
        print("============================== STRClosedPlanner")
        print(f"always_emergency_brake: {self.always_emergency_brake}")
        print(f"pdm_lateral_offsets: {self._lateral_offsets}")
        print(f"pdm_speed_limit_fraction: {self._idm_policies._speed_limit_fractions}")
        print(f"str_lateral_offsets: {self._str_lateral_offsets}")
        print(f"str_speed_limit_fraction: {self._str_idm_policies._speed_limit_fractions}")
        print(f"conservative_factor: {self._scorer.conservative_factor}")
        print(f"comfort_weight: {self._scorer.comfort_weight}")
        print(f"initstable_time: {self.initstable_time}")
        print("==============================")

    def initialize(self, initialization: PlannerInitialization) -> None:
        super().initialize(initialization)
        # str generator
        self._str_generator.initialize(initialization)
        # nuboard debug info
        if self.debug_candidate_trajectories:
            self.candidate_trajectories_list: List[List[List[EgoState]]] = []
            self.proposal_scores_list: List[npt.NDArray[np.float64]] = []
            self.pdm_proposals_num_list: List[int] = []


    def compute_planner_trajectory(
        self, current_input: PlannerInput, str_pred_states=None
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        trajectory = self._get_closed_loop_trajectory(current_input, str_pred_states)

        self._iteration += 1
        return trajectory


    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
        str_pred_states = None,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """
        # t0 = time.time()
        ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )
        speed_limit = self._get_starting_lane(ego_state).speed_limit_mps
        if speed_limit is None:
            speed_limit = float("inf")
            
        # 2. Centerline extraction and proposal update
        if str_pred_states is None:
            str_pred_states, scores = self._str_generator.predict_states(current_input)  # str_proposal as centerline
            str_pred_states = str_pred_states[0]
        self._update_proposal_manager(ego_state, str_pred_states)

        # 3. Generate/Unroll proposals
        pdm_proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )
        
        str_proposals_array = self._strpdm_generator.generate_proposals(
            ego_state, self._observation, self._str_proposal_manager
        )
        
        str_init_proposal_array = np.expand_dims(ego_states_to_state_array(str_pred_states), axis=0) 
        proposals_array = np.concatenate((pdm_proposals_array, str_proposals_array, str_init_proposal_array), axis=0)

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
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
            speed_limit,
        )

        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer, self.always_emergency_brake
        )

        # 6.b Otherwise, extend and output best proposal
        if trajectory is None:
            if self._iteration < self.initstable_time:
                trajectory = InterpolatedTrajectory(str_pred_states)
            else:
                if np.argmax(proposal_scores) == len(pdm_proposals_array) + len(str_proposals_array):
                    trajectory = InterpolatedTrajectory(str_pred_states)
                elif np.argmax(proposal_scores) < len(pdm_proposals_array):
                    trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))
                else:
                    trajectory = self._strpdm_generator.generate_trajectory(np.argmax(proposal_scores) - len(pdm_proposals_array))
        return trajectory
    

    def _update_candidate_trajectories(
        self,
        str_pred_states: List[List[EgoState]],
        proposal_scores: npt.NDArray[np.float64],
    ) -> None:
        # update
        self.candidate_trajectories_list.append(str_pred_states)
        self.proposal_scores_list.append(proposal_scores)
        self.pdm_proposals_num_list.append(0)


    def _update_proposal_manager(self, ego_state: EgoState, str_pred_states):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        current_lane = self._get_starting_lane(ego_state)

        # TODO: Find additional conditions to trigger re-planning
        # create_new_proposals = self._iteration == 0
        create_new_proposals = True

        if create_new_proposals:
            pdm_proposal_paths: List[PDMPath] = self._get_pdm_proposal_paths(current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=pdm_proposal_paths,
                longitudinal_policies=self._idm_policies,
            )
            
            str_proposal_paths: List[PDMPath] = self._get_str_proposal_paths(str_pred_states)

            self._str_proposal_manager = PDMProposalManager(
                lateral_proposals=str_proposal_paths,
                longitudinal_policies=self._str_idm_policies,
            )
        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)
        self._str_proposal_manager.update(current_lane.speed_limit_mps)


    def _get_pdm_proposal_paths(
        self, current_lane
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        output_paths = []
        
        ### pdm proposals
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)
        
        # 1. save centerline path (necessary for progress metric)
        output_paths.append(self._centerline)

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))
                
        return output_paths
    
    
    def _get_str_proposal_paths(
        self, str_pred_states
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        output_paths = []
        
        ### str proposals
        str_centerline_discrete_path = [state.rear_axle for state in str_pred_states]
        str_centerline = PDMPath(str_centerline_discrete_path)
        output_paths.append(str_centerline)

        # 2. add additional paths with lateral offset of centerline
        if self._str_lateral_offsets is not None:
            for str_lateral_offset in self._str_lateral_offsets:
                str_offset_discrete_path = parallel_discrete_path(
                    discrete_path=str_centerline_discrete_path, offset=str_lateral_offset
                )
                output_paths.append(PDMPath(str_offset_discrete_path))
                
        return output_paths
    