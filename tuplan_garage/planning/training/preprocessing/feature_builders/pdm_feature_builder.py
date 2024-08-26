from __future__ import annotations

from typing import List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    TimeDuration,
    TimePoint,
)
from nuplan.planning.metrics.utils.state_extractors import (
    extract_ego_acceleration,
    extract_ego_yaw_rate,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features,
)
from shapely.geometry import Point, LineString

from nuplan.common.actor_state.state_representation import Point2D

from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)

import cv2
import torch
import math
from transformer4planning.utils.nuplan_utils import generate_contour_pts
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from transformer4planning.utils.nuplan_utils import get_angle_of_a_line

class PDMFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder class for PDMOpen and PDMOffset."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        planner: Optional[PDMClosedPlanner],
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
    ):
        """
        Constructor for PDMFeatureBuilder
        :param history_sampling: dataclass for storing trajectory sampling
        :param centerline_samples: number of centerline poses
        :param centerline_interval: interval of centerline poses [m]
        :param planner: PDMClosed planner for correction
        """
        assert (
            type(planner) == PDMClosedPlanner or planner is None
        ), f"PDMFeatureBuilder: Planner must be PDMClosedPlanner or None, but got {type(planner)}"

        self._trajectory_sampling = trajectory_sampling
        self._history_sampling = history_sampling
        self._centerline_samples = centerline_samples
        self._centerline_interval = centerline_interval

        self._planner = planner

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return PDMFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "pdm_features"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> PDMFeature:
        """Inherited, see superclass."""

        past_ego_states = [
            ego_state
            for ego_state in scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=self._history_sampling.time_horizon,
                num_samples=self._history_sampling.num_poses,
            )
        ] + [scenario.initial_ego_state]

        current_input, initialization = self._get_planner_params_from_scenario(scenario)

        return self._compute_feature(past_ego_states, current_input, initialization)

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PDMFeature:
        """Inherited, see superclass."""

        history = current_input.history
        current_ego_state, _ = history.current_state
        past_ego_states = history.ego_states[:-1]

        indices = sample_indices_with_time_horizon(
            self._history_sampling.num_poses, self._history_sampling.time_horizon, history.sample_interval
        )
        past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)] + [
            current_ego_state
        ]

        return self._compute_feature(past_ego_states, current_input, initialization)

    def _get_planner_params_from_scenario(
        self, scenario: AbstractScenario
    ) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        buffer_size = int(2 / scenario.database_interval + 1)

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_scenario(
            buffer_size=buffer_size,
            scenario=scenario,
            observation_type=DetectionsTracks,
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> PDMFeature:
        """
        Creates PDMFeature dataclass based in ego history, and planner input
        :param ego_states: list of ego states
        :param current_input: planner input of current frame
        :param initialization: planner initialization of current frame
        :return: PDMFeature dataclass
        """

        current_ego_state: EgoState = ego_states[-1]
        current_pose: StateSE2 = current_ego_state.rear_axle

        history = current_input.history
        traffic_light_data = current_input.traffic_light_data
        map_name = initialization.map_api.map_name
        map_api = initialization.map_api
        route_blocks_ids = initialization.route_roadblock_ids

        context_length = len(ego_states)  # context_length = 22/23
        past_seconds = 2
        frame_rate = 20
        frame_rate_change = 2
        frame_id = context_length * frame_rate_change - 1  # 22 * 2 = 44 in 20hz
        scenario_start_frame = frame_id - past_seconds * frame_rate  # 44 - 2 * 20 = 4
        past_sample_interval = 2

        sample_frames_in_past_20hz = [frame_id - 40, frame_id - 20, frame_id - 10, frame_id]
        sample_frames_in_past_10hz = [int(frame_id / frame_rate_change) for frame_id in
                                      sample_frames_in_past_20hz]  # length = 8
        # past_frames = int(past_seconds * frame_rate)
        # if context_length < past_frames:
        #     assert False, f"context length is too short, {context_length} {past_frames}"
        # trajectory as format of [(x, y, yaw)]
        oriented_point = np.array([ego_states[-1].rear_axle.x,
                                   ego_states[-1].rear_axle.y,
                                   ego_states[-1].rear_axle.heading]).astype(np.float32)


        road_dic = get_road_dict(map_api, ego_pose_center=Point2D(oriented_point[0], oriented_point[1]))
        sampled_ego_states = [ego_states[i] for i in sample_frames_in_past_10hz]
        ego_trajectory = np.array([(ego_state.rear_axle.x,
                                    ego_state.rear_axle.y,
                                    ego_state.rear_axle.heading) for ego_state in
                                   sampled_ego_states]).astype(np.float32)  # (20, 3)
        ego_shape = np.array([ego_states[-1].waypoint.oriented_box.width,
                              ego_states[-1].waypoint.oriented_box.length]).astype(np.float32)
        observation_buffer = list(history.observation_buffer)
        sampled_observation_buffer = [observation_buffer[i] for i in sample_frames_in_past_10hz]
        agents = [observation.tracked_objects.get_agents() for observation in sampled_observation_buffer]
        statics = [observation.tracked_objects.get_static_objects() for observation in sampled_observation_buffer]

        # corrected_route_ids = route_roadblock_correction(
        #     ego_states[-1],
        #     map_api,
        #     route_blocks_ids,
        # )
        # if len(corrected_route_ids) > 2:
        #     self._route_roadblock_ids = corrected_route_ids
        # else:
        #     print('Route correction in progress before: ', self._route_roadblock_ids)
        #     print('Route correction in progress after: ', corrected_route_ids)

        corrected_route_ids = route_blocks_ids

        high_res_raster, low_res_raster, context_action, agent_rect_pts_local = self.compute_raster_input(
            ego_trajectory, agents, statics, traffic_light_data, ego_shape,
            origin_ego_pose=oriented_point,
            map_name=map_name,
            corrected_route_ids=corrected_route_ids,
            road_dic=road_dic)

        if True: # self._model.config.use_speed:
            speed = np.ones((context_action.shape[0], 3), dtype=np.float32) * -1
            for i in range(context_action.shape[0]):
                current_ego_dynamic = sampled_ego_states[i].dynamic_car_state  # Class of DynamicCarState
                speed[i, :] = [
                    current_ego_dynamic.speed,
                    current_ego_dynamic.acceleration,
                    current_ego_dynamic.angular_velocity]
            context_action = np.concatenate([context_action, speed], axis=1)

        # return {
        #     'high_res_raster': high_res_raster,
        #     'low_res_raster': low_res_raster,
        #     'context_actions': context_action,
        #     'ego_pose': oriented_point,
        #     'agents_rect_local': agent_rect_pts_local,
        # }

        # extract ego vehicle history states
        ego_position = get_ego_position(ego_states)
        ego_velocity = get_ego_velocity(ego_states)
        ego_acceleration = get_ego_acceleration(ego_states)

        # run planner
        self._planner.initialize(initialization)
        trajectory: InterpolatedTrajectory = self._planner.compute_planner_trajectory(
            current_input
        )

        # extract planner trajectory
        future_step_time: TimeDuration = TimeDuration.from_s(
            self._trajectory_sampling.step_time
        )
        future_time_points: List[TimePoint] = [
            trajectory.start_time + future_step_time * (i + 1)
            for i in range(self._trajectory_sampling.num_poses)
        ]
        trajectory_ego_states = trajectory.get_state_at_times(
            future_time_points
        )  # sample to model trajectory

        planner_trajectory = ego_states_to_state_array(
            trajectory_ego_states
        )  # convert to array
        planner_trajectory = planner_trajectory[
            ..., StateIndex.STATE_SE2
        ]  # drop values
        planner_trajectory = convert_absolute_to_relative_se2_array(
            current_pose, planner_trajectory
        )  # convert to relative coords

        # extract planner centerline
        centerline: PDMPath = self._planner._centerline
        current_progress: float = centerline.project(Point(*current_pose.array))
        centerline_progress_values = (
            np.arange(self._centerline_samples, dtype=np.float64)
            * self._centerline_interval
            + current_progress
        )  # distance values to interpolate
        planner_centerline = convert_absolute_to_relative_se2_array(
            current_pose,
            centerline.interpolate(centerline_progress_values, as_array=True),
        )  # convert to relative coords

        return PDMFeature(
            ego_position=ego_position,
            ego_velocity=ego_velocity,
            ego_acceleration=ego_acceleration,
            planner_centerline=planner_centerline,
            planner_trajectory=planner_trajectory,
            high_res_raster=high_res_raster,
            low_res_raster=low_res_raster,
            context_actions=context_action,
            ego_pose=oriented_point,
        )
        
    @staticmethod
    def compute_raster_input(ego_trajectory, agents_seq, statics_seq, traffic_data=None,
                             ego_shape=None, max_dis=300, origin_ego_pose=None, map_name=None,
                             use_speed=True, corrected_route_ids=None, road_dic=None
                             ):
        """
        the first dimension is the sequence length, each timestep include n-items.
        agent_seq and statics_seq are both agents in raster definition
        """
        # origin_ego_pose: (x, y, yaw) in current timestamp
        import copy
        road_types = 20
        agent_type = 8
        traffic_types = 4
        high_res_scale = 4
        low_res_scale = 0.77
        # context_length = past_frames // self.model.config.past_sample_interval  # 4
        context_length = len(ego_trajectory)
        # print("context length", context_length)
        # assert context_length == len(ego_trajectory), f'context length {context_length} != ego trajectory length {len(ego_trajectory)}'
        total_raster_channels = 2 + road_types + traffic_types + agent_type * context_length
        raster_shape = [224, 224, total_raster_channels]

        # WARNING: Not Supporting Goal Points
        rasters_high_res = np.zeros(raster_shape, dtype=np.uint8)
        rasters_low_res = np.zeros(raster_shape, dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        y_inverse = -1 if map_name == "sg-one-north" else 1

        ## channel 0-1: goal route
        cos_, sin_ = math.cos(-origin_ego_pose[2] - math.pi / 2), math.sin(-origin_ego_pose[2] - math.pi / 2)
        route_ids = corrected_route_ids
        for route_id in route_ids:
            if int(route_id) == -1:
                continue
            if int(route_id) not in road_dic:
                print('ERROR: ', route_id, ' not found in road_dic with ', map_name)
                continue
            xyz = road_dic[int(route_id)]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                    abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis):
                continue
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, \
                                                         simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_route = (simplified_xyz * high_res_scale + raster_shape[0] // 2).astype('int32')
            low_res_route = (simplified_xyz * low_res_scale + raster_shape[0] // 2).astype('int32')

            cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
            cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))

            route_lanes = road_dic[int(route_id)]["lower_level"]
            for each_route_lane in route_lanes:
                xyz = road_dic[int(each_route_lane)]["xyz"].copy()
                xyz[:, :2] -= origin_ego_pose[:2]
                pts = list(zip(xyz[:, 0], xyz[:, 1]))
                line = LineString(pts)
                simplified_xyz_line = line.simplify(1)
                simplified_x, simplified_y = simplified_xyz_line.xy
                simplified_xyz = np.ones((len(simplified_x), 2)) * -1
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,
                                                                                                  1].copy() * sin_, simplified_xyz[
                                                                                                                    :,
                                                                                                                    0].copy() * sin_ + simplified_xyz[
                                                                                                                                       :,
                                                                                                                                       1].copy() * cos_
                simplified_xyz[:, 1] *= -1
                simplified_xyz[:, 0] *= y_inverse
                high_res_route = (simplified_xyz * high_res_scale).astype('int32') + 112
                low_res_route = (simplified_xyz * low_res_scale).astype('int32') + 112
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[1], tuple(high_res_route[j, :2]),
                             tuple(high_res_route[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[1], tuple(low_res_route[j, :2]),
                             tuple(low_res_route[j + 1, :2]), (255, 255, 255), 2)

        for i, key in enumerate(road_dic):
            xyz = road_dic[key]["xyz"].copy()
            road_type = int(road_dic[key]['type'])
            xyz[:, :2] -= origin_ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                    abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis):
                continue
            # simplify road vector, can simplify about half of all the points
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,
                                                                                              1].copy() * sin_, simplified_xyz[
                                                                                                                :,
                                                                                                                0].copy() * sin_ + simplified_xyz[
                                                                                                                                   :,
                                                                                                                                   1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_road = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_road = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            if road_type in [5, 17, 18, 19]:
                cv2.fillPoly(
                    rasters_high_res_channels[road_type + 2], np.int32([high_res_road[:, :2]]), (255, 255, 255))
                cv2.fillPoly(
                    rasters_low_res_channels[road_type + 2], np.int32([low_res_road[:, :2]]), (255, 255, 255))
            else:
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[road_type + 2], tuple(high_res_road[j, :2]),
                             tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[road_type + 2], tuple(low_res_road[j, :2]),
                             tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)

        # traffic light
        for each_traffic_light_data in traffic_data:
            traffic_state = int(each_traffic_light_data.status)
            lane_id = int(each_traffic_light_data.lane_connector_id)
            if lane_id not in road_dic:
                continue
            xyz = road_dic[lane_id]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            if not ((abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                    abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis)):
                pts = list(zip(xyz[:, 0], xyz[:, 1]))
                line = LineString(pts)
                simplified_xyz_line = line.simplify(1)
                simplified_x, simplified_y = simplified_xyz_line.xy
                simplified_xyz = np.ones((len(simplified_x), 2)) * -1
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,
                                                                                                  1].copy() * sin_, simplified_xyz[
                                                                                                                    :,
                                                                                                                    0].copy() * sin_ + simplified_xyz[
                                                                                                                                       :,
                                                                                                                                       1].copy() * cos_
                simplified_xyz[:, 1] *= -1
                simplified_xyz[:, 0] *= y_inverse
                high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
                low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
                # traffic state order is GREEN, RED, YELLOW, UNKNOWN
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[2 + road_types + traffic_state],
                             tuple(high_res_traffic[j, :2]),
                             tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[2 + road_types + traffic_state],
                             tuple(low_res_traffic[j, :2]),
                             tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)

        cos_, sin_ = math.cos(-origin_ego_pose[2]), math.sin(-origin_ego_pose[2])
        agent_rect_pts_local = []
        ## agent includes VEHICLE, PEDESTRIAN, BICYCLE, EGO(except)
        for i, each_type_agents in enumerate(agents_seq):
            # i is the time sequence
            current_agent_rect_pts_local = []
            current_agent_pose_unnorm = []
            for j, agent in enumerate(each_type_agents):
                agent_type = int(agent.tracked_object_type.value)
                pose = np.array([agent.box.center.point.x, agent.box.center.point.y,
                                 agent.box.center.heading], dtype=np.float32)
                pose -= origin_ego_pose
                if (abs(pose[0]) > max_dis or abs(pose[1]) > max_dis):
                    continue
                rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                                pose[0] * sin_ + pose[1] * cos_]
                shape = np.array([agent.box.width, agent.box.length], dtype=np.float32)
                if shape[0] == 0 or shape[1] == 0:
                    continue
                shape = np.clip(shape, 0.1, 100.0)
                rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                                direction=-pose[2])
                rect_pts = np.array(rect_pts, dtype=np.float64)
                rect_pts[:, 0] *= y_inverse
                if agent_type != 7:
                    current_agent_rect_pts_local.append(rect_pts)
                    current_agent_pose_unnorm.append(pose)
                rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
                # example: if frame_interval = 10, past frames = 40
                # channel number of [index:0-frame_0, index:1-frame_10, index:2-frame_20, index:3-frame_30, index:4-frame_40]  for agent_type = 0
                # channel number of [index:5-frame_0, index:6-frame_10, index:7-frame_20, index:8-frame_30, index:9-frame_40]  for agent_type = 1
                # ...
                cv2.drawContours(rasters_high_res_channels[
                                     2 + road_types + traffic_types + agent_type * context_length + i],
                                 [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
                cv2.drawContours(rasters_low_res_channels[
                                     2 + road_types + traffic_types + agent_type * context_length + i],
                                 [rect_pts_low_res], -1, (255, 255, 255), -1)

            max_agent_num = 400
            if len(current_agent_rect_pts_local) == 0:
                agent_rect_pts_local.append(np.zeros((max_agent_num, 4, 2)))
                continue
            current_agent_rect_pts_local = np.stack(current_agent_rect_pts_local)
            if len(current_agent_rect_pts_local) > max_agent_num:
                print('agent more than 400 overflowing: ', current_agent_rect_pts_local.shape)
                current_agent_rect_pts_local = current_agent_rect_pts_local[:max_agent_num]
            elif len(current_agent_rect_pts_local) < max_agent_num:
                current_agent_rect_pts_local = np.concatenate([current_agent_rect_pts_local, np.zeros((max_agent_num - len(current_agent_rect_pts_local), 4, 2))], axis=0)

            agent_rect_pts_local.append(current_agent_rect_pts_local)
        agent_rect_pts_local = np.stack(agent_rect_pts_local)  # 4(time steps), 300(agent number), 4(rectangle points), 2(x, y)

        # # check and only keep static agents
        # static_num = 0
        # for i in range(max_agent_num):
        #     first_pt = agent_rect_pts_local[0, i, :, :]
        #     last_pt = agent_rect_pts_local[-1, i, :, :]
        #     if first_pt.sum() == 0 and last_pt.sum() == 0:
        #         continue
        #     if not (abs(first_pt - last_pt).sum() < 0.1):
        #         agent_rect_pts_local[:, i, :, :] = 0
        #     else:
        #         static_num += 1

        recentered_ego_trajectory = ego_trajectory - origin_ego_pose
        for i, pose in enumerate(recentered_ego_trajectory):
            agent_type = 7  # type EGO is 7
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]),
                                            w=ego_shape[0], l=ego_shape[1],
                                            direction=-pose[2])
            rect_pts = np.array(rect_pts, dtype=np.float64)
            rect_pts[:, 0] *= y_inverse
            rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
            cv2.drawContours(rasters_high_res_channels[
                                 2 + road_types + traffic_types + agent_type * context_length + i],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
            cv2.drawContours(rasters_low_res_channels[
                                 2 + road_types + traffic_types + agent_type * context_length + i],
                             [rect_pts_low_res], -1, (255, 255, 255), -1)

        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

        # context_actions computation
        recentered_ego_trajectory = ego_trajectory - origin_ego_pose
        rotated_poses = np.array([recentered_ego_trajectory[:, 0] * cos_ - recentered_ego_trajectory[:, 1] * sin_,
                                  recentered_ego_trajectory[:, 0] * sin_ + recentered_ego_trajectory[:, 1] * cos_,
                                  np.zeros(recentered_ego_trajectory.shape[0]),
                                  recentered_ego_trajectory[:, -1]], dtype=np.float32).transpose(1, 0)
        rotated_poses[:, 1] *= y_inverse

        context_actions = rotated_poses  # (4, 4)
        return rasters_high_res, rasters_low_res, np.array(context_actions, dtype=np.float32), agent_rect_pts_local

def get_road_dict(map_api, ego_pose_center):
    road_dic = {}
    # Collect lane information, following nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils.get_neighbor_vector_map
    # currently NuPlan only supports these map obj classes
    selected_objs = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    selected_objs += [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    selected_objs += [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK]
    selected_objs += [SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]

    all_selected_map_instances = map_api.get_proximal_map_objects(ego_pose_center, 1e8,
                                                                  selected_objs)

    all_selected_objs_to_render = []

    for layer_name in list(all_selected_map_instances.keys()):
        all_selected_obj = all_selected_map_instances[layer_name]
        map_layer_type = layer_name.value
        for selected_obj in all_selected_obj:
            map_obj_id = selected_obj.id
            if map_obj_id in road_dic:
                continue
            speed_limit = 80
            has_traffic_light = -1
            incoming = []
            outgoing = []
            upper_level = []
            lower_level = []
            connector = 0
            if layer_name in [SemanticMapLayer.STOP_LINE]:
                # PED_CROSSING = 0
                # STOP_SIGN = 1
                # TRAFFIC_LIGHT = 2
                # TURN_STOP = 3
                # YIELD = 4
                line_x, line_y = selected_obj.polygon.exterior.coords.xy
                # if selected_obj.stop_line_type not in [0, 1]:
                #     continue
            elif layer_name in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
                line_x, line_y = selected_obj.baseline_path.linestring.coords.xy
                if selected_obj.speed_limit_mps is not None:
                    speed_limit = selected_obj.speed_limit_mps * 3600 / 1609.34  # mps(meters per second) to mph(miles per hour)
                if selected_obj.has_traffic_lights() is not None:
                    has_traffic_light = 1 if selected_obj.has_traffic_lights() else 0
                incoming = [int(obj.id) for obj in selected_obj.incoming_edges]
                outgoing = [int(obj.id) for obj in selected_obj.outgoing_edges]
                upper_level = [int(selected_obj.get_roadblock_id())]
                connector = 1 if layer_name == SemanticMapLayer.LANE_CONNECTOR else 0
            elif layer_name in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
                line_x, line_y = selected_obj.polygon.exterior.coords.xy
                incoming = [int(obj.id) for obj in selected_obj.incoming_edges]
                outgoing = [int(obj.id) for obj in selected_obj.outgoing_edges]
                lower_level = [int(obj.id) for obj in selected_obj.interior_edges]
                connector = 1 if layer_name == SemanticMapLayer.ROADBLOCK_CONNECTOR else 0
            else:
                line_x, line_y = selected_obj.polygon.exterior.coords.xy

            num_of_pts = len(line_x)

            road_xy_np = np.ones([num_of_pts, 3]) * -1
            road_dir_np = np.ones([num_of_pts, 1]) * -1

            for i in range(num_of_pts):
                road_xy_np[i, 0] = line_x[i]
                road_xy_np[i, 1] = line_y[i]
                if i != 0:
                    road_dir_np[i, 0] = get_angle_of_a_line(pt1=[road_xy_np[i - 1, 0], road_xy_np[i - 1, 1]],
                                                            pt2=[road_xy_np[i, 0], road_xy_np[i, 1]])

            new_dic = {
                'dir': road_dir_np, 'type': int(map_layer_type), 'turning': connector,
                'next_lanes': outgoing, 'previous_lanes': incoming,
                'outbound': 0, 'marking': 0,
                'vector_dir': road_dir_np, 'xyz': road_xy_np[:, :3],
                'speed_limit': speed_limit,  # in mph,
                'upper_level': upper_level, 'lower_level': lower_level,
                'render': map_obj_id in all_selected_objs_to_render,
            }
            road_dic[int(map_obj_id)] = new_dic

    # print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
    return road_dic

def get_ego_position(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of relative positions (x, y, θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    ego_poses = build_ego_features(ego_states, reverse=True)
    return ego_poses


def get_ego_velocity(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of ego's velocities (v_x, v_y, v_θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    v_x = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.x for ego_state in ego_states]
    )
    v_y = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.y for ego_state in ego_states]
    )
    v_yaw = extract_ego_yaw_rate(ego_states)
    return np.stack([v_x, v_y, v_yaw], axis=1)


def get_ego_acceleration(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of ego's accelerations (a_x, a_y, a_θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    a_x = extract_ego_acceleration(ego_states, "x")
    a_y = extract_ego_acceleration(ego_states, "y")
    a_yaw = extract_ego_yaw_rate(ego_states, deriv_order=2, poly_order=3)
    return np.stack([a_x, a_y, a_yaw], axis=1)
