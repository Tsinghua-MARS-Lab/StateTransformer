import os
import pickle
import sys
from copy import deepcopy
from typing import List, Type

import cv2
import math
import numpy as np
from scipy import interpolate
import numpy.typing as npt
import shapely
import torch
import copy
from transformers import (HfArgumentParser)
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer_utils import get_last_checkpoint

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.planner.planner_report import PlannerReport
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from transformer4planning.models.model import build_models
from transformer4planning.utils.args import ModelArguments
from omegaconf import DictConfig
import time
from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_absolute_agent_states_from_numpy_poses, _get_fixed_timesteps
from nuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from transformers import (HfArgumentParser)

def generate_contour_pts(center_pt, w, l, direction):
    pt1 = rotate(center_pt, (center_pt[0] - w / 2, center_pt[1] - l / 2), direction, tuple=True)
    pt2 = rotate(center_pt, (center_pt[0] + w / 2, center_pt[1] - l / 2), direction, tuple=True)
    pt3 = rotate(center_pt, (center_pt[0] + w / 2, center_pt[1] + l / 2), direction, tuple=True)
    pt4 = rotate(center_pt, (center_pt[0] - w / 2, center_pt[1] + l / 2), direction, tuple=True)
    return pt1, pt2, pt3, pt4


def rotate(origin, point, angle, tuple=False):
    """
    Rotate a point counter-clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if tuple:
        return (qx, qy)
    else:
        return qx, qy


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle


class ControlTFPlanner(AbstractPlanner):

    requires_scenario: bool = True

    def __init__(self,
                 max_velocity: float = 5.0,
                 use_backup_planner=True,
                 model=None,
                 steering_angle: float = 0.0,
                 map_radius: float = 50,  # following default PDM offset Model
                 **kwargs):

        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.use_backup_planner = use_backup_planner
        self.idm_planner = IDMPlanner(
            target_velocity=max_velocity,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=3.0,
            planned_trajectory_samples=80,
            planned_trajectory_sample_interval=0.1,
            occupancy_map_radius=map_radius
        )
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)
        # model initialization and configuration
        if model is None:
            parser = HfArgumentParser((ModelArguments))
            model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
            model_args.model_name = 'pretrain-gpt-small'
            print('debug model args: ', model_args.model_name, model_args)
            model_args.model_pretrain_name_or_path = config.checkpoint_path
            self.model = build_models(model_args=model_args)
        else:
            self.model = model
        self.warm_up = False
        scenario = kwargs.get('scenario', None)
        self.scenario_type = scenario.scenario_type
        self.scenario_id = scenario.token
        # self.use_gpu = False  # torch.cuda.is_available()
        self.use_gpu = model.device != 'cpu'
        self._iteration = 0
        self.road_dic = None
        del scenario

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        self.initialization = initialization
        self.goal = initialization.mission_goal
        self.route_roadblock_ids = initialization.route_roadblock_ids
        self._map_api = initialization.map_api
        self.idm_planner.initialize(initialization)

    def warmup(self):
        # if self.use_gpu:
        #     self.model.to("cuda")
            # warmup = self.model.generate(
            #                     context_actions=torch.zeros((1, 10, 4), device='cuda'), \
            #                     high_res_raster=torch.zeros((1, 224, 224, 109), device='cuda'), \
            #                     low_res_raster=torch.zeros((1, 224, 224, 109), device='cuda'), \
            #                     trajectory_label=torch.zeros((1, 160, 4)).to('cuda'))
        self.warm_up = True

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input) -> List[AbstractTrajectory]:
        use_backup_planner = self.use_backup_planner
        start = time.time()
        print("count: ", self._iteration, "cuda:", torch.cuda.is_available(), 'scenario type:', self.scenario_type, "device: ", self.model.device)
        self._iteration += 1
        ego_state, _ = current_input.history.current_state
        # 1. fetch data and convert to features
        traffic_data = current_input.traffic_light_data
        history = current_input.history
        ego_states = list(history.ego_state_buffer)  # a list of ego trajectory
        context_length = len(ego_states)  # context_length = 22
        past_seconds = 2
        frame_rate = 20
        frame_rate_change = 2
        frame_id = context_length * frame_rate_change  # 22 * 2 = 44 in 20hz
        scenario_start_frame = frame_id - past_seconds * frame_rate  # 44 - 2 * 20 = 4
        past_sample_interval = int(self.model.model_args.past_sample_interval)  # 5
        sample_frames_in_past_20hz = list(range(scenario_start_frame, frame_id, past_sample_interval))  # length = 8
        sample_frames_in_past_10hz = [int(frame_id / frame_rate_change) for frame_id in sample_frames_in_past_20hz]  # length = 8
        # past_frames = int(past_seconds * frame_rate)
        # if context_length < past_frames:
        #     assert False, f"context length is too short, {context_length} {past_frames}"
        # trajectory as format of [(x, y, yaw)]
        oriented_point = np.array([ego_states[-1].rear_axle.x,
                                   ego_states[-1].rear_axle.y,
                                   ego_states[-1].rear_axle.heading])
        if self.road_dic is None:
            self.road_dic = get_road_dict(self._map_api, ego_pose_center=Point2D(oriented_point[0], oriented_point[1]))
        sampled_ego_states = [ego_states[i] for i in sample_frames_in_past_10hz]
        ego_trajectory = np.array([(ego_state.rear_axle.x,
                                    ego_state.rear_axle.y,
                                    ego_state.rear_axle.heading) for ego_state in sampled_ego_states])  # (20, 3)
        ego_shape = np.array([ego_states[-1].waypoint.oriented_box.width,
                              ego_states[-1].waypoint.oriented_box.length])
        observation_buffer = list(history.observation_buffer)
        sampled_observation_buffer = [observation_buffer[i] for i in sample_frames_in_past_10hz]
        agents = [observation.tracked_objects.get_agents() for observation in sampled_observation_buffer]
        statics = [observation.tracked_objects.get_static_objects() for observation in sampled_observation_buffer]
        high_res_raster, low_res_raster, context_action = self.compute_raster_input(
            ego_trajectory, agents, statics, traffic_data, ego_shape,
            origin_ego_pose=oriented_point)
        time_after_input = time.time() - start
        print("time after raster build", time_after_input)
        # 2. Centerline extraction and proposal update
        # WARNING: It's extremely slow to initialize pdm planner
        # self._update_proposal_manager(ego_state)

        # 3. Generate trajectory
        # 3.1. Check if control point off-road
        # 3.2. If off-road, force key-point to centerline
        # print('inspect: ', self._centerline.interpolate(np.array([0])))

        # compute idm trajectory and scenario flag
        traffic_stop_threshold = 5
        # idm_trajectory, flag, relative_distance = self.idm_planner.compute_planner_trajectory(current_input)  # (17, 3)
        # if flag == "redlight" and relative_distance < traffic_stop_threshold:
        #     return idm_trajectory

        pred_length = int(160 / self.model.model_args.future_sample_interval)
        with torch.no_grad():
            if self.use_gpu:
                device = self.model.device
                output = self.model.generate(
                    context_actions=torch.tensor(context_action[np.newaxis, ...]).to(device),
                    high_res_raster=torch.tensor(high_res_raster[np.newaxis, ...]).to(device),
                    low_res_raster=torch.tensor(low_res_raster[np.newaxis, ...]).to(device),
                    trajectory_label=torch.zeros((1, pred_length, 4)).to(device),
                    map_api=self._map_api,
                    route_ids=self.route_roadblock_ids,
                    ego_pose=oriented_point,
                    road_dic=self.road_dic,
                    map=self._map_api.map_name,
                    scenario_type=self.scenario_type if self.scenario_type is not None else None,
                    # idm_reference_global=idm_trajectory._trajectory if flag == "redlight" or (relative_distance is not None and relative_distance < traffic_stop_threshold) else None
                )
            else:
                output = self.model.generate(
                    context_actions=torch.tensor(context_action[np.newaxis, ...]),
                    high_res_raster=torch.tensor(high_res_raster[np.newaxis, ...]),
                    low_res_raster=torch.tensor(low_res_raster[np.newaxis, ...]),
                    trajectory_label=torch.zeros((1, pred_length, 4)),
                    map_api=self._map_api,
                    route_ids=self.route_roadblock_ids,
                    ego_pose=oriented_point,
                    road_dic=self.road_dic,
                    map=self._map_api.map_name,
                    scenario_type=self.scenario_type if self.scenario_type is not None else None,
                    # idm_reference_global=idm_trajectory._trajectory if flag == "redlight" or (relative_distance is not None and relative_distance < traffic_stop_threshold) else None
                )
            pred_traj = output[0, -pred_length:, :2].detach().cpu()  # (80, 2)
            pred_key_points = output[0, :-pred_length, :2].detach().cpu()

        pred_traj = torch.cat([pred_traj, torch.zeros([pred_length, 1]), torch.zeros([pred_length, 1]),], dim=1).numpy()
                               # torch.ones([pred_length, 1]) * ego_states[-1].rear_axle.heading], dim=1).numpy()
        # print('inspect 240 trajshape: ', pred_traj.shape)
        # print('inspect 241 traj: ', pred_traj)
        # print('inspect 242 kp: ', pred_key_points)
        print("time after gpt", time.time() - start)
        # # post-processing
        low_filter = False
        low_threshold = 0.01
        if low_filter:
            filtered_traj = np.zeros_like(pred_traj)
            last_pose = None
            for idx, each_pose in enumerate(pred_traj):
                if last_pose is None:
                    dx, dy = each_pose[:2]
                    dx = 0 if dx < low_threshold else dx
                    dy = 0 if dy < low_threshold else dy
                    last_pose = filtered_traj[idx, :] = [dx, dy, each_pose[2], each_pose[3]]
                    continue
                dx, dy = each_pose[:2] - pred_traj[idx - 1, :2]
                dx = 0 if dx < low_threshold else dx
                dy = 0 if dy < low_threshold else dy
                last_pose = filtered_traj[idx, :] = [last_pose[0] + dx, last_pose[1] + dy, each_pose[2],
                                                     each_pose[3]]
            pred_traj = filtered_traj

        # print('inspect 263: ', pred_traj)

        # absolute_traj = np.zeros_like(pred_traj)
        # cos_, sin_ = math.cos(-ego_trajectory[-1][2]), math.sin(-ego_trajectory[-1][2])
        # for i in range(pred_traj.shape[0]):
        #     new_x = pred_traj[i, 0] * cos_ + pred_traj[i, 1] * sin_ + ego_trajectory[-1][0]
        #     new_y = pred_traj[i, 1] * cos_ - pred_traj[i, 0] * sin_ + ego_trajectory[-1][1]
        #     absolute_traj[i, 0] = new_x
        #     absolute_traj[i, 1] = new_y
        #     absolute_traj[i, 2] = 0
        #     absolute_traj[i, -1] = ego_trajectory[-1][-1]

        # compare pred trajectory and idm trajectory
        # idm_threshold = 3
        # traffic_stop_threshold = 5
        # agent_stop_threshold = 3
        # ## judge if route is valid:
        # all_nearby_map_instances = self.map_api.get_proximal_map_objects(
        #     Point2D(ego_trajectory[-1, 0], ego_trajectory[-1, 1]),
        #     0.001, [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
        # all_nearby_map_instances_ids = []
        # for each_type in all_nearby_map_instances:
        #     for each_ins in all_nearby_map_instances[each_type]:
        #         all_nearby_map_instances_ids.append(each_ins.id)
        # for each in all_nearby_map_instances_ids:
        #     if each in self.route_roadblock_ids or int(each) in self.route_roadblock_ids:
        #         in_route = True
        #         break
        #     else:
        #         in_route = False
        #         print("route wrong, forbid IDM")
        # use_backup_planner = use_backup_planner and in_route
        # if use_backup_planner:
        #     # compute idm trajectory and scenario flag
        #     idm_trajectory, flag, relative_distance = self.idm_planner.compute_planner_trajectory(current_input)
        #     if flag == "redlight" and relative_distance < traffic_stop_threshold:
        #         return idm_trajectory
        #
        #     out_pts = 0
        #     sample_frames = list(range(10)) + list(range(10, absolute_traj.shape[0], 5))
        #     for i in sample_frames:
        #         all_nearby_map_instances = self.map_api.get_proximal_map_objects(
        #             Point2D(absolute_traj[i, 0], absolute_traj[i, 1]),
        #             0.1, [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
        #         all_nearby_map_instances_ids = []
        #         for each_type in all_nearby_map_instances:
        #             for each_ins in all_nearby_map_instances[each_type]:
        #                 all_nearby_map_instances_ids.append(each_ins.id)
        #         any_in = False
        #         for each in all_nearby_map_instances_ids:
        #             if each in self.route_roadblock_ids or int(each) in self.route_roadblock_ids:
        #                 any_in = True
        #                 break
        #         if not any_in:
        #             out_pts += 1
        #         if out_pts > idm_threshold:
        #             break
        #
        #     if out_pts > idm_threshold:
        #         out_of_route = True
        #         print('OUT OF ROUTE, Use IDM Planner to correct trajectory')
        #
        #         print(time.time() - start)
        #         return idm_trajectory

        # fix indices
        pred_traj[:-1, :] = pred_traj[1:, :]
        pred_traj[-1, :] = pred_traj[-2, :] + (pred_traj[-2, :] - pred_traj[-3, :])

        relative_traj = pred_traj.copy()
        yaw_change_upper_threshold = 0.1
        # generating yaw angle from relative_traj
        dx = relative_traj[4::5, 0] - relative_traj[:-4:5, 0]
        dy = relative_traj[4::5, 1] - relative_traj[:-4:5, 1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        relative_yaw_angles = np.where(distances > 0.1, np.arctan2(dy, dx), 0)
        # accumulate yaw angle
        # relative_yaw_angles = yaw_angles.cumsum()
        relative_yaw_angles_full = np.repeat(relative_yaw_angles, 5, axis=0)
        relative_traj[:, -1] = relative_yaw_angles_full

        # heading = relative_traj[0, -1]
        # for i in range(relative_traj.shape[0]):
        #     if i > 1 and euclidean_distance(relative_traj[i - 1, :2], relative_traj[i, :2]) > 0.1:
        #         delta_heading = get_angle_of_a_line(relative_traj[i - 1, :2], relative_traj[i, :2])
        #         heading = normalize_angle(heading + min(delta_heading, yaw_change_upper_threshold))
        #     relative_traj[i, -1] = heading

        # change relative poses to absolute states
        if relative_traj.shape[1] == 4:
            new = np.zeros((relative_traj.shape[0], 3))
            new[:, :2] = relative_traj[:, :2]
            new[:, -1] = relative_traj[:, -1]
            relative_traj = new

        if self._iteration == 1:
            import random
            input_dic = {
                'context_actions': context_action[np.newaxis, ...],
                'high_res_raster': high_res_raster[np.newaxis, ...],
                'low_res_raster': low_res_raster[np.newaxis, ...],
                'trajectory_label': torch.zeros((1, pred_length, 4))
            }
            print('inspect: ', relative_traj.shape, relative_traj, pred_key_points.shape, pred_key_points)
            # save_raster_with_results(path_to_save='/home/sunq/debug_raster_controller/',
            #                          inputs=input_dic,
            #                          sample_index=0,
            #                          prediction_trajectory=relative_traj,
            #                          file_index=f'{self.scenario_id}-{self._map_api.map_name}',
            #                          prediction_key_point=pred_key_points)

        # reverse again for singapore trajectory
        y_inverse = -1 if self._map_api.map_name == "sg-one-north" else 1
        relative_traj[:, 1] *= y_inverse

        planned_time_points = _get_fixed_timesteps(ego_states[-1], 8, 0.1)
        states = _get_absolute_agent_states_from_numpy_poses(poses=relative_traj,
                                                             ego_history=ego_states,
                                                             timesteps=planned_time_points)
        states.insert(0, ego_states[-1])
        # post process
        # if use_backup_planner and flag == "leadagent" and relative_distance < agent_stop_threshold:
        #     states[1:5] = idm_trajectory.get_sampled_trajectory()[1:5]

        trajectory = InterpolatedTrajectory(states)
        print("time consumed", time.time() - start)
        return trajectory

    def compute_raster_input(self, ego_trajectory, agents_seq, statics_seq, traffic_data=None,
                             ego_shape=None, max_dis=300, origin_ego_pose=None):
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
        # context_length = past_frames // self.model.model_args.past_sample_interval  # 4
        context_length = len(ego_trajectory)
        # assert context_length == len(ego_trajectory), f'context length {context_length} != ego trajectory length {len(ego_trajectory)}'
        total_raster_channels = 1 + road_types + traffic_types + agent_type * context_length
        raster_shape = [224, 224, total_raster_channels]

        rasters_high_res = np.zeros(raster_shape, dtype=np.uint8)
        rasters_low_res = np.zeros(raster_shape, dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        y_inverse = -1 if self._map_api.map_name == "sg-one-north" else 1
        road_dic = copy.deepcopy(self.road_dic)

        ## goal route
        cos_, sin_ = math.cos(-origin_ego_pose[2] - math.pi / 2), math.sin(-origin_ego_pose[2] - math.pi / 2)
        route_ids = self.route_roadblock_ids
        for route_id in route_ids:
            if int(route_id) == -1:
                continue
            if int(route_id) not in road_dic:
                continue
            xyz = road_dic[int(route_id)]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                    abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis):
                continue
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_route = (simplified_xyz * high_res_scale + raster_shape[0] // 2).astype('int32')
            low_res_route = (simplified_xyz * low_res_scale + raster_shape[0] // 2).astype('int32')

            cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
            cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))

        for i, key in enumerate(road_dic):
            xyz = road_dic[key]["xyz"].copy()
            road_type = int(road_dic[key]['type'])
            xyz[:, :2] -= origin_ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                    abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis):
                continue
            # simplify road vector, can simplify about half of all the points
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_road = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_road = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            if road_type in [5, 17, 18, 19]:
                cv2.fillPoly(
                    rasters_high_res_channels[road_type + 1], np.int32([high_res_road[:, :2]]), (255, 255, 255))
                cv2.fillPoly(rasters_low_res_channels[road_type + 1], np.int32([low_res_road[:, :2]]), (255, 255, 255))
            else:
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[road_type + 1], tuple(high_res_road[j, :2]),
                             tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[road_type + 1], tuple(low_res_road[j, :2]),
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
                line = shapely.geometry.LineString(pts)
                simplified_xyz_line = line.simplify(1)
                simplified_x, simplified_y = simplified_xyz_line.xy
                simplified_xyz = np.ones((len(simplified_x), 2)) * -1
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
                simplified_xyz[:, 1] *= -1
                simplified_xyz[:, 0] *= y_inverse
                high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
                low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
                # traffic state order is GREEN, RED, YELLOW, UNKNOWN
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[1 + road_types + traffic_state],
                             tuple(high_res_traffic[j, :2]),
                             tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[1 + road_types + traffic_state],
                             tuple(low_res_traffic[j, :2]),
                             tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)

        cos_, sin_ = math.cos(-origin_ego_pose[2]), math.sin(-origin_ego_pose[2])
        ## agent includes VEHICLE, PEDESTRIAN, BICYCLE, EGO(except)
        for i, each_type_agents in enumerate(agents_seq):
            for j, agent in enumerate(each_type_agents):
                agent_type = int(agent.tracked_object_type.value)
                pose = np.array([agent.box.center.point.x, agent.box.center.point.y, agent.box.center.heading])
                pose -= origin_ego_pose
                if (abs(pose[0]) > max_dis or abs(pose[1]) > max_dis):
                    continue
                rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                                pose[0] * sin_ + pose[1] * cos_]
                shape = np.array([agent.box.width, agent.box.length])
                rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                                direction=-pose[2])
                rect_pts = np.array(rect_pts, dtype=np.int32)
                rect_pts[:, 0] *= y_inverse
                rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
                # example: if frame_interval = 10, past frames = 40
                # channel number of [index:0-frame_0, index:1-frame_10, index:2-frame_20, index:3-frame_30, index:4-frame_40]  for agent_type = 0
                # channel number of [index:5-frame_0, index:6-frame_10, index:7-frame_20, index:8-frame_30, index:9-frame_40]  for agent_type = 1
                # ...
                cv2.drawContours(rasters_high_res_channels[
                                     1 + road_types + traffic_types + agent_type * context_length + i],
                                 [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
                cv2.drawContours(rasters_low_res_channels[
                                     1 + road_types + traffic_types + agent_type * context_length + i],
                                 [rect_pts_low_res], -1, (255, 255, 255), -1)

        for i, each_type_static in enumerate(statics_seq):
            for j, static in enumerate(each_type_static):
                agent_type = int(static.tracked_object_type.value)
                pose = np.array([static.box.center.point.x, static.box.center.point.y, static.box.center.heading])
                pose -= origin_ego_pose
                if (abs(pose[0]) > max_dis or abs(pose[1]) > max_dis):
                    continue
                rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                                pose[0] * sin_ + pose[1] * cos_]
                shape = np.array([static.box.width, static.box.length])
                rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                                direction=-pose[2])
                rect_pts = np.array(rect_pts, dtype=np.int32)
                rect_pts[:, 0] *= y_inverse
                rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
                # example: if frame_interval = 10, past frames = 40
                # channel number of [index:0-frame_0, index:1-frame_10, index:2-frame_20, index:3-frame_30, index:4-frame_40]  for agent_type = 0
                # channel number of [index:5-frame_0, index:6-frame_10, index:7-frame_20, index:8-frame_30, index:9-frame_40]  for agent_type = 1
                # ...
                cv2.drawContours(rasters_high_res_channels[
                                     1 + road_types + traffic_types + agent_type * context_length + i],
                                 [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
                cv2.drawContours(rasters_low_res_channels[
                                     1 + road_types + traffic_types + agent_type * context_length + i],
                                 [rect_pts_low_res], -1, (255, 255, 255), -1)

        recentered_ego_trajectory = ego_trajectory - origin_ego_pose
        for i, pose in enumerate(recentered_ego_trajectory):
            agent_type = 7  # type EGO is 7
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]),
                                            w=ego_shape[0], l=ego_shape[1],
                                            direction=-pose[2])
            rect_pts = np.array(rect_pts, dtype=np.int32)
            rect_pts[:, 0] *= y_inverse
            rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
            cv2.drawContours(rasters_high_res_channels[
                                 1 + road_types + traffic_types + agent_type * context_length + i],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0] // 2
            cv2.drawContours(rasters_low_res_channels[
                                 1 + road_types + traffic_types + agent_type * context_length + i],
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
        # for i in range(0, len(rotated_poses) - 1):
        #     action = rotated_poses[i]
        #     context_actions.append(action)
        #
        # context_actions = np.array(context_actions, dtype=np.float32)
        context_actions = rotated_poses  # (4, 4)

        result_dict = {'high_res_raster': rasters_high_res, 'low_res_raster': rasters_low_res,
                       'context_actions': rotated_poses}
        # print('saving raster to debug_raster ', context_length)
        # save_raster(result_dic=result_dict,
        #             debug_raster_path='/home/sunq/debug_raster/',
        #             agent_type_num=agent_type,
        #             past_frames_num=context_length,
        #             image_file_name=f'{self._iteration}',
        #             split=f'nuplan_simulation',
        #             high_scale=high_res_raster_scale,
        #             low_scale=low_res_raster_scale)

        # print('inspect feature: ', context_actions[:, 2], context_actions[-1], context_actions)

        return rasters_high_res, rasters_low_res, np.array(context_actions, dtype=np.float32)

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """
        Generate a report containing runtime stats from the planner.
        By default, returns a report containing the time-series of compute_trajectory runtimes.
        :param clear_stats: whether or not to clear stored stats after creating report.
        :return: report containing planner runtime stats.
        """
        report = PlannerReport(compute_trajectory_runtimes=self._compute_trajectory_runtimes)
        if self.use_gpu:
            self.model.to("cpu")
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
        return report


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





if __name__ == "__main__":
    from visulization.checkraster import *

    with open("history.pkl", "rb") as f:
        input = pickle.load(f)

    with open("init.pkl", "rb") as f:
        initial = pickle.load(f)
    map_api = initial.map_api
    ego_pose = (input.ego_state_buffer[-1].waypoint.center.x, \
                input.ego_state_buffer[-1].waypoint.center.y)
    ego_pose = Point2D(ego_pose[0], ego_pose[1])
    radius = 500
    # map_objects = map_api.get_available_map_objects()
    # map_raster_objects = map_api.get_available_raster_layers()
    selected_objs = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    selected_objs += [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    selected_objs += [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK]
    selected_objs += [SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]
    # map_objects = map_api.get_proximal_map_objects(ego_pose, radius, selected_objs)
    # road_dict = get_road_dict(map_api, ego_pose)
    planner = ControlTFPlanner(10, 0.1, np.array([5, 5]))
    planner.initialize(initial)


    def test(planner, history):
        ego_states = history.ego_state_buffer
        context_length = len(ego_states)
        ego_trajectory = np.array([(ego_states[i].waypoint.center.x, \
                                    ego_states[i].waypoint.center.y, \
                                    ego_states[i].waypoint.heading) for i in range(context_length)])
        ego_shape = np.array([ego_states[-1].waypoint.oriented_box.width, \
                              ego_states[-1].waypoint.oriented_box.length])
        agents = [history.observation_buffer[i].tracked_objects.get_agents() for i in range(context_length)]
        statics = [history.observation_buffer[i].tracked_objects.get_static_objects() for i in range(context_length)]
        # road_dic = get_road_dict(planner.map_api, Point2D(ego_trajectory[-1][0], ego_trajectory[-1][1]))
        with open("road_dic.pickle", "rb") as f:
            road_dic = pickle.load(f)
        high_res_raster, low_res_raster, context_action = planner.compute_raster_input(
            ego_trajectory, agents, statics, road_dic, ego_shape)
        visulize_raster("visulization/rasters/nuplan", "high_res", high_res_raster)
        visulize_raster("visulization/rasters/nuplan", "low_res", low_res_raster)
        visulize_trajectory("visulization/rasters/nuplan", trajectory=context_action, scale=4)
        visulize_trajectory("visulization/rasters/nuplan", trajectory=context_action, scale=0.77)
        print("done")


    test(planner, input)


    test(planner, input)
    # planner.compute_planner_trajectory(input)

def save_raster(result_dic, debug_raster_path, agent_type_num, past_frames_num, image_file_name, split,
                high_scale, low_scale):
    # save rasters
    path_to_save = debug_raster_path
    # check if path not exist, create
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        file_number = 0
    else:
        file_number = len(os.listdir(path_to_save))
        if file_number > 1000:
            return
    image_shape = None
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = result_dic[each_key]
        goal = each_img[:, :, 0]
        road = each_img[:, :, :21]
        traffic_lights = each_img[:, :, 21:25]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = np.random.randint(0, 255, size=(21, 3)) * 0.5
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float32)
        image_shape = target_image.shape
        for i in range(21):
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()

        for i in range(past_frames_num):
            for j in range(agent_type_num):
                # if j == 7:
                #     print('debug', np.sum(agent[:, :, j * 9 + i]), agent[:, :, j * 9 + i])
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '.png'), target_image)
    for each_key in ['context_actions']:
        pts = result_dic[each_key]
        for scale in [high_scale, low_scale]:
            target_image = np.zeros(image_shape, dtype=np.float32)
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, :] = [255, 255, 255]
            cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '_' + str(scale) +'.png'), target_image)
    print('length of action and labels: ', result_dic['context_actions'].shape)
    print('debug images saved to: ', path_to_save, file_number)


def save_raster_with_results(path_to_save,
                             inputs, sample_index,
                             prediction_trajectory,
                             file_index,
                             high_scale=4, low_scale=0.77,
                             prediction_key_point=None,
                             prediction_key_point_by_gen=None,
                             prediction_trajectory_by_gen=None):
    import cv2
    # save rasters
    image_shape = None

    # check if path not exist, create
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        file_number = 0
    else:
        file_number = len(os.listdir(path_to_save))
        if file_number > 1000:
            return

    image_to_save = {
        'high_res_raster': None,
        'low_res_raster': None
    }
    past_frames_num = inputs['context_actions'][sample_index].shape[0]
    agent_type_num = 8
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = inputs[each_key][sample_index]
        goal = each_img[:, :, 0]
        road = each_img[:, :, :21]
        traffic_lights = each_img[:, :, 21:25]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = np.random.randint(0, 255, size=(21, 3)) * 0.5
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float)
        image_shape = target_image.shape
        for i in range(21):
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                # if j == 7:
                #     print('debug', np.sum(agent[:, :, j * 9 + i]), agent[:, :, j * 9 + i])
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        if 'high' in each_key:
            scale = high_scale
        elif 'low' in each_key:
            scale = low_scale
        # draw context actions, and trajectory label
        for each_traj_key in ['context_actions', 'trajectory_label']:
            pts = inputs[each_traj_key][sample_index]
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    if 'actions' in each_traj_key:
                        target_image[x, y, :] = [255, 0, 0]
                    # elif 'label' in each_traj_key:
                    #     target_image[x, y, :] = [0, 0, 255]

        # draw prediction trajectory
        for i in range(prediction_trajectory.shape[0]):
            if i % 5 != 0:
                continue
            x = int(prediction_trajectory[i, 0] * scale) + target_image.shape[0] // 2
            y = int(prediction_trajectory[i, 1] * scale) + target_image.shape[1] // 2
            if x < target_image.shape[0] and y < target_image.shape[1]:
                target_image[x, y, :2] == 255

        # draw key points
        if prediction_key_point is not None:
            for i in range(prediction_key_point.shape[0]):
                x = int(prediction_key_point[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, 1:] == 255

        # draw prediction key points during generation
        if prediction_key_point_by_gen is not None:
            for i in range(prediction_key_point_by_gen.shape[0]):
                x = int(prediction_key_point_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, 1:] == 255

        # draw prediction trajectory by generation
        if prediction_trajectory_by_gen is not None:
            for i in range(prediction_trajectory_by_gen.shape[0]):
                x = int(prediction_trajectory_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_trajectory_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, :] += 100
        target_image = np.clip(target_image, 0, 255)
        image_to_save[each_key] = target_image

    for each_key in image_to_save:
        cv2.imwrite(os.path.join(path_to_save, 'test' + '_' + str(file_index) + '_' + str(each_key) + '.png'), image_to_save[each_key])
    print('length of action and labels: ',
          inputs['context_actions'][sample_index].shape, inputs['trajectory_label'][sample_index].shape)
    print('debug planner images aug9-1413 saved to: ', path_to_save, file_index)