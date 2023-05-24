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
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from transformer4planning.models.model import build_models
from transformer4planning.utils import ModelArguments

count = 0

def generate_contour_pts(center_pt, w, l, direction):
    pt1 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]-l/2), direction, tuple=True)
    pt2 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]-l/2), direction, tuple=True)
    pt3 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]+l/2), direction, tuple=True)
    pt4 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]+l/2), direction, tuple=True)
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

def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)


class ControlTFPlanner(AbstractPlanner):
    """
    Planner with Pretrained Control Transformer
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 use_backup_planner = True,
                 model = None,
                 planning_interval = 1,
                 steering_angle: float = 0.0,
                 controller=None,
                 **kwargs):
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.horizon_seconds_time = horizon_seconds
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.sampling_time_time = sampling_time
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.use_backup_planner = use_backup_planner
        self.planning_interval = planning_interval
        self.idm_planner = IDMPlanner(
            target_velocity=10,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=3.0,
            planned_trajectory_samples=16,
            planned_trajectory_sample_interval=0.5,
            occupancy_map_radius=40
        )
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)
        # model initialization and configuration
        assert model is not None
        self.model = model
        self.frequency = 5
        if "PerfectTrackingController" in controller:
            self.mode = "openloop"
        else:
            self.mode = "closedloop"         
        

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        self.initialization = initialization
        self.goal = initialization.mission_goal
        self.route_roadblock_ids = initialization.route_roadblock_ids
        self.map_api = initialization.map_api
        self.idm_planner.initialize(initialization)
        self.road_dic = get_road_dict(self.map_api, ego_pose_center=Point2D(0, 0))
        if torch.cuda.is_available():
            warmup = self.model(intended_maneuver_vector=torch.zeros((1), dtype=torch.int32).to('cuda'), \
                                current_maneuver_vector=torch.zeros((1, 12), dtype=torch.float32).to('cuda'), \
                                context_actions=torch.zeros((1, 10, 4), device='cuda'), \
                                high_res_raster=torch.zeros((1, 224, 224, 109), device='cuda'), \
                                low_res_raster=torch.zeros((1, 224, 224, 109), device='cuda'), \
                                trajectory_label=torch.zeros((1, 160, 4)).to('cuda'))

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        import time
        global count
        use_backup_planner = self.use_backup_planner
        count += 1
        start=time.time()
        print("count: ", count, "cuda:", torch.cuda.is_available(), "mode:", self.mode)
        history = current_input.history
        ego_states = history.ego_state_buffer  # a list of ego trajectory
        context_length = len(ego_states)
        # trajectory as format of [(x, y, yaw)]
        ego_trajectory = np.array([(ego_states[i].rear_axle.x, \
                                    ego_states[i].rear_axle.y, \
                                    ego_states[i].rear_axle.heading) for i in range(context_length)])
        ego_shape = np.array([ego_states[-1].waypoint.oriented_box.width, \
                              ego_states[-1].waypoint.oriented_box.length])
        agents = [history.observation_buffer[i].tracked_objects.get_agents() for i in range(context_length)]
        statics = [history.observation_buffer[i].tracked_objects.get_static_objects() for i in range(context_length)]
        high_res_raster, low_res_raster, context_action = self.compute_raster_input(
            ego_trajectory, agents, statics, self.road_dic, ego_shape, max_dis=500, context_frequency=self.frequency)
        time_after_input = time.time() - start
        print("time after ratser build", time_after_input)
        if time_after_input > 0.65:
            use_backup_planner = False
            print("forbid idm planner")
        if torch.cuda.is_available():
            output = self.model(intended_maneuver_vector=torch.zeros((1), dtype=torch.int32).to('cuda'), \
                                current_maneuver_vector=torch.zeros((1, 12), dtype=torch.float32).to('cuda'), \
                                context_actions=torch.tensor(context_action).unsqueeze(0).to('cuda'), \
                                high_res_raster=torch.tensor(high_res_raster).unsqueeze(0).to('cuda'), \
                                low_res_raster=torch.tensor(low_res_raster).unsqueeze(0).to('cuda'),
                                trajectory_label=torch.zeros((1, 160, 4)).to('cuda'))
            # pred_traj [80, 4]
            try:
                pred_traj = output[-1][-1].squeeze(0).detach().cpu().numpy()
            except:
                pred_traj = output.logits.squeeze(0).detach().cpu().numpy()
        else:
            output = self.model(intended_maneuver_vector=torch.zeros((1), dtype=torch.int32), \
                                current_maneuver_vector=torch.zeros((1, 12), dtype=torch.float32), \
                                context_actions=torch.tensor(context_action).unsqueeze(0), \
                                high_res_raster=torch.tensor(high_res_raster).unsqueeze(0), \
                                low_res_raster=torch.tensor(low_res_raster).unsqueeze(0),
                                trajectory_label=torch.zeros((1, 160, 4)))
            try:
                pred_traj = output[-1][-1].squeeze(0).detach().cpu().numpy()
            except:
                pred_traj = output.logits.squeeze(0).detach().cpu().numpy()
        print("time after gpt", time.time() - start)
        # # post-processing
        low_filter = True
        low_threshold = 0.01
        if low_filter:
            filtered_traj = np.zeros_like(pred_traj)
            last_pose = None
            for idx, each_pose in enumerate(pred_traj):
                if last_pose is None:
                    dx, dy = each_pose[:2] - np.zeros(2)
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

        absolute_traj = np.zeros_like(pred_traj)
        cos_, sin_ = math.cos(-ego_trajectory[-1][2]), math.sin(-ego_trajectory[-1][2])
        for i in range(pred_traj.shape[0]):
            new_x = pred_traj[i, 0].copy() * cos_ + pred_traj[i, 1].copy() * sin_ + ego_trajectory[-1][0]
            new_y = pred_traj[i, 1].copy() * cos_ - pred_traj[i, 0].copy() * sin_ + ego_trajectory[-1][1]
            absolute_traj[i, 0] = new_x
            absolute_traj[i, 1] = new_y
            absolute_traj[i, 2] = pred_traj[i, -1]
        
        idm_threshold = 1
        traffic_stop_threshold = 5
        agent_stop_threshold = 3
        if use_backup_planner:
            # compute idm trajectory and scenario flag
            trajectory, flag, relative_distance = self.idm_planner.compute_planner_trajectory(current_input)
            
            if flag == "redlight" and relative_distance < traffic_stop_threshold and self.mode == "closedloop":
                return trajectory
            elif flag == "leadagent" and relative_distance < agent_stop_threshold and self.mode == "closedloop": 
                return trajectory
            # check if out of boundary
            out_pts = 0
            sample_frames = list(range(10)) + list(range(10, absolute_traj.shape[0], 5)) 
            # TODO: open loop & close loop
            for i in sample_frames:
                all_nearby_map_instances = self.map_api.get_proximal_map_objects(
                    Point2D(absolute_traj[i, 0], absolute_traj[i, 1]),
                    0.1, [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
                all_nearby_map_instances_ids = []
                for each_type in all_nearby_map_instances:
                    for each_ins in all_nearby_map_instances[each_type]:
                        all_nearby_map_instances_ids.append(each_ins.id)
                any_in = False
                for each in all_nearby_map_instances_ids:
                    if each in self.route_roadblock_ids or int(each) in self.route_roadblock_ids:
                        any_in = True
                        break
                if not any_in:
                    out_pts += 1
                if out_pts > idm_threshold:
                    break
            
            if out_pts > idm_threshold:
                out_of_route = True
                print('OUT OF ROUTE, Use IDM Planner to correct trajectory')
                
                print(time.time() - start)
                return trajectory

        relative_traj = pred_traj.copy()
        # generating yaw angle from points
        yaw_change_upper_threshold = 0.1
        prev_delta_heading = 0
        prev_pt = np.zeros(2)
        scrolling_frame_idx = 0
        for i in range(pred_traj.shape[0]):
            if i <= scrolling_frame_idx and i != 0:
                relative_traj[i, -1] = prev_delta_heading
            else:
                # scrolling forward
                for j in range(pred_traj.shape[0] - i):
                    dist = euclidean_distance(prev_pt, relative_traj[i + j, :2])
                    delta_heading = get_angle_of_a_line(prev_pt, relative_traj[i + j, :2])
                    if dist > low_threshold and delta_heading - prev_delta_heading < yaw_change_upper_threshold:
                        prev_pt = relative_traj[i + j, :2]
                        prev_delta_heading = delta_heading
                        scrolling_frame_idx = i + j
                        relative_traj[i, -1] = delta_heading
                        break
        # change relative poses to absolute states
        from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_absolute_agent_states_from_numpy_poses,_get_fixed_timesteps
        if relative_traj.shape[1] == 4:
            new = np.zeros((relative_traj.shape[0], 3))
            new[:, :2] = relative_traj[:, :2]
            new[:, -1] = relative_traj[:, -1]
            relative_traj = new

        # print('test: ', self.planning_interval, self.horizon_seconds_time)
        planned_time_points = []
        for i in range(0, relative_traj.shape[0]):
            planned_time_points.append(TimePoint(ego_states[-1].time_point.time_us + 1e5 * i ).time_s)

        states = _get_absolute_agent_states_from_numpy_poses(poses=relative_traj,
                                                             ego_history=ego_states,
                                                             timesteps=planned_time_points)
        trajectory = InterpolatedTrajectory(states)
        print("time consumed", time.time()-start)
        return trajectory

        # DEPERATED CODES BELOW !!
        #
        # # build output
        # ego_state = history.ego_states[-1]
        # state = EgoState(
        #     car_footprint=ego_state.car_footprint,
        #     dynamic_car_state=DynamicCarState.build_from_rear_axle(
        #         ego_state.car_footprint.rear_axle_to_center_dist,
        #         ego_state.dynamic_car_state.rear_axle_velocity_2d,
        #         self.acceleration,
        #     ),
        #     tire_steering_angle=self.steering_angle,
        #     is_in_auto_mode=True,
        #     time_point=ego_state.time_point
        # )
        # trajectory: List[EgoState] = [state]
        # for i in range(0, next_world_coor_points.shape[0]):
        #     new_time_point = TimePoint(state.time_point.time_us + 1e5)
        #     # velocity and acceleration computation
        #     # self.last_velocity = self.curret_velocity
        #     # if i == 0:
        #     #     self.current_velocity = next_world_coor_points[0, :2] - ego_trajectory[-1][:2]
        #     # else:
        #     #     self.current_velocity = next_world_coor_points[i, :2] - next_world_coor_points[i-1, :2]
        #     # self.acceleration = (self.curret_velocity - self.last_velocity)/0.1
        #     state = EgoState.build_from_rear_axle(
        #         rear_axle_pose=StateSE2(next_world_coor_points[i, 0],
        #                         next_world_coor_points[i, 1],
        #                         #ego_trajectory[-1, -1]
        #                         next_world_coor_points[i][-1]
        #                         ),
        #         rear_axle_velocity_2d=StateVector2D(0, 0),
        #         rear_axle_acceleration_2d=StateVector2D(0, 0),
        #         tire_steering_angle=state.tire_steering_angle,
        #         time_point=new_time_point,
        #         vehicle_parameters=state.car_footprint.vehicle_parameters,
        #         is_in_auto_mode=True,
        #         angular_vel=state.dynamic_car_state.angular_velocity,
        #         angular_accel=state.dynamic_car_state.angular_acceleration
        #     )
        #     state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
        #     state._time_point = new_time_point
        #     trajectory.append(state)
        #
        # return InterpolatedTrajectory(trajectory)

    def compute_raster_input(self, ego_trajectory, agents_seq, statics_seq, road_dic, ego_shape=None, max_dis=500, context_frequency=5):
        """
        the first dimension is the sequence length, each timestep include n-items.
        agent_seq and statics_seq are both agents in raster definition
        """
        ego_pose = ego_trajectory[-1]  # (x, y, yaw) in current timestamp
        cos_, sin_ = math.cos(-ego_pose[2]), math.sin(-ego_pose[2])

        ## hyper initilization
        total_road_types = 20
        total_agent_types = 8
        high_res_raster_scale = 4
        low_res_raster_scale = 0.77
        context_length = 2 * context_frequency + 1
        total_raster_channels = 1 + total_road_types + total_agent_types * context_length
        rasters_high_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_low_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        # downsampling from ego_trajectory, agent_seq and statics_seq in 4 hz case
        if context_frequency == 4:
            downsample_indexs = [2, 5, 7, 10, 12, 15, 18, 20, 22]
            downsample_agents_seq = list()
            downsample_statics_seq = list()
            for i in downsample_indexs:
                downsample_agents_seq.append(agents_seq[i].copy())
                downsample_statics_seq.append(statics_seq[i].copy())
            del agents_seq, statics_seq
            agents_seq = downsample_agents_seq
            statics_seq = downsample_statics_seq
            new_ego_trajectory = list()
            for idx in range(0, len(ego_trajectory)-1):
                new_ego_trajectory.append(ego_trajectory[idx])
                new_ego_trajectory.append((ego_trajectory[idx] + ego_trajectory[idx + 1]) / 2)
            new_ego_trajectory.append(ego_pose)
            ego_trajectory = np.array(new_ego_trajectory)[-41::][::5].copy()
        elif context_frequency == 5:
            agents_seq = agents_seq[2::2]
            statics_seq = statics_seq[2::2]
            ego_trajectory = ego_trajectory[-21::2]
        # goal channel
        ## goal point
        if self.goal is None:
            relative_goal = ego_pose
        else:
            relative_goal = np.array([self.goal.x, self.goal.y, self.goal.heading]) - ego_pose
        rotated_goal_pose = [relative_goal[0] * cos_ - relative_goal[1] * sin_,
                             relative_goal[0] * sin_ + relative_goal[1] * cos_,
                             relative_goal[2]]
        goal_contour = generate_contour_pts((rotated_goal_pose[1], rotated_goal_pose[0]), w=ego_shape[0],
                                            l=ego_shape[1],
                                            direction=-rotated_goal_pose[2])
        goal_contour = np.array(goal_contour, dtype=np.int32)
        goal_contour_high_res = int(high_res_raster_scale) * goal_contour + 112
        cv2.drawContours(rasters_high_res_channels[0], [goal_contour_high_res], -1, (255, 255, 255), -1)
        goal_contour_low_res = (low_res_raster_scale * goal_contour).astype(np.int64) + 112
        cv2.drawContours(rasters_low_res_channels[0], [goal_contour_low_res], -1, (255, 255, 255), -1)
        ## goal route
        cos_, sin_ = math.cos(-ego_pose[2] - math.pi / 2), math.sin(-ego_pose[2] - math.pi / 2)
        route_ids = self.route_roadblock_ids
        routes = [road_dic[int(route_id)] for route_id in route_ids]
        for route in routes:
            xyz = route["xyz"].copy()
            xyz[:, :2] -= ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis):
                continue
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            high_res_route = simplified_xyz * high_res_raster_scale
            low_res_route = simplified_xyz * low_res_raster_scale
            high_res_route = high_res_route.astype('int32')
            low_res_route = low_res_route.astype('int32')
            high_res_route += 112
            low_res_route += 112
            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[0], tuple(high_res_route[j, :2]),
                        tuple(high_res_route[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[0], tuple(low_res_route[j, :2]),
                        tuple(low_res_route[j + 1, :2]), (255, 255, 255), 2)
        # road element computation
        cos_, sin_ = math.cos(-ego_pose[2] - math.pi / 2), math.sin(-ego_pose[2] - math.pi / 2)
        # road_key_to_del = list()
        for i, key in enumerate(road_dic):
            xyz = road_dic[key]["xyz"].copy()
            road_type = int(road_dic[key]['type'])
            xyz[:, :2] -= ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                    abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis):
                # if (abs(xyz[0, 0]) > 2 * max_dis and abs(xyz[-1, 0]) > 2 * max_dis) or (
                #     abs(xyz[0, 1]) > 2 * max_dis and abs(xyz[-1, 1]) > 2 * max_dis):
                #     road_key_to_del.append(key)               
                continue
            # simplify road vector, can simplify about half of all the points
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, \
                                                        simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            high_res_road = simplified_xyz * high_res_raster_scale
            low_res_road = simplified_xyz * low_res_raster_scale
            high_res_road = high_res_road.astype('int32') + 112
            low_res_road = low_res_road.astype('int32') + 112

            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[road_type + 1], tuple(high_res_road[j, :2]),
                         tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[road_type + 1], tuple(low_res_road[j, :2]),
                         tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)
        
        # for key in road_key_to_del:
        #     del road_dic[key]
        # agent element computation
        ## statics include CZONE_SIGN,BARRIER,TRAFFIC_CONE,GENERIC_OBJECT,
        cos_, sin_ = math.cos(-ego_pose[2]), math.sin(-ego_pose[2])
        # for i, statics in enumerate(statics_seq):
        #     for j, static in enumerate(statics):
        #         static_type = static.tracked_object_type.value
        #         pose = np.array([static.box.center.point.x, static.box.center.point.y, static.box.center.heading])
        #         pose -= ego_pose
        #         if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
        #             continue
        #         rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
        #                         pose[0] * sin_ + pose[1] * cos_]
        #         shape = np.array([static.box.width, static.box.length])
        #         rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
        #                                         direction=-pose[2])
        #         rect_pts = np.array(rect_pts, dtype=np.int32)
        #         rect_pts_high_res = int(high_res_raster_scale) * rect_pts + 112
        #         cv2.drawContours(rasters_high_res_channels[1 + total_road_types + static_type * context_length + i],
        #                          [rect_pts_high_res], -1, (255, 255, 255), -1)
        #         # draw on low resolution
        #         rect_pts_low_res = (low_res_raster_scale * rect_pts).astype(np.int64) + 112
        #         cv2.drawContours(rasters_low_res_channels[1 + total_road_types + static_type * context_length + i],
        #                          [rect_pts_low_res], -1, (255, 255, 255), -1)

        ## agent includes VEHICLE, PEDESTRIAN, BICYCLE, EGO(except)
        for i, agents in enumerate(agents_seq):
            for j, agent in enumerate(agents):
                agent_type = agent.tracked_object_type.value
                pose = np.array([agent.box.center.point.x, agent.box.center.point.y, agent.box.center.heading])
                pose -= ego_pose
                if (abs(pose[0]) > max_dis or abs(pose[1]) > max_dis):
                    continue
                rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                                pose[0] * sin_ + pose[1] * cos_]
                shape = np.array([agent.box.width, agent.box.length])
                rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                                direction=-pose[2])
                rect_pts = np.array(rect_pts, dtype=np.int32)
                rect_pts_high_res = int(high_res_raster_scale) * rect_pts + 112
                cv2.drawContours(rasters_high_res_channels[1 + total_road_types + agent_type * context_length + i],
                                 [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = (low_res_raster_scale * rect_pts).astype(np.int64) + 112
                cv2.drawContours(rasters_low_res_channels[1 + total_road_types + agent_type * context_length + i],
                                 [rect_pts_low_res], -1, (255, 255, 255), -1)

        for i, pose in enumerate(copy.deepcopy(ego_trajectory)):
            agent_type = 7  # type EGO is 7
            pose -= ego_pose
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]),
                                            w=ego_shape[0], l=ego_shape[1],
                                            direction=-pose[2])
            rect_pts = np.int0(rect_pts)
            rect_pts_high_res = int(high_res_raster_scale) * rect_pts + 112
            cv2.drawContours(rasters_high_res_channels[1 + total_road_types + agent_type * context_length + i], [rect_pts_high_res],
                             -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = int(low_res_raster_scale) * rect_pts + 112
            cv2.drawContours(rasters_low_res_channels[1 + total_road_types + agent_type * context_length + i], [rect_pts_low_res],
                             -1, (255, 255, 255), -1)

        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

        # context_actions computation
        context_actions = list()
        ego_poses = ego_trajectory - ego_pose
        rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                                  ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                                  np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
        for i in range(len(rotated_poses) - 1):
            # action = rotated_poses[i+1] - rotated_poses[i] # old model, #it later@5.18
            action = rotated_poses[i]
            context_actions.append(action)

        return rasters_high_res, rasters_low_res, np.array(context_actions, dtype=np.float32)
    
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
        #road_dic = get_road_dict(planner.map_api, Point2D(ego_trajectory[-1][0], ego_trajectory[-1][1]))
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
    # planner.compute_planner_trajectory(input)
