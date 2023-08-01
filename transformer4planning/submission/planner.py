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
from transformer4planning.utils import ModelArguments
from omegaconf import DictConfig
import time

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
            target_velocity=8,
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
        if isinstance(model, dict) or isinstance(model, DictConfig):
            self.multi_city = True
        else:
            self.multi_city = False
        self.model = model
        self.warm_up = False       
        

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        self.initialization = initialization
        self.goal = initialization.mission_goal
        self.route_roadblock_ids = initialization.route_roadblock_ids
        self.map_api = initialization.map_api
        self.idm_planner.initialize(initialization)
        self.road_dic = get_road_dict(self.map_api, ego_pose_center=Point2D(0, 0))
        print("map_name", self.map_api.map_name, self.multi_city)
        if self.multi_city: 
            if "boston" in self.map_api.map_name:#us-ma-boston
                print("boston")
                self.model = self.model["boston"]
            elif "pittsburgh" in self.map_api.map_name:#us-pa-pittsburgh-hazelwood
                print("pittsburgh")
                self.model = self.model["pittsburgh"]
            elif "sg" in self.map_api.map_name:#sg-one-north
                print("singapore")
                self.model = self.model["singapore"]
            elif "vegas" in self.map_api.map_name:#us-nv-las-vegas-strip 
                print("vegas")
                self.model = self.model["vegas"]
            else:
                raise ValueError("Map is not invalid")
    
    def warmup(self):
        if torch.cuda.is_available():
            self.model.to("cuda")
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

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        global count
        use_backup_planner = self.use_backup_planner
        count += 1
        start=time.time()
        print("count: ", count, "cuda:", torch.cuda.is_available())
        traffic_data = current_input.traffic_light_data
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
            ego_trajectory, agents, statics, self.road_dic, traffic_data, ego_shape, max_dis=300, map=self.map_api.map_name)
        time_after_input = time.time() - start
        print("time after ratser build", time_after_input)
        if time_after_input > 0.65:
            use_backup_planner = False
            print("forbid idm planner")
        pred_length = int(160 / self.model.model_args.future_sample_interval)
        if torch.cuda.is_available():
            output = self.model.generate(
                                context_actions=torch.tensor(context_action).unsqueeze(0).to('cuda'), \
                                high_res_raster=torch.tensor(high_res_raster).unsqueeze(0).to('cuda'), \
                                low_res_raster=torch.tensor(low_res_raster).unsqueeze(0).to('cuda'),
                                trajectory_label=torch.zeros((1, pred_length, 4)).to('cuda'))
            # pred_traj [160, 4]
            try:
                pred_traj = output[:, -pred_length:].squeeze(0).detach().cpu()
            except:
                pred_traj = output.logits.squeeze(0).detach().cpu()
        else:
            output = self.model.generate(
                                context_actions=torch.tensor(context_action).unsqueeze(0), \
                                high_res_raster=torch.tensor(high_res_raster).unsqueeze(0), \
                                low_res_raster=torch.tensor(low_res_raster).unsqueeze(0),
                                trajectory_label=torch.zeros((1, pred_length, 4)))
            try:
                pred_traj = output[:, -pred_length:].squeeze(0).detach().cpu()
            except:
                pred_traj = output.logits.squeeze(0).detach().cpu()
        # insert_pred_traj = np.zeros((2 * len(pred_traj), 4))
        # for i in range(len(pred_traj)):
        #     if i == 0:
        #         insert_pred_traj[2 * i, :2] = pred_traj[i, :] / 2
        #     else:
        #         insert_pred_traj[2 * i, :2] = (pred_traj[i, :] + pred_traj[i - 1 , :]) / 2
        #     insert_pred_traj[2 * i + 1, :2] = pred_traj[i, :]
        # pred_traj = insert_pred_traj
        pred_traj = torch.cat([pred_traj, torch.zeros([pred_length, 1]), torch.ones([pred_length, 1])*ego_states[-1].rear_axle.heading], dim=1).numpy()
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
            new_x = pred_traj[i, 0] * cos_ + pred_traj[i, 1] * sin_ + ego_trajectory[-1][0]
            new_y = pred_traj[i, 1] * cos_ - pred_traj[i, 0] * sin_ + ego_trajectory[-1][1]
            absolute_traj[i, 0] = new_x
            absolute_traj[i, 1] = new_y
            absolute_traj[i, 2] = 0
            absolute_traj[i, -1] = ego_trajectory[-1][-1]
        
        # compare pred trajectory and idm trajectory
        idm_threshold = 3
        traffic_stop_threshold = 5
        agent_stop_threshold = 3
        ## judge if route is valid:
        all_nearby_map_instances = self.map_api.get_proximal_map_objects(
                    Point2D(ego_trajectory[-1, 0], ego_trajectory[-1, 1]),
                    0.001, [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
        all_nearby_map_instances_ids = []
        for each_type in all_nearby_map_instances:
            for each_ins in all_nearby_map_instances[each_type]:
                all_nearby_map_instances_ids.append(each_ins.id)
        for each in all_nearby_map_instances_ids:
            if each in self.route_roadblock_ids or int(each) in self.route_roadblock_ids:
                in_route = True
                break
            else:
                in_route = False
                print("route wrong, forbid IDM")
        use_backup_planner = use_backup_planner and in_route
        if use_backup_planner:
            # compute idm trajectory and scenario flag
            idm_trajectory, flag, relative_distance = self.idm_planner.compute_planner_trajectory(current_input)
            if flag == "redlight" and relative_distance < traffic_stop_threshold:
                return idm_trajectory
            
            out_pts = 0
            sample_frames = list(range(10)) + list(range(10, absolute_traj.shape[0], 5)) 
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
                return idm_trajectory
            
        # generating yaw angle from points
        relative_traj = pred_traj.copy()
        yaw_change_upper_threshold = 0.1
        prev_delta_heading = 0
        prev_pt = np.zeros(2)
        scrolling_frame_idx = 0
        for i in range(pred_traj.shape[0]):
            if i <= scrolling_frame_idx and i != 0:
                relative_traj[i, -1] = prev_delta_heading
            else:
                # scrolling forward
                relative_traj[i, -1] = prev_delta_heading
                for j in range(pred_traj.shape[0] - i):
                    dist = euclidean_distance(prev_pt, relative_traj[i + j, :2])
                    delta_heading = get_angle_of_a_line(prev_pt, relative_traj[i + j, :2])
                    if dist > low_threshold and delta_heading - prev_delta_heading < yaw_change_upper_threshold:
                        prev_pt = relative_traj[i + j, :2]
                        prev_delta_heading = delta_heading
                        scrolling_frame_idx = i + j
                        relative_traj[i, -1] = delta_heading
                        break
        absolute_traj[:, -1] += relative_traj[:, -1]
        # change relative poses to absolute states
        from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_absolute_agent_states_from_numpy_poses,_get_fixed_timesteps
        if relative_traj.shape[1] == 4:
            new = np.zeros((relative_traj.shape[0], 3))
            new[:, :2] = relative_traj[:, :2]
            new[:, -1] = relative_traj[:, -1]
            relative_traj = new

        planned_time_points = _get_fixed_timesteps(ego_states[-1], 8, 0.1)
        states = _get_absolute_agent_states_from_numpy_poses(poses=relative_traj,
                                                             ego_history=ego_states,
                                                             timesteps=planned_time_points)
        states.insert(0, ego_states[-1])
        # post process   
        if use_backup_planner and flag == "leadagent" and relative_distance < agent_stop_threshold:
            states[1:5] = idm_trajectory.get_sampled_trajectory()[1:5]

        trajectory = InterpolatedTrajectory(states)
        print("time consumed", time.time()-start)
        return trajectory
        
    #TODO: add traffic channel
    def compute_raster_input(self, ego_trajectory, agents_seq, statics_seq, road_dic, traffic_data=None, ego_shape=None, max_dis=300, context_frequency=None, map="sg-one-north"):
        """
        the first dimension is the sequence length, each timestep include n-items.
        agent_seq and statics_seq are both agents in raster definition
        """
        ego_pose = ego_trajectory[-1]  # (x, y, yaw) in current timestamp
        cos_, sin_ = math.cos(-ego_pose[2]), math.sin(-ego_pose[2])

        ## hyper initilization
        total_road_types = 20
        total_agent_types = 8
        total_traffic_types = 4
        high_res_raster_scale = 4
        low_res_raster_scale = 0.77
        context_length = 2 * context_frequency if context_frequency is not None else int(10/self.model.model_args.past_sample_interval) * 2
        total_raster_channels = 1 + total_road_types + total_traffic_types + total_agent_types * context_length
        rasters_high_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_low_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        raster_shape = np.array([224, 224])
        high_res_scale = 4
        low_res_scale = 0.77
        y_inverse = -1 if map == "sg-one-north" else 1
        # downsampling from ego_trajectory, agent_seq and statics_seq in 4 hz case
        
        agents_seq = agents_seq[-20::self.model.model_args.past_sample_interval]
        statics_seq = statics_seq[-20::self.model.model_args.past_sample_interval]
        ego_trajectory = ego_trajectory[-21::self.model.model_args.past_sample_interval]
        traffic_seq = traffic_data[-20::self.model.model_args.past_sample_interval]
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
        rotation_matrix = np.array([[cos_, -sin_], [sin_, cos_]])
        route_ids = self.route_roadblock_ids
        # if self.multi_city:
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
            simplified_xyz[:, 1] *= y_inverse
            high_res_route = (simplified_xyz * high_res_scale + raster_shape[0] // 2).astype('int32')
            low_res_route = (simplified_xyz * low_res_scale + raster_shape[0] // 2).astype('int32')

            cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
            cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))

        for i, key in enumerate(road_dic):
            xyz = road_dic[key]["xyz"].copy()
            road_type = int(road_dic[key]['type'])
            xyz[:, :2] -= ego_pose[:2]
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
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 1] *= y_inverse
            high_res_road = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_road = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            if road_type in [5, 17, 18, 19]:
                cv2.fillPoly(rasters_high_res_channels[road_type + 1], np.int32([high_res_road[:, :2]]), (255, 255, 255))
                cv2.fillPoly(rasters_low_res_channels[road_type + 1], np.int32([low_res_road[:, :2]]), (255, 255, 255))
            else:
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[road_type + 1], tuple(high_res_road[j, :2]),
                            tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[road_type + 1], tuple(low_res_road[j, :2]),
                            tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)
        
        # traffic light
        traffic_state = int(traffic_seq[-1].status)
        lane_id = traffic_seq[-1].lane_connector_id
        xyz = road_dic[lane_id] ["xyz"].copy()
        xyz[:, :2] -= ego_pose[:2]
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
            simplified_xyz[:, 1] *= y_inverse
            high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            # traffic state order is GREEN, RED, YELLOW, UNKNOWN
            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[1 + total_road_types + traffic_state],
                        tuple(high_res_traffic[j, :2]),
                        tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[1 + total_road_types + traffic_state],
                        tuple(low_res_traffic[j, :2]),
                        tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)

        cos_, sin_ = math.cos(-ego_pose[2]), math.sin(-ego_pose[2])

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
                cv2.drawContours(rasters_high_res_channels[1 + total_road_types + total_traffic_types + agent_type * context_length + i],
                                 [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = (low_res_raster_scale * rect_pts).astype(np.int64) + 112
                cv2.drawContours(rasters_low_res_channels[1 + total_road_types + total_traffic_types + agent_type * context_length + i],
                                 [rect_pts_low_res], -1, (255, 255, 255), -1)

        recentered_ego_trajectory = ego_trajectory - ego_pose
        for i, pose in enumerate(copy.deepcopy(recentered_ego_trajectory[:-1])):
            agent_type = 7  # type EGO is 7
            # pose -= ego_pose
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]),
                                            w=ego_shape[0], l=ego_shape[1],
                                            direction=-pose[2])
            rect_pts = np.int0(rect_pts)
            rect_pts_high_res = int(high_res_raster_scale) * rect_pts + 112
            cv2.drawContours(rasters_high_res_channels[1 + total_road_types + total_traffic_types + agent_type * context_length + i], [rect_pts_high_res],
                             -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = int(low_res_raster_scale) * rect_pts + 112
            cv2.drawContours(rasters_low_res_channels[1 + total_road_types + total_traffic_types + agent_type * context_length + i], [rect_pts_low_res],
                             -1, (255, 255, 255), -1)

        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

        # context_actions computation
        context_actions = list()
        # ego_poses = ego_trajectory - ego_pose
        rotated_poses = np.array([recentered_ego_trajectory[:, 0] * cos_ - recentered_ego_trajectory[:, 1] * sin_,
                                  recentered_ego_trajectory[:, 0] * sin_ + recentered_ego_trajectory[:, 1] * cos_,
                                  np.zeros(recentered_ego_trajectory.shape[0]), recentered_ego_trajectory[:, -1]]).transpose((1, 0))
        for i in range(0, len(rotated_poses) - 1):
            # if not self.multi_city:
            #     action = rotated_poses[i+1] - rotated_poses[i] # old model, #it later@5.18
            # else:
            action = rotated_poses[i]
            context_actions.append(action)

        return rasters_high_res, rasters_low_res, np.array(context_actions, dtype=np.float32)
    
    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """
        Generate a report containing runtime stats from the planner.
        By default, returns a report containing the time-series of compute_trajectory runtimes.
        :param clear_stats: whether or not to clear stored stats after creating report.
        :return: report containing planner runtime stats.
        """
        report = PlannerReport(compute_trajectory_runtimes=self._compute_trajectory_runtimes)
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
