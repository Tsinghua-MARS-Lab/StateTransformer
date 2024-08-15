import logging
import math

import numpy as np
from shapely.geometry import Point, LineString
from nuplan_simulation.planner_utils import *

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

# packages for STR model
from transformer4planning.models.backbone.str_base import build_model_from_path
from nuplan.common.actor_state.state_representation import Point2D
from transformer4planning.utils.nuplan_utils import get_angle_of_a_line
from nuplan_simulation.route_corrections.route_utils import route_roadblock_correction

import cv2
import torch

class Planner(AbstractPlanner):
    def __init__(self, model_path, device, all_road_dic={}, scenarios=None):
        self._future_horizon = T  # [s]
        self._step_interval = DT  # [s]
        self._N_points = int(T / DT)
        self._model_path = model_path
        self.road_dic = None
        self._device = device
        self.all_road_dic = all_road_dic
        self.scenarios = scenarios

        self.iteration = 0

    def name(self) -> str:
        return "STR-Planner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization, **kwargs):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        model = kwargs.get('model', None)
        if model is None:
            self._initialize_model()
        else:
            self._model = model
        # self._trajectory_planner = TreePlanner(self._device, self._encoder, self._decoder)

    def _initialize_model(self):
        self._model = build_model_from_path(self._model_path)
        self._model.to(self._device)
        self._model.eval()
        print('model built on ', self._model.device)

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def compute_planner_trajectory(self, current_input: PlannerInput):
        assert False
        # Extract iteration, history, and traffic light
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state

        y_inverse = -1 if self._map_api.map_name == "sg-one-north" else 1
        # Construct input features
        start_time = time.perf_counter()
        # parse PlannerInput to model sample. costs about 0.5s per iteration
        sample = self.inputs_to_model_sample(history, traffic_light_data)

        if self.gt_trajs is not None:
            relative_traj = np.zeros((1, self._N_points, 4))
            # normalize gt trajectory
            trajectory_label_15s = self.gt_trajs.copy()
            origin_ego_pose = sample['ego_pose']
            sin_, cos_ = np.sin(-origin_ego_pose[-1]), np.cos(-origin_ego_pose[-1])
            trajectory_label_15s -= origin_ego_pose
            traj_x = trajectory_label_15s[:, 0].copy()
            traj_y = trajectory_label_15s[:, 1].copy()
            trajectory_label_15s[:, 0] = traj_x * cos_ - traj_y * sin_
            trajectory_label_15s[:, 1] = traj_x * sin_ + traj_y * cos_
            trajectory_label_15s[:, 1] *= y_inverse
            relative_traj[0, :] = trajectory_label_15s[iteration, :]
        else:
            # features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)
            with torch.no_grad():
                # print("start generating trajectory with gpu?", self.use_gpu)
                if self._model.device != 'cpu':
                    device = self._model.device
                    prediction_generation = self._model.generate(
                        context_actions=torch.tensor(sample['context_actions'][np.newaxis, ...]).to(device),
                        high_res_raster=torch.tensor(sample['high_res_raster'][np.newaxis, ...]).to(device),
                        low_res_raster=torch.tensor(sample['low_res_raster'][np.newaxis, ...]).to(device),
                        trajectory_label=torch.zeros((1, self._N_points, 4)).to(device),
                        map_api=self._map_api,
                        route_ids=self._route_roadblock_ids,
                        ego_pose=sample['ego_pose'],
                        road_dic=self.road_dic,
                        map=self._map_api.map_name,
                        # idm_reference_global=idm_reference_trajectory._trajectory
                    )
                    pred_traj = prediction_generation['traj_logits'].detach.cpu()
                    if 'key_points_logits' in prediction_generation:
                        pred_key_points = prediction_generation['key_points_logits'].detach.cpu()
                    else:
                        pred_key_points = None
                else:
                    prediction_generation = self._model.generate(
                        context_actions=torch.tensor(sample['context_actions'][np.newaxis, ...]),
                        high_res_raster=torch.tensor(sample['high_res_raster'][np.newaxis, ...]),
                        low_res_raster=torch.tensor(sample['low_res_raster'][np.newaxis, ...]),
                        trajectory_label=torch.zeros((1, self._N_points, 4)),
                        map_api=self._map_api,
                        route_ids=self._route_roadblock_ids,
                        ego_pose=sample['ego_pose'],
                        road_dic=self.road_dic,
                        map_name=self._map_api.map_name
                        # idm_reference_global=idm_reference_trajectory._trajectory
                    )
                    pred_traj = prediction_generation['traj_logits'].numpy()[0]  # (80, 2) or (80, 4)
                    if 'key_points_logits' in prediction_generation:
                        pred_key_points = prediction_generation['key_points_logits'].numpy()[0]
                    else:
                        pred_key_points = None
                if self._model.config.skip_yaw_norm:
                    oriented_yaw = sample['ego_pose'][-1]
                    # rotate prediction based on oriented yaw
                    sin_, cos_ = np.sin(-oriented_yaw), np.cos(-oriented_yaw)
                    rotated_pred_traj = pred_traj.copy()
                    rotated_pred_traj[:, 0] = pred_traj[:, 0] * cos_ - pred_traj[:, 1] * sin_
                    rotated_pred_traj[:, 1] = pred_traj[:, 0] * sin_ + pred_traj[:, 1] * cos_
                    pred_traj = rotated_pred_traj

            # post-processing
            low_filter = False
            low_threshold = 0.1
            if low_filter:
                filtered_traj = np.zeros_like(pred_traj)
                last_pose = None
                for idx, each_pose in enumerate(pred_traj):
                    if last_pose is None:
                        dx, dy = each_pose[:2]
                        dx = 0 if dx < low_threshold else dx
                        dy = 0 if dy < low_threshold else dy
                        last_pose = filtered_traj[idx, :] = [dx, dy, each_pose[2]]
                        continue
                    dx, dy = each_pose[:2] - pred_traj[idx - 1, :2]
                    dx = 0 if dx < low_threshold else dx
                    dy = 0 if dy < low_threshold else dy
                    last_pose = filtered_traj[idx, :] = [last_pose[0] + dx, last_pose[1] + dy, each_pose[2]]
                pred_traj = filtered_traj
            relative_traj = pred_traj.copy()


        # change relative poses to absolute states
        if relative_traj.shape[1] == 4:
            new = np.zeros((relative_traj.shape[0], 3))
            new[:, :2] = relative_traj[:, :2]
            new[:, -1] = relative_traj[:, -1]
            relative_traj = new

        # reverse again for singapore trajectory
        relative_traj[:, 1] *= y_inverse

        plan = relative_traj

        # Get starting block
        # starting_block = None
        # cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        # closest_distance = math.inf
        #
        # for block in self._route_roadblocks:
        #     for edge in block.interior_edges:
        #         distance = edge.polygon.distance(Point(cur_point))
        #         if distance < closest_distance:
        #             starting_block = block
        #             closest_distance = distance
        #
        #     if np.isclose(closest_distance, 0):
        #         break
        #
        # # Get traffic light lanes
        # traffic_light_lanes = []
        # for data in traffic_light_data:
        #     id_ = str(data.lane_connector_id)
        #     if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
        #         lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
        #         traffic_light_lanes.append(lane_conn)
        #
        # # Tree policy planner
        # try:
        #     plan = self._trajectory_planner.plan(iteration, ego_state, features, starting_block, self._route_roadblocks,
        #                                          self._candidate_lane_edge_ids, traffic_light_lanes, observation)
        # except Exception as e:
        #     print("Error in planning")
        #     print(e)
        #     plan = np.zeros((self._N_points, 3))

        # Convert relative poses to absolute states and wrap in a trajectory object
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)
        # print(f'Step {iteration + 1} with timestamp {current_input.iteration.time_s} Planning time: {time.perf_counter() - start_time:.3f} s')
        return trajectory

    def compute_planner_trajectory_in_batch(self, model_samples, map_names, ego_states_in_batch, route_ids, road_dics=None):
        # Construct input features
        map_y_inverse = [-1 if map_name == "sg-one-north" else 1 for map_name in map_names]
        y_inverse = np.array(map_y_inverse)[:, np.newaxis]
        batch_size = model_samples['context_actions'].shape[0]

        one_second_correction = False
        one_second_correction_with_kp = False

        agents_rect_local = model_samples['agents_rect_local']  # batch_size, time_steps, max_agent_num, 4, 2 (x, y)

        if self._model.config.sim_eval_with_gt:
            assert self.scenarios is not None
            relative_traj = np.zeros((batch_size, self._N_points, 3))
            print("WARNING: testing with ground truth trajectory")
            origin_ego_pose = model_samples['ego_pose']

            from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
            for i in range(batch_size):
                ego_initial = ego_states_in_batch[i][-1].rear_axle
                gt_absolute_traj = self.scenarios[i].get_ego_future_trajectory(
                    iteration=self.iteration,
                    num_samples=self._N_points,
                    time_horizon=self._future_horizon,
                )
                from nuplan.common.actor_state.state_representation import StateSE2, Point2D, StateVector2D
                # relative_gt_traj = convert_absolute_to_relative_poses(StateSE2(origin_ego_pose[i, 0], origin_ego_pose[i, 1], origin_ego_pose[i, 2]),
                #                                                       [state.rear_axle for state in gt_absolute_traj])

                relative_gt_traj = convert_absolute_to_relative_poses(ego_initial,
                                                                      [state.rear_axle for state in gt_absolute_traj])
                #
                # print('testing: ', relative_gt_traj.shape, self.iteration, relative_gt_traj[:5, :])
                relative_traj[i, :, :] = relative_gt_traj[:, :]
                # relative_traj[i, :10, :] = relative_gt_traj[:10, :]

            # from transformer4planning.trainer import save_raster
            # for i in range(batch_size):
            #     if 'sg-one' not in map_names[i]:
            #         continue
            #     save_raster(inputs=model_samples,
            #                 sample_index=i,
            #                 file_index=str(self.iteration) + map_names[i],
            #                 prediction_trajectory=relative_traj[i],
            #                 path_to_save='./debug_raster')
        else:
            if one_second_correction or one_second_correction_with_kp:
                # temporory debug by passing gt
                assert self.scenarios is not None
                relative_traj = np.zeros((batch_size, self._N_points, 3))
                print("WARNING: testing with ground truth 1s KP")
                origin_ego_pose = model_samples['ego_pose']
                from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
                for i in range(batch_size):
                    ego_initial = ego_states_in_batch[i][-1].rear_axle
                    gt_absolute_traj = self.scenarios[i].get_ego_future_trajectory(
                        iteration=self.iteration,
                        num_samples=self._N_points,
                        time_horizon=self._future_horizon,
                    )
                    from nuplan.common.actor_state.state_representation import StateSE2, Point2D, StateVector2D
                    # relative_gt_traj = convert_absolute_to_relative_poses(StateSE2(origin_ego_pose[i, 0], origin_ego_pose[i, 1], origin_ego_pose[i, 2]),
                    #                                                       [state.rear_axle for state in gt_absolute_traj])
                    relative_gt_traj = convert_absolute_to_relative_poses(ego_initial,
                                                                          [state.rear_axle for state in
                                                                           gt_absolute_traj])
                    # print('testing: ', relative_gt_traj.shape, self.iteration, relative_gt_traj[:5, :])
                    relative_traj[i, :, :] = relative_gt_traj[:, :]
                gt_1s_kp = relative_traj[:, 10, :]
            # features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

            # set up a timer
            start_time = time.perf_counter()

            with torch.no_grad():
                # print("start generating trajectory with gpu?", self.use_gpu)
                if self._model.device != 'cpu':
                    gt_1s_kp = torch.tensor(gt_1s_kp[..., :2]).float().to(self._model.device).unsqueeze(1) if one_second_correction_with_kp else None
                    device = self._model.device
                    prediction_generation = self._model.generate(
                        context_actions=torch.tensor(model_samples['context_actions']).to(device),
                        high_res_raster=torch.tensor(model_samples['high_res_raster']).to(device),
                        low_res_raster=torch.tensor(model_samples['low_res_raster']).to(device),
                        trajectory_label=torch.zeros((batch_size, self._N_points, 4)).to(device),
                        route_ids=route_ids,
                        ego_pose=model_samples['ego_pose'],
                        road_dic=road_dics,
                        map=map_names,
                        agents_rect_local=torch.tensor(agents_rect_local).to(device),
                        gt_1s_kp=gt_1s_kp if one_second_correction else None,
                    )
                    pred_traj = prediction_generation['traj_logits'].detach().cpu().float().numpy()
                    if 'key_points_logits' in prediction_generation:
                        pred_key_points = prediction_generation['key_points_logits'].detach().cpu().numpy()
                    else:
                        pred_key_points = None
                else:
                    prediction_generation = self._model.generate(
                        context_actions=torch.tensor(model_samples['context_actions']),
                        high_res_raster=torch.tensor(model_samples['high_res_raster']),
                        low_res_raster=torch.tensor(model_samples['low_res_raster']),
                        trajectory_label=torch.zeros((batch_size, self._N_points, 4)),
                        route_ids=route_ids,
                        ego_pose=model_samples['ego_pose'],
                        road_dic=road_dics,
                        map=map_names,
                        agents_rect_local=torch.tensor(agents_rect_local),
                        gt_1s_kp=gt_1s_kp if one_second_correction else None,
                    )
                    pred_traj = prediction_generation['traj_logits'].float().numpy()  # (80, 2) or (80, 4)
                    if 'key_points_logits' in prediction_generation:
                        pred_key_points = prediction_generation['key_points_logits'].numpy()[0]
                    else:
                        pred_key_points = None

                try:
                    if self._model.config.simulate_with_5f_smoothing:
                        # make a smoothing on the first 5 frames of the trajectory
                        pred_traj[:, 0, :2] = pred_traj[:, 5, :2] / 5.0
                except:
                    pass

                if self._model.config.skip_yaw_norm:
                    oriented_yaws = model_samples['ego_pose'][:, -1]
                    # rotate prediction based on oriented yaw
                    sin_, cos_ = np.sin(-oriented_yaws - math.pi / 2), np.cos(-oriented_yaws - math.pi / 2)
                    rotated_pred_traj = pred_traj.copy()
                    for i in range(batch_size):
                        rotated_pred_traj[i, :, 0] = pred_traj[i, :, 0] * cos_[i] - pred_traj[i, :, 1] * sin_[i]
                        rotated_pred_traj[i, :, 1] = pred_traj[i, :, 0] * sin_[i] + pred_traj[i, :, 1] * cos_[i]
                    pred_traj = rotated_pred_traj

            # print the timer
            print(f'Model inference time: {time.perf_counter() - start_time:.3f} s')

            # post-processing
            relative_traj = pred_traj.copy()  # [batch_size, 80, 4]

            # StateSE2 requires a dimension of 3 instead of 4
            if relative_traj.shape[-1] == 4:
                new = np.zeros(relative_traj.shape)[:, :, :3]
                new[:, :, :2] = relative_traj[:, :, :2]
                new[:, :, -1] = relative_traj[:, :, -1]
                relative_traj = new  # [batch_size, 80, 3]

            if one_second_correction:
                one_second_delta = relative_traj[:, 10, :] - gt_1s_kp
                # linearly interpolate the trajectory from 0 to 1s
                for i in range(10):
                    relative_traj[:, i, :] -= one_second_delta * i / 10
                for i in range(10, 20):
                    relative_traj[:, i, :] -= one_second_delta * (20 - i) / 10

            # from transformer4planning.trainer import save_raster
            # for i in range(batch_size):
            #     if 'sg-one' not in map_names[i]:
            #         continue
            #     save_raster(inputs=model_samples,
            #                 sample_index=i,
            #                 file_index=str(self.iteration) + map_names[i],
            #                 prediction_trajectory=relative_traj[i].copy(),
            #                 path_to_save='./debug_raster')

            # reverse again for singapore trajectory
            relative_traj[:, :, 1] *= y_inverse

        trajectories = []
        for i in range(relative_traj.shape[0]):
            plan = relative_traj[i]
            # Convert relative poses to absolute states and wrap in a trajectory object
            states = transform_predictions_to_states(plan, ego_states_in_batch[i], self._future_horizon, DT)
            trajectories.append(InterpolatedTrajectory(states))
        self.iteration += 1
        return trajectories

    def inputs_to_model_sample(self, history, traffic_light_data, map_name):
        ego_states = list(history.ego_state_buffer)  # a list of ego trajectory

        context_length = len(ego_states)  # context_length = 22/23
        past_seconds = 2
        frame_rate = 20
        frame_rate_change = 2
        frame_id = context_length * frame_rate_change - 1  # 22 * 2 = 44 in 20hz
        scenario_start_frame = frame_id - past_seconds * frame_rate  # 44 - 2 * 20 = 4
        past_sample_interval = 2
        # past_sample_interval = int(self.model.config.past_sample_interval)  # 5
        if self._model.config.selected_exponential_past:
            # sample_frames_in_past_20hz = [scenario_start_frame + 0, scenario_start_frame + 20,
            #                               scenario_start_frame + 30, frame_id]
            sample_frames_in_past_20hz = [frame_id - 40, frame_id - 20,
                                          frame_id - 10, frame_id]
        elif self._model.config.current_frame_only:
            sample_frames_in_past_20hz = [frame_id]
        else:
            sample_frames_in_past_20hz = list(range(scenario_start_frame, frame_id, past_sample_interval))  # length = 8
        sample_frames_in_past_10hz = [int(frame_id / frame_rate_change) for frame_id in
                                      sample_frames_in_past_20hz]  # length = 8
        # past_frames = int(past_seconds * frame_rate)
        # if context_length < past_frames:
        #     assert False, f"context length is too short, {context_length} {past_frames}"
        # trajectory as format of [(x, y, yaw)]
        oriented_point = np.array([ego_states[-1].rear_axle.x,
                                   ego_states[-1].rear_axle.y,
                                   ego_states[-1].rear_axle.heading]).astype(np.float32)


        if self._model.config.skip_yaw_norm:
            oriented_yaw = oriented_point[-1]
            oriented_point[-1] = 0

        if self.road_dic is None:
            self.road_dic = get_road_dict(self._map_api, ego_pose_center=Point2D(oriented_point[0], oriented_point[1]),
                                          all_road_dic=self.all_road_dic)
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

        print('Route correction in progress. before: ', self._route_roadblock_ids)
        self._route_roadblock_ids = route_roadblock_correction(
            ego_states[-1],
            self._map_api,
            self._route_roadblock_ids,
        )
        print('Route correction in progress. after: ', self._route_roadblock_ids)
        corrected_route_ids = self._route_roadblock_ids
        high_res_raster, low_res_raster, context_action, agent_rect_pts_local = self.compute_raster_input(
            ego_trajectory, agents, statics, traffic_light_data, ego_shape,
            origin_ego_pose=oriented_point,
            map_name=map_name,
            corrected_route_ids=corrected_route_ids)

        if self._model.config.use_speed:
            speed = np.ones((context_action.shape[0], 3), dtype=np.float32) * -1
            for i in range(context_action.shape[0]):
                current_ego_dynamic = sampled_ego_states[i].dynamic_car_state  # Class of DynamicCarState
                speed[i, :] = [
                    current_ego_dynamic.speed,
                    current_ego_dynamic.acceleration,
                    current_ego_dynamic.angular_velocity]
            context_action = np.concatenate([context_action, speed], axis=1)

        if self._model.config.skip_yaw_norm:
            oriented_point[-1] = oriented_yaw

        return {
            'high_res_raster': high_res_raster,
            'low_res_raster': low_res_raster,
            'context_actions': context_action,
            'ego_pose': oriented_point,
            'agents_rect_local': agent_rect_pts_local,
        }

    def compute_raster_input(self, ego_trajectory, agents_seq, statics_seq, traffic_data=None,
                             ego_shape=None, max_dis=300, origin_ego_pose=None, map_name=None,
                             use_speed=True, corrected_route_ids=None
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
        y_inverse = -1 if self._map_api.map_name == "sg-one-north" else 1

        ## channel 0-1: goal route
        cos_, sin_ = math.cos(-origin_ego_pose[2] - math.pi / 2), math.sin(-origin_ego_pose[2] - math.pi / 2)
        route_ids = corrected_route_ids if corrected_route_ids is not None else self._route_roadblock_ids
        for route_id in route_ids:
            if int(route_id) == -1:
                continue
            if int(route_id) not in self.road_dic:
                print('ERROR: ', route_id, ' not found in road_dic with ', self._map_api.map_name)
                continue
            xyz = self.road_dic[int(route_id)]["xyz"].copy()
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

            route_lanes = self.road_dic[int(route_id)]["lower_level"]
            for each_route_lane in route_lanes:
                xyz = self.road_dic[int(each_route_lane)]["xyz"].copy()
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

        for i, key in enumerate(self.road_dic):
            xyz = self.road_dic[key]["xyz"].copy()
            road_type = int(self.road_dic[key]['type'])
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
            if lane_id not in self.road_dic:
                continue
            xyz = self.road_dic[lane_id]["xyz"].copy()
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

        # check and only keep static agents
        static_num = 0
        for i in range(max_agent_num):
            first_pt = agent_rect_pts_local[0, i, :, :]
            last_pt = agent_rect_pts_local[-1, i, :, :]
            if first_pt.sum() == 0 and last_pt.sum() == 0:
                continue
            if not (abs(first_pt - last_pt).sum() < 0.1):
                agent_rect_pts_local[:, i, :, :] = 0
            else:
                static_num += 1

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

        result_dict = {'high_res_raster': rasters_high_res, 'low_res_raster': rasters_low_res,
                       'context_actions': rotated_poses}

        return rasters_high_res, rasters_low_res, np.array(context_actions, dtype=np.float32), agent_rect_pts_local


def get_road_dict(map_api, ego_pose_center, all_road_dic={}):
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
            # if int(map_obj_id) in all_road_dic:
            #     road_dic[int(map_obj_id)] = all_road_dic[int(map_obj_id)]
            #     continue
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
            # all_road_dic[int(map_obj_id)] = new_dic

    # print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
    return road_dic


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