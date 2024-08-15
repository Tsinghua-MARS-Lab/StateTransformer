import os
import math
from typing import List

import cv2
import torch
from shapely.geometry import LineString

from transformers import HfArgumentParser
from transformer4planning.utils.args import ModelArguments
from transformer4planning.models.backbone.str_base import build_models

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)

from nuplan_simulation.planner_utils import *
from nuplan_simulation.planner import get_road_dict, generate_contour_pts


class STRTrajectoryGenerator:
    def __init__(self, model_path):
        self._future_horizon = T  # [s]
        self._step_interval = DT  # [s]
        self._N_points = int(T / DT)
        self.iteration = 0
        self.road_dic = None
        self.all_road_dic = {}

        self._model_path = model_path
        self._initialize_model()

    def _initialize_model(self):
        parser = HfArgumentParser((ModelArguments))
        # load model args from config.json
        config_path = os.path.join(self._model_path, "config.json")
        if not os.path.exists(config_path):
            print(
                "WARNING config.json not found in checkpoint path, using default model args ",
                config_path,
            )
            model_args = parser.parse_args_into_dataclasses(
                return_remaining_strings=True
            )[0]
        else:
            (model_args,) = parser.parse_json_file(config_path, allow_extra_keys=True)
            model_args.model_pretrain_name_or_path = self._model_path
            model_args.model_name = model_args.model_name.replace(
                "scratch", "pretrained"
            )
        print(
            "model loaded with args: ",
            model_args,
            model_args.model_name,
            self._model_path,
        )
        self._model = build_models(model_args=model_args)
        self._model.cuda()
        self._model.eval()
        print("model built on ", self._model.device)

    def _initialize_route_plan(self, route_roadblock_ids: List[str]):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id
            for block in self._route_roadblocks
            if block
            for edge in block.interior_edges
        ]

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)

    def predict_states(self, planner_input: PlannerInput) -> List[EgoState]:
        model_samples = [
            self._inputs_to_model_sample(
                history=planner_input.history,
                traffic_light_data=list(planner_input.traffic_light_data),
                map_name=self._map_api.map_name,
            )
        ]

        samples_in_batch = {
            key: np.stack([sample[key] for sample in model_samples])
            for key in model_samples[0].keys()
        }

        trajectory = self._compute_str_trajectory(
            model_samples=samples_in_batch, map_names=[self._map_api.map_name]
        )[0]

        states = transform_predictions_to_states(
            trajectory,
            planner_input.history.ego_states,
            self._future_horizon,
            self._step_interval,
        )

        return states

    def _compute_str_trajectory(self, model_samples, map_names):
        # Construct input features
        map_y_inverse = [
            -1 if map_name == "sg-one-north" else 1 for map_name in map_names
        ]
        y_inverse = np.array(map_y_inverse)[:, np.newaxis]
        batch_size = model_samples["context_actions"].shape[0]

        # features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)
        with torch.no_grad():
            device = self._model.device
            prediction_generation = self._model.generate(
                context_actions=torch.tensor(model_samples["context_actions"]).to(
                    device
                ),
                high_res_raster=torch.tensor(model_samples["high_res_raster"]).to(
                    device
                ),
                low_res_raster=torch.tensor(model_samples["low_res_raster"]).to(device),
                trajectory_label=torch.zeros((batch_size, self._N_points, 4)).to(
                    device
                ),
            )
            pred_traj = (
                prediction_generation["traj_logits"].detach().cpu().float().numpy()
            )

            try:
                if self._model.config.simulate_with_5f_smoothing:
                    # make a smoothing on the first 5 frames of the trajectory
                    pred_traj[:, 0, :2] = pred_traj[:, 5, :2] / 5.0
            except:
                pass

            if self._model.config.skip_yaw_norm:
                oriented_yaws = model_samples["oriented_point"][:, -1]
                # rotate prediction based on oriented yaw
                sin_, cos_ = np.sin(-oriented_yaws - math.pi / 2), np.cos(
                    -oriented_yaws - math.pi / 2
                )
                rotated_pred_traj = pred_traj.copy()
                for i in range(batch_size):
                    rotated_pred_traj[i, :, 0] = (
                        pred_traj[i, :, 0] * cos_[i] - pred_traj[i, :, 1] * sin_[i]
                    )
                    rotated_pred_traj[i, :, 1] = (
                        pred_traj[i, :, 0] * sin_[i] + pred_traj[i, :, 1] * cos_[i]
                    )
                pred_traj = rotated_pred_traj

        # post-processing
        relative_traj = pred_traj.copy()  # [batch_size, 80, 4]

        # StateSE2 requires a dimension of 3 instead of 4
        if relative_traj.shape[-1] == 4:
            new = np.zeros(relative_traj.shape)[:, :, :3]
            new[:, :, :2] = relative_traj[:, :, :2]
            new[:, :, -1] = relative_traj[:, :, -1]
            relative_traj = new  # [batch_size, 80, 3]

        # reverse again for singapore trajectory
        relative_traj[:, :, 1] *= y_inverse
        self.iteration += 1

        return relative_traj

    def _inputs_to_model_sample(self, history, traffic_light_data, map_name):
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
            sample_frames_in_past_20hz = [
                frame_id - 40,
                frame_id - 20,
                frame_id - 10,
                frame_id,
            ]
        elif self._model.config.current_frame_only:
            sample_frames_in_past_20hz = [frame_id]
        else:
            sample_frames_in_past_20hz = list(
                range(scenario_start_frame, frame_id, past_sample_interval)
            )  # length = 8
        sample_frames_in_past_10hz = [
            int(frame_id / frame_rate_change) for frame_id in sample_frames_in_past_20hz
        ]  # length = 8
        # past_frames = int(past_seconds * frame_rate)
        # if context_length < past_frames:
        #     assert False, f"context length is too short, {context_length} {past_frames}"
        # trajectory as format of [(x, y, yaw)]
        oriented_point = np.array(
            [
                ego_states[-1].rear_axle.x,
                ego_states[-1].rear_axle.y,
                ego_states[-1].rear_axle.heading,
            ]
        ).astype(np.float32)

        if self._model.config.skip_yaw_norm:
            oriented_yaw = oriented_point[-1]
            oriented_point[-1] = 0

        if self.road_dic is None:
            self.road_dic = get_road_dict(
                self._map_api,
                ego_pose_center=Point2D(oriented_point[0], oriented_point[1]),
                all_road_dic=self.all_road_dic,
            )
        sampled_ego_states = [ego_states[i] for i in sample_frames_in_past_10hz]
        ego_trajectory = np.array(
            [
                (
                    ego_state.rear_axle.x,
                    ego_state.rear_axle.y,
                    ego_state.rear_axle.heading,
                )
                for ego_state in sampled_ego_states
            ]
        ).astype(
            np.float32
        )  # (20, 3)
        ego_shape = np.array(
            [
                ego_states[-1].waypoint.oriented_box.width,
                ego_states[-1].waypoint.oriented_box.length,
            ]
        ).astype(np.float32)
        observation_buffer = list(history.observation_buffer)
        sampled_observation_buffer = [
            observation_buffer[i] for i in sample_frames_in_past_10hz
        ]
        agents = [
            observation.tracked_objects.get_agents()
            for observation in sampled_observation_buffer
        ]
        statics = [
            observation.tracked_objects.get_static_objects()
            for observation in sampled_observation_buffer
        ]

        # corrected_route_ids = route_roadblock_correction(
        #     ego_states[-1],
        #     self._map_api,
        #     self._route_roadblock_ids,
        # )
        corrected_route_ids = self._route_roadblock_ids
        high_res_raster, low_res_raster, context_action = self.compute_raster_input(
            ego_trajectory,
            agents,
            statics,
            traffic_light_data,
            ego_shape,
            origin_ego_pose=oriented_point,
            map_name=map_name,
            corrected_route_ids=corrected_route_ids,
        )

        if self._model.config.use_speed:
            speed = np.ones((context_action.shape[0], 3), dtype=np.float32) * -1
            for i in range(context_action.shape[0]):
                current_ego_dynamic = sampled_ego_states[
                    i
                ].dynamic_car_state  # Class of DynamicCarState
                speed[i, :] = [
                    current_ego_dynamic.speed,
                    current_ego_dynamic.acceleration,
                    current_ego_dynamic.angular_velocity,
                ]
            context_action = np.concatenate([context_action, speed], axis=1)

        if self._model.config.skip_yaw_norm:
            oriented_point[-1] = oriented_yaw

        return {
            "high_res_raster": high_res_raster,
            "low_res_raster": low_res_raster,
            "context_actions": context_action,
            "oriented_point": oriented_point,
        }

    def compute_raster_input(
        self,
        ego_trajectory,
        agents_seq,
        statics_seq,
        traffic_data=None,
        ego_shape=None,
        max_dis=300,
        origin_ego_pose=None,
        map_name=None,
        use_speed=True,
        corrected_route_ids=None,
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
        total_raster_channels = (
            2 + road_types + traffic_types + agent_type * context_length
        )
        raster_shape = [224, 224, total_raster_channels]

        # WARNING: Not Supporting Goal Points
        rasters_high_res = np.zeros(raster_shape, dtype=np.uint8)
        rasters_low_res = np.zeros(raster_shape, dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        y_inverse = -1 if self._map_api.map_name == "sg-one-north" else 1

        ## channel 0-1: goal route
        cos_, sin_ = math.cos(-origin_ego_pose[2] - math.pi / 2), math.sin(
            -origin_ego_pose[2] - math.pi / 2
        )
        route_ids = (
            corrected_route_ids
            if corrected_route_ids is not None
            else self._route_roadblock_ids
        )
        for route_id in route_ids:
            if int(route_id) == -1:
                continue
            if int(route_id) not in self.road_dic:
                print(
                    "ERROR: ",
                    route_id,
                    " not found in road_dic with ",
                    self._map_api.map_name,
                )
                continue
            xyz = self.road_dic[int(route_id)]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis
            ):
                continue
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = (
                simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_,
                simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_,
            )
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_route = (
                simplified_xyz * high_res_scale + raster_shape[0] // 2
            ).astype("int32")
            low_res_route = (
                simplified_xyz * low_res_scale + raster_shape[0] // 2
            ).astype("int32")

            cv2.fillPoly(
                rasters_high_res_channels[0],
                np.int32([high_res_route[:, :2]]),
                (255, 255, 255),
            )
            cv2.fillPoly(
                rasters_low_res_channels[0],
                np.int32([low_res_route[:, :2]]),
                (255, 255, 255),
            )

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
                simplified_xyz[:, 0], simplified_xyz[:, 1] = (
                    simplified_xyz[:, 0].copy() * cos_
                    - simplified_xyz[:, 1].copy() * sin_,
                    simplified_xyz[:, 0].copy() * sin_
                    + simplified_xyz[:, 1].copy() * cos_,
                )
                simplified_xyz[:, 1] *= -1
                simplified_xyz[:, 0] *= y_inverse
                high_res_route = (simplified_xyz * high_res_scale).astype("int32") + 112
                low_res_route = (simplified_xyz * low_res_scale).astype("int32") + 112
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(
                        rasters_high_res_channels[1],
                        tuple(high_res_route[j, :2]),
                        tuple(high_res_route[j + 1, :2]),
                        (255, 255, 255),
                        2,
                    )
                    cv2.line(
                        rasters_low_res_channels[1],
                        tuple(low_res_route[j, :2]),
                        tuple(low_res_route[j + 1, :2]),
                        (255, 255, 255),
                        2,
                    )

        for i, key in enumerate(self.road_dic):
            xyz = self.road_dic[key]["xyz"].copy()
            road_type = int(self.road_dic[key]["type"])
            xyz[:, :2] -= origin_ego_pose[:2]
            if (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis) or (
                abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis
            ):
                continue
            # simplify road vector, can simplify about half of all the points
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = (
                simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_,
                simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_,
            )
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_road = (simplified_xyz * high_res_scale).astype(
                "int32"
            ) + raster_shape[0] // 2
            low_res_road = (simplified_xyz * low_res_scale).astype(
                "int32"
            ) + raster_shape[0] // 2
            if road_type in [5, 17, 18, 19]:
                cv2.fillPoly(
                    rasters_high_res_channels[road_type + 2],
                    np.int32([high_res_road[:, :2]]),
                    (255, 255, 255),
                )
                cv2.fillPoly(
                    rasters_low_res_channels[road_type + 2],
                    np.int32([low_res_road[:, :2]]),
                    (255, 255, 255),
                )
            else:
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(
                        rasters_high_res_channels[road_type + 2],
                        tuple(high_res_road[j, :2]),
                        tuple(high_res_road[j + 1, :2]),
                        (255, 255, 255),
                        2,
                    )
                    cv2.line(
                        rasters_low_res_channels[road_type + 2],
                        tuple(low_res_road[j, :2]),
                        tuple(low_res_road[j + 1, :2]),
                        (255, 255, 255),
                        2,
                    )

        # traffic light
        for each_traffic_light_data in traffic_data:
            traffic_state = int(each_traffic_light_data.status)
            lane_id = int(each_traffic_light_data.lane_connector_id)
            if lane_id not in self.road_dic:
                continue
            xyz = self.road_dic[lane_id]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            if not (
                (abs(xyz[0, 0]) > max_dis and abs(xyz[-1, 0]) > max_dis)
                or (abs(xyz[0, 1]) > max_dis and abs(xyz[-1, 1]) > max_dis)
            ):
                pts = list(zip(xyz[:, 0], xyz[:, 1]))
                line = LineString(pts)
                simplified_xyz_line = line.simplify(1)
                simplified_x, simplified_y = simplified_xyz_line.xy
                simplified_xyz = np.ones((len(simplified_x), 2)) * -1
                simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
                simplified_xyz[:, 0], simplified_xyz[:, 1] = (
                    simplified_xyz[:, 0].copy() * cos_
                    - simplified_xyz[:, 1].copy() * sin_,
                    simplified_xyz[:, 0].copy() * sin_
                    + simplified_xyz[:, 1].copy() * cos_,
                )
                simplified_xyz[:, 1] *= -1
                simplified_xyz[:, 0] *= y_inverse
                high_res_traffic = (simplified_xyz * high_res_scale).astype(
                    "int32"
                ) + raster_shape[0] // 2
                low_res_traffic = (simplified_xyz * low_res_scale).astype(
                    "int32"
                ) + raster_shape[0] // 2
                # traffic state order is GREEN, RED, YELLOW, UNKNOWN
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(
                        rasters_high_res_channels[2 + road_types + traffic_state],
                        tuple(high_res_traffic[j, :2]),
                        tuple(high_res_traffic[j + 1, :2]),
                        (255, 255, 255),
                        2,
                    )
                    cv2.line(
                        rasters_low_res_channels[2 + road_types + traffic_state],
                        tuple(low_res_traffic[j, :2]),
                        tuple(low_res_traffic[j + 1, :2]),
                        (255, 255, 255),
                        2,
                    )

        cos_, sin_ = math.cos(-origin_ego_pose[2]), math.sin(-origin_ego_pose[2])
        ## agent includes VEHICLE, PEDESTRIAN, BICYCLE, EGO(except)
        for i, each_type_agents in enumerate(agents_seq):
            for j, agent in enumerate(each_type_agents):
                agent_type = int(agent.tracked_object_type.value)
                pose = np.array(
                    [
                        agent.box.center.point.x,
                        agent.box.center.point.y,
                        agent.box.center.heading,
                    ]
                ).astype(np.float32)
                pose -= origin_ego_pose
                if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
                    continue
                rotated_pose = [
                    pose[0] * cos_ - pose[1] * sin_,
                    pose[0] * sin_ + pose[1] * cos_,
                ]
                shape = np.array([agent.box.width, agent.box.length])
                rect_pts = generate_contour_pts(
                    (rotated_pose[1], rotated_pose[0]),
                    w=shape[0],
                    l=shape[1],
                    direction=-pose[2],
                )
                rect_pts = np.array(rect_pts, dtype=np.int32)
                rect_pts[:, 0] *= y_inverse
                rect_pts_high_res = (high_res_scale * rect_pts).astype(
                    np.int64
                ) + raster_shape[0] // 2
                # example: if frame_interval = 10, past frames = 40
                # channel number of [index:0-frame_0, index:1-frame_10, index:2-frame_20, index:3-frame_30, index:4-frame_40]  for agent_type = 0
                # channel number of [index:5-frame_0, index:6-frame_10, index:7-frame_20, index:8-frame_30, index:9-frame_40]  for agent_type = 1
                # ...
                cv2.drawContours(
                    rasters_high_res_channels[
                        2 + road_types + traffic_types + agent_type * context_length + i
                    ],
                    [rect_pts_high_res],
                    -1,
                    (255, 255, 255),
                    -1,
                )
                # draw on low resolution
                rect_pts_low_res = (low_res_scale * rect_pts).astype(
                    np.int64
                ) + raster_shape[0] // 2
                cv2.drawContours(
                    rasters_low_res_channels[
                        2 + road_types + traffic_types + agent_type * context_length + i
                    ],
                    [rect_pts_low_res],
                    -1,
                    (255, 255, 255),
                    -1,
                )

        recentered_ego_trajectory = ego_trajectory - origin_ego_pose
        for i, pose in enumerate(recentered_ego_trajectory):
            agent_type = 7  # type EGO is 7
            rotated_pose = [
                pose[0] * cos_ - pose[1] * sin_,
                pose[0] * sin_ + pose[1] * cos_,
            ]
            rect_pts = generate_contour_pts(
                (rotated_pose[1], rotated_pose[0]),
                w=ego_shape[0],
                l=ego_shape[1],
                direction=-pose[2],
            )
            rect_pts = np.array(rect_pts, dtype=np.int32)
            rect_pts[:, 0] *= y_inverse
            rect_pts_high_res = (high_res_scale * rect_pts).astype(
                np.int64
            ) + raster_shape[0] // 2
            cv2.drawContours(
                rasters_high_res_channels[
                    2 + road_types + traffic_types + agent_type * context_length + i
                ],
                [rect_pts_high_res],
                -1,
                (255, 255, 255),
                -1,
            )
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(
                np.int64
            ) + raster_shape[0] // 2
            cv2.drawContours(
                rasters_low_res_channels[
                    2 + road_types + traffic_types + agent_type * context_length + i
                ],
                [rect_pts_low_res],
                -1,
                (255, 255, 255),
                -1,
            )

        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

        # context_actions computation
        recentered_ego_trajectory = ego_trajectory - origin_ego_pose
        rotated_poses = np.array(
            [
                recentered_ego_trajectory[:, 0] * cos_
                - recentered_ego_trajectory[:, 1] * sin_,
                recentered_ego_trajectory[:, 0] * sin_
                + recentered_ego_trajectory[:, 1] * cos_,
                np.zeros(recentered_ego_trajectory.shape[0]),
                recentered_ego_trajectory[:, -1],
            ],
            dtype=np.float32,
        ).transpose(1, 0)
        rotated_poses[:, 1] *= y_inverse

        context_actions = rotated_poses  # (4, 4)

        result_dict = {
            "high_res_raster": rasters_high_res,
            "low_res_raster": rasters_low_res,
            "context_actions": rotated_poses,
        }

        return (
            rasters_high_res,
            rasters_low_res,
            np.array(context_actions, dtype=np.float32),
        )
