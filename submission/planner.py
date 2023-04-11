from typing import List, Type

import numpy as np
import numpy.typing as npt
import cv2, math, pickle, os, sys
import shapely
import torch
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters, VehicleParameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import  InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer, StopLineType, VectorLayer

from transformers import (HfArgumentParser)
from transformers.trainer_utils import get_last_checkpoint
from transformers.configuration_utils import PretrainedConfig
from models.model import TransfoXLModelNuPlan
from runner import ModelArguments, DataTrainingArguments

class SimplePlanner(AbstractPlanner):
    """
    Planner going straight
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 steering_angle: float = 0.0):
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        pass

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        # Extract iteration and history
        with open("example.pkl", "wb") as f:
            pickle.dump(current_input, f)
        iteration = current_input.iteration
        print("iteration:", iteration)
        history = current_input.history
        print("state buffer length", len(history.ego_state_buffer))
        print("ego trajectory", [(history.ego_state_buffer[i].waypoint.center.x, history.ego_state_buffer[i].waypoint.center.y) for i in range(len(history.ego_state_buffer))])
        # print("observations_buffer", history.observation_buffer[0])
        # print("sample_interval", history.sample_interval)
        agent_center = history.observation_buffer[0].tracked_objects.get_agents()[0].box.center
        print("observation number", len(history.observation_buffer))
        print("agent number:", [len(history.observation_buffer[i].tracked_objects.get_agents()) for i in range(len(history.observation_buffer))])
        print(f"agent box example: ({agent_center.point.x}, {agent_center.point.y}, {agent_center.heading}) ", )
        # print("agent velocity example ", history.observation_buffer[0].tracked_objects.get_agents()[0].velocity)
        # print("agent future example ", history.observation_buffer[0].tracked_objects.get_agents()[0].predictions)
        # print("agent past example ", history.observation_buffer[0].tracked_objects.get_agents()[0].past_trajectory)
        
        static_center = history.observation_buffer[0].tracked_objects.get_static_objects()[0].box.center 
        print("static number:", [len(history.observation_buffer[i].tracked_objects.get_static_objects()) for i in range(len(history.observation_buffer))])
        print(f"static box example: ({static_center.point.x}, {static_center.point.y}, {static_center.heading})" )
        print("static type example", history.observation_buffer[0].tracked_objects.get_static_objects()[0].tracked_object_type)
        

        ego_state = history.ego_states[-1]
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                self.acceleration,
            ),
            tire_steering_angle=self.steering_angle,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for _ in np.arange(
            iteration.time_us + self.sampling_time.time_us,
            iteration.time_us + self.horizon_seconds.time_us,
            self.sampling_time.time_us,
        ):
            if state.dynamic_car_state.speed > self.max_velocity:
                accel = self.max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=state.rear_axle,
                    rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
                    rear_axle_acceleration_2d=StateVector2D(accel, 0),
                    tire_steering_angle=state.tire_steering_angle,
                    time_point=state.time_point,
                    vehicle_parameters=state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=state.dynamic_car_state.angular_velocity,
                    angular_accel=state.dynamic_car_state.angular_acceleration,
                )

            state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)

        return InterpolatedTrajectory(trajectory)

class ControlTFPlanner(AbstractPlanner):
    """
    Planner with Pretrained Control Transformer
    """
    def __init__(self, 
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 steering_angle: float = 0.0):
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.samping_time = TimePoint(int(sampling_time * 1e6))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        
        # model initialization and configuration 
        parser = HfArgumentParser((ModelArguments))
        model_args = parser.parse_args_into_dataclasses()[0]
        self.model = TransfoXLModelNuPlan.from_pretrained(model_args.model_pretrain_name_or_path, \
                                                          model_args=model_args)
        self.load_state_dict()

    def load_state_dict(self):
        checkpoint = get_last_checkpoint("checkpoints/perinstance_nsm_pose_traj") 
        if os.path.isfile(os.path.join(checkpoint, "config.json")):
            config = PretrainedConfig.from_json_file(os.path.join(checkpoint, "config.json"))
        if os.path.isfile(os.path.join(checkpoint, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location="cpu")
            self.model.load_state_dict(state_dict)
        else:
            raise RuntimeError("No checkpoint in current directory")
    
    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        self.initialization = initialization
        self.goal = initialization.mission_goal
        self.route_roadblock_ids = initialization.route_roadblock_ids
        self.map_api = initialization.map_api

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        history = current_input
        ego_states = history.ego_state_buffer # a list of ego trajectory
        context_length = len(ego_states)
        # trajectory as format of [(x, y, yaw)]
        ego_trajectory = np.array([(ego_states[i].waypoint.center.x, \
                                    ego_states[i].waypoint.center.y, \
                                    ego_states[i].waypoint.heading) for i in range(context_length)])
        ego_shape = np.array([ego_states[-1].waypoint.oriented_box.height, \
                              ego_states[-1].waypoint.oriented_box.width])
        agents = [history.observation_buffer[i].tracked_objects.get_agents() for i in range(context_length)] 
        statics = [history.observation_buffer[i].tracked_objects.get_static_objects() for i in range(context_length)]
        road_dic = get_road_dict(self.map_api, Point2D(ego_trajectory[-1][0], ego_trajectory[-1][1]))
        high_res_raster, low_res_raster, context_action = self.compute_raster_input(ego_trajectory, agents, statics, road_dic, ego_shape)
        output = self.model.predict(

        )

    def compute_raster_input(self, ego_trajectory, agents_seq, statics_seq, road_dic, ego_shape=None, max_dis=500):
        """
        the first dimension is the sequence length, each timestep include n-items.
        agent_seq and statics_seq are both agents in raster definition
        """
        ego_pose = ego_trajectory[-1] # (x, y, yaw) in current timestamp
        cos_, sin_ = math.cos(ego_pose[2]), math.sin(ego_pose[2])

        ## hyper initilization
        total_road_types = 20
        total_agent_types = 8
        context_length = 9
        high_res_raster_scale = 4
        low_res_raster_scale = 0.77

        total_raster_channels = 1 + total_road_types + total_agent_types * 9
        rasters_high_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_low_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        # downsampling from ego_trajectory, agent_seq and statics_seq
        downsample_indexs = [0, 2, 5, 7, 10, 12, 15, 18, 21]
        downsample_agents_seq = list()
        downsample_statics_seq = list()
        downsample_ego_trajectory = list()
        for i in downsample_indexs:
            downsample_agents_seq.append(agents_seq[i].copy())
            downsample_statics_seq.append(statics_seq[i].copy())
            downsample_ego_trajectory.append(ego_trajectory[i].copy())
        del agents_seq, statics_seq, ego_trajectory
        agents_seq = downsample_agents_seq
        ego_trajectory = downsample_ego_trajectory
        statics_seq = downsample_statics_seq
        # goal channel
        relative_goal = np.array([self.goal.x, self.goal.y, self.goal.heading]) - ego_pose
        rotated_goal_pose = [relative_goal[0] * cos_ - relative_goal[1] * sin_,
                             relative_goal[0] * sin_ + relative_goal[1] * cos_,
                             relative_goal[2]]
        goal_contour = cv2.boxPoints(((rotated_goal_pose[0], rotated_goal_pose[1]),
                                      (ego_shape[0], ego_shape[1]), rotated_goal_pose[2]))
        goal_contour = np.int0(goal_contour)
        goal_contour_high_res = int(high_res_raster_scale) * goal_contour
        cv2.drawContours(rasters_high_res_channels[0], [goal_contour_high_res], -1, (255, 255, 255), -1)
        goal_contour_low_res = int(low_res_raster_scale) * goal_contour
        cv2.drawContours(rasters_low_res_channels[0], [goal_contour_low_res], -1, (255, 255, 255), -1)

        # road element computation
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
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0] * cos_ - simplified_xyz[:, 1] * sin_, simplified_xyz[:, 0] * sin_ + simplified_xyz[:, 1] * cos_

            high_res_road = simplified_xyz * high_res_raster_scale
            low_res_road = simplified_xyz * low_res_raster_scale
            high_res_road = high_res_road.astype('int32')
            low_res_road = low_res_road.astype('int32')

            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[road_type + 1], tuple(high_res_road[j, :2]), tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[road_type + 1], tuple(low_res_road[j, :2]), tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)
        
        # agent element computation
        ## statics include CZONE_SIGN,BARRIER,TRAFFIC_CONE,GENERIC_OBJECT,
        # TODO:merge the statics，agents and ego agents
        # total_agents_seq = list()
        # for agents, statics in zip(agents_seq, statics_seq):
            # total_agents = list()
            # total_agents.extend(agents)
            # total_agents.extend(statics)
            # total_agents_seq.append(total_agents)
        for i, statics in enumerate(statics_seq):
            for j, static in enumerate(statics):
                static_type = static.tracked_object_type.value
                pose = np.array([static.box.center.point.x, static.box.center.point.y, static.box.center.heading])
                pose -= ego_pose
                if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
                    continue
                rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                                pose[0] * sin_ + pose[1] * cos_]    
                shape = np.array([static.box.height, static.box.width])
                rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
                            (shape[0], shape[1]), pose[2]))
                rect_pts = np.int0(rect_pts)
                rect_pts_high_res = int(high_res_raster_scale) * rect_pts
                cv2.drawContours(rasters_high_res_channels[1 + total_road_types + static_type * 9 + i], [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = int(low_res_raster_scale) * rect_pts
                cv2.drawContours(rasters_low_res_channels[1 + total_road_types + static_type * 9 + i], [rect_pts_low_res], -1, (255, 255, 255), -1)

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
                shape = np.array([agent.box.height, agent.box.width])
                rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
                            (shape[0], shape[1]), pose[2]))
                rect_pts = np.int0(rect_pts)
                rect_pts_high_res = int(high_res_raster_scale) * rect_pts
                cv2.drawContours(rasters_high_res_channels[1 + total_road_types + agent_type * 9 + i], [rect_pts_high_res], -1, (255, 255, 255), -1)
                # draw on low resolution
                rect_pts_low_res = int(low_res_raster_scale) * rect_pts
                cv2.drawContours(rasters_low_res_channels[1 + total_road_types + agent_type * 9 + i], [rect_pts_low_res], -1, (255, 255, 255), -1)
        
        for i, pose in enumerate(ego_trajectory):
            agent_type = 7 # type EGO is 7
            pose -= ego_pose
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                                pose[0] * sin_ + pose[1] * cos_]
            rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
                            (shape[0], shape[1]), pose[2]))
            rect_pts = np.int0(rect_pts)
            rect_pts_high_res = int(high_res_raster_scale) * rect_pts
            cv2.drawContours(rasters_high_res_channels[1 + total_road_types + agent_type * 9 + i], [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = int(low_res_raster_scale) * rect_pts
            cv2.drawContours(rasters_low_res_channels[1 + total_road_types + agent_type * 9 + i], [rect_pts_low_res], -1, (255, 255, 255), -1)

        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

        # context_actions computation
        context_actions = list()
        for i in range(len(ego_trajectory) - 1):
            action = ego_trajectory[i + 1] - ego_trajectory[i]
            action = np.insert(action, 2, 0)
            context_actions.append(action)

        return rasters_high_res, rasters_low_res, np.array(context_actions, dtype=np.float32)
    
def get_road_dict(map_api, ego_pose_center):
        road_dic = {}
        traffic_dic = {}
        all_map_obj = map_api.get_available_map_objects()

        # Collect lane information, following nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils.get_neighbor_vector_map

        # currently NuPlan only supports these map obj classes
        selected_objs = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        selected_objs += [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        selected_objs += [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK]
        selected_objs += [SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]

        all_selected_map_instances = map_api.get_proximal_map_objects(ego_pose_center, 999999,
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
                    if selected_obj.stop_line_type not in [0, 1]:
                        continue
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
                        road_dir_np[i, 0] = util.get_angle_of_a_line(pt1=[road_xy_np[i-1, 0], road_xy_np[i-1, 1]],
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
    from nuplan.common.actor_state.state_representation import Point2D
    import interactive_sim.envs.util as util
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
    planner.compute_planner_trajectory(input)

    
