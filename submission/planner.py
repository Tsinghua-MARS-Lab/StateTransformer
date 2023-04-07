from typing import List, Type

import numpy as np
import numpy.typing as npt
import cv2, math, pickle
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters, VehicleParameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import  InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel


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
        self.model = None
    
    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        pass

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
        agents = [history.observation_buffer[i].tracked_objects.get_agents() for i in range(context_length)] 
        statics = [history.observation_buffer[i].tracked_objects.get_static_objects() for i in range(context_length)]
        result = self.compute_raster_input(ego_trajectory, agents, statics)

    def compute_raster_input(self, ego_trajectory, agents_seq, statics_seq, map=None, max_dis=500):
        """
        the first dimension is the sequence length, each timestep include n-items.
        agent_seq and statics_seq are both agents in raster definition
        """
        ego_pose = ego_trajectory[-1] # (x, y, yaw) in current timestamp
        cos_, sin_ = math.cos(ego_pose[2]), math.sin(ego_pose[2])

        ## 
        total_road_types = 20
        total_agent_types = 8
        high_res_raster_scale = 4
        low_res_raster_scale = 0.77

        total_raster_channels = 1 + total_road_types + total_agent_types * 9
        rasters_high_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_low_res = np.zeros([224, 224, total_raster_channels], dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        
        # road element computation
        
        
        # agent element computation
        ## statics include CZONE_SIGN,BARRIER,TRAFFIC_CONE,GENERIC_OBJECT,
        # TODO:merge the statics and agents
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
        
        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

        return rasters_high_res, rasters_low_res

if __name__ == "__main__":
    with open("history.pkl", "rb") as f:
        input = pickle.load(f)

    planner = ControlTFPlanner(10, 0.1, np.array([5, 5]))
    planner.compute_planner_trajectory(input)

