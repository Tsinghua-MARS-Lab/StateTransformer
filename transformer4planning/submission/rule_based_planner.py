import logging
import pickle
from copy import deepcopy
from typing import List, Type

import random
import math
import numpy as np
import numpy.typing as npt
import time

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from transformer4planning.utils import *
from transformer4planning.libs.classes import SudoInterpolator, GoalSetter, Agent
import transformer4planning.libs.plan_helper as plan_helper

current_frame_idx = 20

"""
Settings for the rule-based planner
"""
S0 = 3
T = 0.25  # 1.5  # reaction time when following
DELTA = 4  # the power term in IDM
PLANNING_HORIZON = 5  # in frames
PREDICTION_HTZ = 10  # prediction_htz
T_HEADWAY = 0.2
A_SPEEDUP_DESIRE = 0.3  # A
A_SLOWDOWN_DESIRE = 1.5  # B
XPT_SHRESHOLD = 0.7
MINIMAL_DISTANCE_PER_STEP = 0.05
MINIMAL_DISTANCE_TO_TRAVEL = 4
# MINIMAL_DISTANCE_TO_RESCALE = -999 #0.1
REACTION_AFTER = 200  # in frames
MINIMAL_SCALE = 0.3
MAX_DEVIATION_FOR_PREDICTION = 4
TRAFFIC_LIGHT_COLLISION_SIZE = 2

MINIMAL_SPEED_TO_TRACK_ORG_GOAL = 5
MINIMAL_DISTANCE_TO_GOAL = 15
PRINT_TIMER = False

DEFAULT_SPEED = 75  # in mph
constant_v = False


def mph_to_meterpersecond(mph):
    return mph * 0.4472222222


def get_angle(x, y):
    return math.atan2(y, x)


def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle


def calculate_yaw_from_states(trajectory, default_yaw):
    time_frames, _ = trajectory.shape
    pred_yaw = np.zeros([time_frames])
    for i in range(time_frames - 1):
        pose_p = trajectory[i + 1]
        pose = trajectory[i]
        delta_x = pose_p[0] - pose[0]
        delta_y = pose_p[1] - pose[1]
        dis = np.sqrt(delta_x * delta_x + delta_y * delta_y)
        if dis > 1:
            angel = get_angle(delta_x, delta_y)
            pred_yaw[i] = angel
            default_yaw = angel
        else:
            pred_yaw[i] = default_yaw
    return pred_yaw


def change_axis(yaw):
    return - yaw - math.pi / 2


def find_closest_lane(current_state, my_current_pose,
                      ignore_intersection_lane=False,
                      include_unparallel=True,
                      selected_lanes=[],
                      valid_lane_types=[1, 2],
                      excluded_lanes=[]):
    """
    :param current_state: extract lanes from it
    :param my_current_pose: current pose for searching
    :param selected_lanes: only search lanes in this list and ignore others
    :param include_unparallel: return lanes without yaw difference checking
    :param ignore_intersection_lane: ignore lanes in an intersection, not implemented yet
    """
    # find a closest lane for a state
    closest_dist = 999999
    closest_dist_no_yaw = 999999
    closest_dist_threshold = 5
    closest_lane = None
    closest_lane_no_yaw = None
    closest_lane_pt_no_yaw_idx = None
    closest_lane_pt_idx = None

    current_lane = None
    current_closest_pt_idx = None
    dist_to_lane = None

    for each_lane in current_state['road']:
        if each_lane in excluded_lanes:
            continue
        if len(selected_lanes) > 0 and each_lane not in selected_lanes:
            continue
        if isinstance(current_state['road'][each_lane]['type'], int):
            if current_state['road'][each_lane]['type'] not in valid_lane_types:
                continue
        else:
            if current_state['road'][each_lane]['type'][0] not in valid_lane_types:
                continue
        road_xy = current_state['road'][each_lane]['xyz'][:, :2]
        if road_xy.shape[0] < 3:
            continue
        for j, each_xy in enumerate(road_xy):
            road_yaw = current_state['road'][each_lane]['dir'][j]
            dist = euclidean_distance(each_xy, my_current_pose[:2])
            yaw_diff = abs(normalize_angle(my_current_pose[3] - road_yaw))
            if dist < closest_dist_no_yaw:
                closest_lane_no_yaw = each_lane
                closest_dist_no_yaw = dist
                closest_lane_pt_no_yaw_idx = j
            if yaw_diff < math.pi / 180 * 20 and dist < closest_dist_threshold:
                if dist < closest_dist:
                    closest_lane = each_lane
                    closest_dist = dist
                    closest_lane_pt_idx = j

    if closest_lane is not None:
        current_lane = closest_lane
        current_closest_pt_idx = closest_lane_pt_idx
        dist_to_lane = closest_dist
        # distance_threshold = max(7, max(7 * my_current_v_per_step, dist_to_lane))
    elif closest_lane_no_yaw is not None and include_unparallel:
        current_lane = closest_lane_no_yaw
        current_closest_pt_idx = closest_lane_pt_no_yaw_idx
        dist_to_lane = closest_dist_no_yaw
        # distance_threshold = max(10, dist_to_lane)
    # else:
    #     logging.warning(f'No current lane founded: {agent_id}')
    # return
    return current_lane, current_closest_pt_idx, dist_to_lane


class RuleBasedPlanner(AbstractPlanner):
    """
    Planner without Learning Methods
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 target_velocity=None,
                 min_gap_to_lead_agent=None,
                 steering_angle: float = 0.0,
                 **kwargs):

        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.samping_time = TimePoint(int(sampling_time * 1e6))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle

        # parameters for rule-based planner
        self.planning_horizon = 160  # horizon_seconds
        self.scenario_frame_number = 0
        # TODO: get rid of the predictor
        self.method_testing = 2
        self.test_task = 2
        self.all_relevant = False
        self.follow_loaded_relation = False
        self.predict_relations_for_env = False
        self.target_lanes = [0, 0]  # lane_index, point_index
        self.routed_traj = {}
        self.frame_rate = 20
        self.current_on_road = True
        self.dataset = 'NuPlan'
        self.valid_lane_types = [1, 2] if self.dataset == 'Waymo' else [0, 11]
        self.vehicle_types = [1] if self.dataset == 'Waymo' else [0, 7]
        self.past_lanes = {}
        self.navigator = None
        self.current_lane = None
        self.previous_lanes_in_path = []

        self.dbg = {}
        self.available_routes = {}
        self.goal_setter = GoalSetter()
        self.predict_env_for_ego_collisions = True
        self.planning_to = 0
        self.predict_relations_for_ego = True

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
        global current_frame_idx
        current_frame_idx += 1
        print("current_frame_idx: ", current_frame_idx)
        history = current_input
        ego_states = history.ego_state_buffer  # a list of ego trajectory
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

        agent_dic = get_agent_dict(ego_states, agents, statics)
        current_state = {
            'agent': agent_dic,
            'road': road_dic,
            'route': self.route_roadblock_ids
        }

        # load scenario data
        ego_agent_id = 'ego'
        my_current_pose = current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1]
        my_current_v_per_step = euclidean_distance(
            current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, :2],
            current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 6, :2]) / 5 / 2 # 2 for frame rate change
        total_time_frame = current_state['agent'][ego_agent_id]['pose'].shape[0]
        my_target_speed = DEFAULT_SPEED / self.frame_rate  # change this to the speed limit of the current lane

        # TODO: connect goal info and deal with a goal of None
        goal_pt, goal_yaw = ((self.goal.x, self.goal.y), self.goal.heading) if self.goal is not None else ((0, 0), 0)

        goal_lane, _, _ = find_closest_lane(
            current_state=current_state,
            my_current_pose=[goal_pt[0], goal_pt[1], -1, goal_yaw],
            valid_lane_types=self.valid_lane_types,
        )

        my_interpolators, my_interpolated_marginal_trajectories, current_routes = self.plan_marginal_trajectories(
            current_state=current_state,
            current_frame_idx=current_frame_idx,
            ego_agent_id=ego_agent_id,
            my_current_pose=my_current_pose,
            my_current_v_per_step=my_current_v_per_step
        )

        # deal with interactions
        # 1. make predictions
        other_agent_traj, other_agent_ids, prior_agent_traj, prior_agent_ids = self.make_predictions(
            current_state=current_state,
            current_frame_idx=current_frame_idx,
            ego_agent_id=ego_agent_id)

        self.dbg["plan_ego/other_agent_ids"] = other_agent_ids
        self.dbg["plan_ego/other_agent_traj"] = other_agent_traj
        self.dbg["plan_ego/prior_agent_ids"] = prior_agent_ids
        self.dbg["plan_ego/prior_agent_traj"] = prior_agent_traj

        # ego_org_traj = my_interpolated_trajectory

        def check_traffic_light(ego_org_traj):
            total_time_frame = self.planning_horizon
            for current_time in range(total_time_frame):
                if current_frame_idx + current_time < 90:
                    traffic_light_ending_pts = self.get_traffic_light_collision_pts(current_state=current_state,
                                                                                    current_frame_idx=current_frame_idx + min(
                                                                                        5, current_time))
                else:
                    traffic_light_ending_pts = []
                ego_pose = ego_org_traj[current_time]
                if abs(ego_pose[0]) < 1.1 and abs(ego_pose[1]) < 1.1:
                    continue
                ego_agent = Agent(x=ego_pose[0],
                                  y=ego_pose[1],
                                  yaw=ego_pose[3],
                                  length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                  width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                  agent_id=ego_agent_id)

                # check if ego agent is running a red light
                if abs(ego_org_traj[-1, 0] + 1) < 0.01 or abs(ego_org_traj[0, 0] + 1) < 0.01:
                    ego_dist = 0
                else:
                    ego_dist = euclidean_distance(ego_org_traj[-1, :2], ego_org_traj[0, :2])
                if abs(ego_org_traj[60, 3] + 1) < 0.01:
                    ego_turning_right = False
                else:
                    ego_yaw_diff = -normalize_angle(ego_org_traj[60, 3] - ego_org_traj[0, 3])
                    ego_running_red_light = False
                    if math.pi / 180 * 15 < ego_yaw_diff and abs(ego_org_traj[60, 3] + 1) > 0.01:
                        ego_turning_right = True
                    else:
                        ego_turning_right = False

                if not ego_turning_right and ego_dist > 10:
                    for tl_pt in traffic_light_ending_pts:
                        dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                               length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                               width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                               agent_id=99999)
                        running = check_collision(
                            checking_agent=ego_agent,
                            target_agent=dummy_tf_agent)
                        if running:
                            ego_running_red_light = True
                            return current_time
            return None

        def detect_conflicts_and_solve(others_trajectory, target_agent_ids, always_yield=False,
                                       ego_org_traj=my_interpolated_marginal_trajectories[0]):
            total_time_frame = self.planning_horizon
            my_reactors = []
            for current_time in range(total_time_frame):
                if current_frame_idx + current_time < self.planning_to:
                    traffic_light_ending_pts = self.get_traffic_light_collision_pts(current_state=current_state,
                                                                                    current_frame_idx=current_frame_idx + min(
                                                                                        5, current_time))
                else:
                    traffic_light_ending_pts = []
                ego_running_red_light = False
                ego_time_length = ego_org_traj.shape[0]

                if current_time >= ego_time_length:
                    print("break ego length: ", current_time, ego_time_length)
                    break
                ego_pose = ego_org_traj[current_time]
                if ego_pose[0] == -1.0 and ego_pose[1] == -1.0:
                    continue
                ego_agent = Agent(x=ego_pose[0],
                                  y=ego_pose[1],
                                  yaw=ego_pose[3],
                                  length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                  width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                  agent_id=ego_agent_id)
                # check if ego agent is running a red light
                if ego_org_traj[-1, 0] == -1.0 or ego_org_traj[0, 0] == -1.0:
                    ego_dist = 0
                else:
                    ego_dist = euclidean_distance(ego_org_traj[-1, :2], ego_org_traj[0, :2])
                if ego_org_traj[-1, 3] == -1.0:
                    ego_turning_right = False
                else:
                    ego_yaw_diff = -normalize_angle(ego_org_traj[-1, 3] - ego_org_traj[0, 3])
                    ego_running_red_light = False
                    if math.pi / 180 * 15 < ego_yaw_diff and ego_org_traj[-1, 3] != -1.0:
                        ego_turning_right = True
                    else:
                        ego_turning_right = False
                if not ego_turning_right and ego_dist > 10:
                    for tl_pt in traffic_light_ending_pts:
                        dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                               length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                               width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                               agent_id=99999)
                        running = check_collision(
                            checking_agent=ego_agent,
                            target_agent=dummy_tf_agent)
                        if running:
                            ego_running_red_light = True
                            break

                if ego_running_red_light:
                    earliest_collision_idx = current_time
                    collision_point = ego_org_traj[current_time, :2]
                    earliest_conflict_agent = 99999
                    target_speed = 0
                    each_other_traj, detected_relation = None, None
                    return [earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, None,
                            None]

                for j, each_other_traj in enumerate(others_trajectory):
                    target_agent_id = target_agent_ids[j]

                    # Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
                    target_type = int(current_state['agent'][target_agent_id]['type'])
                    if target_type not in self.vehicle_types:
                        target_shape = [max(2, current_state['agent'][target_agent_id]['shape'][0][0]),
                                        max(6, current_state['agent'][target_agent_id]['shape'][0][1])]
                    else:
                        target_shape = [max(1, current_state['agent'][target_agent_id]['shape'][0][0]),
                                        max(1, current_state['agent'][target_agent_id]['shape'][0][1])]

                    if target_agent_id in my_reactors:
                        continue
                    total_frame_in_target = each_other_traj.shape[0]
                    if current_time > total_frame_in_target - 1:
                        continue
                    target_pose = each_other_traj[current_time]
                    if target_pose[0] == -1.0 or target_pose[1] == -1.0:
                        continue

                    # check if target agent is running a red light
                    yaw_diff = normalize_angle(each_other_traj[-1, 3] - each_other_traj[0, 3])
                    dist = euclidean_distance(each_other_traj[-1, :2], each_other_traj[0, :2])
                    target_agent = Agent(x=target_pose[0],
                                         y=target_pose[1],
                                         yaw=target_pose[3],
                                         length=target_shape[1],
                                         width=target_shape[0],
                                         agent_id=target_agent_id)
                    # check target agent is stopping for a red light
                    running_red_light = False
                    if dist > 10:
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                   length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   agent_id=99999)
                            running = check_collision(
                                checking_agent=target_agent,
                                target_agent=dummy_tf_agent)
                            if running:
                                # check if they are on two sides of the red light
                                ego_tf_yaw = get_angle_of_a_line(pt1=ego_pose[:2], pt2=tl_pt[:2])
                                target_tf_yae = get_angle_of_a_line(pt1=target_pose[:2], pt2=tl_pt[:2])
                                if abs(normalize_angle(ego_tf_yaw - target_tf_yae)) < math.pi / 2:
                                    running_red_light = True
                                    break

                    if running_red_light and not constant_v:
                        continue

                    # check collision with ego vehicle
                    has_collision = check_collision(checking_agent=ego_agent,
                                                          target_agent=target_agent)

                    if current_time < ego_time_length - 1:
                        ego_pose2 = ego_org_traj[current_time + 1]
                        ego_agent2 = Agent(x=(ego_pose2[0] + ego_pose[0]) / 2,
                                           y=(ego_pose2[1] + ego_pose[1]) / 2,
                                           yaw=get_angle_of_a_line(ego_pose[:2], ego_pose2[:2]),
                                           length=max(2, euclidean_distance(ego_pose2[:2], ego_pose[:2])),
                                           width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                           agent_id=ego_agent_id)
                        if current_time < total_time_frame - 1:
                            target_pose2 = each_other_traj[current_time + 1]
                            target_agent2 = Agent(x=target_pose2[0],
                                                  y=target_pose2[1],
                                                  yaw=target_pose2[3],
                                                  length=target_shape[1],
                                                  width=target_shape[0],
                                                  agent_id=target_agent_id)
                            has_collision |= check_collision(checking_agent=ego_agent2,
                                                                   target_agent=target_agent2)
                        else:
                            has_collision |= check_collision(checking_agent=ego_agent2,
                                                                   target_agent=target_agent)

                    if has_collision:
                        if not always_yield:
                            # FORWARD COLLISION CHECKINGS
                            target_pose_0 = each_other_traj[0]
                            target_agent_0 = Agent(x=target_pose_0[0],
                                                   y=target_pose_0[1],
                                                   yaw=target_pose_0[3],
                                                   length=target_shape[1],
                                                   width=target_shape[0],
                                                   agent_id=target_agent_id)
                            collision_0 = False
                            for fcc_time in range(total_time_frame):
                                ego_pose = ego_org_traj[fcc_time]
                                if ego_pose[0] == -1.0 and ego_pose[1] == -1.0:
                                    continue
                                ego_agent = Agent(x=ego_pose[0],
                                                  y=ego_pose[1],
                                                  yaw=ego_pose[3],
                                                  length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                                  width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                                  agent_id=ego_agent_id)

                                collision_0 |= check_collision(ego_agent, target_agent_0)
                                if collision_0:
                                    break

                            ego_pose_0 = ego_org_traj[0]
                            ego_agent_0 = Agent(x=ego_pose_0[0],
                                                y=ego_pose_0[1],
                                                yaw=ego_pose_0[3],
                                                length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                                width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                                agent_id=ego_agent_id)
                            collision_1 = False
                            for fcc_time in range(total_time_frame):
                                target_pose = each_other_traj[fcc_time]
                                if target_pose[0] == -1.0 or target_pose[1] == -1.0:
                                    continue
                                target_agent = Agent(x=target_pose[0],
                                                     y=target_pose[1],
                                                     yaw=target_pose[3],
                                                     length=target_shape[1],
                                                     width=target_shape[0],
                                                     agent_id=target_agent_id)

                                collision_1 |= check_collision(target_agent, ego_agent_0)
                                if collision_1:
                                    break
                            # collision_1 = check_collision(target_agent, ego_agent_0)
                            # collision_1 |= check_collision(target_agent2, ego_agent_0)

                            if collision_0 and self.predict_with_rules:
                                # yield
                                detected_relation = [[target_agent_id, ego_agent_id, 'FCC']]
                            elif collision_1 and self.predict_with_rules:
                                # pass
                                my_reactors.append(target_agent_id)
                                continue
                            else:
                                # check relation (MOCK WITHOUT PREDICTION)
                                # if collision, solve conflict
                                detected_relation = []
                                if [ego_agent_id, target_agent_id] in detected_relation:
                                    if [target_agent_id, ego_agent_id] in detected_relation:
                                        # bi-directional relations, still yield
                                        pass
                                    else:
                                        # not to yield, and skip conflict
                                        my_reactors.append(target_agent_id)
                                        continue
                        else:
                            detected_relation = [[target_agent_id, ego_agent_id, 'Prior']]

                        copy = []
                        for each_r in detected_relation:
                            if len(each_r) == 2:
                                copy.append([each_r[0], each_r[1], 'predict'])
                            else:
                                copy.append(each_r)
                        detected_relation = copy

                        earliest_collision_idx = current_time
                        collision_point = ego_org_traj[current_time, :2]
                        earliest_conflict_agent = target_agent_id

                        if total_frame_in_target - current_time > 5:
                            target_speed = euclidean_distance(each_other_traj[current_time, :2],
                                                              each_other_traj[current_time + 5, :2]) / 5
                        elif current_time > 5:
                            target_speed = euclidean_distance(each_other_traj[current_time - 5, :2],
                                                              each_other_traj[current_time, :2]) / 5
                        else:
                            target_speed = 0
                        return [earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed,
                                each_other_traj, detected_relation]

            return None

        earliest_collision_idx = None
        collision_point = None
        earliest_conflict_agent = None
        target_speed = None
        detected_relation = None

        tf_light_frame_idx = None
        # tf_light_frame_idx = check_traffic_light(my_interpolated_marginal_trajectories[0])

        # process prior collision pairs
        progress_for_all_traj = []
        trajectory_to_mark = []
        interpolators = []
        rsts_to_yield = []
        routes_to_yield = []
        target_length = []
        to_yield = []
        has_goal_index = None

        for i, each_route in enumerate(current_routes):
            if goal_lane in each_route:
                has_goal_index = i
                break

        for i, each_traj in enumerate(my_interpolated_marginal_trajectories):
            my_interpolator = my_interpolators[i]
            selected_route = current_routes[i]
            traj_to_mark_this_traj = None
            to_yield_this_traj = False
            rst = detect_conflicts_and_solve(others_trajectory=prior_agent_traj,
                                             target_agent_ids=prior_agent_ids,
                                             always_yield=True, ego_org_traj=each_traj)
            if rst is not None and rst[5] is not None:
                earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, each_other_traj, detected_relation = rst
                if each_other_traj is not None:
                    traj_to_mark_this_traj = each_other_traj
                to_yield_this_traj = True
            # check collisions with not prior collisions
            rst = detect_conflicts_and_solve(other_agent_traj, other_agent_ids,
                                             always_yield=(not self.predict_relations_for_ego),
                                             ego_org_traj=each_traj)
            if rst is not None and len(rst) == 6:
                if to_yield_this_traj:
                    if rst[0] < earliest_collision_idx:
                        consider = True
                    else:
                        consider = False
                else:
                    consider = True
                    to_yield_this_traj = True
                if consider:
                    earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, each_other_traj, detected_relation = rst
                    if each_other_traj is not None:
                        traj_to_mark_this_traj = each_other_traj

            if not to_yield_this_traj:
                # then take current trajectory
                progress_for_all_traj.append(-1)
                trajectory_to_mark.append(None)
                rsts_to_yield.append(None)
                interpolators.append(None)
                routes_to_yield.append(None)
                target_length.append(None)
                to_yield.append(False)
            else:
                progress_for_all_traj.append(earliest_collision_idx)
                # mark other's trajectory to yield
                trajectory_to_mark.append(traj_to_mark_this_traj)
                rsts_to_yield.append(
                    [earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed,
                     each_other_traj, detected_relation])
                interpolators.append(my_interpolator)
                routes_to_yield.append(selected_route)
                if earliest_conflict_agent == 99999:
                    # shape for traffic light
                    target_length.append(2)
                else:
                    target_shapes = current_state['agent'][earliest_conflict_agent]['shape']
                    if len(target_shapes.shape) == 2:
                        if target_shapes.shape[0] == 1:
                            target_length.append(target_shapes[0, 1])
                        else:
                            try:
                                target_length.append(target_shapes[earliest_collision_idx, 1])
                            except:
                                print("Unknown shape size: ", target_shapes.shape)
                                target_length.append(target_shapes[0, 1])
                    else:
                        target_length.append(target_shapes[1])
                to_yield.append(True)

        ego_to_yield = False
        if has_goal_index is not None:
            # choose goal route
            index_to_select = has_goal_index
            ego_to_yield |= to_yield[has_goal_index]
        else:
            any_not_yield = False
            not_yield_index = None
            for i, each_yield in enumerate(to_yield):
                any_not_yield |= not each_yield
                if any_not_yield:
                    not_yield_index = i
                    break
            if any_not_yield:
                ego_to_yield = False
                index_to_select = not_yield_index
            else:
                # all yields, choose furtheest one
                ego_to_yield = True
                index_to_select = progress_for_all_traj.index(max(progress_for_all_traj))

        if ego_to_yield:
            earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, each_other_traj, detected_relation = \
                rsts_to_yield[index_to_select]
            my_interpolator = interpolators[index_to_select]
            my_traj = my_interpolated_marginal_trajectories[index_to_select]
            selected_route = routes_to_yield[index_to_select]
            S0 = target_length[index_to_select] / 2 * 1.5
        else:
            # my_interpolators = interpolators[index_to_select]
            my_traj = my_interpolated_marginal_trajectories[index_to_select]
            selected_route = current_routes[index_to_select]

        if len(selected_route) > 0:
            current_lane = selected_route[0]
            current_lane_speed_limit = current_state['road'][current_lane]['speed_limit'] if current_lane in \
                                                                                             current_state[
                                                                                                 'road'] and 'speed_limit' in \
                                                                                             current_state['road'][
                                                                                                 current_lane] else None
            if current_lane_speed_limit is not None:
                my_target_speed = mph_to_meterpersecond(current_lane_speed_limit) / self.frame_rate

        if earliest_collision_idx is not None and (
                tf_light_frame_idx is None or earliest_collision_idx < tf_light_frame_idx):
            # data to save
            if detected_relation is not None:
                if 'relations_per_frame_ego' not in current_state:
                    current_state['relations_per_frame_ego'] = {}
                for dt in range(self.planning_interval):
                    if (current_frame_idx + dt) not in current_state['relations_per_frame_ego']:
                        current_state['relations_per_frame_ego'][current_frame_idx + dt] = []
                    current_state['relations_per_frame_ego'][current_frame_idx + dt] += detected_relation
        elif tf_light_frame_idx is not None:
            earliest_collision_idx = tf_light_frame_idx
            # collision_point = ego_org_traj[earliest_collision_idx, :2]
            earliest_conflict_agent = 99999
            target_speed = 0
            detected_relation = None
            ego_to_yield = True

        if ego_to_yield:
            distance_to_minuse = S0
            if earliest_conflict_agent == 99999:
                distance_to_minuse = 0.1
            distance_to_travel = my_interpolator.get_distance_with_index(
                earliest_collision_idx) - distance_to_minuse
            stopping_point = my_interpolator.interpolate(distance_to_travel - distance_to_minuse)[:2]
            if distance_to_travel < MINIMAL_DISTANCE_PER_STEP:
                # if euclidean_distance(my_traj[0, :2],
                #                       stopping_point) < MINIMAL_DISTANCE_TO_TRAVEL or distance_to_travel < MINIMAL_DISTANCE_TO_TRAVEL or my_current_v_per_step < 0.1:
                planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                    my_current_speed=my_current_v_per_step,
                                                                    desired_speed=my_target_speed,
                                                                    emergency_stop=True)
                # planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                #                                           reactor_traj=my_traj,
                #                                           reactor_interpolator=my_interpolator,
                #                                           scale=scale,
                #                                           current_v_per_step=my_current_v_per_step,
                #                                           target_speed=my_target_speed)
            elif my_current_v_per_step < 1 / self.frame_rate and euclidean_distance(my_traj[0, :2], my_traj[-1,
                                                                                                    :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                    my_current_speed=my_current_v_per_step,
                                                                    desired_speed=my_target_speed,
                                                                    hold_still=True)
            else:
                planed_traj = self.adjust_speed_for_collision(interpolator=my_interpolator,
                                                              distance_to_end=distance_to_travel,
                                                              current_v=my_current_v_per_step,
                                                              end_point_v=min(my_current_v_per_step * 0.8,
                                                                              target_speed))
                assert len(planed_traj.shape) > 1, planed_traj.shape
                # my_interpolator = SudoInterpolator(my_traj, my_current_pose)
                # planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                #                                           reactor_traj=my_traj,
                #                                           reactor_interpolator=my_interpolator,
                #                                           scale=1,
                #                                           current_v_per_step=my_current_v_per_step,
                #                                           target_speed=my_target_speed)
        else:
            if euclidean_distance(my_traj[0, :2], my_traj[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                    my_current_speed=my_current_v_per_step,
                                                                    desired_speed=my_target_speed,
                                                                    hold_still=True)
            else:
                planed_traj = my_traj

        if planed_traj.shape[0] < self.planning_horizon:
            planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                my_current_speed=my_current_v_per_step,
                                                                desired_speed=my_target_speed,
                                                                hold_still=True)
        assert planed_traj.shape[0] >= self.planning_horizon, planed_traj.shape
        pred_traj = planed_traj[:self.planning_horizon, :]

        # # rotate pred_traj
        # rot_eye = np.eye(pred_traj.shape[1])
        # rot = np.array(
        #     [[np.cos(ego_trajectory[-1][2]), -np.sin(ego_trajectory[-1][2])],
        #      [np.sin(ego_trajectory[-1][2]), np.cos(ego_trajectory[-1][2])]]
        # )
        # rot_eye[:2, :2] = rot
        # pred_traj = rot_eye @ pred_traj.T
        # pred_traj = pred_traj.T

        # build output
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
            time_point=ego_state.time_point
        )
        trajectory: List[EgoState] = [state]
        for i in range(0, pred_traj.shape[0] - 1):
            new_time_point = TimePoint(state.time_point.time_us + 1e5)
            state = EgoState.build_from_center(
                center=StateSE2(pred_traj[i + 1, 0] + ego_trajectory[-1][0],
                                pred_traj[i + 1, 1] + ego_trajectory[-1][1],
                                ego_trajectory[-1][2]),
                center_velocity_2d=StateVector2D((pred_traj[i + 1, 0] - pred_traj[i, 0]) / 0.1,
                                                 (pred_traj[i + 1, 1] - pred_traj[i, 1]) / 0.1),
                center_acceleration_2d=StateVector2D(0, 0),
                tire_steering_angle=state.tire_steering_angle,
                time_point=new_time_point,
                vehicle_parameters=state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=state.dynamic_car_state.angular_velocity,
                angular_accel=state.dynamic_car_state.angular_acceleration
            )
            trajectory.append(state)
        return InterpolatedTrajectory(trajectory[::2])

    def get_trajectory_from_interpolator(self, my_interpolator, my_current_speed, a_per_step=None,
                                         check_turning_dynamics=True, desired_speed=31,  # desired_speed in meters per second
                                         emergency_stop=False, hold_still=False,
                                         agent_id=None, a_scale_turning=0.7, a_scale_not_turning=0.9):
        total_frames = int(self.planning_horizon)
        total_pts_in_interpolator = my_interpolator.trajectory.shape[0]
        trajectory = np.ones((total_frames, 4)) * -1
        # get proper speed for turning
        largest_yaw_change = -1
        largest_yaw_change_idx = None
        if check_turning_dynamics and not emergency_stop:
            for i in range(min(200, total_pts_in_interpolator - 2)):
                if my_interpolator.trajectory[i, 0] == -1.0 or my_interpolator.trajectory[i+1, 0] == -1.0 or my_interpolator.trajectory[i+2, 0] == -1.0:
                    continue
                current_yaw = normalize_angle(get_angle_of_a_line(pt1=my_interpolator.trajectory[i, :2], pt2=my_interpolator.trajectory[i+1, :2]))
                next_yaw = normalize_angle(get_angle_of_a_line(pt1=my_interpolator.trajectory[i+1, :2], pt2=my_interpolator.trajectory[i+2, :2]))
                dist = euclidean_distance(pt1=my_interpolator.trajectory[i, :2], pt2=my_interpolator.trajectory[i+1, :2])
                yaw_diff = abs(normalize_angle(next_yaw - current_yaw))
                if yaw_diff > largest_yaw_change and 0.04 < yaw_diff < math.pi / 2 * 0.9 and 100 > dist > 0.3:
                    largest_yaw_change = yaw_diff
                    largest_yaw_change_idx = i
            proper_speed_minimal = max(5, math.pi / 3 / largest_yaw_change)  # calculate based on 20m/s turning for 12s a whole round with a 10hz data in m/s
            proper_speed_minimal_per_frame = proper_speed_minimal / self.frame_rate
            if largest_yaw_change_idx is not None:
                deceleration_frames = max(0, largest_yaw_change_idx - abs(my_current_speed - proper_speed_minimal_per_frame) / (A_SLOWDOWN_DESIRE / self.frame_rate / self.frame_rate / 2))
            else:
                deceleration_frames = 99999
        if agent_id is not None:
            pass
        dist_past = 0
        current_speed = my_current_speed
        for i in range(total_frames):
            if current_speed < 0.1:
                low_speed_a_scale = 1 * self.frame_rate
            else:
                low_speed_a_scale = 0.1 * self.frame_rate
            if hold_still:
                trajectory[i] = my_interpolator.interpolate(0)
                continue
            elif emergency_stop:
                current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate
            elif largest_yaw_change_idx is not None:
                proper_speed_minimal_per_frame = max(0.5, min(proper_speed_minimal_per_frame, 5))
                if largest_yaw_change_idx >= i >= deceleration_frames:
                    if current_speed > proper_speed_minimal_per_frame:
                        current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate / 2
                    else:
                        current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_not_turning * low_speed_a_scale
                elif i < deceleration_frames:
                    if current_speed < desired_speed / self.frame_rate:
                        # if far away from the turnings and current speed is smaller than 15m/s, then speed up
                        # else keep current speed
                        if a_per_step is not None:
                            current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                        else:
                            current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_turning * low_speed_a_scale
                elif i > largest_yaw_change_idx:
                    if current_speed > proper_speed_minimal_per_frame:
                        current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate
                    else:
                        if a_per_step is not None:
                            current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                        else:
                            current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_turning * low_speed_a_scale
            else:
                if current_speed < desired_speed / self.frame_rate:
                    if a_per_step is not None:
                        current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                    else:
                        current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_not_turning * low_speed_a_scale  # accelerate with 0.2 of desired acceleration
            current_speed = max(0, current_speed)
            dist_past += current_speed
            trajectory[i] = my_interpolator.interpolate(dist_past)
        return trajectory

    def set_ego_route(self, current_pose, current_state, goal_pose, route=None, previous_route=None,
                      use_nuplan_route=True, agent_id='ego'):
        road_dic = current_state['road']
        # TODO: Add U-Turn into Search
        # search possible routes from current pose to the goal point
        # 1. if no given route, get the closest road block
        # 2. get all lanes as starting lanes from the current road block
        # 3. get goal lane
        # 4. search for possible road blocks connecting from current road block to goal lane block
        # 5. if no connecting road blocks, connecting current pose with the goal point and throw a warning
        closest_road_block, dist_to_road = self.map_api.get_distance_to_nearest_map_object(
            point=Point2D(current_pose[0], current_pose[1]),
            layer=SemanticMapLayer.ROADBLOCK)
        closest_road_blockc, dist_to_roadc = self.map_api.get_distance_to_nearest_map_object(
            point=Point2D(current_pose[0], current_pose[1]),
            layer=SemanticMapLayer.ROADBLOCK_CONNECTOR)
        closest_road_block = int(closest_road_block) if dist_to_road < dist_to_roadc else int(closest_road_blockc)
        # use given route if available
        if route is not None:
            if closest_road_block in route:
                current_index = route.index(closest_road_block)
                if current_index > 0:
                    return [route[current_index:]], closest_road_block == route[-1]
                else:
                    return [route[1:]], closest_road_block == route[-1]
            else:
                print('Got wrong current block from the map api')
                return [], False
        print('No Route Given, Current Road Block: ', closest_road_block, dist_to_road, dist_to_roadc)
        # if no given route, search for possible routes with goal pose
        if goal_pose is not None:
            # get goal lane and search for possible routes
            goal_lane, _, dist_to_goal_lane = find_closest_lane(
                current_state=current_state,
                my_current_pose=goal_pose,
                valid_lane_types=self.valid_lane_types)
            goal_roads = road_dic[goal_lane]['upper_level']
            max_searching_length = 100
            if closest_road_block not in goal_roads:
                checking_roads = [[closest_road_block]]
                counter = 0
                visited_roads = [closest_road_block]
                while len(checking_roads) > 0:
                    if counter > max_searching_length:
                        break
                    current_route = checking_roads.pop(0)
                    ending_road = current_route[-1]
                    if ending_road in goal_roads:
                        return [current_route], closest_road_block in goal_roads
                    lanes = road_dic[ending_road]['lower_level']
                    for each_lane in lanes:
                        if each_lane not in road_dic:
                            continue
                        next_lanes = road_dic[each_lane]['next_lanes']
                        for each_next_lane in next_lanes:
                            if each_next_lane not in road_dic:
                                continue
                            each_road = road_dic[each_next_lane]['upper_level'][0]
                            if each_road not in visited_roads:
                                checking_roads.append(current_route + [each_road])
                                visited_roads.append(each_road)
                    counter += 1
            print("WARNING: Route not found from current pose to goal point! Using NuPlan's Route.", goal_roads)
            # no route found to goal point, check given route
            org_goal_road = goal_roads
            goal_road = None
            looping_route = []
            if previous_route is not None:
                looping_route = previous_route.copy()
            elif route is not None and len(route) > 10:
                looping_route = route[10:].copy()
            while len(looping_route) > 0:
                current_road = looping_route.pop()
                if current_road in road_dic:
                    goal_road = current_road
                    break
            if goal_road is not None:
                checking_roads = [[closest_road_block]]
                counter = 0
                visited_roads = [closest_road_block]
                while len(checking_roads) > 0:
                    if counter > max_searching_length:
                        break
                    current_route = checking_roads.pop(0)
                    ending_road = current_route[-1]
                    if ending_road == goal_road:
                        return [current_route], org_goal_road == closest_road_block
                    lanes = road_dic[ending_road]['lower_level']
                    for each_lane in lanes:
                        if each_lane not in road_dic:
                            continue
                        next_lanes = road_dic[each_lane]['next_lanes']
                        for each_next_lane in next_lanes:
                            if each_next_lane not in road_dic:
                                continue
                            each_road = road_dic[each_next_lane]['upper_level'][0]
                            if each_road not in visited_roads:
                                checking_roads.append(current_route + [each_road])
                                visited_roads.append(each_road)
                    counter += 1
        else:
            print(f'Empty goal pose for {agent_id}')

        print("WARNING: Route not found from current pose to given NuPlan Route End! ", goal_road,
              goal_road in road_dic)

        if not use_nuplan_route or agent_id != 'ego':
            print("Return an empty list.")
            return [], False
        if agent_id == 'ego':
            print("Using NuPlan's Route.")
            # route not found, try to use given route
            # 1. if given route, loop all lanes in given route
            # 2. find the closest lane in these lanes and its closest road block
            # 3. return that road block and its consecutive blocks in the given route
            lanes_in_routes = []
            for each_road_block in route:
                if each_road_block not in road_dic:
                    continue
                lanes_in_routes += road_dic[each_road_block]['lower_level']
            current_lane, current_closest_pt_idx, dist_to_lane = plan_helper.find_closest_lane(
                current_state=current_state,
                my_current_pose=current_pose,
                selected_lanes=lanes_in_routes,
                valid_lane_types=self.valid_lane_types)
            current_road_block = road_dic[current_lane]['upper_level'][0]
            current_road_index = route.index(current_road_block)
            return [route[current_road_index:]], org_goal_road == closest_road_block
        else:
            return [], False

    def filter_current_speed(self, my_current_v_per_step):
        if my_current_v_per_step > 1 * self.frame_rate:
            my_current_v_per_step = 0.1 * self.frame_rate
        elif my_current_v_per_step < 0.001 * self.frame_rate:
            my_current_v_per_step = 0
        my_current_v_per_step = np.clip(my_current_v_per_step, a_min=0, a_max=7)
        return my_current_v_per_step

    def plan_marginal_trajectories(self, current_state, my_current_pose,
                                   my_current_v_per_step,
                                   current_frame_idx=0,
                                   ego_agent_id=None,
                                   agent_id='ego', ):
        # TODO:
        # 1. get route_road_ids
        # 2. get current_v
        # 3. pack current_state

        my_current_v_per_step = self.filter_current_speed(my_current_v_per_step)

        if PRINT_TIMER:
            last_tic = time.perf_counter()
        if self.goal_setter.data is None:
            # initialize goal setter
            self.goal_setter(new_data=current_state, )
        goal_pt, goal_yaw = self.goal_setter.get_goal(current_data=current_state,
                                                      agent_id=agent_id)
        goal_pose = [goal_pt[0], goal_pt[1], 0, goal_yaw] if goal_pt is not None else None
        if agent_id in self.available_routes and len(self.available_routes[agent_id]) > 0:
            available_routes, current_goal_block = self.set_ego_route(current_pose=my_current_pose,
                                                                      current_state=current_state,
                                                                      previous_route=self.available_routes[agent_id][0],
                                                                      route=current_state['route'],
                                                                      goal_pose=goal_pose,
                                                                      agent_id=agent_id)
        else:
            available_routes, current_goal_block = self.set_ego_route(current_pose=my_current_pose,
                                                                      route=current_state['route'],
                                                                      current_state=current_state,
                                                                      goal_pose=goal_pose,
                                                                      agent_id=agent_id)
            # register route to self not predicting anymore
            self.available_routes[agent_id] = available_routes
        print('Got routes ', available_routes, ' for agent ', agent_id)
        if PRINT_TIMER:
            print(f"Time spent on first lane search:  {time.perf_counter() - last_tic:04f}s")

        interpolators = []
        marginal_trajectories = []
        routes = []

        proper_v_for_cbc = plan_helper.mph_to_meterpersecond(70)

        if len(available_routes) == 0:
            # drive a straight line and stop
            print("WARNING: No Lanes to travel at all")
            routes_in_lanes = []
            yaw = - normalize_angle(my_current_pose[3] + math.pi / 2)
            delta = 50
            x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + my_current_pose[1]
            path = [my_current_pose[:2], [x, y]]
            my_interpolator = SudoInterpolator(np.array(path), my_current_pose)
            my_interpolated_trajectory = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                               my_current_speed=my_current_v_per_step,
                                                                               # in meters per step
                                                                               desired_speed=0,  # in meters per second
                                                                               agent_id=agent_id)
            my_traj = my_interpolated_trajectory[:, :2]
            my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
            interpolators.append(my_interpolator)
            marginal_trajectories.append(my_interpolated_trajectory)
            # routes returned is a list of lanes to travel
            routes.append(routes_in_lanes)
        else:
            for each_available_route in available_routes:
                # each route is a list of roadblocks
                # only consider lane changing in the first block
                if PRINT_TIMER:
                    last_tic = time.perf_counter()

                first_block = each_available_route[0]
                all_lanes_in_block_zero = current_state['road'][first_block]['lower_level'].copy()

                lanes_to_travel = []

                # 1. add lanes to travel for keeping current lane
                current_lane, current_closest_pt_idx, dist_to_lane = plan_helper.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=my_current_pose,
                    selected_lanes=all_lanes_in_block_zero,
                    valid_lane_types=self.valid_lane_types)

                lanes_to_travel.append([current_lane, current_closest_pt_idx, dist_to_lane])

                # 2. add lanes to travel for 2 neighbor lanes
                lanes_to_loop = all_lanes_in_block_zero
                lanes_to_loop.remove(current_lane)
                if len(lanes_to_loop) < 3:
                    # 0, 1, or 2
                    for each_lane in lanes_to_loop:
                        current_lane, current_closest_pt_idx, dist_to_lane = plan_helper.find_closest_lane(
                            current_state=current_state,
                            my_current_pose=my_current_pose,
                            selected_lanes=[each_lane],
                            valid_lane_types=self.valid_lane_types)
                        lanes_to_travel.append([current_lane, current_closest_pt_idx, dist_to_lane])
                else:
                    # 3 or more, add 2 other closest lanes
                    for _ in range(2):
                        current_lane, current_closest_pt_idx, dist_to_lane = plan_helper.find_closest_lane(
                            current_state=current_state,
                            my_current_pose=my_current_pose,
                            selected_lanes=[lanes_to_loop],
                            valid_lane_types=self.valid_lane_types)
                        lanes_to_travel.append([current_lane, current_closest_pt_idx, dist_to_lane])
                        lanes_to_loop.remove(current_lane)

                if current_goal_block:
                    goal_lane, goal_closest_pt_idx, dist_to_goal_lane = plan_helper.find_closest_lane(
                        current_state=current_state,
                        my_current_pose=goal_pose,
                        valid_lane_types=self.valid_lane_types)
                    print("current goal block: ", lanes_to_travel, goal_lane)
                    goal_lane_info = []
                    for each_info in lanes_to_travel:
                        if each_info[0] == goal_lane:
                            goal_lane_info = each_info
                    if len(goal_lane_info) > 0:
                        lanes_to_travel = [goal_lane_info]

                road_dic = current_state['road']
                current_speed_limit = plan_helper.mph_to_meterpersecond(70)

                for each_starting_lane in lanes_to_travel:
                    path = [my_current_pose[:2]]
                    current_lane, current_closest_pt_idx, dist_to_lane = each_starting_lane
                    routes_in_lanes = []
                    for i in range(10):
                        # loop next ten lanes
                        current_lane_length = road_dic[current_lane]['xyz'].shape[0]
                        if current_lane_length - current_closest_pt_idx >= 5:
                            if i == 0:
                                path += road_dic[current_lane]['xyz'][current_closest_pt_idx + 1:, :2].tolist()
                            else:
                                path += road_dic[current_lane]['xyz'][:, :2].tolist()
                            routes_in_lanes.append(current_lane)
                            if road_dic[current_lane]['speed_limit'] is not None:
                                current_speed_limit = plan_helper.mph_to_meterpersecond(
                                    road_dic[current_lane]['speed_limit'])
                            else:
                                current_speed_limit = 31  # 31 meters per hour = 70 miles per hour
                        if i > len(each_available_route) - 2:
                            break
                        next_road_block = each_available_route[i + 1]
                        next_lanes_on_route = road_dic[next_road_block]['lower_level']
                        minimal_dist = 100000
                        closest_next_lane = next_lanes_on_route[0]
                        for each_next_lane in next_lanes_on_route:
                            if each_next_lane in road_dic:
                                starting_pt = road_dic[each_next_lane]['xyz'][0, :2]
                                dist = euclidean_distance(starting_pt, path[-1])
                                if dist < minimal_dist:
                                    minimal_dist = dist
                                    closest_next_lane = each_next_lane

                        current_lane, current_closest_pt_idx, dist_to_lane = plan_helper.find_closest_lane(
                            current_state=current_state,
                            my_current_pose=my_current_pose,
                            selected_lanes=[closest_next_lane],
                            valid_lane_types=self.valid_lane_types)

                    if len(routes_in_lanes) < 5:
                        # if don't have enough lanes before goal, search random lanes after goal to fill up to 5 lanes
                        for i in range(5):
                            next_lanes = road_dic[current_lane]['next_lanes']
                            any_available_lane = False
                            for any_next_lane in next_lanes:
                                if any_next_lane in road_dic:
                                    current_lane = any_next_lane
                                    if current_lane_length - current_closest_pt_idx >= 5 and i == 0:
                                        continue
                                    if not (len(routes_in_lanes) == 0 and i == 0):
                                        path += road_dic[any_next_lane]['xyz'][:, :2].tolist()
                                        routes_in_lanes.append(any_next_lane)
                                        any_available_lane = True
                                        break
                            # if not any_available_lane:
                            #     break

                    proper_v_for_cbc = current_speed_limit
                    if len(routes_in_lanes) == 0:
                        print("WARNING: No Lanes to travel at all")
                        p1 = my_current_pose[:2]
                        yaw = - normalize_angle(my_current_pose[3] + math.pi / 2)
                        delta = 50
                        x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + \
                               my_current_pose[1]
                        p2 = [x, y]
                        p3 = p2
                        x, y = -math.sin(yaw) * delta + p2[0], -math.cos(yaw) * delta + p2[1]
                        p4 = [x, y]
                        connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4,
                                                                        v=my_current_v_per_step * 0.1)
                        path = [path[0]] + connection_traj[:, :2].tolist()
                        proper_v_for_cbc *= 0.1
                    else:
                        current_lane, current_closest_pt_idx, dist_to_lane = plan_helper.find_closest_lane(
                            current_state=current_state,
                            my_current_pose=my_current_pose,
                            selected_lanes=routes_in_lanes,
                            valid_lane_types=self.valid_lane_types)

                        # throw away several first points due to map_api's bugs
                        path = [path[0]] + path[30:]
                        if dist_to_lane > 1:
                            # adding CBC to join
                            advance_index = max(20, int(10 * my_current_v_per_step))
                            counter = 0
                            max_turning_angle = 25
                            max_turning_yaw = math.pi / 180 * max_turning_angle
                            proper_v_for_cbc = my_current_v_per_step * 0.8
                            p1 = my_current_pose[:2]
                            yaw12 = - normalize_angle(my_current_pose[3] + math.pi / 2)
                            delta0 = max(2, int(10 * my_current_v_per_step))
                            while True:
                                skip_connector = False
                                p4_on_connection = True
                                if advance_index >= len(path) - 2:
                                    break
                                p4 = path[1 + advance_index]
                                while p4_on_connection and skip_connector:
                                    closest_road_block, dist_to_road = self.map_api.get_distance_to_nearest_map_object(
                                        point=Point2D(p4[0], p4[1]),
                                        layer=SemanticMapLayer.ROADBLOCK)
                                    closest_road_blockc, dist_to_roadc = self.map_api.get_distance_to_nearest_map_object(
                                        point=Point2D(p4[0], p4[1]),
                                        layer=SemanticMapLayer.ROADBLOCK_CONNECTOR)
                                    if dist_to_road < dist_to_roadc:
                                        p4_on_connection = False
                                    else:
                                        advance_index += 5
                                        if advance_index >= len(path) - 2:
                                            break
                                        p4 = path[1 + advance_index]
                                # yaw34 = normalize_angle(- normalize_angle(get_angle_of_a_line(p4, path[2 + advance_index]) + math.pi / 2) + math.pi)
                                yaw34 = - normalize_angle(
                                    get_angle_of_a_line(p4, path[2 + advance_index]) + math.pi / 2)
                                # yaw34 = -normalize_angle(
                                #     road_dic[current_lane]['dir'][current_closest_pt_idx] + math.pi / 2)
                                delta1 = euclidean_distance(p4, my_current_pose[:2]) / 4

                                x, y = -math.sin(yaw12) * delta0 + my_current_pose[0], -math.cos(yaw12) * delta0 + \
                                       my_current_pose[1]
                                p2 = [x, y]
                                x, y = math.sin(yaw34) * delta1 + p4[0], math.cos(yaw34) * delta1 + p4[1]
                                p3 = [x, y]
                                connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4,
                                                                                v=proper_v_for_cbc)
                                if connection_traj.shape[0] < 10:
                                    advance_index += 5
                                    advance_index = min(len(path) - 1, advance_index)
                                    continue
                                starting_yaw_change = get_angle_of_a_line(connection_traj[5, :2],
                                                                                connection_traj[6,
                                                                                :2]) - get_angle_of_a_line(
                                    connection_traj[0, :2], connection_traj[1, :2])
                                ending_yaw_change = get_angle_of_a_line(connection_traj[-7, :2],
                                                                              connection_traj[-6,
                                                                              :2]) - get_angle_of_a_line(
                                    connection_traj[-2, :2], connection_traj[-1, :2])
                                if abs(starting_yaw_change) > max_turning_yaw:
                                    if delta0 >= euclidean_distance(p1, p4):
                                        print("P1;P2 too far, breaking loop")
                                        # path = [path[0]] + connection_traj[:, :2].tolist() + path[1 + advance_index:]
                                        # break
                                    else:
                                        delta0 += 1
                                if abs(ending_yaw_change) > max_turning_yaw:
                                    advance_index += 10
                                    advance_index = min(len(path) - 1, advance_index)
                                if abs(starting_yaw_change) <= max_turning_yaw and abs(
                                        ending_yaw_change) <= max_turning_yaw:
                                    path = [path[0]] + connection_traj[:, :2].tolist() + path[1 + advance_index:]
                                    break
                                else:
                                    proper_v_for_cbc *= 0.8
                                if counter > 20:
                                    print("not enough iterations for CBC under max turning yaw, ", starting_yaw_change,
                                          ending_yaw_change, max_turning_yaw, advance_index, delta1)
                                    break
                                counter += 1

                    my_interpolator = SudoInterpolator(np.array(path), my_current_pose)
                    my_interpolated_trajectory = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                                       my_current_speed=my_current_v_per_step,
                                                                                       # in meters per step
                                                                                       desired_speed=proper_v_for_cbc,
                                                                                       # in meters per second
                                                                                       agent_id=agent_id)
                    my_traj = my_interpolated_trajectory[:, :2]
                    my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
                    interpolators.append(my_interpolator)
                    marginal_trajectories.append(my_interpolated_trajectory)

                    # add one interpolator for each path (at maximum of 3)
                    # interpolator = SudoInterpolator(trajectory=np.array(path),
                    #                                 current_pose=my_current_pose)
                    # interpolators.append(interpolator)
                    # marginal_trajectory = []
                    #
                    # for t in range(self.planning_horizon):
                    #     marginal_trajectory.append(
                    #         interpolator.interpolate(distance=current_speed_limit / self.frame_rate * t))
                    # marginal_trajectories.append(marginal_trajectory)
                    # routes returned is a list of lanes to travel
                    routes.append(routes_in_lanes)

                if PRINT_TIMER:
                    print(f"Time spent on one path generation:  {time.perf_counter() - last_tic:04f}s")

        # self.dbg["plan_marginal_trajectories/my_trajs"] = my_trajs
        # self.dbg["plan_marginal_trajectories/current_routes"] = routes
        # self.dbg["plan_marginal_trajectories/my_interpolators"] = interpolators
        # self.dbg["plan_marginal_trajectories/my_interpolated_marginal_trajectories"] = marginal_trajectories

        if len(self.previous_lanes_in_path) > 0 and len(routes) > 0:
            better_route_index = 0
            for index, route in enumerate(routes):
                if len(route) > 0 and route[0] == self.previous_lanes_in_path[0]:
                    better_route_index = index
                    break
            if better_route_index != 0:
                interpolator = interpolators.pop(better_route_index)
                marginal_trajectory = marginal_trajectories.pop(better_route_index)
                route = routes.pop(better_route_index)
                interpolators.insert(0, interpolator)
                marginal_trajectories.insert(0, marginal_trajectory)
                routes.insert(0, route)

        assert len(marginal_trajectories) > 0, f'No Available Navigation Paths? {routes}'

        return interpolators, marginal_trajectories, routes
    
    def make_predictions(self, current_state, current_frame_idx, ego_agent_id):
        other_agent_traj = []
        other_agent_ids = []
        # prior for ped and cyclist
        prior_agent_traj = []
        prior_agent_ids = []

        if self.predict_env_for_ego_collisions:
            if False:
                # predict on all agents for detection checking
                self.online_predictor.marginal_predict(current_frame=current_frame_idx, selected_agent_ids='all',
                                                       current_data=current_state)
                for each_agent_id in self.online_predictor.data['predicting']['marginal_trajectory'].keys():
                    if each_agent_id == ego_agent_id:
                        continue
                    # if k = 1
                    k = 6
                    for n in range(k):
                        pred_traj = self.online_predictor.data['predicting']['marginal_trajectory'][each_agent_id]['rst'][n]
                        total_frames_in_pred = pred_traj.shape[0]
                        pred_traj_with_yaw = np.ones((total_frames_in_pred, 4)) * -1
                        pred_traj_with_yaw[:, :2] = pred_traj[:, :]
                        for t in range(total_frames_in_pred):
                            if t == total_frames_in_pred - 1:
                                pred_traj_with_yaw[t, 3] = pred_traj_with_yaw[t - 1, 3]
                            else:
                                pred_traj_with_yaw[t, 3] = get_angle_of_a_line(pt1=pred_traj[t, :2],
                                                                                     pt2=pred_traj[t + 1, :2])
                        other_agent_traj.append(pred_traj_with_yaw)
                        other_agent_ids.append(each_agent_id)
            else:
                # use constant velocity
                for each_agent_id in current_state['agent']:
                    if each_agent_id == ego_agent_id:
                        continue
                    varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                    # varies = [1]
                    for v in varies:
                        delta_x = (current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 0] -
                                   current_state['agent'][each_agent_id]['pose'][current_frame_idx - 6, 0]) / 5
                        delta_y = (current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 1] -
                                   current_state['agent'][each_agent_id]['pose'][current_frame_idx - 6, 1]) / 5
                        pred_traj_with_yaw = np.ones((int(self.planning_horizon), 4)) * -1
                        pred_traj_with_yaw[:, 3] = current_state['agent'][each_agent_id]['pose'][
                            current_frame_idx - 1, 3]
                        for t in range(int(self.planning_horizon)):
                            pred_traj_with_yaw[t, 0] = current_state['agent'][each_agent_id]['pose'][
                                                           current_frame_idx - 1, 0] + t * delta_x * v
                            pred_traj_with_yaw[t, 1] = current_state['agent'][each_agent_id]['pose'][
                                                           current_frame_idx - 1, 1] + t * delta_y * v
                        # always yield with constant v
                        prior_agent_traj.append(pred_traj_with_yaw)
                        prior_agent_ids.append(each_agent_id)
        else:
            for each_agent in current_state['agent']:
                if each_agent == ego_agent_id:
                    continue
                each_agent_pose = current_state['agent'][each_agent]['pose']
                # check distance
                if euclidean_distance(current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, :2],
                                      each_agent_pose[current_frame_idx - 1, :2]) > 500 and \
                        current_state['agent'][ego_agent_id]['pose'][
                            current_frame_idx - 1, 0] != -1:  # 20m for 1 second on 70km/h
                    continue

                # 'predict' its trajectory by following lanes
                if int(current_state['agent'][each_agent]['type']) not in self.vehicle_types:
                    # for pedestrians or bicycles
                    if each_agent_pose[current_frame_idx - 1, 0] == -1.0 or \
                            each_agent_pose[current_frame_idx - 6, 0] == -1.0:
                        continue
                    # for non-vehicle types agent
                    delta_x = (each_agent_pose[current_frame_idx - 1, 0] -
                               each_agent_pose[current_frame_idx - 6, 0]) / 5
                    delta_y = (each_agent_pose[current_frame_idx - 1, 1] -
                               each_agent_pose[current_frame_idx - 6, 1]) / 5
                    if delta_x < 1 and delta_y < 1:
                        traj_with_yaw = np.ones((self.planning_horizon, 4)) * -1
                        traj_with_yaw[:, 3] = each_agent_pose[current_frame_idx - 1, 3]
                        traj_with_yaw[:, :] = each_agent_pose[current_frame_idx, :]
                        other_agent_ids.append(each_agent)
                        other_agent_traj.append(traj_with_yaw)
                        continue
                    else:
                        varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                        predict_horizon = 39  # in frames
                        for mul in varies:
                            traj_with_yaw = np.ones((self.planning_horizon, 4)) * -1
                            traj_with_yaw[:, 3] = each_agent_pose[current_frame_idx - 1, 3]
                            traj_with_yaw[:, :] = each_agent_pose[current_frame_idx, :]
                            for i in range(predict_horizon):
                                # constant v with variations
                                traj_with_yaw[i + 1, 0] = traj_with_yaw[i, 0] + min(0.5, delta_x * mul)
                                traj_with_yaw[i + 1, 1] = traj_with_yaw[i, 1] + min(0.5, delta_y * mul)
                            other_agent_ids.append(each_agent)
                            other_agent_traj.append(traj_with_yaw)
                else:
                    # for vehicles
                    if each_agent_pose[current_frame_idx - 1, 0] == -1.0 or \
                            each_agent_pose[current_frame_idx - 6, 0] == -1.0 or \
                            each_agent_pose[current_frame_idx - 11, 0] == -1.0:
                        continue

                    # for vehicle types agents
                    each_agent_current_pose = each_agent_pose[current_frame_idx - 1]
                    each_agent_current_v_per_step = euclidean_distance(
                        each_agent_pose[current_frame_idx - 1, :2],
                        each_agent_pose[current_frame_idx - 6, :2]) / 5
                    each_agent_current_a_per_step = (euclidean_distance(
                        each_agent_pose[current_frame_idx - 1, :2],
                        each_agent_pose[current_frame_idx - 6, :2]) / 5 - euclidean_distance(
                        each_agent_pose[current_frame_idx - 6, :2],
                        each_agent_pose[current_frame_idx - 11, :2]) / 5) / 5

                    if each_agent_current_v_per_step < 0.05:
                        steady_in_past = True
                        # for static vehicles, skip prediction
                        traj_with_yaw = np.ones((int(self.planning_horizon), 4)) * -1
                        traj_with_yaw[:, 3] = each_agent_pose[current_frame_idx - 1, 3]
                        traj_with_yaw[:, :] = each_agent_pose[current_frame_idx, :]
                        prior_agent_traj.append(traj_with_yaw)
                        prior_agent_ids.append(each_agent)
                        continue


                    if each_agent_current_v_per_step > 1 * self.frame_rate:
                        each_agent_current_v_per_step = 0.1 * self.frame_rate
                    # get the route for each agent, you can use your prediction model here
                    if each_agent_current_v_per_step < 0.025 * self.frame_rate:
                        each_agent_current_v_per_step = 0
                    if each_agent_current_a_per_step > 0.05 * self.frame_rate:
                        each_agent_current_a_per_step = 0.03 * self.frame_rate

                    # 1. find the closest lane
                    current_lane, current_closest_pt_idx, dist_to_lane, _ = self.find_closes_lane(
                        current_state=current_state,
                        agent_id=each_agent,
                        my_current_v_per_step=each_agent_current_v_per_step,
                        my_current_pose=each_agent_current_pose,
                        no_unparallel=False,
                        return_list=False)

                    # detect parking vehicles
                    steady_in_past = euclidean_distance(
                        each_agent_pose[current_frame_idx - 1, :2],
                        each_agent_pose[current_frame_idx - 10, :2]) < 3

                    if each_agent_current_v_per_step < 0.05 and (dist_to_lane is None or dist_to_lane > 2) and steady_in_past:
                        dummy_steady = np.repeat(
                            each_agent_pose[current_frame_idx - 1, :][np.newaxis, :], self.planning_horizon,
                            axis=0)
                        prior_agent_ids.append(each_agent)
                        prior_agent_traj.append(dummy_steady)
                        current_state['agent'][each_agent]['marking'] = "Parking"
                        continue

                    # 2. search all possible route from this lane and add trajectory from the lane following model
                    # random shooting for all possible routes
                    if current_lane in current_state['road'] and 'speed_limit' in current_state['road'][current_lane]:
                        speed_limit = current_state['road'][current_lane]['speed_limit']
                        my_target_speed = speed_limit if speed_limit is not None else mph_to_meterpersecond(DEFAULT_SPEED) / self.frame_rate
                    else:
                        my_target_speed = mph_to_meterpersecond(DEFAULT_SPEED) / self.frame_rate

                    routes = []
                    for _ in range(self.frame_rate):
                        lanes_in_a_route = [current_lane]
                        current_looping = current_lane
                        route_traj_left = np.array(
                            current_state['road'][current_looping]['xyz'][current_closest_pt_idx + self.frame_rate:, :2], ndmin=2)
                        next_lanes = current_state['road'][current_looping]['next_lanes']
                        while len(next_lanes) > 0 and len(lanes_in_a_route) < 5:
                            lanes_in_a_route.append(current_looping)
                            current_looping = random.choice(next_lanes)
                            if current_looping not in current_state['road']:
                                continue
                            next_lanes = current_state['road'][current_looping]['next_lanes']
                            route_traj_left = np.concatenate(
                                (route_traj_left, current_state['road'][current_looping]['xyz'][:, :2]))
                        if lanes_in_a_route not in routes:
                            routes.append(lanes_in_a_route)
                            varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                            for mul in varies:
                                other_interpolator = SudoInterpolator(route_traj_left.copy(), each_agent_current_pose)
                                traj_with_yaw = self.get_trajectory_from_interpolator(
                                    my_interpolator=other_interpolator,
                                    my_current_speed=each_agent_current_v_per_step * mul,
                                    a_per_step=each_agent_current_a_per_step,
                                    desired_speed=my_target_speed,
                                    check_turning_dynamics=False)
                                other_agent_traj.append(traj_with_yaw)
                                other_agent_ids.append(each_agent)
        return other_agent_traj, other_agent_ids, prior_agent_traj, prior_agent_ids
    
    def find_closes_lane(self, current_state, agent_id, my_current_v_per_step, my_current_pose, no_unparallel=False,
                         return_list=False, current_route=[]):
        # find a closest lane to trace
        closest_dist = 999999
        closest_dist_no_yaw = 999999
        closest_dist_threshold = 5
        closest_lane = None
        closest_lane_no_yaw = None
        closest_lane_pt_no_yaw_idx = None
        closest_lane_pt_idx = None

        current_lane = None
        current_closest_pt_idx = None
        dist_to_lane = None
        distance_threshold = None

        closest_lanes_same_dir = []
        closest_lanes_idx_same_dir = []

        for each_lane in current_state['road']:
            if len(current_route) > 0 and each_lane not in current_route:
                continue

            if isinstance(current_state['road'][each_lane]['type'], int):
                if current_state['road'][each_lane]['type'] not in self.valid_lane_types:
                    continue
            else:
                if current_state['road'][each_lane]['type'][0] not in self.valid_lane_types:
                    continue

            road_xy = current_state['road'][each_lane]['xyz'][:, :2]
            if road_xy.shape[0] < 3:
                continue
            current_lane_closest_dist = 999999
            current_lane_closest_idx = None

            for j, each_xy in enumerate(road_xy):
                road_yaw = current_state['road'][each_lane]['dir'][j]
                dist = euclidean_distance(each_xy, my_current_pose[:2])
                yaw_diff = abs(normalize_angle(my_current_pose[3] - road_yaw))
                if dist < closest_dist_no_yaw:
                    closest_lane_no_yaw = each_lane
                    closest_dist_no_yaw = dist
                    closest_lane_pt_no_yaw_idx = j
                if yaw_diff < math.pi / 180 * 20 and dist < closest_dist_threshold:
                    if dist < closest_dist:
                        closest_lane = each_lane
                        closest_dist = dist
                        closest_lane_pt_idx = j
                    if dist < current_lane_closest_dist:
                        current_lane_closest_dist = dist
                        current_lane_closest_idx = j

            # classify current agent as a lane changer or not:
            if my_current_v_per_step > 0.1 and 0.5 < current_lane_closest_dist < 3.2 and each_lane not in closest_lanes_same_dir and current_state['road'][each_lane]['turning'] == 0:
                closest_lanes_same_dir.append(each_lane)
                closest_lanes_idx_same_dir.append(current_lane_closest_idx)

        if closest_lane is not None and not 0.5 < closest_dist < 3.2:
            closest_lanes_same_dir = []
            closest_lanes_idx_same_dir = []


        if closest_lane is not None:
            current_lane = closest_lane
            current_closest_pt_idx = closest_lane_pt_idx
            dist_to_lane = closest_dist
            distance_threshold = max(7, max(7 * my_current_v_per_step, dist_to_lane))
        elif closest_lane_no_yaw is not None and not no_unparallel:
            current_lane = closest_lane_no_yaw
            current_closest_pt_idx = closest_lane_pt_no_yaw_idx
            dist_to_lane = closest_dist_no_yaw
            distance_threshold = max(10, dist_to_lane)
        else:
            logging.warning(f'No current lane founded: {agent_id}')
            # return
        if return_list:
            if len(closest_lanes_same_dir) > 0:
                return closest_lanes_same_dir, closest_lanes_idx_same_dir, dist_to_lane, distance_threshold
            else:
                return [current_lane], [current_closest_pt_idx], dist_to_lane, distance_threshold
        else:
            return current_lane, current_closest_pt_idx, dist_to_lane, distance_threshold

    def get_traffic_light_collision_pts(self, current_state, current_frame_idx,
                                        continue_time_threshold=5):
        tl_dics = current_state['traffic_light']
        road_dics = current_state['road']
        traffic_light_ending_pts = []
        for lane_id in tl_dics.keys():
            if lane_id == -1:
                continue
            tl = tl_dics[lane_id]
            # get the position of the end of this lane
            # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4, Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
            try:
                tl_state = tl["state"][current_frame_idx]
            except:
                tl_state = tl["state"][0]

            if tl_state in [1, 4, 7]:
                end_of_tf_checking = min(len(tl["state"]), current_frame_idx + continue_time_threshold)
                all_red = True
                for k in range(current_frame_idx, end_of_tf_checking):
                    if tl["state"][k] not in [1, 4, 7]:
                        all_red = False
                        break
                if all_red:
                    for seg_id in road_dics.keys():
                        if lane_id == seg_id:
                            road_seg = road_dics[seg_id]
                            if self.dataset == 'Waymo':
                                if road_seg["type"] in [1, 2, 3]:
                                    if len(road_seg["dir"].shape) < 1:
                                        continue
                                    if road_seg['turning'] == 1 and tl_state in [4, 7]:
                                        # can do right turn with red light
                                        continue
                                    end_point = road_seg["xyz"][0][:2]
                                    traffic_light_ending_pts.append(end_point)
                                break
                            elif self.dataset == 'NuPlan':
                                end_point = road_seg["xyz"][0][:2]
                                traffic_light_ending_pts.append(end_point)
                                break
                            else:
                                assert False, f'Unknown dataset in env planner - {self.dataset}'
        return traffic_light_ending_pts

def get_road_dict(map_api, ego_pose_center):
    road_dic = {}
    traffic_dic = {}
    all_map_obj = map_api.get_available_map_objects()

    # Collect lane information, following nuplan.planning.training.preprocessing.feature_builders.vector_builder_get_neighbor_vector_map

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


def get_agent_dict(ego_states, other_agents, statics):
    selected_agent_types = None

    # VEHICLE = 0, 'vehicle'
    # PEDESTRIAN = 1, 'pedestrian'
    # BICYCLE = 2, 'bicycle'
    # TRAFFIC_CONE = 3, 'traffic_cone'
    # BARRIER = 4, 'barrier'
    # CZONE_SIGN = 5, 'czone_sign'
    # GENERIC_OBJECT = 6, 'generic_object'
    # EGO = 7, 'ego'

    agent_dic = {}
    context_length = len(ego_states)

    new_dic = {'pose': np.ones([context_length, 4]) * -1,
               'shape': np.ones([context_length, 3]) * -1,
               'speed': np.ones([context_length, 2]) * -1,
               'type': 0,
               'is_sdc': 0, 'to_predict': 0}

    # pack ego
    agent_dic['ego'] = deepcopy(new_dic)
    poses_np = agent_dic['ego']['pose']
    shapes_np = agent_dic['ego']['shape']
    speeds_np = agent_dic['ego']['speed']
    # past
    for t in range(context_length):
        poses_np[t, :] = [ego_states[t].waypoint.center.x,
                          ego_states[t].waypoint.center.y,
                          0,
                          ego_states[t].waypoint.heading]  # x, y, z, yaw
        shapes_np[t, :] = [ego_states[-1].waypoint.oriented_box.width, ego_states[-1].waypoint.oriented_box.height,
                           0]  # width, height
        speeds_np[t, :] = [ego_states[t].waypoint.velocity.x, ego_states[t].waypoint.velocity.y]

    for t in range(context_length):
        for each_agent in other_agents[t]:
            agent_token = each_agent.track_token
            if agent_token not in agent_dic:
                agent_dic[agent_token] = deepcopy(new_dic)
            agent_dic[agent_token]['pose'][t, :] = [each_agent.center.x, each_agent.center.y, 0, each_agent.center.heading]
            agent_dic[agent_token]['shape'][t, :] = [each_agent.box.width, each_agent.box.length, each_agent.box.height]
            agent_dic[agent_token]['speed'][t, :] = [each_agent.velocity.x, each_agent.velocity.y]

        for each_agent in statics[t]:
            agent_token = each_agent.track_token
            if agent_token not in agent_dic:
                agent_dic[agent_token] = deepcopy(new_dic)
            agent_dic[agent_token]['pose'][t, :] = [each_agent.center.x, each_agent.center.y, 0, each_agent.center.heading]
            agent_dic[agent_token]['shape'][t, :] = [each_agent.box.width, each_agent.box.length, each_agent.box.height]
            agent_dic[agent_token]['speed'][t, :] = [each_agent.velocity.x, each_agent.velocity.y]

    return agent_dic


if __name__ == "__main__":
    with open("/home/shiduozhang/Project/transformer4planning/history.pkl", "rb") as f:
        input = pickle.load(f)

    with open("/home/shiduozhang/Project/transformer4planning/init.pkl", "rb") as f:
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
    planner = RuleBasedPlanner(10, 0.1, np.array([5, 5]))
    planner.initialize(initial)
    trajectory = planner.compute_planner_trajectory(input)
