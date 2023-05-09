MINIMAL_DISTANCE_PER_STEP = 0.05
from transformer4planning.utils import *
from typing import List
import math
import numpy as np

def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle

def get_angle(x, y):
    return math.atan2(y, x)

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

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

class GoalSetter:
    def __init__(self):
        self.data = None

    def __call__(self, *args, **kwargs):
        self.data = kwargs['new_data']

    def get_goal(self, current_data, agent_id, dataset='Waymo') -> List:
        # get last valid point as the goal point
        agent_dic = current_data['agent'][agent_id]
        #agent_dic = current_data['predicting']['original_trajectory'][agent_id]
        yaw = None
        point = None
        if dataset == 'Waymo':
            # Waymo
            for frame_idx in range(1, 80):
                if yaw is not None:
                    break
                if agent_dic['pose'][-frame_idx][0] != -1 and agent_dic['pose'][-frame_idx][1] != -1:
                    point = [agent_dic['pose'][-frame_idx][0], agent_dic['pose'][-frame_idx][1]]
                    yaw = agent_dic['pose'][-frame_idx][3]
                    break
        elif dataset == 'NuPlan':
            # NuPlan
            assert 'ego_goal' in current_data, 'Goal Setter: Not found goal in data dic'
            goal = current_data['ego_goal']
            if agent_id == 'ego' and goal is not None:
                point = [goal[0], goal[1]]
                yaw = goal[3]
            else:
                for frame_idx in range(1, 180):
                    if yaw is not None:
                        break
                    if agent_dic['pose'][-frame_idx][0] != -1 and agent_dic['pose'][-frame_idx][1] != -1:
                        point = [agent_dic['pose'][-frame_idx][0], agent_dic['pose'][-frame_idx][1]]
                        yaw = agent_dic['pose'][-frame_idx][3]
                        break
                if point is None:
                    if agent_id == 'ego':
                        # print('ERROR: goal point is none ', agent_dic['pose'], agent_id)
                        print('[Static goal] ERROR: goal point is none ', agent_id)
                    point = [0, 0]
                    yaw = 0
        return [point, yaw]

class SudoInterpolator:
    def __init__(self, trajectory, current_pose):
        self.trajectory = trajectory
        self.current_pose = current_pose

    def interpolate(self, distance: float, starting_from=None, debug=False):
        if starting_from is not None:
            assert False, 'not implemented'
        else:
            pose = self.trajectory.copy()
        if distance <= MINIMAL_DISTANCE_PER_STEP:
            return self.current_pose
        total_frame, _ = pose.shape
        # assert distance > 0, distance
        distance_input = distance
        for i in range(total_frame):
            if i == 0:
                pose1 = self.current_pose[:2]
                pose2 = pose[0, :2]
            else:
                pose1 = pose[i - 1, :2]
                pose2 = pose[i, :2]
            next_step = euclidean_distance(pose1, pose2)
            if debug:
                print(
                    f"{i} {next_step} {distance} {total_frame} {self.current_pose}")
            if next_step >= MINIMAL_DISTANCE_PER_STEP:
                if distance > next_step and i != total_frame - 1:
                    distance -= next_step
                    continue
                else:
                    return self.get_state_from_poses(pose1, pose2, distance, next_step)
                    # x = (pose2[0] - pose1[0]) * distance / next_step + pose1[0]
                    # y = (pose2[1] - pose1[1]) * distance / next_step + pose1[1]
                    # yaw = utils.normalize_angle(get_angle_of_a_line(pt1=pose1, pt2=pose2))
                    # return [x, y, 0, yaw]
        if distance_input > distance:
            # hide it outshoot
            # logging.warning(f'Over shooting while planning!!!')
            return self.get_state_from_poses(pose1, pose2, distance, next_step)
        else:
            # return current pose if trajectory not moved at all
            pose1 = self.current_pose[:2]
            pose2 = pose[0, :2]
            return self.get_state_from_poses(pose1, pose2, 0, 0.01)

    
    def get_state_from_poses(self, pose1, pose2, mul, divider):
        x = (pose2[0] - pose1[0]) * mul / (divider + 0.0001) + pose1[0]
        y = (pose2[1] - pose1[1]) * mul / (divider + 0.0001) + pose1[1]
        yaw = normalize_angle(get_angle_of_a_line(pt1=pose1, pt2=pose2))
        return [x, y, 0, yaw]

    def get_distance_with_index(self, index: int):
        distance = 0
        if index != 0:
            pose = self.trajectory.copy()
            total_frame, _ = pose.shape
            for i in range(total_frame):
                if i >= index != -1:
                    # pass -1 to travel through all indices
                    break
                elif i == 0:
                    step = euclidean_distance(
                        self.current_pose[:2], pose[i, :2])
                else:
                    step = euclidean_distance(pose[i, :2], pose[i-1, :2])
                if step > MINIMAL_DISTANCE_PER_STEP:
                    distance += step
        return distance

    def get_speed_with_index(self, index: int):
        if index != 0:
            p_t = self.trajectory[index, :2]
            p_t1 = self.trajectory[index - 1, :2]
            speed_per_step = euclidean_distance(p_t, p_t1)
            return speed_per_step
        else:
            return None
        
class Agent:
    def __init__(self,
                 # init location, angle, velocity
                 x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, length=4.726, width=1.842, agent_id=None, color=None):
        self.x = x  # px
        self.y = y
        self.yaw = self.yaw_changer(yaw)
        self.vx = vx  # px/frame
        self.vy = vy
        self.length = max(1, length)
        self.width = max(1, width)
        self.agent_polys = []
        self.crashed = False
        self.agent_id = agent_id
        self.color = color

    def yaw_changer(self, yaw):
        return -yaw