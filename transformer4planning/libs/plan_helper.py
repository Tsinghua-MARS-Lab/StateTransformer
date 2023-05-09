import interactive_sim.envs.util as utils
import math
import numpy as np


def mph_to_meterpersecond(mph):
    return mph * 0.4472222222


def get_angle(x, y):
    return math.atan2(y, x)


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


def calculate_yaw_from_states(trajectory, default_yaw):
    time_frames, _ = trajectory.shape
    pred_yaw = np.zeros([time_frames])
    for i in range(time_frames - 1):
        pose_p = trajectory[i+1]
        pose = trajectory[i]
        delta_x = pose_p[0] - pose[0]
        delta_y = pose_p[1] - pose[1]
        dis = np.sqrt(delta_x*delta_x + delta_y*delta_y)
        if dis > 1:
            angel = get_angle(delta_x, delta_y)
            pred_yaw[i] = angel
            default_yaw = angel
        else:
            pred_yaw[i] = default_yaw
    return pred_yaw


def change_axis(yaw):
    return - yaw - math.pi/2


def get_current_pose_and_v(current_state, agent_id, current_frame_idx):
    my_current_pose = current_state['agent'][agent_id]['pose'][current_frame_idx - 1]
    if current_state['agent'][agent_id]['pose'][current_frame_idx - 1, 0] == -1 or current_state['agent'][agent_id]['pose'][current_frame_idx - 6, 0] == -1:
        my_current_v_per_step = 0
        print("Past invalid for ", agent_id, " and setting v to 0")
    else:
        my_current_v_per_step = euclidean_distance(current_state['agent'][agent_id]['pose'][current_frame_idx - 1, :2],
                                                   current_state['agent'][agent_id]['pose'][current_frame_idx - 6, :2]) / 5
    return my_current_pose, my_current_v_per_step


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
            yaw_diff = abs(utils.normalize_angle(my_current_pose[3] - road_yaw))
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


def search_neighbour_lanes(current_pose, current_state, dist_threshold=2,
                           valid_lane_types=[1, 2]):
    lanes_to_return = []
    for each_lane in current_state['road']:
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
            dist = euclidean_distance(each_xy, current_pose[:2])
            yaw_diff = abs(utils.normalize_angle(current_pose[3] - road_yaw))
            if yaw_diff < math.pi / 180 * 20 and dist < dist_threshold:
                lanes_to_return.append(each_lane)
                continue
    return lanes_to_return

def compute_trajectory_derivatives(traj, frame_rate=20):
    dt = 1.0 / frame_rate
    traj = np.array(traj)
    num = len(traj)
    results = {}
    results["v_abs"] = ((traj[1:, :2] - traj[:-1, :2]) ** 2).sum(axis=1) ** 0.5 / dt
    results["dist"] = np.concatenate([[0], np.cumsum(results["v_abs"] * dt)])
    results["v"] = ((traj[1:, :2] - traj[:-1, :2]) * np.stack([np.cos(traj[:-1, 3]), np.sin(traj[:-1, 3])], axis=1)).sum(axis=1) / dt
    results["a"] = (results["v"][1:] - results["v"][:-1]) / dt
    results["j"] = (results["a"][1:] - results["a"][:-1]) / dt
    results["omega"] = angle_diff(traj[1:, 3], traj[:-1, 3]) / dt
    results["kappa"] = results["omega"] / np.maximum(1e-6, results["v_abs"])
    return results

# Returns an angle wrapped to [-pi, pi).
def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

# Returns an angle difference in [-pi, pi).
def angle_diff(theta0, theta1):
    return wrap_angle(theta0 - theta1)