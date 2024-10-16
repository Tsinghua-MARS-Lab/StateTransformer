import copy
import math
import numpy as np
from typing import List
import pickle
import pandas as pd
from shapely import geometry

def rotate_array(origin, points, angle, tuple=False):
    """
    Rotate a numpy array of points counter-clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    assert isinstance(points, type(np.array([]))), type(points)
    ox, oy = origin
    px = points[:, 0]
    py = points[:, 1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if tuple:
        return (qx, qy)
    else:
        rst_array = np.zeros_like(points)
        rst_array[:, 0] = qx
        rst_array[:, 1] = qy
        return rst_array


def change_coordination(target_point, ego_center, ego_to_global=False):
    target_point_new = target_point.copy()
    if ego_to_global:
        cos_, sin_ = math.cos(ego_center[-1]), math.sin(ego_center[-1])
        # ego to global
        new_x, new_y = target_point_new[0] * cos_ - target_point_new[1] * sin_, \
                       target_point_new[0] * sin_ + target_point_new[1] * cos_
        target_point_new[0], target_point_new[1] = new_x, new_y
        target_point_new[:2] += ego_center[:2]
    else:
        cos_, sin_ = math.cos(-ego_center[-1]), math.sin(-ego_center[-1])
        target_point_new[:2] -= ego_center[:2]
        # global to ego
        new_x, new_y = target_point_new[0] * cos_ - target_point_new[1] * sin_, \
                       target_point_new[0] * sin_ + target_point_new[1] * cos_
        target_point_new[0], target_point_new[1] = new_x, new_y
    return target_point_new


def get_closest_lane_on_route(pred_key_point_global,
                              route_ids,
                              road_dic):
    # loop over all the route ids
    pred_key_point_global_copy = copy.deepcopy(pred_key_point_global)
    route_lanes = []
    for each_route_block in route_ids:
        route_lanes += road_dic[each_route_block]['lower_level']
    route_lane_pts = []
    lane_ids = []
    for each_lane in route_lanes:
        if road_dic[each_lane]['type'] not in [0, 11]:
            continue
        route_lane_pts.append(road_dic[each_lane]['xyz'][:, :2])
        lane_ids += [each_lane] * road_dic[each_lane]['xyz'].shape[0]
    # concatenate all points in the list in one dimension
    route_lane_pts_np = np.concatenate(route_lane_pts, axis=0)
    # get the closest point over all of the selected lanes
    dist = np.linalg.norm(route_lane_pts_np - pred_key_point_global_copy[:2], axis=1)
    closest_index = np.argmin(dist)
    closest_lane_id = lane_ids[closest_index]
    return closest_lane_id, dist[closest_index]


def get_closest_lane_point_on_route(pred_key_point_global,
                                    route_ids,
                                    road_dic):
    # loop over all the route ids
    pred_key_point_global_copy = copy.deepcopy(pred_key_point_global)
    route_lanes = []
    for each_route_block in route_ids:
        route_lanes += road_dic[each_route_block]['lower_level']
    route_lane_pts = []
    lane_ids = []
    for each_lane in route_lanes:
        if road_dic[each_lane]['type'] not in [0, 11]:
            continue
        route_lane_pts.append(road_dic[each_lane]['xyz'][:, :2])
        lane_ids += [each_lane] * road_dic[each_lane]['xyz'].shape[0]
    # concatenate all points in the list in one dimension
    route_lane_pts_np = np.concatenate(route_lane_pts, axis=0)
    # get the closest point over all of the selected lanes
    dist = np.linalg.norm(route_lane_pts_np - pred_key_point_global_copy[:2], axis=1)
    closest_index = np.argmin(dist)
    closest_lane_point = route_lane_pts_np[closest_index]

    # check if point in route road block
    closest_lane_id = lane_ids[closest_index]
    closest_road_block = None
    for each_route_block in road_dic[closest_lane_id]['upper_level']:
        if each_route_block in route_ids:
            closest_road_block = each_route_block
            break
    assert closest_road_block is not None, 'closest_road_block is None'
    line = geometry.LineString(road_dic[closest_road_block]['xyz'][:, :2])
    point = geometry.Point(pred_key_point_global_copy[:2])
    polygon = geometry.Polygon(line)
    on_road = polygon.contains(point)

    return closest_lane_point, dist[closest_index], on_road


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

def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

def check_collision(checking_agent, target_agent):
    # return check_collision_for_two_agents_dense_scipy(checking_agent, target_agent)  # slower
    # return check_collision_for_two_agents_dense(checking_agent, target_agent)
    return check_collision_for_two_agents_rotate_and_dist_check(checking_agent=checking_agent,
                                                                target_agent=target_agent)

def check_collision_for_two_agents_rotate_and_dist_check(checking_agent, target_agent, vertical_margin=0.7, vertical_margin2=0.7, horizon_margin=0.7):
    # center_c = [checking_agent.x, checking_agent.y]
    # center_t = [target_agent.x, target_agent.y]

    length_sum_top_threshold = checking_agent.length + target_agent.length
    if checking_agent.x == -1 or target_agent.x == -1:
        return False
    if abs(checking_agent.x - target_agent.x) > length_sum_top_threshold:
        return False
    if abs(checking_agent.y - target_agent.y) > length_sum_top_threshold:
        return False

    if euclidean_distance([checking_agent.x, checking_agent.y], [target_agent.x, target_agent.y]) <= (checking_agent.width + target_agent.width)/2:
        return True
    collision_box_t = [(target_agent.x - target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y - target_agent.length/2 * vertical_margin2 - checking_agent.y),
                       (target_agent.x - target_agent.width / 2 * horizon_margin - checking_agent.x,
                        target_agent.y - checking_agent.y),
                       (target_agent.x - target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y + target_agent.length/2 * vertical_margin2 - checking_agent.y),
                       (target_agent.x + target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y + target_agent.length/2 * vertical_margin2 - checking_agent.y),
                       (target_agent.x + target_agent.width / 2 * horizon_margin - checking_agent.x,
                        target_agent.y - checking_agent.y),
                       (target_agent.x + target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y - target_agent.length/2 * vertical_margin2 - checking_agent.y)]
    rotated_checking_box_t = rotate_array(origin=(target_agent.x - checking_agent.x, target_agent.y - checking_agent.y),
                                          points=np.array(collision_box_t),
                                          angle=normalize_angle( - target_agent.yaw))
    rotated_checking_box_t = np.insert(rotated_checking_box_t, 0, [target_agent.x - checking_agent.x, target_agent.y - checking_agent.y], 0)

    rotated_checking_box_t = rotate_array(origin=(0, 0),
                                          points=np.array(rotated_checking_box_t),
                                          angle=normalize_angle( - checking_agent.yaw))

    rst = False
    for idx, pt in enumerate(rotated_checking_box_t):
        x, y = pt
        if abs(x) < checking_agent.width/2 * horizon_margin and abs(y) < checking_agent.length/2 * vertical_margin:
            rst = True
            # print('test: ', idx)
            break
    return rst


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle

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


FDE_THRESHHOLD = 8 # keep same with nuplan simulation
ADE_THRESHHOLD = 8 # keep same with nuplan simulation
HEADING_ERROR_THRESHHOLD = 0.8 # keep same with nuplan simulation
MISS_RATE_THRESHHOLD = 0.3
MISS_THRESHHOLD = [6, 8, 16]
DISPLACEMENT_WEIGHT = 1
HEADING_WEIGHT = 2


def compute_average_score(horizon_3, horizon_5, horizon_8, threshold):
    avg_value =  np.mean((np.array(horizon_3) + np.array(horizon_5) + np.array(horizon_8))) / 3
    score = max(1 - avg_value/threshold, 0)
    return score    


def compute_scenario_score(eval_results: List, scenario_id: int):
    """
    :param scenario: list of item (eval result dictionary)
    :return:
    """
    ade3, ade5, ade8 = list(), list(), list()
    fde3, fde5, fde8 = list(), list(), list()
    ahe3, ahe5, ahe8 = list(), list(), list()
    fhe3, fhe5, fhe8 = list(), list(), list()
    ismiss = list()
    frames = list()
    for i, item in enumerate(eval_results):
        ade3.append(item["ade_horizon3_gen"])
        ade5.append(item["ade_horizon5_gen"])
        ade8.append(item["ade_horizon8_gen"])
        # FDE
        fde3.append(item["fde_horizon3_gen"])
        fde5.append(item["fde_horizon5_gen"])
        fde8.append(item["fde_horizon8_gen"])
        # AHE
        ahe3.append(item["ahe_horizon3_gen"])
        ahe5.append(item["ahe_horizon5_gen"])
        ahe8.append(item["ahe_horizon8_gen"])
        # FHE
        fhe3.append(item["fhe_horizon3_gen"])
        fhe5.append(item["fhe_horizon5_gen"])
        fhe8.append(item["fhe_horizon8_gen"])
        # miss
        ismiss.append(item["miss_score"])
        # frame_id
        frames.append(item["frame_id"])
    ade_score = compute_average_score(ade3, ade5, ade8, ADE_THRESHHOLD)
    fde_score = compute_average_score(fde3, fde5, fde8, FDE_THRESHHOLD)
    ahe_score = compute_average_score(ahe3, ahe5, ahe8, HEADING_ERROR_THRESHHOLD)
    fhe_score = compute_average_score(fhe3, fhe5, fhe8, HEADING_ERROR_THRESHHOLD)
    miss = sum(ismiss)
    if miss >= 5:
        miss_score = 0
    else:
        miss_score = 1
    score =(ade_score + fde_score + ahe_score * 2 + fhe_score * 2) / 6
    
    data_to_return = dict(
        # file_name = VALIDATION_LIST[int(scenario[0]["file_id"])],
        scenario15s_id=scenario_id,
        ahe_score=ahe_score,
        ade_score=ade_score,
        fhe_score=fhe_score,
        fde_score=fde_score,
        miss_score=miss_score,
        score=miss_score * score,
    )
    return data_to_return
    
        
def compute_scores(data):
    scenarios = dict()
    data_frame = pd.DataFrame(data)
    for i in range(len(data_frame)):
        # group eval results by scenario
        item = data_frame.iloc[i]
        int_scenario15s_id = int(item['scenario15s_id'])
        if int_scenario15s_id not in scenarios.keys():
            scenarios[int_scenario15s_id] = list()
        scenarios[int_scenario15s_id].append(item.to_dict())
    scores = list()
    results = list()
    miss_scores = list()
    for scenario_id in scenarios:
        eval_results = scenarios[scenario_id]
        result_dic = compute_scenario_score(eval_results, scenario_id)
        score = result_dic["score"]
        miss_score = result_dic["miss_score"]
        results.append(result_dic)
        scores.append(score)
        miss_scores.append(miss_score)
    avg_score = np.average(scores)
    avg_miss_score = np.average(miss_scores)
    # pd.DataFrame(results).to_csv("medium_full_test.csv")
    return avg_score, avg_miss_score
