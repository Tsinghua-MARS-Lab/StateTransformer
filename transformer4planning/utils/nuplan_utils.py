import math
import numpy as np
from typing import List
import pickle
import pandas as pd
import numpy as np
import os

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

VALIDATION_LIST = os.listdir("/public/MARS/datasets/nuPlan/nuplan-v1.1/data/cache/val")

def compute_average_score(horizon_3, horizon_5, horizon_8, threshold):
    avg_value =  np.mean((np.array(horizon_3) + np.array(horizon_5) + np.array(horizon_8))) / 3
    score = max(1 - avg_value/threshold, 0)
    return score    
    
def compute_scenario_score(scenario:List):
    ade3, ade5, ade8 = list(), list(), list()
    fde3, fde5, fde8 = list(), list(), list()
    ahe3, ahe5, ahe8 = list(), list(), list()
    fhe3, fhe5, fhe8 = list(), list(), list()
    ismiss = list()
    frames = list()
    for i, item in enumerate(scenario):
        ade3.append(item["ade_horison3_gen"])
        ade5.append(item["ade_horison5_gen"])
        ade8.append(item["ade_horison8_gen"])
        # FDE
        fde3.append(item["fde_horison3_gen"])
        fde5.append(item["fde_horison5_gen"])
        fde8.append(item["fde_horison8_gen"])
        # AHE
        ahe3.append(item["ahe_horison3_gen"])
        ahe5.append(item["ahe_horison5_gen"])
        ahe8.append(item["ahe_horison8_gen"])
        # FHE
        fhe3.append(item["fhe_horison3_gen"])
        fhe5.append(item["fhe_horison5_gen"])
        fhe8.append(item["fhe_horison8_gen"])
        # miss
        ismiss.append(item["miss_score"])
        # frame_id
        frames.append(item["frame_id"])
    ade_score = compute_average_score(ade3, ade5, ade8, ADE_THRESHHOLD)
    fde_score = compute_average_score(fde3, fde5, fde8, FDE_THRESHHOLD)
    ahe_score = compute_average_score(ahe3, ahe5, ahe8, HEADING_ERROR_THRESHHOLD)
    fhe_score = compute_average_score(fhe3, fhe5, fhe8, HEADING_ERROR_THRESHHOLD)
    miss = sum(ismiss)
    if miss >=5:
        miss_score = 0
    else:
        miss_score = 1
    score =(ade_score + fde_score + ahe_score * 2 + fhe_score * 2) / 6
    
    data_to_return = dict(
        file_name = VALIDATION_LIST[int(scenario[0]["file_id"])],
        ahe_score = ahe_score,
        ade_score = ade_score,
        fhe_score = fhe_score,
        fde_score = fde_score,
        miss_score = miss_score,
        score = miss_score * score,
    )
    return data_to_return
    
        
def compute_scores(data):
    scenarios = dict()
    data_frame = pd.DataFrame(data)
    print(len(data_frame))
    for i in range(len(data_frame)):
        item = data_frame.iloc[i]
        item["frame_id"] = int(item["frame_id"])
        if int(item["file_id"]) not in scenarios.keys():
            scenarios[int(item["file_id"])] = list()    
        scenarios[item["file_id"]].append(item.to_dict())
    for key, value in scenarios.items():
        scenarios[key] =  sorted(value, key=lambda x: x['frame_id'], reverse=False)
    print("scenario spiltted")
    scenarios_to_compute = list()
    for key, value in scenarios.items():
        if len(value) % 15 == 0:
            for i in range(len(value)//15):
                scenarios_to_compute.append(value[15 * i:15 * (i + 1)])
        else:
            start_id = value[0]["frame_id"]
            new_scenario = list()
            for i in range(len(value)):
                frame_id = value[i]["frame_id"]
                if frame_id - start_id <= 280:
                    new_scenario.append(value[i])
                else:
                    scenarios_to_compute.append(new_scenario)
                    new_scenario = [value[i]]
                    start_id = frame_id
    print(len(scenarios_to_compute))
    scores = list()
    results = list()
    miss_scores = list()
    for scenario in scenarios_to_compute:
        result_dic = compute_scenario_score(scenario) 
        score = result_dic["score"]
        miss_score = result_dic["miss_score"]
        results.append(result_dic)
        scores.append(score)
        miss_scores.append(miss_score)
    avg_score = np.average(scores)
    avg_miss_score = np.average(miss_scores)
    # pd.DataFrame(results).to_csv("medium_full_test.csv")
    return avg_score, avg_miss_score
