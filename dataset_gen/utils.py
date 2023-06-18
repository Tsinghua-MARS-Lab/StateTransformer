import math
import numpy as np
from PIL import Image, ImageDraw

SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD = 20
SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD = 0.1

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


def tuple_recenter(point, window_w, window_h, tuple=True):
    x, y = point
    if tuple:
        return (x+window_w/2, y+window_h/2)
    else:
        return x+window_w/2, y+window_h/2


def generate_contour_pts(center_pt, w, l, direction):
    pt1 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]-l/2), direction, tuple=True)
    pt2 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]-l/2), direction, tuple=True)
    pt3 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]+l/2), direction, tuple=True)
    pt4 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]+l/2), direction, tuple=True)
    return pt1, pt2, pt3, pt4


def generate_contour_pts_with_direction(center_pt, w, l, direction):
    pt1 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]-l/2), direction, tuple=True)
    pt2 = rotate(center_pt, (center_pt[0], center_pt[1] - l / 2 * 1.2), direction, tuple=True)
    pt3 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]-l/2), direction, tuple=True)
    pt4 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]+l/2), direction, tuple=True)
    pt5 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]+l/2), direction, tuple=True)
    return pt1, pt2, pt3, pt4, pt5


def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)


def manhattan_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return abs(x_1-x_2)+abs(y_1-y_2)


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle


def is_point_in_box(point_tuple, box_two_points):
    x, y = point_tuple
    box_1, box_2 = box_two_points
    upper_left_x, upper_left_y = box_1
    lower_right_x, lower_right_y = box_2
    if (lower_right_x - x) * (upper_left_x - x) <= 0:
        if (lower_right_y - y) * (upper_left_y - y) <= 0:
            return True
    return False


def is_point_in_box_with_angel(point_tuple, box_four_points):
    angles = []
    sum_result = 0
    for pt in box_four_points:
        angles.append(get_angle_of_a_line(point_tuple, pt))
    angles.append(angles[0])
    for i in range(len(angles)-1):
        sum_result += abs(normalize_angle(angles[i+1] - angles[i]))
    if sum_result >= math.pi*1.99:
        return True
    else:
        return False


def check_collision(checking_agent, target_agent):
    return check_collision_for_two_agents_rotate_and_dist_check(checking_agent=checking_agent,
                                                                target_agent=target_agent)


def check_collision_for_two_agents(checking_agent, target_agent, vertical_margin=1, vertical_margin2=1, horizon_margin=0.9):
    center_c = [checking_agent.x, checking_agent.y]
    center_t = [target_agent.x, target_agent.y]
    length_sum_top_threshold = checking_agent.length + target_agent.length
    if checking_agent.x == -1 or target_agent.x == -1:
        return False
    if abs(checking_agent.x - target_agent.x) > length_sum_top_threshold:
        return False
    if abs(checking_agent.y - target_agent.y) > length_sum_top_threshold:
        return False
    collision_box_c = [(checking_agent.x - checking_agent.width/2 * horizon_margin,
                        checking_agent.y - checking_agent.length/2 * vertical_margin),
                       (checking_agent.x - checking_agent.width/2 * horizon_margin,
                        checking_agent.y + checking_agent.length/2 * vertical_margin),
                       (checking_agent.x + checking_agent.width/2 * horizon_margin,
                        checking_agent.y + checking_agent.length/2 * vertical_margin),
                       (checking_agent.x + checking_agent.width/2 * horizon_margin,
                        checking_agent.y - checking_agent.length/2 * vertical_margin)]
    rotated_checking_box_c = rotate_array(origin=(checking_agent.x, checking_agent.y),
                                          points=np.array(collision_box_c),
                                          angle=normalize_angle(checking_agent.yaw))

    collision_box_t = [(target_agent.x - target_agent.width/2 * horizon_margin,
                        target_agent.y - target_agent.length/2 * vertical_margin2),
                       (target_agent.x - target_agent.width/2 * horizon_margin,
                        target_agent.y + target_agent.length/2 * vertical_margin2),
                       (target_agent.x + target_agent.width/2 * horizon_margin,
                        target_agent.y + target_agent.length/2 * vertical_margin2),
                       (target_agent.x + target_agent.width/2 * horizon_margin,
                        target_agent.y - target_agent.length/2 * vertical_margin2)]
    rotated_checking_box_t = rotate_array(origin=(target_agent.x, target_agent.y),
                                          points=np.array(collision_box_t),
                                          angle=normalize_angle(target_agent.yaw))
    c_rst = check_collision_for_two_center_points(rotated_checking_box_c, rotated_checking_box_t, center_c, center_t)

    return c_rst

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


def check_collision_three_points_distance(checking_agent, target_agent, diameter_scale=1):

    anchor_points_a = [[checking_agent.x, checking_agent.y],
                       [checking_agent.x, checking_agent.y - checking_agent.length/2 + checking_agent.width/2],
                       [checking_agent.x, checking_agent.y + checking_agent.length/2 - checking_agent.width/2]]
    anchor_points_b = [[target_agent.x, target_agent.y],
                       [target_agent.x, target_agent.y - target_agent.length/2 + target_agent.width/2],
                       [target_agent.x, target_agent.y + target_agent.length/2 - target_agent.width/2]]
    anchor_points_a = rotate_array(origin=(checking_agent.x, checking_agent.y),
                                   points=np.array(anchor_points_a),
                                   angle=normalize_angle(-checking_agent.yaw))
    anchor_points_b = rotate_array(origin=(target_agent.x, target_agent.y),
                                   points=np.array(anchor_points_b),
                                   angle=normalize_angle(-target_agent.yaw))
    for each_a in anchor_points_a:
        for each_b in anchor_points_b:
            if each_a[0] == -1 or each_a[1] == -1 or each_b[0] == -1 or each_b[1] == -1 or checking_agent.width == -1 or target_agent.width == -1:
                continue
            if euclidean_distance(each_a, each_b) < (checking_agent.width + target_agent.width) / 2 * diameter_scale:
                return True

    return False


def check_collision_two_methods(checking_agent, target_agent, vertical_margin=0.7, vertical_margin2=0.7, horizon_margin=0.7, diameter_scale=1):
    distance_rst = check_collision_for_two_agents_rotate_and_dist_check(checking_agent, target_agent,
                                                                        vertical_margin, vertical_margin2, horizon_margin)
    three_pts_rst = check_collision_three_points_distance(checking_agent, target_agent, diameter_scale)
    # if distance_rst != three_pts_rst:
    #     print(f"COLLISION CHECK NOT THE SAME: \n {distance_rst} {three_pts_rst}\n ", checking_agent.agent_id, checking_agent.x, checking_agent.y, checking_agent.yaw, checking_agent.width, checking_agent.length)
    #     print(f" ", target_agent.agent_id, target_agent.x, target_agent.y, target_agent.yaw, target_agent.width, target_agent.length)
    return distance_rst | three_pts_rst
    # return three_pts_rst
    # return distance_rst


def check_collision_for_point_in_path(pt1, size1, yaw1, pt2, size2, yaw2, vertical_margin=1):
    x1, y1 = pt1
    x2, y2 = pt2
    width1, length1 = size1
    width2, length2 = size2
    collision_box_c = [(x1 - width1/2,
                        y1 - length1/2 * vertical_margin),
                       (x1 - width1/2,
                        y1 + length1/2 * vertical_margin),
                       (x1 + width1/2,
                        y1 + length1/2 * vertical_margin),
                       (x1 + width1/2,
                        y1 - length1/2 * vertical_margin)]
    rotated_checking_box_c = []
    for pt in collision_box_c:
        rotated_checking_box_c.append(rotate(origin=(x1, y1),
                                             point=pt,
                                             angle=normalize_angle(yaw1 + math.pi / 2),
                                             tuple=True))

    collision_box_t = [(x2 - width2/2,
                        y2 - length2/2 * vertical_margin),
                       (x2 - width2/2,
                        y2 + length2/2 * vertical_margin),
                       (x2 + width2/2,
                        y2 + length2/2 * vertical_margin),
                       (x2 + width2/2,
                        y2 - length2/2 * vertical_margin)]
    rotated_checking_box_t = []
    for pt in collision_box_t:
        rotated_checking_box_t.append(rotate(origin=(x2, y2),
                                             point=pt,
                                             angle=normalize_angle(yaw2 + math.pi / 2),
                                             tuple=True))
    return check_collision_for_two_center_points(rotated_checking_box_c, rotated_checking_box_t, pt1, pt2)


def check_collision_for_two_center_points(rotated_checking_box_c, rotated_checking_box_t, center_c, center_t):
    # check each point in/out of the box
    collision = False
    for pt in rotated_checking_box_t:
        if is_point_in_box_with_angel(pt, rotated_checking_box_c) | is_point_in_box_with_angel(center_t, rotated_checking_box_c):
            collision = True
            break
    return collision


def get_possible_destinations(agent, direction, map, scale, window_size):
    destinations = []
    out_intersection, out_lane = agent.spawn_position
    total_roads_number = len(map.roads)
    if direction == "L":
        target_intersection = (out_intersection - 1) % total_roads_number
    elif direction == "R":
        target_intersection = (out_intersection + 1) % total_roads_number
    elif direction == "F":
        target_intersection = (out_intersection + 2) % total_roads_number
    else:
        print("get_possible_destinations - Error & Exiting: unknown direction ", direction)
    total_lanes_target = map.roads[target_intersection]["in_number"]
    target_direction = normalize_angle(map.roads[target_intersection]["direction"] + math.pi)
    for i in range(total_lanes_target):
        target_pt = map.get_point_from_map(intersection=target_intersection, lane=i, scale=scale, window_size=window_size)
        destinations.append([target_pt, target_direction])
    return destinations


def destinations_to_paths_batch(agent, destinations, scale, frame_rate, target_v, target_a=0):
    paths = []
    for destination in destinations:
        t_state = [(destination[0][0]/scale, destination[0][1]/scale), target_v, target_a, destination[1]]
        paths.append(agent.gen_trajectory_agent2pt(scale=scale, frame_rate=frame_rate, t_state=t_state))
    return paths


def get_extended_point(starting_point, direction, extend):
    return starting_point[0] + math.sin(direction) * extend, starting_point[1] - math.cos(direction) * extend


def search_same_way_lanes(one_inbound_lane_id, road_dic, in_or_out=0, marking=0):
    # in_or_out: 0=inbound provided, 1=outbound provided
    outbound_lanes = []
    inbound_lanes = []
    # search from these inbound lanes
    xy_np = road_dic[one_inbound_lane_id]["xyz"][:, :2]
    dir_np = road_dic[one_inbound_lane_id]["dir"]
    if len(xy_np.shape) < 1 or len(dir_np.shape) < 1:
        return None
    if in_or_out:
        # outbound given
        entry_pt = xy_np[-1]
        entry_dir = dir_np[-2]
    else:
        # inbound given
        entry_pt = xy_np[0]
        entry_dir = dir_np[0]

    entry_pts_list = [entry_pt]
    out_pts_list = [entry_pt]
    pt_dist_threshold = SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD
    dir_threshold = SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD
    if in_or_out:
        # outbound given
        entry_dir = normalize_angle(entry_dir + math.pi)

    for road_seg_key in road_dic.keys():
        # if road_seg_key == tl_key:
        #     continue
        if road_dic[road_seg_key]["type"] not in [1, 2]:
            continue
        target_xy_np = road_dic[road_seg_key]["xyz"][:, :2]
        target_dir_np = road_dic[road_seg_key]["dir"]
        if len(target_xy_np.shape) < 1 or len(target_dir_np.shape) < 1:
            continue
        if target_xy_np.shape[0] < 2:
            print("ERROR: lane target_xy_np size too short. ", road_seg_key, target_xy_np)
            continue
        if target_dir_np.shape[0] < 3:
            # print("ERROR: lane target_dir_np size too short. ", road_seg_key, target_dir_np)
            # [[-1.71647068], [0.]]
            continue
        target_seg_entry_pt = target_xy_np[0]
        target_starting_dir = target_dir_np[0]
        if abs(normalize_angle(float(target_starting_dir) - float(entry_dir))) < dir_threshold:
            for one_entry_pt in entry_pts_list:
                # disth, distv = handvdistance(one_entry_pt, target_seg_entry_pt, entry_dir)
                dist = euclidean_distance(one_entry_pt, target_seg_entry_pt)
                if dist < pt_dist_threshold:
                    # if abs(disth) < pt_dist_threshold and abs(distv) < 5:
                    inbound_lanes.append(road_seg_key)
                    entry_pts_list.append(target_seg_entry_pt)
                    if marking:
                        road_dic[road_seg_key]["marking"] = 4
                    break
        target_seg_ending_pt = target_xy_np[-1]
        target_ending_dir = target_dir_np[-2]
        if abs(normalize_angle(float(target_ending_dir) - float(normalize_angle(entry_dir + math.pi)))) < dir_threshold:
            for one_entry_pt in out_pts_list:
                dist = euclidean_distance(one_entry_pt, target_seg_ending_pt)
                if dist < pt_dist_threshold:
                    outbound_lanes.append(road_seg_key)
                    out_pts_list.append(target_seg_ending_pt)
                    if marking:
                        road_dic[road_seg_key]["marking"] = 5
                    break

    return [outbound_lanes, inbound_lanes]


def mark_agents_mode(agent_dic, current_frame=0):
    for each_agent_id in agent_dic:
        its_trajectory = agent_dic[each_agent_id]['pose']
        current_mode = get_current_mode(its_trajectory, current_frame, each_agent_id)
        agent_dic[each_agent_id]['current_mode'] = current_mode
    return agent_dic


def check_pt_valid(pt):
    x, y = pt
    return abs(x + 1) < 0.01 or abs(y + 1) < 0.01


def get_current_mode(traj, current_frame, agent_id=0):
    '''
    traj=gt_trajectory[0, :, :], current_frame=10+i
    modes: 0=straight, 1=left turning, 2=right turning, 3=stopping, None=no steady mode detected
    '''
    current_looping_idx = current_frame
    total_frames = traj.shape[0]
    current_mode = None
    current_mode_counter = 0

    degree_threshold = 5
    steady_threshold = 1
    time_span = 15

    # print("test 506: ", total_frames, current_looping_idx)

    for i in range(total_frames - current_looping_idx - time_span):
        target_idx = current_frame + i
        if check_pt_valid(traj[target_idx, :2]) or check_pt_valid(traj[target_idx + time_span, :2]):
            # invalid
            continue
        dist = euclidean_distance(traj[target_idx, :2], traj[target_idx + time_span, :2])
        # print("test0: ", target_idx, dist)
        if dist < steady_threshold:
            # steady mode
            if current_mode is None:
                current_mode = 3
                current_mode_counter = 1
            elif current_mode == 3:
                current_mode_counter += 1
            else:
                current_mode = 3
                current_mode_counter = 1
        else:
            current_yaw = traj[target_idx, 3]
            next_yaw = traj[target_idx + time_span, 3]
            yaw_diff = normalize_angle(next_yaw - current_yaw)
            # if agent_id in [33, 37]:
            #     print("test1: ", yaw_diff, next_yaw, current_yaw, current_mode, current_mode_counter)
            if abs(yaw_diff) < math.pi / 180 * degree_threshold:
                if current_mode is None:
                    current_mode = 0
                    current_mode_counter = 1
                elif current_mode == 0:
                    current_mode_counter += 1
                else:
                    current_mode = 0
                    current_mode_counter = 1
            elif math.pi / 180 * degree_threshold <= yaw_diff <= math.pi:
                if current_mode is None:
                    current_mode = 2
                    current_mode_counter = 1
                elif current_mode == 2:
                    current_mode_counter += 1
                else:
                    current_mode = 2
                    current_mode_counter = 1
            elif -math.pi <= yaw_diff <= -math.pi / 180 * degree_threshold:
                if current_mode is None:
                    current_mode = 1
                    current_mode_counter = 1
                elif current_mode == 1:
                    current_mode_counter += 1
                else:
                    current_mode = 1
                    current_mode_counter = 1

        if current_mode_counter >= time_span:
            return current_mode

    return None
