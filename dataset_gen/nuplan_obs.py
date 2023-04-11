import numpy as np
import shapely
import cv2
import math

def get_observation_for_nsm(observation_kwargs, data_dic, scenario_frame_number, total_frames, nsm_result=None):
    """
    only for NuPlan dataset, rasters and vectors are not flexible to change
    Rester shapes should be 2D: width and height
    Speed: tested on MAC M1 on 2Hz, 6 examples per second with two rasters, 7 examples per second without rasters,
    Size: about 1.3GB per file on 2Hz with two rasters, 40MB per file without rasters
    return:
        'intended_maneuver_vector': (1),
        'current_maneuver_vector': (12),
        'trajectory_label': (t, 4)  # t = future_frame_num
        'intended_maneuver_label': (1),  # after interval
        'current_maneuver_label': (12),  # after interval
        'high_res_raster': (w, h, 1 + n + m * t)
        'low_res_raster': (w, h, 1 + n + m * t)
    {
        'goal_of_current_frame_raster': (w, h),
        'map_raster': (n, w, h),  # n is the number of road types and traffic lights types
        'ego_past_raster': (t, w, h)  # t = past_frame_num // frame_sample_interval + 1 (including the current frame)
        'other_past_raster': (t, m-1, w, h)  # m is the number of agent types
    }
    """

    # hyper parameters setting
    max_dis = observation_kwargs["max_dis"]
    high_res_raster_shape = observation_kwargs["high_res_raster_shape"]
    low_res_raster_shape = observation_kwargs["low_res_raster_shape"]
    assert len(high_res_raster_shape) == len(low_res_raster_shape) == 2, f'{high_res_raster_shape}, {low_res_raster_shape}'
    high_res_raster_scale = observation_kwargs["high_res_raster_scale"]
    low_res_raster_scale = observation_kwargs["low_res_raster_scale"]

    result_to_return = {}
    past_frames_number = observation_kwargs["past_frame_num"]
    future_frames_number = observation_kwargs["future_frame_num"]
    frame_sample_interval = observation_kwargs["frame_sample_interval"]

    ego_pose = data_dic["agent"]["ego"]["pose"][scenario_frame_number].copy()
    cos_, sin_ = math.cos(ego_pose[3]), math.sin(ego_pose[3])

    total_road_types = 20
    total_agent_types = 8
    sample_frames = list(range(scenario_frame_number - past_frames_number, scenario_frame_number, frame_sample_interval))
    sample_frames.append(scenario_frame_number)
    total_raster_channels = 1 + total_road_types + total_agent_types * len(sample_frames)

    rasters_high_res = np.zeros([high_res_raster_shape[0],
                                 high_res_raster_shape[1],
                                 total_raster_channels], dtype=np.uint8)
    rasters_low_res = np.zeros([low_res_raster_shape[0],
                                low_res_raster_shape[1],
                                total_raster_channels], dtype=np.uint8)
    rasters_high_res_channels = cv2.split(rasters_high_res)
    rasters_low_res_channels = cv2.split(rasters_low_res)

    # trajectory label
    trajectory_label = data_dic['agent']['ego']['pose'][
                       scenario_frame_number - 1:scenario_frame_number + future_frames_number + 1, :].copy()
    trajectory_label -= ego_pose
    trajectory_label[:, 0] = trajectory_label[:, 0] * cos_ - trajectory_label[:, 1] * sin_
    trajectory_label[:, 1] = trajectory_label[:, 0] * sin_ + trajectory_label[:, 1] * cos_
    result_to_return['trajectory_label'] = trajectory_label[2:, :]
    action_label_scale = observation_kwargs["action_label_scale"]
    result_to_return['action_label'] = np.array((trajectory_label[1, :2] - trajectory_label[0, :2])*action_label_scale, dtype=np.int32)


    # WARNING: Import actions first
    # goal action example: [{'action': <ActionLabel.Stop: 3>, 'weight': 1}]
    if nsm_result is not None:
        current_goal_maneuver = nsm_result['goal_actions_weights_per_frame'][scenario_frame_number][0]['action']
        result_to_return['intended_maneuver_vector'] = current_goal_maneuver.value - 1
        target_goal_maneuver = nsm_result['goal_actions_weights_per_frame'][scenario_frame_number+frame_sample_interval][0]['action']
        result_to_return['intended_maneuver_label'] = target_goal_maneuver.value - 1
        current_action_weights = np.zeros(12, dtype=np.float32)
        for each_current_action in nsm_result['current_actions_weights_per_frame'][scenario_frame_number]:
            current_action_index = each_current_action['action'].value - 1
            current_action_weights[current_action_index] = each_current_action['weight']
        result_to_return['current_maneuver_vector'] = current_action_weights
        target_current_action_weights = np.zeros(12, dtype=np.float32)
        for each_current_action in nsm_result['current_actions_weights_per_frame'][scenario_frame_number+frame_sample_interval]:
            current_action_index = each_current_action['action'].value - 1
            target_current_action_weights[current_action_index] = each_current_action['weight']
        result_to_return['current_maneuver_label'] = target_current_action_weights
    else:
        result_to_return["intended_maneuver_label"] = None
        result_to_return["intended_maneuver_vector"] = None
        result_to_return["current_maneuver_label"] = None
        result_to_return["current_maneuver_vector"] = None

    # sample and draw the goal
    goal_sample_frame = min(total_frames - 1, scenario_frame_number + 20 * 20)
    goal_point = data_dic['agent']['ego']['pose'][goal_sample_frame, :].copy()
    shape = data_dic['agent']['ego']['shape'][scenario_frame_number, :]
    goal_point -= ego_pose
    rotated_goal_pose = [goal_point[0] * cos_ - goal_point[1] * sin_,
                         goal_point[0] * sin_ + goal_point[1] * cos_,
                         0, goal_point[3]]
    goal_contour = cv2.boxPoints(((rotated_goal_pose[0], rotated_goal_pose[1]),
                                  (shape[0], shape[1]), rotated_goal_pose[3]))
    goal_contour = np.int0(goal_contour)
    goal_contour_high_res = int(high_res_raster_scale) * goal_contour
    cv2.drawContours(rasters_high_res_channels[0], [goal_contour_high_res], -1, (255, 255, 255), -1)
    goal_contour_low_res = int(low_res_raster_scale) * goal_contour
    cv2.drawContours(rasters_low_res_channels[0], [goal_contour_low_res], -1, (255, 255, 255), -1)

    # 'map_raster': (n, w, h),  # n is the number of road types and traffic lights types
    for i, key in enumerate(data_dic['road']):
        xyz = data_dic["road"][key]["xyz"].copy()
        road_type = int(data_dic['road'][key]['type'])
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

    for i, key in enumerate(data_dic['agent']):
        for j, sample_frame in enumerate(sample_frames):
            pose = data_dic['agent'][key]['pose'][sample_frame, :].copy()
            pose -= ego_pose
            if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
                continue
            agent_type = int(data_dic['agent'][key]['type'])
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            shape = data_dic['agent'][key]['shape'][scenario_frame_number, :]
            rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
                                      (shape[0], shape[1]), pose[3]))
            rect_pts = np.int0(rect_pts)
            # draw on high resolution
            rect_pts_high_res = int(high_res_raster_scale) * rect_pts
            cv2.drawContours(rasters_high_res_channels[1 + total_road_types + agent_type * len(sample_frames) + j], [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = int(low_res_raster_scale) * rect_pts
            cv2.drawContours(rasters_low_res_channels[1 + total_road_types + agent_type * len(sample_frames) + j], [rect_pts_low_res], -1, (255, 255, 255), -1)

    rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
    rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

    result_to_return['high_res_raster'] = np.array(rasters_high_res, dtype=bool)
    result_to_return['low_res_raster'] = np.array(rasters_low_res, dtype=bool)
    # context action computation
    context_actions = list()
    ego_poses = data_dic["agent"]["ego"]["pose"]
    for i in range(len(sample_frames) - 1):
        action = ego_poses[sample_frames[i + 1]] - ego_poses[sample_frames[i]]
        context_actions.append(action)
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
        
    return result_to_return

if __name__ == '__main__':
    import pickle
    with open("../labels/intentions/nuplan_boston/training.wtime.0-100.iter0.pickle", "rb")  as f:
        nsm = pickle.load(f)
    with open("data_dic.pkl", "rb") as f:
        data_dic = pickle.load(f)
    total_frames = data_dic["agent"]["ego"]["pose"].shape[0]
    observation_encode_kwargs = dict(
            visible_dis=30,
            max_dis=80,
            scale=1.0,
            stride=5,
            raster_scale=1.0,
            past_frame_num=40,
            future_frame_num=160,
            high_res_raster_shape=[224, 224],  # for high resolution image, we cover 50 meters for delicated short-term actions
            high_res_raster_scale=4.0,
            low_res_raster_shape=[224, 224],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
            low_res_raster_scale=0.77,
            frame_sample_interval=5,
            action_label_scale=100
        )
    result = get_observation_for_nsm(observation_encode_kwargs, data_dic, 88, total_frames)
    with open("dataset_example.pkl", "wb") as f:
        pickle.dump(result, f)
    

