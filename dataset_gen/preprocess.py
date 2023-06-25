import numpy as np
import pickle
import math
import cv2
import shapely
import os
from functools import partial

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

def preprocess(dataset, dic_path, autoregressive=False):
    if autoregressive:
        preprocess_function = None
    else:
        preprocess_function = partial(static_coor_rasterize, datapath=dic_path)
    target_datasets= dataset.map(preprocess_function, 
                    batch_size=os.cpu_count(), drop_last_batch=True, 
                    writer_batch_size=10, num_proc=os.cpu_count())
    return target_datasets

def augmentation():
    pass

def static_coor_rasterize(sample, datapath, raster_shape=(224, 224),
                            frame_rate=20, past_seconds=2, future_seconds=8,
                            high_res_scale=4, low_res_scale=0.77, frame_sample_interval=4,
                            road_types=20, agent_types=8, traffic_types=4):
    """
    coordinate is the ego pose at frame_id
    parameters:
        road_dic: a dictionary stores all the road elements, for one city
        agent_dic: a dictionary stores all the agents elements at each timestamp, for one file
        road_ids: a list of all the visible road elements id, for one scenario
        agent_ids: a list of all the visible agents at each timestamp, for one scenario
        traffic_ids: a list of all the visible traffic elements at each timestamp, for one scenario
        route_ids: a list of all the route ids, for one scenario 
        frame_id: the scenairo's frame number in one file  
    """
    filename = sample["file_name"]
    map = sample["map"]
    with open(os.path.join(datapath, f"{map}.pkl"), "rb") as f:
        road_dic = pickle.load(f)
    with open(os.path.join(datapath, f"agent_dic/{filename}.pkl"), "rb") as f:
        data_dic = pickle.load(f)  
        agent_dic = data_dic["agent_dic"]
        traffic_dic = data_dic["traffic_dic"]          
    road_ids = sample["road_ids"]
    agent_ids = sample["agent_ids"]
    traffic_ids = sample["traffic_ids"]
    route_ids = sample["route_ids"]
    frame_id = sample["frame_id"].item()
    # initialize rasters
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate
    sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))
    origin_ego_pose = agent_dic["ego"]["pose"][frame_id]

    total_raster_channels = 1 + road_types + traffic_types + agent_types * len(sample_frames)
    rasters_high_res = np.zeros([raster_shape[0],
                                raster_shape[1],
                                total_raster_channels], dtype=np.uint8)
    rasters_low_res = np.zeros([raster_shape[0],
                                raster_shape[1],
                                total_raster_channels], dtype=np.uint8)
    rasters_high_res_channels = cv2.split(rasters_high_res)
    rasters_low_res_channels = cv2.split(rasters_low_res)


    # route raster
    cos_, sin_ = math.cos(-origin_ego_pose[3] - math.pi / 2), math.sin(-origin_ego_pose[3] - math.pi / 2)
    for route_id in route_ids:
        xyz = road_dic[route_id.item()]["xyz"].copy()
        xyz[:, :2] -= origin_ego_pose[:2]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        high_res_route = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
        low_res_route = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
        cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
        cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))
    # road raster
    for road_id in road_ids:
        xyz = road_dic[road_id.item()]["xyz"].copy()
        road_type = int(road_dic[road_id.item()]["type"])
        xyz[:, :2] -= origin_ego_pose[:2]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        high_res_road = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
        low_res_road = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
        if road_type in [5, 17, 18, 19]:
            cv2.fillPoly(rasters_high_res_channels[road_type + 1], np.int32([high_res_road[:, :2]]), (255, 255, 255))
            cv2.fillPoly(rasters_low_res_channels[road_type + 1], np.int32([low_res_road[:, :2]]), (255, 255, 255))
        else:
            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[road_type + 1], tuple(high_res_road[j, :2]),
                        tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[road_type + 1], tuple(low_res_road[j, :2]),
                        tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)
    # traffic raster
    for traffic_id in traffic_ids:
        xyz = road_dic[int(traffic_id.item())]["xyz"].copy()
        xyz[:, :2] -= origin_ego_pose[:2]
        traffic_state = traffic_dic[int(traffic_id.item())]["state"]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
        low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            # traffic state order is GREEN, RED, YELLOW, UNKNOWN
        for j in range(simplified_xyz.shape[0] - 1):
            cv2.line(rasters_high_res_channels[1 + road_types + traffic_state], \
                    tuple(high_res_traffic[j, :2]),
                    tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
            cv2.line(rasters_low_res_channels[1 + road_types + traffic_state], \
                    tuple(low_res_traffic[j, :2]),
                    tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)
    # agent raster
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    for i, sample_frame in enumerate(sample_frames):
        for _, agent_id in enumerate(agent_ids[i]):
            pose = agent_dic[agent_id]['pose'][sample_frame, :].copy()
            if pose[0] < 0 and pose[1] < 0:
                continue
            pose -= origin_ego_pose
            agent_type = int(agent_dic[agent_id]['type'])
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            shape = agent_dic[agent_id]['shape'][frame_id, :]
            # rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
            #   (shape[1], shape[0]), np.rad2deg(pose[3])))
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                            direction=-pose[3])
            rect_pts = np.array(rect_pts, dtype=np.int32)

            # draw on high resolution
            rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0]//2
        
            cv2.drawContours(rasters_high_res_channels[1 + road_types + traffic_types + agent_type * len(sample_frames) + i],
                            [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0]//2
            cv2.drawContours(rasters_low_res_channels[1 + road_types + traffic_types + agent_type * len(sample_frames) + i],
                            [rect_pts_low_res], -1, (255, 255, 255), -1)
    
    # context action computation
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    context_actions = list()
    ego_poses = agent_dic["ego"]["pose"] - origin_ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                              ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                              np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))

    for i in range(len(sample_frames) - 1):
        action = rotated_poses[sample_frames[i]]
        context_actions.append(action)
    
    # future trajectory 
    trajectory_label = agent_dic['ego']['pose'][
                       frame_id:scenario_end_frame + 1, :].copy()
    trajectory_label -= origin_ego_pose
    traj_x = trajectory_label[:, 0].copy()
    traj_y = trajectory_label[:, 1].copy()
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_

    result_to_return = dict()
    result_to_return["high_res_raster"] = rasters_high_res
    result_to_return["low_res_raster"] = rasters_low_res
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
    result_to_return['trajectory_label'] = trajectory_label[1:, :]
    
    return result_to_return

if __name__ == "__main__":
    import datasets
    index_dataset = datasets.load_from_disk("/home/shiduozhang/nuplan/online_debug/boston_index_debug")
    file_names = set(index_dataset["file_name"])
    exm = static_coor_rasterize(index_dataset[0], datapath="/home/shiduozhang/nuplan/online_debug")
    preprocess_function = partial(static_coor_rasterize, datapath="/home/shiduozhang/nuplan/online_debug")
    dataset = index_dataset.map(preprocess_function, batch_size=10, drop_last_batch=True, writer_batch_size=10, num_proc=10)
    print(dataset)