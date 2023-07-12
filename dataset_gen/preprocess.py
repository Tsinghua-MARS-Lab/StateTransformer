import numpy as np
import pickle
import math
import cv2
import shapely
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from transformers import FeatureExtractionMixin
from torch.utils.data._utils.collate import default_collate
from transformer4planning.utils import generate_contour_pts

def preprocess(dataset, dic_path, autoregressive=False):
    """
    This function is only designed to call dataset.map() function to preprocess the dataset without augmentations.
    """
    if autoregressive:
        preprocess_function = partial(autoregressive_rasterize, data_path=dic_path)
    else:
        preprocess_function = partial(static_coor_rasterize, data_path=dic_path)
    target_datasets= dataset.map(preprocess_function, 
                    batch_size=os.cpu_count(), drop_last_batch=True, 
                    writer_batch_size=10, num_proc=os.cpu_count())
    return target_datasets

def nuplan_collate_func(batch, dic_path=None, autoregressive=False, **encode_kwargs):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    # padding for tensor data
    expected_padding_keys = ["road_ids", "route_ids", "traffic_ids"]
    agent_id_lengths = list()
    for i, d in enumerate(batch):
        agent_id_lengths.append(len(d["agent_ids"]))
    max_agent_id_length = max(agent_id_lengths)
    for i, d in enumerate(batch):
        agent_ids = d["agent_ids"]
        agent_ids.extend(["null"] * (max_agent_id_length - len(agent_ids)))
        batch[i]["agent_ids"] = agent_ids
    padded_tensors = dict()
    for key in expected_padding_keys:
        tensors = [data[key] for data in batch]
        padded_tensors[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
        for i, _ in enumerate(batch):
            batch[i][key] = padded_tensors[key][i]

    # online rasterize
    if autoregressive:
        map_func = partial(autoregressive_rasterize, data_path=dic_path, **encode_kwargs)
    else:
        map_func = partial(static_coor_rasterize, data_path=dic_path, **encode_kwargs)
    # with ThreadPoolExecutor(max_workers=len(batch)) as executor:
    #     new_batch = list(executor.map(map_func, batch))
    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)
    
    # process as data dictionary
    result = dict()
    for key in new_batch[0].keys():
        result[key] = default_collate([d[key] for d in new_batch])
    return result

def waymo_collate_func(batch, dic_path=None, autoregressive=False, **encode_kwargs):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    # online rasterize TODO:autoregressive function transfer
    if autoregressive:
        map_func = partial(waymo_preprocess)
    else:
        raise NotImplementedError
    # with ThreadPoolExecutor(max_workers=len(batch)) as executor:
    #     batch = list(executor.map(map_func, batch))
    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)
    
    # process as data dictionary
    result = dict()
    for key in new_batch[0].keys():
        result[key] = torch.cat([d[key] for d in new_batch], dim=0)
    return result

def augmentation():
    pass


def save_raster(result_dic, debug_raster_path, agent_type_num, past_frames_num, image_file_name,
                high_scale, low_scale):
    # save rasters
    path_to_save = debug_raster_path
    # check if path not exist, create
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        file_number = 0
    else:
        file_number = len(os.listdir(path_to_save))
        if file_number > 200:
            return
    image_shape = None
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = result_dic[each_key]
        goal = each_img[:, :, 0]
        road = each_img[:, :, :21]
        traffic_lights = each_img[:, :, 21:25]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = np.random.randint(0, 255, size=(21, 3)) * 0.5
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float)
        image_shape = target_image.shape
        for i in range(21):
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()

        # print('test: ', past_frames_num, agent_type_num, agent.shape)
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                # if j == 7:
                #     print('debug', np.sum(agent[:, :, j * 9 + i]), agent[:, :, j * 9 + i])
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        cv2.imwrite(os.path.join(path_to_save, image_file_name + '_' + str(each_key) + '.png'), target_image)
    for each_key in ['context_actions', 'trajectory_label']:
        pts = result_dic[each_key]
        for scale in [high_scale, low_scale]:
            target_image = np.zeros(image_shape, dtype=np.float)
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, :] = [255, 255, 255]
            cv2.imwrite(os.path.join(path_to_save, image_file_name + '_' + str(each_key) + '_' + str(scale) +'.png'), target_image)
    print('debug images saved to: ', path_to_save, file_number)


def static_coor_rasterize(sample, data_path, raster_shape=(224, 224),
                          frame_rate=20, past_seconds=2, future_seconds=8,
                          high_res_scale=4, low_res_scale=0.77,
                          road_types=20, agent_types=8, traffic_types=4,
                          past_sample_interval=4, future_sample_interval=4,
                          debug_raster_path=None):
    """
    :param sample: a dictionary containing the following keys:
        - file_name: the name of the file
        - map: the name of the map, ex: us-ma-boston
        - split: the split, train, val or test
        - road_ids: the ids of the road elements
        - agent_ids: the ids of the agents in string
        - traffic_ids: the ids of the traffic lights
        - traffic_status: the status of the traffic lights
        - route_ids: the ids of the routes
        - frame_id: the frame id of the current frame
        - debug_raster_path: if a debug_path past, will save rasterized images to disk, warning: will slow down the process
    :param data_path: the root path to load pickle files
    """
    # filename = sample["file_name"].item()
    # map = sample["map"].item()
    # split = sample["split"].item()
    # road_ids = sample["road_ids"].item()
    # agent_ids = sample["agent_ids"].item()
    # traffic_light_ids = sample["traffic_ids"].item()
    # traffic_light_states = sample["traffic_status"].item()
    # route_ids = sample["route_ids"].item()
    # frame_id = sample["frame_id"].item()

    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]
    road_ids = sample["road_ids"].tolist()
    agent_ids = sample["agent_ids"]  # list of strings
    traffic_light_ids = sample["traffic_ids"].tolist()
    traffic_light_states = sample["traffic_status"].tolist()
    route_ids = sample["route_ids"].tolist()
    frame_id = sample["frame_id"]

    if map == 'sg-one-north':
        y_inverse = -1
    else:
        y_inverse = 1

    # clean traffic ids, for legacy reasons, there might be -1 in the list
    traffic_light_ids = [x for x in traffic_light_ids if x != -1]
    assert len(traffic_light_ids) == len(traffic_light_states), f'length of ids is not same as length of states, ids: {traffic_light_ids}, states: {traffic_light_states}'

    if os.path.exists(os.path.join(data_path, "map", f"{map}.pkl")):
        with open(os.path.join(data_path, "map", f"{map}.pkl"), "rb") as f:
            road_dic = pickle.load(f)
    else:
        print(f"Error: cannot load map {map} from {data_path}")
        return None

    # load agent and traffic dictionaries
    if os.path.exists(os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")):
        with open(os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl"), "rb") as f:
            data_dic = pickle.load(f)
            agent_dic = data_dic["agent_dic"]
    else:
        print(f"Error: cannot load {filename} from {data_path}")
        return None

    # calculate frames to sample
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate
    # for example,
    # [10, 11, ..., 10+(2+8)*20=210], interval=10
    # frames_to_sample = [10, 20, 30, .., 210]
    # [10, 11, ..., 10+(2+8)*20=210], past_interval=10, future_interval=1
    # frames_to_sample = [10, 20, 30, 31, .., 209, 210]
    sample_frames_in_past = list(range(scenario_start_frame, frame_id, past_sample_interval))
    sample_frames_in_future = list(range(frame_id, scenario_end_frame, future_sample_interval))
    sample_frames = sample_frames_in_past + sample_frames_in_future
    # sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))

    # initialize rasters
    origin_ego_pose = agent_dic["ego"]["pose"][frame_id].copy()
    if np.isinf(origin_ego_pose[0]) or np.isinf(origin_ego_pose[1]):
        assert False, f"Error: ego pose is inf {origin_ego_pose}, not enough precision while generating dictionary"
    # channels:
    # 0: route raster
    # 1-20: road raster
    # 21-24: traffic raster
    # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
    total_raster_channels = 1 + road_types + traffic_types + agent_types * len(sample_frames_in_past)

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
        if int(route_id) == -1:
            continue
        xyz = road_dic[int(route_id)]["xyz"].copy()
        xyz[:, :2] -= origin_ego_pose[:2]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        simplified_xyz[:, 1] *= y_inverse
        high_res_route = (simplified_xyz * high_res_scale + raster_shape[0] // 2).astype('int32')
        low_res_route = (simplified_xyz * low_res_scale + raster_shape[0] // 2).astype('int32')

        cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
        cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))
    # road raster
    for road_id in road_ids:
        if int(road_id) == -1:
            continue
        xyz = road_dic[int(road_id)]["xyz"].copy()
        road_type = int(road_dic[int(road_id)]["type"])
        xyz[:, :2] -= origin_ego_pose[:2]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        simplified_xyz[:, 1] *= y_inverse
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
    # traffic channels drawing
    for idx, traffic_id in enumerate(traffic_light_ids):
        traffic_state = int(traffic_light_states[idx])
        if int(traffic_id) == -1 or int(traffic_id) not in list(road_dic.keys()):
            continue
        xyz = road_dic[int(traffic_id)]["xyz"].copy()
        xyz[:, :2] -= origin_ego_pose[:2]
        # traffic_state = traffic_dic[traffic_id.item()]["state"]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        simplified_xyz[:, 1] *= y_inverse
        high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
        low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
        # traffic state order is GREEN, RED, YELLOW, UNKNOWN
        for j in range(simplified_xyz.shape[0] - 1):
            cv2.line(rasters_high_res_channels[1 + road_types + traffic_state],
                     tuple(high_res_traffic[j, :2]),
                     tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
            cv2.line(rasters_low_res_channels[1 + road_types + traffic_state],
                     tuple(low_res_traffic[j, :2]),
                     tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)
    # agent raster
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    for _, agent_id in enumerate(agent_ids):
        if agent_id == "null":
            continue
        if agent_id not in list(agent_dic.keys()):
            print('unknown agent id', agent_id)
            continue
        for i, sample_frame in enumerate(sample_frames_in_past):
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
            rect_pts[:, 1] *= y_inverse
            # draw on high resolution
            rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0]//2
            # example: if frame_interval = 10, past frames = 40
            # channel number of [index:0-frame_0, index:1-frame_10, index:2-frame_20, index:3-frame_30, index:4-frame_40]  for agent_type = 0
            # channel number of [index:5-frame_0, index:6-frame_10, index:7-frame_20, index:8-frame_30, index:9-frame_40]  for agent_type = 1
            # ...
            cv2.drawContours(rasters_high_res_channels[1 + road_types + traffic_types + agent_type * len(sample_frames_in_past) + i],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0]//2
            cv2.drawContours(rasters_low_res_channels[1 + road_types + traffic_types + agent_type * len(sample_frames_in_past) + i],
                             [rect_pts_low_res], -1, (255, 255, 255), -1)

    # context action computation
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    context_actions = list()
    ego_poses = agent_dic["ego"]["pose"] - origin_ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                              ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                              np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
    rotated_poses[:, 1] *= y_inverse
    for i in sample_frames_in_past:
        action = rotated_poses[i]
        context_actions.append(action)

    # future trajectory
    # check if samples in the future is beyond agent_dic['ego']['pose'] length
    if sample_frames_in_future[-1] >= agent_dic['ego']['pose'].shape[0]:
        print('sample index beyond length of agent_dic: ', sample_frames_in_future[-1], agent_dic['ego']['pose'].shape[0])
        return None

    trajectory_label = agent_dic['ego']['pose'][sample_frames_in_future, :].copy()
    trajectory_label -= origin_ego_pose
    traj_x = trajectory_label[:, 0].copy()
    traj_y = trajectory_label[:, 1].copy()
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_
    trajectory_label[:, 1] *= y_inverse

    rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
    rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)

    result_to_return = dict()
    result_to_return["high_res_raster"] = np.array(rasters_high_res, dtype=bool)
    result_to_return["low_res_raster"] = np.array(rasters_low_res, dtype=bool)
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
    result_to_return['trajectory_label'] = trajectory_label.astype(np.float32)

    if debug_raster_path is not None:
        # check if path not exist, create
        if not os.path.exists(debug_raster_path):
            os.makedirs(debug_raster_path)
        image_file_name = sample['file_name'] + '_' + str(int(sample['frame_id']))
        save_raster(result_to_return, debug_raster_path, agent_types, len(sample_frames_in_past), image_file_name,
                    high_res_scale, low_res_scale)

    return result_to_return


def autoregressive_rasterize(sample, data_path, raster_shape=(224, 224),
                             frame_rate=20, past_seconds=2, future_seconds=8,
                             high_res_scale=4, low_res_scale=0.77,
                             road_types=20, agent_types=8, traffic_types=4,
                             past_sample_interval=4, future_sample_interval=4):
    """
    :param sample: a dictionary containing the following keys:
        - file_name: the name of the file
        - map: the name of the map, ex: us-ma-boston
        - split: the split, train, val or test
        - road_ids: the ids of the road elements
        - agent_ids: the ids of the agents in string
        - traffic_ids: the ids of the traffic lights
        - traffic_status: the status of the traffic lights
        - route_ids: the ids of the routes
        - frame_id: the frame id of the current frame
    :param data_path: the root path to load pickle files
    """
    filename = sample["file_name"].item()
    map = sample["map"].item()
    split = sample["split"].item()
    road_ids = sample["road_ids"].item()
    agent_ids = sample["agent_ids"].item()
    traffic_light_ids = sample["traffic_ids"].item()
    traffic_light_states = sample["traffic_status"].item()
    route_ids = sample["route_ids"].item()
    frame_id = sample["frame_id"].item()

    if os.path.exists(os.path.join(data_path, "map", f"{map}.pkl")):
        with open(os.path.join(data_path, "map", f"{map}.pkl"), "rb") as f:
            road_dic = pickle.load(f)
    else:
        print(f"Error: cannot load map {map} from {data_path}")
        return None

    # load agent and traffic dictionaries
    if os.path.exists(os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")):
        with open(os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl"), "rb") as f:
            data_dic = pickle.load(f)
            agent_dic = data_dic["agent_dic"]
    else:
        print(f"Error: cannot load {filename} from {data_path}")
        return None

    # calculate frames to sample
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate
    # for example,
    # [10, 11, ..., 10+(2+8)*20=210], interval=10
    # frames_to_sample = [10, 20, 30, .., 210]
    # [10, 11, ..., 10+(2+8)*20=210], past_interval=10, future_interval=1
    # frames_to_sample = [10, 20, 30, 31, .., 209, 210]
    sample_frames_in_past = list(range(scenario_start_frame, frame_id, past_sample_interval))
    sample_frames_in_future = list(range(frame_id, scenario_end_frame, future_sample_interval))
    sample_frames = sample_frames_in_past + sample_frames_in_future
    # sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))
    # initialize rasters
    total_raster_channels = 1 + road_types + traffic_types + agent_types
    trajectory_list = list()
    high_res_rasters_list = list()
    low_res_rasters_list = list()

    for frame in sample_frames:
        # update ego position
        ego_pose = agent_dic["ego"]["pose"][frame].copy()
        cos_, sin_ = math.cos(-ego_pose[3]), math.sin(-ego_pose[3])
        # trajectory label
        trajectory_label = agent_dic['ego']['pose'][frame + frame_sample_interval].copy()
        trajectory_label -= ego_pose
        traj_x = trajectory_label[0].copy()
        traj_y = trajectory_label[1].copy()
        trajectory_label[0] = traj_x * cos_ - traj_y * sin_
        trajectory_label[1] = traj_x * sin_ + traj_y * cos_
        trajectory_list.append(trajectory_label)

        # rasters encode
        rasters_high_res = np.zeros([raster_shape[0],
                                     raster_shape[1],
                                     total_raster_channels], dtype=np.uint8)
        rasters_low_res = np.zeros([raster_shape[0],
                                    raster_shape[1],
                                    total_raster_channels], dtype=np.uint8)
        rasters_high_res_channels = cv2.split(rasters_high_res)
        rasters_low_res_channels = cv2.split(rasters_low_res)
        # static roads elements drawing
        cos_, sin_ = math.cos(-ego_pose[3] - math.pi / 2), math.sin(-ego_pose[3] - math.pi / 2)
        for route_id in route_ids:
            if int(route_id) == -1:
                continue
            xyz = road_dic[int(route_id)]["xyz"].copy()
            xyz[:, :2] -= ego_pose[:2]
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            high_res_route = simplified_xyz * high_res_scale
            low_res_route = simplified_xyz * low_res_scale
            high_res_route = (high_res_route + raster_shape[0] // 2).astype('int32')
            low_res_route = (low_res_route + raster_shape[0] // 2).astype('int32')
            cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
            cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))

        # road channels drawing
        for road_id in road_ids:
            if int(road_id) == -1:
                continue
            xyz = road_dic[int(road_id)]["xyz"].copy()
            road_type = int(road_dic[int(road_id)]["type"])
            xyz[:, :2] -= ego_pose[:2]
            # simplify road vector, can simplify about half of all the points
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            high_res_road = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_road = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2

            if road_type in [5, 17, 18, 19]:
                cv2.fillPoly(
                    rasters_high_res_channels[road_type + 1], np.int32([high_res_road[:, :2]]), (255, 255, 255))
                cv2.fillPoly(
                    rasters_low_res_channels[road_type + 1], np.int32([low_res_road[:, :2]]), (255, 255, 255))
            else:
                for j in range(simplified_xyz.shape[0] - 1):
                    cv2.line(rasters_high_res_channels[road_type + 1], tuple(high_res_road[j, :2]),
                             tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                    cv2.line(rasters_low_res_channels[road_type + 1], tuple(low_res_road[j, :2]),
                             tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)
        # traffic channels drawing
        for idx, traffic_id in enumerate(traffic_light_ids):
            traffic_state = int(traffic_light_states[idx])
            if int(traffic_id) == -1 or int(traffic_id) not in list(road_dic.keys()):
                continue
            xyz = road_dic[int(traffic_id)]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ \
                                                         - simplified_xyz[:,1].copy() * sin_, \
                                                         simplified_xyz[:,0].copy() * sin_ \
                                                         + simplified_xyz[:,1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            # traffic state order is GREEN, RED, YELLOW, UNKNOWN
            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[1 + road_types + traffic_state],
                         tuple(high_res_traffic[j, :2]),
                         tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[1 + road_types + traffic_state],
                         tuple(low_res_traffic[j, :2]),
                         tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)
        # draw agent
        cos_, sin_ = math.cos(-ego_pose[3]), math.sin(-ego_pose[3])
        for agent_id in agent_ids:
            if agent_id == "null":
                continue
            pose = agent_dic[agent_id]['pose'][frame, :].copy()
            if pose[0] < 0 and pose[1] < 0:
                continue
            pose[:] -= ego_pose[:]
            agent_type = int(agent_dic[agent_id]['type'])
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            shape = agent_dic[agent_id]['shape'][frame, :]
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                            direction=-pose[3])
            rect_pts = np.array(rect_pts, dtype=np.int32)

            # draw on high resolution
            rect_pts_high_res = int(high_res_scale) * rect_pts + raster_shape[0] // 2
            cv2.drawContours(rasters_high_res_channels[1 + road_types + agent_type],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int32) + raster_shape[0] // 2
            cv2.drawContours(rasters_low_res_channels[1 + road_types + agent_type],
                             [rect_pts_low_res], -1, (255, 255, 255), -1)

        rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
        rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)
        high_res_rasters_list.append(rasters_high_res)
        low_res_rasters_list.append(rasters_low_res)


    
    result_to_return = {}
    result_to_return['trajectory'] = np.array(trajectory_list)
    # squeeze raster for less space occupy and faster disk write
    result_to_return['high_res_raster'] = np.array(high_res_rasters_list, dtype=bool).transpose(1, 2, 0, 3).reshape(224, 224, -1)
    result_to_return['low_res_raster'] = np.array(low_res_rasters_list, dtype=bool).transpose(1, 2, 0, 3).reshape(224, 224, -1)
    return result_to_return

def waymo_preprocess(sample, dynamic_center=False):
    filename = sample["file_name"]
    with open(filename, "rb") as f:
        info = pickle.load(f)

    track_infos = info['track_infos']

    track_index_to_predict = torch.tensor(info['tracks_to_predict']['track_index'])
    agent_trajs = torch.from_numpy(track_infos['trajs'])  # (num_objects, num_timestamp, 10)
    num_egos = track_index_to_predict.shape[0]

    num_agents, num_frames, num_attrs = agent_trajs.shape
    agent_trajs_res = torch.zeros((num_egos, 512, num_frames, num_attrs))
    if dynamic_center:
        agent_trajs_res[:, :, 0, 0], agent_trajs_res[:, :, 0, 1] = 0, 0
        for frame in range(num_frames):
            center, heading = agent_trajs[track_index_to_predict, frame, :3], agent_trajs[track_index_to_predict, frame, 6]

            if frame < num_frames - 1: 
                agent_trajs_res[:, :num_agents, frame + 1, :] = transform_trajs_to_center_coords(agent_trajs[:, frame + 1, :], center, heading, 6, dynamic_center)

            if frame == 10:
                map_polylines_data, map_polylines_mask = create_map_data_for_center_objects(
                    center_objects=center, heading=heading, map_infos=info['map_infos'],
                    center_offset=(30.0, 0),
                )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)
            
            for i, track in enumerate(track_index_to_predict):
                if agent_trajs[track, frame, -1] == False:
                    if frame < num_frames - 1: agent_trajs_res[i, :, frame + 1, -1] = False
    else:
        center, heading = agent_trajs[track_index_to_predict, 10, :3], agent_trajs[track_index_to_predict, 10, 6]
        agent_trajs_res[:, :num_agents, :, :] = transform_trajs_to_center_coords(agent_trajs, center, heading, 6, dynamic_center)
        map_polylines_data, map_polylines_mask = create_map_data_for_center_objects(
                    center_objects=center, heading=heading, map_infos=info['map_infos'],
                    center_offset=(30.0, 0),
                )
    
    agent_trajs_res = agent_trajs_res.permute(0, 2, 1, 3)
    map_data = torch.from_numpy(map_polylines_data)
    map_mask = torch.from_numpy(map_polylines_mask)
    ret_dict = {
        "agent_trajs": agent_trajs_res,
        "track_index_to_predict": track_index_to_predict.view(-1, 1),
        "map_polyline": map_data, 
        "map_polylines_mask": map_mask,
        }
    
    return ret_dict

def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, no_time_dim=False):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    if no_time_dim:
        num_objects, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_attrs).repeat(num_center_objects, 1,  1)
        obj_trajs[:, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, :]
        obj_trajs[:, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, 2)

        obj_trajs[:, :, heading_index] -= center_heading[:, None]
    else:
        num_objects, num_frame, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_frame, num_attrs).repeat(num_center_objects, 1,  1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_frame, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    return obj_trajs

def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

    # # CHECK the results
    # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
    # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
    # assert center_dist.max() < 10
    return ret_polylines, ret_polylines_mask

def create_map_data_for_center_objects(center_objects, heading, map_infos, center_offset):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """
    num_center_objects = center_objects.shape[0]

    # transform object coordinates by center objects
    def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-heading
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
            angle=-heading
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
        xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    polylines = torch.from_numpy(map_infos['all_polylines'].copy())

    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines.numpy(), point_sampled_interval=1,
        vector_break_dist_thresh=1.0,
        num_points_each_polyline=20,
    )  # (num_polylines, num_points_each_polyline, 9), (num_polylines, num_points_each_polyline)

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = 768
    map_polylines = torch.zeros((num_center_objects, num_of_src_polylines, batch_polylines.shape[-2], batch_polylines.shape[-1]))
    map_polylines_mask = torch.zeros((num_center_objects, num_of_src_polylines, batch_polylines.shape[-2]))
    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
        center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
        center_offset_rot = rotate_points_along_z(
            points=center_offset_rot.view(num_center_objects, 1, 2),
            angle=heading
        ).view(num_center_objects, 2)

        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

        dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
        topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
        map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines[:, :len(batch_polylines), :, :] = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
        map_polylines_mask[:, :len(batch_polylines), :] = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )

    map_polylines = map_polylines.numpy()
    map_polylines_mask = map_polylines_mask.numpy()

    return map_polylines, map_polylines_mask

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa,  sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
