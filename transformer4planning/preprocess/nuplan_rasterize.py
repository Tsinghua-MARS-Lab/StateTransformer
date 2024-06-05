import numpy as np
import pickle
import math, random
import cv2
import shapely.geometry
import os
import torch
from functools import partial

from torch.utils.data._utils.collate import default_collate
from transformer4planning.utils.nuplan_utils import generate_contour_pts, normalize_angle, change_coordination
from transformer4planning.utils import nuplan_utils
from transformer4planning.utils.common_utils import save_raster

def nuplan_rasterize_collate_func(batch, dic_path=None, autoregressive=False, **encode_kwargs):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    # padding for tensor data
    expected_padding_keys = ["road_ids", "route_ids", "traffic_ids"]
    # expected_padding_keys = ["route_ids", "traffic_ids"]
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

    if len(new_batch) == 0:
        return {}
    
    # process as data dictionary
    result = dict()
    for key in new_batch[0].keys():
        if key is None:
            continue
        list_of_dvalues = []
        for d in new_batch:
            if d[key] is not None:
                list_of_dvalues.append(d[key])
            elif key == "scenario_type":
                list_of_dvalues.append('Unknown')
            else:
                print('Error: None value', key, d[key])   # scenario_type might be none for older dataset
        result[key] = default_collate(list_of_dvalues)
    return result


def static_coor_rasterize(sample, data_path, raster_shape=(224, 224),
                          frame_rate=20, past_seconds=2, future_seconds=8,
                          high_res_scale=4, low_res_scale=0.77,
                          road_types=20, agent_types=8, traffic_types=4,
                          past_sample_interval=2, future_sample_interval=2,
                          debug_raster_path=None, all_maps_dic=None, agent_dic=None,
                          frequency_change_rate=2,
                          previous_prediction_to_step=None,
                          **kwargs):
    """
    WARNING: frame_rate has been change to 10 as default to generate new dataset pickles, this is automatically processed by hard-coded logits
    :param sample: a dictionary containing the following keys:
        - file_name: the name of the file
        - map: the name of the map, ex: us-ma-boston
        - split: the split, train, val or test
        - road_ids: the ids of the road elements
        - agent_ids: the ids of the agents in string
        - traffic_ids: the ids of the traffic lights
        - traffic_status: the status of the traffic lights
        - route_ids: the ids of the routes
        - frame_id: the frame id of the current frame, this is the global index which is irrelevant to frame rate of agent_dic pickles (20Hz)
        - debug_raster_path: if a debug_path past, will save rasterized images to disk, warning: will slow down the process
    :param data_path: the root path to load pickle files
    starting_frame, ending_frame, sample_frame in 20Hz,
    """
    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]
    frame_id = sample["frame_id"]  # current frame of this sample

    if split == 'val14_1k':
        split = 'val'
    elif split == 'test_hard14_index':
        split = 'test'

    if "road_ids" not in sample.keys():
        assert False
    else:
        road_ids = sample["road_ids"]
        if not isinstance(road_ids, list):
            road_ids = road_ids.tolist()

    if "traffic_ids" not in sample.keys():
        traffic_light_ids = []
    else:
        traffic_light_ids = sample["traffic_ids"]
        if not isinstance(traffic_light_ids, list):
            traffic_light_ids = traffic_light_ids.tolist()
    if "traffic_status" not in sample.keys():
        traffic_light_states = []
    else:
        traffic_light_states = sample["traffic_status"].tolist()
        if not isinstance(traffic_light_states, list):
            traffic_light_states = traffic_light_states.tolist()
    route_ids = sample["route_ids"].tolist()
    if not isinstance(route_ids, list):
        route_ids = route_ids.tolist()

    if map == 'sg-one-north':
        y_inverse = -1
    else:
        y_inverse = 1

    ego_point = sample["mission_goal"].numpy() if "mission_goal" in sample.keys() and kwargs.get('use_mission_goal', False) else None

    # clean traffic ids, for legacy reasons, there might be -1 in the list
    traffic_light_ids = [x for x in traffic_light_ids if x != -1]
    assert len(traffic_light_ids) == len(traffic_light_states), f'length of ids is not same as length of states, ids: {traffic_light_ids}, states: {traffic_light_states}'

    if all_maps_dic is None:
        # print('loading map from disk ', map)
        if os.path.exists(os.path.join(data_path, "map", f"{map}.pkl")):
            with open(os.path.join(data_path, "map", f"{map}.pkl"), "rb") as f:
                road_dic = pickle.load(f)
        else:
            print(f"Error: cannot load map {map} from {data_path}")
            return None
        # print('loading map from disk done ', map)
    else:
        if map in all_maps_dic:
            road_dic = all_maps_dic[map]
        else:
            print(f"Error: cannot load map {map} from all_maps_dic, {list(all_maps_dic.keys())}")
            return None


    # load agent and traffic dictionaries
    # WARNING: load some pickles can be extremely slow costs about 500-1000 seconds for las vegas map
    # if agent_dic is not None:
    #     pass
    if filename is not None:
        path_per_city = os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")
        path_all_city = os.path.join(data_path, f"{split}", f"all_cities", f"{filename}.pkl")
        if os.path.exists(path_per_city):
            pickle_path = path_per_city
        elif os.path.exists(path_all_city):
            pickle_path = path_all_city
        else:
            print(f"Error: cannot load {path_per_city} from {data_path} with {map} or {path_all_city}")
            return None

        if os.path.exists(pickle_path):
            # current_time = time.time()
            # print('loading data from disk ', split, map, filename)
            with open(pickle_path, "rb") as f:
                data_dic = pickle.load(f)
                if 'agent_dic' in data_dic:
                    agent_dic = data_dic["agent_dic"]
                elif 'agent' in data_dic:
                    agent_dic = data_dic['agent']
                else:
                    raise ValueError(f'cannot find agent_dic or agent in pickle file, keys: {data_dic.keys()}')
            # time_spent = time.time() - current_time
            # print('loading data from disk done ', split, map, filename, time_spent, 'total frames: ', agent_dic['ego']['pose'].shape[0])
            # if split == 'test':
            #     print('loading data from disk done ', split, map, filename, 'total frames: ', agent_dic['ego']['pose'].shape[0])
    else:
        assert False, 'either filename or agent_dic should be provided for online process'

    augment_frame_id = kwargs.get('augment_index', 0)
    if previous_prediction_to_step is not None:
        if augment_frame_id != 0 and 'train' in split:
            frame_id = sample['old_frame_id']  # keep frame id as augmented as previous step
        if kwargs.get('finetuning_with_stepping_random_steps', False):
            aug_frame_offset = random.randint(0, 20)
        elif kwargs.get('finetuning_with_stepping_minimal_step', False):
            aug_frame_offset = 2
        else:
            aug_frame_offset = 20
        frame_id += aug_frame_offset
    elif augment_frame_id != 0 and 'train' in split:
        frame_id += random.randint(-augment_frame_id - 1, augment_frame_id)
        frame_id = max(frame_id, past_seconds * frame_rate)

    # if new version of data, using relative frame_id
    relative_frame_id = True if 'starting_frame' in agent_dic['ego'] else False

    if previous_prediction_to_step is not None or ("train" in split and kwargs.get('augment_current_pose_rate', 0) > 0):
        # copy agent_dic before operating to it
        ego_pose_agent_dic = agent_dic['ego']['pose'].copy()
    else:
        ego_pose_agent_dic = agent_dic['ego']['pose']

    if previous_prediction_to_step is not None:
        # WIP, about to give up in next submit
        # updating agent_dic with previous prediction
        # 1. turn prediction to global frame
        assert sample['old_file_name'] == sample['file_name'], f'old file name {sample["old_file_name"]} is not same as current file name {sample["file_name"]}'
        # assert abs(sample['old_frame_id'] - sample["frame_id"] + 20) == 0, f'old frame id {sample["old_frame_id"]} is not same as current frame id {frame_id}'
        assert previous_prediction_to_step.shape[
                   0] == 80, f'previous_prediction_to_step shape {previous_prediction_to_step.shape}'
        if isinstance(previous_prediction_to_step, torch.Tensor):
            previous_prediction_to_step = previous_prediction_to_step.cpu().detach().numpy()
        global_previous_prediction_to_step = previous_prediction_to_step.copy()
        # label y was reversed for y_inverse
        global_previous_prediction_to_step[:, 1] *= y_inverse
        assert relative_frame_id, 'previous prediction should be used with relative frame id'
        agent_starting_frame = agent_dic['ego']['starting_frame']
        previous_ego_pose_origin = ego_pose_agent_dic[
                                   (frame_id - agent_starting_frame - 20) // frequency_change_rate,
                                   :].copy()
        global_previous_prediction_to_step[:11, :2] = nuplan_utils.rotate_array(
            origin=np.array([0, 0]),
            points=global_previous_prediction_to_step[:11, :2],
            angle=previous_ego_pose_origin[-1]
        )
        global_previous_prediction_to_step += previous_ego_pose_origin
        assert - np.pi <= previous_ego_pose_origin[-1] <= np.pi, f'{previous_ego_pose_origin[-1]}'
        prediction_offset = global_previous_prediction_to_step[10, :] - ego_pose_agent_dic[
                                   (frame_id - agent_starting_frame) // frequency_change_rate,
                                   :].copy()
        prediction_offset[-1] = nuplan_utils.normalize_angle(prediction_offset[-1])

        # 2. update agent_dic with previous prediction
        ego_pose_agent_dic[(frame_id - agent_starting_frame) // frequency_change_rate - aug_frame_offset // frequency_change_rate:
        (frame_id - agent_starting_frame) // frequency_change_rate, :] = global_previous_prediction_to_step[:aug_frame_offset // frequency_change_rate, :]

        if kwargs.get('finetuning_with_stepping', False):
            # updating future poses for labels after stepping
            # Note this will not work if we predict without involving dynamic models, the trajectory will fly everywhere
            frames_left_in_prediction = 80 - aug_frame_offset // frequency_change_rate
            decay = np.ones((frames_left_in_prediction, 4)) * np.linspace(1, 0, frames_left_in_prediction).reshape(-1, 1)
            ego_pose_agent_dic[(frame_id - agent_starting_frame) // frequency_change_rate: (frame_id - agent_starting_frame) // frequency_change_rate + frames_left_in_prediction, :] += decay * prediction_offset[:]
            # for i in range(70):
            #     target_frame = (frame_id - agent_starting_frame) // frequency_change_rate + i
            #     if i > 0:
            #         scale = 1 - 70.0 / i
            #     else:
            #         scale = 1
            #     ego_pose_agent_dic[target_frame, :2] += prediction_offset[:2] * scale
            #     ego_pose_agent_dic[target_frame, -1] += prediction_offset[-1] * scale
            #     ego_pose_agent_dic[target_frame, -1] = nuplan_utils.normalize_angle(ego_pose_agent_dic[target_frame, -1])

    # calculate frames to sample
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate
    # for example,
    if kwargs.get('selected_exponential_past', True):
        # 2s, 1s, 0.5s, 0s
        # sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30]
        sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30, frame_id]
    elif kwargs.get('current_frame_only', False):
        sample_frames_in_past = [frame_id]
    else:
        # [10, 11, ...., 10+(2+8)*20=210], past_interval=2, future_interval=2, current_frame=50
        # sample_frames_in_past = [10, 12, 14, ..., 48], number=(50-10)/2=20
        sample_frames_in_past = list(range(scenario_start_frame, frame_id, past_sample_interval))  # add current frame in the end
    # sample_frames_in_future = [52, 54, ..., 208, 210], number=(210-50)/2=80
    sample_frames_in_future = list(range(frame_id + future_sample_interval, scenario_end_frame + future_sample_interval, future_sample_interval))  # + one step to avoid the current frame

    sample_frames = sample_frames_in_past + sample_frames_in_future
    # sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))

    # augment current position
    aug_current = 0
    aug_rate = kwargs.get('augment_current_pose_rate', 0)
    if "train" in split and aug_rate > 0 and random.random() < aug_rate:
        speed_per_step = nuplan_utils.euclidean_distance(
            ego_pose_agent_dic[frame_id // frequency_change_rate, :2],
            ego_pose_agent_dic[frame_id // frequency_change_rate - 5, :2]) / 5.0
        aug_x = 0.3 * speed_per_step
        aug_y = 0.3 * speed_per_step
        aug_yaw = 0.05  # 360 * 0.05 = 18 degree
        dx = (random.random() * 2 - 1) * aug_x
        dy = (random.random() * 2 - 1) * aug_y
        dyaw = (random.random() * 2 * np.pi - np.pi) * aug_yaw
        ego_pose_agent_dic[frame_id//frequency_change_rate, 0] += dx
        ego_pose_agent_dic[frame_id//frequency_change_rate, 1] += dy
        ego_pose_agent_dic[frame_id//frequency_change_rate, -1] += dyaw
        aug_current = 1
        # linearly project the past poses
        # generate a numpy array decaying from 1 to 0 with shape of 80, 4
        decay = np.ones((80, 4)) * np.linspace(1, 0, 80).reshape(-1, 1)
        decay[:, 0] *= dx
        decay[:, 1] *= dy
        decay[:, 2] *= 0
        decay[:, 3] *= dyaw
        ego_pose_agent_dic[frame_id // frequency_change_rate: frame_id // frequency_change_rate + 80, :] += decay

        # generate a numpy array raising from 0 to 1 with the shape of 20, 4
        raising = np.ones((20, 4)) * np.linspace(0, 1, 20).reshape(-1, 1)
        raising[:, 0] *= dx
        raising[:, 1] *= dy
        raising[:, 2] *= 0
        raising[:, 3] *= dyaw
        ego_pose_agent_dic[frame_id // frequency_change_rate - 21: frame_id // frequency_change_rate - 1, :] += raising

    # initialize rasters
    origin_ego_pose = ego_pose_agent_dic[frame_id//frequency_change_rate].copy()  # hard-coded resample rate 2

    if "agent_ids" not in sample.keys():
        if 'agent_ids_index' in sample.keys():
            agent_ids = []
            all_agent_ids = list(agent_dic.keys())
            for each_agent_index in sample['agent_ids_index']:
                if each_agent_index == -1:
                    continue
                if each_agent_index > len(all_agent_ids):
                    print(f'Warning: agent_ids_index is larger than agent_dic {each_agent_index} {len(all_agent_ids)}')
                    continue
                agent_ids.append(all_agent_ids[each_agent_index])
            assert 'ego' in agent_ids, 'ego should be in agent_ids'
        else:
            assert False
        # print('Warning: agent_ids not in sample keys')
        # agent_ids = []
        # max_dis = 300
        # for each_agent in agent_dic:
        #     starting_frame = agent_dic[each_agent]['starting_frame']
        #     target_frame = frame_id - starting_frame
        #     if target_frame < 0 or frame_id >= agent_dic[each_agent]['ending_frame']:
        #         continue
        #     pose = agent_dic[each_agent]['pose'][target_frame//frequency_change_rate, :].copy()
        #     if pose[0] < 0 and pose[1] < 0:
        #         continue
        #     pose -= origin_ego_pose
        #     if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
        #         continue
        #     agent_ids.append(each_agent)
    else:
        agent_ids = sample["agent_ids"]  # list of strings

    # num_frame = torch.div(frame_id, frequency_change_rate, rounding_mode='floor')
    # origin_ego_pose = agent_dic["ego"]["pose"][num_frame].copy()  # hard-coded resample rate 2
    if np.isinf(origin_ego_pose[0]) or np.isinf(origin_ego_pose[1]):
        assert False, f"Error: ego pose is inf {origin_ego_pose}, not enough precision while generating dictionary"
    # channels:
    # 0-2: route raster
    # 3-22: road raster
    # 23-26: traffic raster
    # 27-91: agent raster (64=8 (agent_types) * 8 (sample_frames_in_past))
    route_channel = 2
    route_channel += 1 if kwargs.get('use_mission_goal', False) else 0

    total_raster_channels = route_channel + road_types + traffic_types + agent_types * len(sample_frames_in_past)
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
        # raster route blocks
        xyz = road_dic[int(route_id)]["xyz"].copy()
        xyz[:, :2] -= origin_ego_pose[:2]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        simplified_xyz[:, 0] *= y_inverse
        high_res_route = (simplified_xyz * high_res_scale + raster_shape[0] // 2).astype('int32')
        low_res_route = (simplified_xyz * low_res_scale + raster_shape[0] // 2).astype('int32')

        cv2.fillPoly(rasters_high_res_channels[0], np.int32([high_res_route[:, :2]]), (255, 255, 255))
        cv2.fillPoly(rasters_low_res_channels[0], np.int32([low_res_route[:, :2]]), (255, 255, 255))

        # raster route lanes
        route_lanes = road_dic[int(route_id)]["lower_level"]
        for each_route_lane in route_lanes:
            xyz = road_dic[int(each_route_lane)]["xyz"].copy()
            xyz[:, :2] -= origin_ego_pose[:2]
            pts = list(zip(xyz[:, 0], xyz[:, 1]))
            line = shapely.geometry.LineString(pts)
            simplified_xyz_line = line.simplify(1)
            simplified_x, simplified_y = simplified_xyz_line.xy
            simplified_xyz = np.ones((len(simplified_x), 2)) * -1
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
            simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:, 1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
            simplified_xyz[:, 1] *= -1
            simplified_xyz[:, 0] *= y_inverse
            high_res_route = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_route = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[1], tuple(high_res_route[j, :2]),
                         tuple(high_res_route[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[1], tuple(low_res_route[j, :2]),
                         tuple(low_res_route[j + 1, :2]), (255, 255, 255), 2)

        # raster ego point
        if ego_point is not None and random.random() <= kwargs.get("mission_goal_dropout", 1):
            ego_point[:2] -= origin_ego_pose[:2]
            ego_point[0], ego_point[1] = ego_point[0].copy() * cos_ - ego_point[1].copy() * sin_, ego_point[0].copy() * sin_ + ego_point[1].copy() * cos_
            ego_point[1] *= -1
            ego_point[0] *= y_inverse
            high_res_ego_point = (ego_point * high_res_scale).astype('int32') + raster_shape[0] // 2
            low_res_ego_point = (ego_point * low_res_scale).astype('int32') + raster_shape[0] // 2
            cv2.circle(rasters_high_res_channels[2], tuple(high_res_ego_point[:2]), 3, (255, 255, 255), -1)
            cv2.circle(rasters_low_res_channels[2], tuple(low_res_ego_point[:2]), 3, (255, 255, 255), -1)

    # road raster
    for road_id in road_ids:
        if int(road_id) == -1:
            continue
        if int(road_id) not in road_dic:
            print('Warning: road_id not in road_dic! ', road_id)
            continue
        xyz = road_dic[int(road_id)]["xyz"].copy()
        road_type = int(road_dic[int(road_id)]["type"])
        assert 0 <= road_type < road_types, f'road_type {road_type} is larger than road_types {road_types}'
        xyz[:, :2] -= origin_ego_pose[:2]
        pts = list(zip(xyz[:, 0], xyz[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy
        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_xyz[:, 0].copy() * cos_ - simplified_xyz[:,1].copy() * sin_, simplified_xyz[:, 0].copy() * sin_ + simplified_xyz[:, 1].copy() * cos_
        simplified_xyz[:, 1] *= -1
        simplified_xyz[:, 0] *= y_inverse
        high_res_road = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
        low_res_road = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
        if road_type in [5, 17, 18, 19]:
            cv2.fillPoly(rasters_high_res_channels[road_type + route_channel], np.int32([high_res_road[:, :2]]), (255, 255, 255))
            cv2.fillPoly(rasters_low_res_channels[road_type + route_channel], np.int32([low_res_road[:, :2]]), (255, 255, 255))
        else:
            for j in range(simplified_xyz.shape[0] - 1):
                cv2.line(rasters_high_res_channels[road_type + route_channel], tuple(high_res_road[j, :2]),
                        tuple(high_res_road[j + 1, :2]), (255, 255, 255), 2)
                cv2.line(rasters_low_res_channels[road_type + route_channel], tuple(low_res_road[j, :2]),
                        tuple(low_res_road[j + 1, :2]), (255, 255, 255), 2)
    # traffic channels drawing
    for idx, traffic_id in enumerate(traffic_light_ids):
        traffic_state = int(traffic_light_states[idx])
        if int(traffic_id) == -1 or int(traffic_id) not in list(road_dic.keys()):
            continue
        assert 0 <= traffic_state < traffic_types, f'traffic_state {traffic_state} is larger than traffic_types {traffic_types}'
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
        simplified_xyz[:, 0] *= y_inverse
        high_res_traffic = (simplified_xyz * high_res_scale).astype('int32') + raster_shape[0] // 2
        low_res_traffic = (simplified_xyz * low_res_scale).astype('int32') + raster_shape[0] // 2
        # traffic state order is GREEN, RED, YELLOW, UNKNOWN
        for j in range(simplified_xyz.shape[0] - 1):
            cv2.line(rasters_high_res_channels[route_channel + road_types + traffic_state],
                     tuple(high_res_traffic[j, :2]),
                     tuple(high_res_traffic[j + 1, :2]), (255, 255, 255), 2)
            cv2.line(rasters_low_res_channels[route_channel + road_types + traffic_state],
                     tuple(low_res_traffic[j, :2]),
                     tuple(low_res_traffic[j + 1, :2]), (255, 255, 255), 2)
    # agent raster
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])

    for _, agent_id in enumerate(agent_ids):
        if agent_id == "null" or agent_id == 'ego':
            continue
        if agent_id not in list(agent_dic.keys()):
            print('unknown agent id', agent_id, type(agent_id))
            continue
        for i, sample_frame in enumerate(sample_frames_in_past):
            if relative_frame_id:
                agent_starting_frame = agent_dic[agent_id]['starting_frame']
                agent_ending_frame = agent_dic[agent_id]['ending_frame']
                if sample_frame < agent_starting_frame:
                    continue
                if agent_ending_frame != -1 and sample_frame >= agent_ending_frame:
                    continue
                pose = agent_dic[agent_id]['pose'][(sample_frame - agent_starting_frame)//frequency_change_rate, :].copy()  # Hard-coded frequency change
                shape = agent_dic[agent_id]['shape'][(sample_frame - agent_starting_frame)//frequency_change_rate, :]
            else:
                pose = agent_dic[agent_id]['pose'][sample_frame, :].copy()
                shape = agent_dic[agent_id]['shape'][sample_frame, :]
            if pose[0] < 0 and pose[1] < 0:
                continue
            pose -= origin_ego_pose
            agent_type = int(agent_dic[agent_id]['type'])
            assert 0 <= agent_type < agent_types, f'agent_type {agent_type} is larger than agent_types {agent_types}'
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
            # rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
            #   (shape[1], shape[0]), np.rad2deg(pose[3])))
            rect_pts = generate_contour_pts((rotated_pose[1], rotated_pose[0]), w=shape[0], l=shape[1],
                                            direction=-pose[3])
            rect_pts = np.array(rect_pts, dtype=np.int32)
            rect_pts[:, 0] *= y_inverse
            # draw on high resolution
            rect_pts_high_res = (high_res_scale * rect_pts).astype(np.int64) + raster_shape[0]//2
            # example: if frame_interval = 10, past frames = 40
            # channel number of [index:0-frame_0, index:1-frame_10, index:2-frame_20, index:3-frame_30, index:4-frame_40]  for agent_type = 0
            # channel number of [index:5-frame_0, index:6-frame_10, index:7-frame_20, index:8-frame_30, index:9-frame_40]  for agent_type = 1
            # ...
            cv2.drawContours(rasters_high_res_channels[route_channel + road_types + traffic_types + agent_type * len(sample_frames_in_past) + i],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_scale * rect_pts).astype(np.int64) + raster_shape[0]//2
            cv2.drawContours(rasters_low_res_channels[route_channel + road_types + traffic_types + agent_type * len(sample_frames_in_past) + i],
                             [rect_pts_low_res], -1, (255, 255, 255), -1)

    # context action computation
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    context_actions = list()
    ego_poses = ego_pose_agent_dic - origin_ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                              ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                              np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
    rotated_poses[:, 1] *= y_inverse

    if kwargs.get('use_speed', True):
        # speed, old data dic does not have speed key
        speed = agent_dic['ego']['speed']  # v, a, angular_v
        if speed.shape[0] == ego_poses.shape[0] * 2:
            speed = speed[::2, :]
        for i in sample_frames_in_past:
            selected_pose = rotated_poses[i // frequency_change_rate]  # hard-coded frequency change
            selected_pose[-1] = normalize_angle(selected_pose[-1])
            action = np.concatenate((selected_pose, speed[i // frequency_change_rate]))
            context_actions.append(action)
    else:
        for i in sample_frames_in_past:
            action = rotated_poses[i//frequency_change_rate]  # hard-coded frequency change
            action[-1] = normalize_angle(action[-1])
            context_actions.append(action)

    # future trajectory
    # check if samples in the future is beyond agent_dic['ego']['pose'] length
    if relative_frame_id:
        sample_frames_in_future = (np.array(sample_frames_in_future, dtype=int) - agent_dic['ego']['starting_frame']) // frequency_change_rate
    if sample_frames_in_future[-1] >= ego_pose_agent_dic.shape[0]:
        # print('sample index beyond length of agent_dic: ', sample_frames_in_future[-1], agent_dic['ego']['pose'].shape[0])
        return None

    result_to_return = dict()

    trajectory_label = ego_pose_agent_dic[sample_frames_in_future, :].copy()

    # get a planning trajectory from a CBC constant velocity planner
    # if kwargs.get('use_cbc_planner', False):
    #     from transformer4planning.rule_based_planner.nuplan_base_planner import MultiPathPlanner
    #     planner = MultiPathPlanner(road_dic=road_dic)
    #     planning_result = planner.plan_marginal_trajectories(
    #         my_current_pose=origin_ego_pose,
    #         my_current_v_mph=agent_dic['ego']['speed'][frame_id//frequency_change_rate, 0],
    #         route_in_blocks=sample['route_ids'].numpy().tolist(),
    #     )
    #     _, marginal_trajectories, _ = planning_result
    #     result_to_return['cbc_planning'] = marginal_trajectories
    if kwargs.get('use_cv_planner', False):
        from transformer4planning.rule_based_planner.nuplan_base_planner import MultiPathPlanner
        planner = MultiPathPlanner(road_dic=road_dic)
        planning_result = planner.plan_cv_trajectory(
            my_current_pose=origin_ego_pose,
            my_prev_step_pose=ego_pose_agent_dic[frame_id//frequency_change_rate - frequency_change_rate, :].copy()
        )
        result_to_return['cv_planning'] = planning_result
        if kwargs.get('use_cv_planner_stage1', False):
            trajectory_label -= origin_ego_pose
        else:
            trajectory_label -= result_to_return['cv_planning']
    else:
        trajectory_label -= origin_ego_pose
    traj_x = trajectory_label[:, 0].copy()
    traj_y = trajectory_label[:, 1].copy()
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_
    trajectory_label[:, 1] *= y_inverse

    rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
    rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)


    result_to_return["high_res_raster"] = np.array(rasters_high_res, dtype=bool)
    result_to_return["low_res_raster"] = np.array(rasters_low_res, dtype=bool)
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
    result_to_return['trajectory_label'] = trajectory_label.astype(np.float32)

    del rasters_high_res_channels
    del rasters_low_res_channels
    del rasters_high_res
    del rasters_low_res
    del trajectory_label

    # print('inspect: ', result_to_return["context_actions"].shape)

    camera_image_encoder = kwargs.get('camera_image_encoder', None)
    if camera_image_encoder is not None and 'test' not in split:
        import PIL.Image
        # load images
        if 'train' in split:
            images_folder = kwargs.get('train_camera_image_folder', None)
        elif 'val' in split:
            images_folder = kwargs.get('val_camera_image_folder', None)
        else:
            raise ValueError('split not recognized: ', split)

        images_paths = sample['images_path']
        if images_folder is None or len(images_paths) == 0:
            print('images_folder or images_paths not valid', images_folder, images_paths, filename, map, split, frame_id)
            return None
        if len(images_paths) != 8:
            # clean duplicate cameras
            camera_dic = {}
            for each_image_path in images_paths:
                camera_key = each_image_path.split('/')[1]
                camera_dic[camera_key] = each_image_path
            if len(list(camera_dic.keys())) != 8 or len(list(camera_dic.values())) != 8:
                print('images_paths length not valid, short? ', camera_dic, images_paths, camera_dic, filename, map, split, frame_id)
                return None
            else:
                images_paths = list(camera_dic.values())
            assert len(images_paths) == 8, images_paths

        # check if image exists
        one_image_path = os.path.join(images_folder, images_paths[0])
        if not os.path.exists(one_image_path):
            print('image folder not exist: ', one_image_path)
            return None
        else:
            images = []
            for image_path in images_paths:
                image = PIL.Image.open(os.path.join(images_folder, image_path))
                image.thumbnail((1080 // 4, 1920 // 4))
                # image = image.resize((1080//4, 1920//4))
                # image = cv2.imread(os.path.join(images_folder, image_path))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image is None:
                    print('image is None: ', os.path.join(images_folder, image_path))
                images.append(np.array(image, dtype=np.float32))

            # shape: 8(cameras), 1080, 1920, 3
            result_to_return['camera_images'] = np.array(images, dtype=np.float32)
            del images

    if debug_raster_path is not None and previous_prediction_to_step is not None:
    # if debug_raster_path is not None:
        # check if path not exist, create
        if not os.path.exists(debug_raster_path):
            os.makedirs(debug_raster_path)
        image_file_name = sample['file_name'] + '_' + str(int(sample['frame_id']))
        # if split == 'test':
        if map == 'sg-one-north':
            save_result = save_raster(result_to_return, debug_raster_path, agent_types, len(sample_frames_in_past),
                                      image_file_name, split, high_res_scale, low_res_scale)
            if save_result and 'images_path' in sample:
                # copy camera images
                for camera in sample['images_path']:
                    import shutil
                    path_to_save = split + '_' + image_file_name + '_' + str(os.path.basename(camera))
                    shutil.copy(os.path.join(images_folder, camera), os.path.join(debug_raster_path, path_to_save))

    result_to_return["file_name"] = sample['file_name']
    result_to_return["map"] = sample['map']
    result_to_return["split"] = sample['split']
    result_to_return["frame_id"] = sample['frame_id']
    result_to_return["scenario_type"] = 'Unknown'
    if 'scenario_type' in sample:
        result_to_return["scenario_type"] = sample['scenario_type']
    if 'scenario_id' in sample:
        result_to_return["scenario_id"] = sample['scenario_id']
    if 't0_frame_id' in sample:
        result_to_return["t0_frame_id"] = sample['t0_frame_id']
    if 'intentions' in sample and kwargs.get('use_proposal', False):
        result_to_return["intentions"] = sample['intentions']
    # try:
    #     result_to_return["scenario_type"] = sample["scenario_type"]
    # except:
    #     # to be compatible with older version dataset without scenario_type
    #     pass
    # try:
    #     result_to_return["scenario_id"] = sample["scenario_id"]
    # except:
    #     pass

    result_to_return["route_ids"] = sample['route_ids']
    result_to_return["aug_current"] = aug_current
    # print('inspect shape: ', result_to_return['trajectory_label'].shape, result_to_return["context_actions"].shape)
    if kwargs.get('closed_loop_eval', False) or kwargs.get('finetuning_with_stepping', False):
        for each_key in sample:
            # somehow the agent_ids in string will not be properly gathered across the batches
            if each_key not in ['road_ids', 'old_file_name', 'old_frame_id']:
                continue
            if each_key in result_to_return:
                continue
            result_to_return[each_key] = sample[each_key]
        result_to_return["data_path"] = data_path
        result_to_return["old_file_name"] = sample['file_name']
        result_to_return["old_frame_id"] = sample['frame_id']

        if 'agent_ids' in sample:
            size_of_agent_ids = len(sample['agent_ids'])
            agent_ids_index = []
            all_agent_ids = list(agent_dic.keys())
            for each_agent_id in sample['agent_ids']:
                if each_agent_id in all_agent_ids:
                    agent_ids_index.append(all_agent_ids.index(each_agent_id))
            short = size_of_agent_ids - len(agent_ids_index)
            agent_ids_index += [-1] * short
            # result_to_return["agent_ids_index"] = torch.tensor(agent_ids_index, dtype=torch.int64, device=sample['road_ids'].device)
            result_to_return["agent_ids_index"] = torch.tensor(agent_ids_index, dtype=torch.int64, device='cpu')

    del agent_dic
    del road_dic
    del ego_pose_agent_dic

    return result_to_return


def step_and_rasterize(high_res, low_res, previous_offset):
    """
    previous_offset: the offset of the previous frame, in the form of [x, y, 0, yaw]
    """
    dx, dy, _, dyaw = previous_offset
    # convert raster numpy array to cv2 image
    # move
    high_res_M = np.float32([[1, 0, -dx*4.0], [0, 1, -dy*4.0]])
    low_res_M = np.float32([[1, 0, -dx*0.77], [0, 1, -dy*0.77]])
    shifted_high_res = cv2.warpAffine(high_res, high_res_M, (high_res.shape[1], high_res.shape[0]))
    shifted_low_res = cv2.warpAffine(low_res, low_res_M, (low_res.shape[1], low_res.shape[0]))
    # rotate
    high_res_M = cv2.getRotationMatrix2D((high_res.shape[1]//2, high_res.shape[0]//2), -dyaw, 1)
    low_res_M = cv2.getRotationMatrix2D((low_res.shape[1]//2, low_res.shape[0]//2), -dyaw, 1)
    rotated_high_res = cv2.warpAffine(shifted_high_res, high_res_M, (shifted_high_res.shape[1], shifted_high_res.shape[0]))
    rotated_low_res = cv2.warpAffine(shifted_low_res, low_res_M, (shifted_low_res.shape[1], shifted_low_res.shape[0]))
    return rotated_high_res, rotated_low_res


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
        print(f"Error Raster Preprocess: cannot load map {map} from {data_path}")
        return None

    # load agent and traffic dictionaries
    if os.path.exists(os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")):
        with open(os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl"), "rb") as f:
            data_dic = pickle.load(f)
            agent_dic = data_dic["agent_dic"]
    else:
        print(f"Error Raster Preprocess: cannot load {filename} from {data_path}")
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