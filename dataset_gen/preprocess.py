import numpy as np
import pickle
import math
import cv2
import shapely
import os, time
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

    if len(new_batch) == 0:
        return None
    
    # process as data dictionary
    result = dict()
    for key in new_batch[0].keys():
        result[key] = default_collate([d[key] for d in new_batch])
    return result

def waymo_collate_func(batch, dic_path=None, dic_valid_path=None, interactive=False):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    split = batch[0]["split"]
    if split == "train":
        assert dic_path is not None
        data_path = dic_path
    elif split ==  "test":
        assert dic_valid_path is not None
        data_path = dic_valid_path
    else:
        raise NotImplementedError
    
    map_func = partial(waymo_preprocess, interactive=interactive, data_path=data_path)
    # with ThreadPoolExecutor(max_workers=len(batch)) as executor:
    #     batch = list(executor.map(map_func, batch))
    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)
    
    # process as data dictionary
    input_dict = dict()
    for key in new_batch[0].keys():
        input_list = []
        if key in ["agent_trajs", "map_polylines", "map_polylines_mask"]:
            dims = [b[key].shape[1] for b in new_batch]
            max_dim = max(dims)

            for idx_b, b in enumerate(new_batch):
                padded_input_size = [n for n in b[key].shape]
                padded_input_size[1] = max_dim
                padded_input = torch.zeros((padded_input_size))
                for idx_s, scene in enumerate(b[key]):
                    padded_input[idx_s, :dims[idx_b]] = scene

                input_list.append(padded_input)
        else:
            input_list = [d[key] for d in new_batch]

        if key in ["scenario_id", "center_objects_type"]:
            input_dict[key] = np.concatenate(input_list, axis=0).reshape(-1)
        else:
            input_dict[key] = torch.cat(input_list, dim=0)

    return {"input_dict": input_dict}

def augmentation():
    pass


def save_raster(result_dic, debug_raster_path, agent_type_num, past_frames_num, image_file_name, split,
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
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float32)
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
        cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '.png'), target_image)
    for each_key in ['context_actions', 'trajectory_label']:
        pts = result_dic[each_key]
        for scale in [high_scale, low_scale]:
            target_image = np.zeros(image_shape, dtype=np.float32)
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, :] = [255, 255, 255]
            cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '_' + str(scale) +'.png'), target_image)
    print('length of action and labels: ', result_dic['context_actions'].shape, result_dic['trajectory_label'].shape)
    print('debug images saved to: ', path_to_save, file_number)


def static_coor_rasterize(sample, data_path, raster_shape=(224, 224),
                          frame_rate=20, past_seconds=2, future_seconds=8,
                          high_res_scale=4, low_res_scale=0.77,
                          road_types=20, agent_types=8, traffic_types=4,
                          past_sample_interval=4, future_sample_interval=4,
                          debug_raster_path=None, all_maps_dic=None, all_pickles_dic=None,
                          frequency_change_rate=2):
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
        - frame_id: the frame id of the current frame, this is the global index which is irrelevant to frame rate of agent_dic pickles
        - debug_raster_path: if a debug_path past, will save rasterized images to disk, warning: will slow down the process
    :param data_path: the root path to load pickle files
    """
    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]
    frame_id = sample["frame_id"]
    road_ids = sample["road_ids"].tolist()
    agent_ids = sample["agent_ids"]  # list of strings
    traffic_light_ids = sample["traffic_ids"].tolist()
    traffic_light_states = sample["traffic_status"].tolist()
    route_ids = sample["route_ids"].tolist()

    if map == 'sg-one-north':
        y_inverse = -1
    else:
        y_inverse = 1

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
    pickle_path = os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")
    # if pickle_path not in all_pickles_dic:
    #     if os.path.exists(pickle_path):
    #         # current_time = time.time()
    #         # print('loading data from disk ', split, map, filename)
    #         with open(pickle_path, "rb") as f:
    #             data_dic = pickle.load(f)
    #             if 'agent_dic' in data_dic:
    #                 agent_dic = data_dic["agent_dic"]
    #             elif 'agent' in data_dic:
    #                 agent_dic = data_dic['agent']
    #             else:
    #                 raise ValueError(f'cannot find agent_dic or agent in pickle file, keys: {data_dic.keys()}')
    #         # time_spent = time.time() - current_time
    #         # print('loading data from disk done ', split, map, filename, time_spent, 'total frames: ', agent_dic['ego']['pose'].shape[0])
    #         if split == 'test':
    #             print('loading data from disk done ', split, map, filename, 'total frames: ', agent_dic['ego']['pose'].shape[0])
    #         all_pickles_dic[pickle_path] = agent_dic
    #     else:
    #         print(f"Error: cannot load {filename} from {data_path} with {map}")
    #         return None
    # else:
    #     agent_dic = all_pickles_dic[pickle_path]

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
        print(f"Error: cannot load {filename} from {data_path} with {map}")
        return None

    # if new version of data, using relative frame_id
    relative_frame_id = True if 'starting_frame' in agent_dic['ego'] else False

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
    origin_ego_pose = agent_dic["ego"]["pose"][frame_id//frequency_change_rate].copy()  # hard-coded resample rate 2
    # num_frame = torch.div(frame_id, frequency_change_rate, rounding_mode='floor')
    # origin_ego_pose = agent_dic["ego"]["pose"][num_frame].copy()  # hard-coded resample rate 2
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
            rotated_pose = [pose[0] * cos_ - pose[1] * sin_,
                            pose[0] * sin_ + pose[1] * cos_]
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
        action = rotated_poses[i//frequency_change_rate]  # hard-coded frequency change
        context_actions.append(action)

    # future trajectory
    # check if samples in the future is beyond agent_dic['ego']['pose'] length
    if relative_frame_id:
        sample_frames_in_future = (np.array(sample_frames_in_future, dtype=int) - agent_dic['ego']['starting_frame']) // frequency_change_rate
    if sample_frames_in_future[-1] >= agent_dic['ego']['pose'].shape[0]:
        # print('sample index beyond length of agent_dic: ', sample_frames_in_future[-1], agent_dic['ego']['pose'].shape[0])
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
        if split == 'test':
            save_raster(result_to_return, debug_raster_path, agent_types, len(sample_frames_in_past), image_file_name, split,
                        high_res_scale, low_res_scale)

    result_to_return["file_name"] = sample['file_name']
    result_to_return["map"] = sample['map']
    result_to_return["split"] = sample['split']
    result_to_return["frame_id"] = sample['frame_id']
    try:
        result_to_return["scenario_type"] = sample["scenario_type"]
    except:
        # to be compatible with older version dataset without scenario_type
        pass
    result_to_return["route_ids"] = sample['route_ids']

    # print('inspect shape: ', result_to_return['trajectory_label'].shape, result_to_return["context_actions"].shape)

    del agent_dic
    del road_dic

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

def waymo_preprocess(sample, interactive=False, data_path=None):
    filename = sample["file_name"]
    with open(os.path.join(data_path, filename), "rb") as f:
        info = pickle.load(f)

    scenario_id = sample["scenario_id"]
    data = info[scenario_id]
    
    agent_trajs = data['agent_trajs']  # (num_objects, num_timestamp, 10)
    current_time_index = data["current_time_index"]
    track_index_to_predict = data['track_index_to_predict'].to(torch.long)
    map_polyline = torch.from_numpy(data["map_polyline"])

    num_ego = len(track_index_to_predict)

    center_objects_world = agent_trajs[track_index_to_predict, current_time_index]
    center, heading = center_objects_world[..., :3], center_objects_world[..., 6]
    agent_trajs_res = transform_to_center(agent_trajs, center, heading, heading_index=6)
    map_polylines_data = transform_to_center(map_polyline, center, heading, no_time_dim=True)
    map_polylines_mask = torch.from_numpy(data["map_polylines_mask"]).unsqueeze(0).repeat(len(track_index_to_predict), 1, 1, )

    ret_dict = {
        "agent_trajs": agent_trajs_res,
        "track_index_to_predict": track_index_to_predict.view(-1, 1),
        "map_polylines": map_polylines_data, 
        "map_polylines_mask": map_polylines_mask,
        "current_time_index": torch.tensor(current_time_index, dtype=torch.int32).repeat(num_ego).view(-1, 1),
        # for evaluation
        "scenario_id": np.array([scenario_id] * num_ego).reshape(-1, 1),
        "center_objects_world": center_objects_world,
        "center_gt_trajs_src": agent_trajs[track_index_to_predict],
        "center_objects_id": torch.tensor(data['center_objects_id'], dtype=torch.int32).view(-1, 1),
        "center_objects_type": np.array(data['center_objects_type']).reshape(-1, 1),
        }
    
    return ret_dict

def transform_to_center(obj_trajs, center_xyz, center_heading, heading_index=None, no_time_dim=False):
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
        obj_trajs[:, :, 0:2] -= center_xyz[:, None, :2]
        obj_trajs[:, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, 2)

        if heading_index is not None: obj_trajs[:, :, heading_index] -= center_heading[:, None]
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

        if heading_index is not None: obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    return obj_trajs

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
