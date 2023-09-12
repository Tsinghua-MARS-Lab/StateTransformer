import numpy as np
import pickle
import os
import torch
from functools import partial
from torch.utils.data._utils.collate import default_collate

def waymo_collate_func(batch, data_path=None, use_intention=False):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    data_path = os.path.join(data_path, batch[0]["split"])
    
    map_func = partial(waymo_preprocess_mtr, use_intention=use_intention, data_path=data_path)
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
        if key in ["agent_trajs", "agent_trajs_input", "map_polylines", "map_polylines_mask", "map_polylines_center",]:
            dims = [b[key].shape[1] for b in new_batch]
            max_dim = max(dims)

            for idx_b, b in enumerate(new_batch):
                padded_input_size = [n for n in b[key].shape]
                padded_input_size[1] = max_dim
                padded_input = torch.zeros((padded_input_size)) if key != "polyline_index" else torch.ones((padded_input_size)) * -1
                for idx_s, scene in enumerate(b[key]):
                    padded_input[idx_s, :dims[idx_b]] = scene

                input_list.append(padded_input)
        else:
            input_list = [d[key] for d in new_batch]

        if key in ["scenario_id", "center_objects_type"]:
            input_dict[key] = default_collate(np.concatenate(input_list, axis=0))
        else:
            input_dict[key] = torch.cat(input_list, dim=0)

    return input_dict

def waymo_preprocess(sample, use_intention=False, data_path=None):
    filename = sample["file_name"]
    with open(os.path.join(data_path, filename), "rb") as f:
        info = pickle.load(f)

    scenario_id = sample["scenario_id"]
    data = info[scenario_id]
    
    agent_trajs = data['agent_trajs']  # (num_objects, num_timestamp, 10)
    current_time_index = data["current_time_index"]
    track_index_to_predict = torch.tensor([sample["ego_index"]]).to(torch.long)

    object_id, object_type = [], []
    if len(data["track_index_to_predict"]) == 1:
        assert data["track_index_to_predict"][0] == track_index_to_predict
        object_id.append(data['center_objects_id'])
        object_type.append(data['center_objects_type'])
    else:
        assert len(data["track_index_to_predict"]) == len(data['center_objects_id'])
        for idx in track_index_to_predict:
            index_of_track = np.where(data["track_index_to_predict"] == idx)
            object_id.append(data['center_objects_id'][index_of_track])
            object_type.append(str(data['center_objects_type'][index_of_track].item()))
            
    map_polyline = torch.from_numpy(data["map_polyline"])
    num_ego = len(track_index_to_predict)

    center_objects_world = agent_trajs[track_index_to_predict, current_time_index]
    center, heading = center_objects_world[..., :3], center_objects_world[..., 6]
    agent_trajs_res = transform_to_center(agent_trajs, center, heading, heading_index=6)
    map_polylines_data = transform_to_center(map_polyline, center, heading, no_time_dim=True)
    map_polylines_mask = torch.from_numpy(data["map_polylines_mask"]).unsqueeze(0).repeat(len(track_index_to_predict), 1, 1)
    polyline_index = data["polyline_index"]
    polyline_index.append(len(map_polyline))
    polyline_index = torch.tensor(polyline_index, dtype=torch.int32).unsqueeze(0).repeat(num_ego, 1)

    if use_intention:
        with open("/home/ldr/workspace/transformer4planning/data/waymo/cluster_64_center_dict.pkl", "rb") as f:
            intention_points_dict = pickle.load(f)
        intention_points = torch.tensor([intention_points_dict[agent_type] for agent_type in object_type], dtype=torch.float32)
    else:
        intention_points = torch.zeros((num_ego, ), dtype=torch.float32)
        

    ret_dict = {
        "agent_trajs": agent_trajs_res,
        "track_index_to_predict": track_index_to_predict.view(-1, 1),
        "map_polylines": map_polylines_data,
        "polyline_index": polyline_index,
        "map_polylines_mask": map_polylines_mask,
        "current_time_index": torch.tensor(current_time_index, dtype=torch.int32).repeat(num_ego).view(-1, 1),
        # for evaluation
        "scenario_id": [scenario_id] * num_ego,
        "center_objects_world": center_objects_world,
        "center_gt_trajs_src": agent_trajs[track_index_to_predict],
        "center_objects_id": torch.tensor(object_id, dtype=torch.int32).view(-1, 1),
        "center_objects_type": object_type,
        # for intention points
        "intention_points": intention_points
        }
    
    return ret_dict

def waymo_preprocess_mtr(sample, use_intention=False, data_path=None):
    scenario_id = sample["scenario_id"]
    with open(os.path.join(data_path, f'sample_{scenario_id}.pkl'), 'rb') as f:
        data = pickle.load(f)

    track_infos = data['track_infos']
    agent_trajs = torch.from_numpy(track_infos['trajs'])  # (num_objects, num_timestamp, 10)
    current_time_index = data["current_time_index"]
    track_index_to_predict = torch.tensor([sample["ego_index"]]).to(torch.long)

    object_id, object_type = [], []
    obj_ids = np.array(track_infos['object_id'])
    obj_types = np.array(track_infos['object_type'])
    object_id.append(obj_ids[track_index_to_predict])
    object_type.append(obj_types[track_index_to_predict])

    center_objects_world = agent_trajs[track_index_to_predict, current_time_index]
    center, heading = center_objects_world[..., :3], center_objects_world[..., 6]

    agent_trajs_res = transform_to_center(agent_trajs, center, heading, heading_index=6)

    obj_trajs = agent_trajs_res[:, :, :current_time_index + 1, :]
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
    ## generate the attributes for each object
    object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
    object_onehot_mask[:, obj_types == 'TYPE_PEDESTRIAN', :, 1] = 1  # TODO: CHECK THIS TYPO
    object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
    object_onehot_mask[torch.arange(num_center_objects), track_index_to_predict, :, 3] = 1
    object_onehot_mask[:, data['sdc_track_index'], :, 4] = 1

    object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = torch.tensor(data['timestamps_seconds'][:current_time_index + 1], dtype=torch.float32)

    object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = torch.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = torch.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = torch.cat((
        obj_trajs[:, :, :, 0:6], 
        object_onehot_mask,
        object_time_embedding, 
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9], 
        acce,
    ), dim=-1)
    
    map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
                center_objects=center_objects_world, map_infos=data['map_infos'],
                center_offset=(30.0, 0),
            )
    
    num_ego = len(track_index_to_predict)
    if use_intention:
        with open("/home/ldr/workspace/transformer4planning/data/waymo/cluster_64_center_dict.pkl", "rb") as f:
            intention_points_dict = pickle.load(f)
        intention_points = torch.tensor([intention_points_dict[agent_type] for agent_type in object_type], dtype=torch.float32)
    else:
        intention_points = torch.zeros((num_ego, ), dtype=torch.float32)

    ret_dict = {
        "agent_trajs": agent_trajs_res,
        "agent_trajs_input": ret_obj_trajs,
        "track_index_to_predict": track_index_to_predict.view(-1, 1),
        "map_polylines": map_polylines_data,
        "map_polylines_mask": map_polylines_mask,
        "map_polylines_center": map_polylines_center,
        "current_time_index": torch.tensor(current_time_index, dtype=torch.int32).repeat(num_ego).view(-1, 1),
        # for evaluation
        "scenario_id": [scenario_id] * num_ego,
        "center_objects_world": center_objects_world,
        "center_gt_trajs_src": agent_trajs[track_index_to_predict],
        "center_objects_id": torch.tensor(object_id, dtype=torch.int32).view(-1, 1),
        "center_objects_type": object_type,
        # for intention points
        "intention_points": intention_points
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

def create_map_data_for_center_objects(center_objects, map_infos, center_offset):
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
            angle=-center_objects[:, 6]
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
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
    center_objects = center_objects

    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines.numpy(), point_sampled_interval=1,
        vector_break_dist_thresh=1.0,
        num_points_each_polyline=20,
    )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = 768

    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
        center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
        center_offset_rot = rotate_points_along_z(
            points=center_offset_rot.view(num_center_objects, 1, 2),
            angle=center_objects[:, 6]
        ).view(num_center_objects, 2)

        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

        dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
        topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
        map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
        map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )

    temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
    map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

    map_polylines = map_polylines
    map_polylines_mask = map_polylines_mask
    map_polylines_center = map_polylines_center

    return map_polylines, map_polylines_mask, map_polylines_center