import numpy as np
import pickle
import os
import torch
from functools import partial
from torch.utils.data._utils.collate import default_collate

def waymo_collate_func(batch, data_path=None, interaction=False):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    data_path = os.path.join(data_path, batch[0]["split"])
    
    map_func = partial(waymo_preprocess, interaction=interaction, data_path=data_path)
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
            input_dict[key] = default_collate(np.concatenate(input_list, axis=0))
        else:
            input_dict[key] = torch.cat(input_list, dim=0)

    if interaction: input_dict["agents_num_per_scenario"] = [len(d["track_index_to_predict"]) for d in new_batch]

    return input_dict

def waymo_preprocess(sample, interaction=False, data_path=None):
    filename = sample["file_name"]
    with open(os.path.join(data_path, filename), "rb") as f:
        info = pickle.load(f)

    scenario_id = sample["scenario_id"]
    data = info[scenario_id]
    
    agent_trajs = data['agent_trajs']  # (num_objects, num_timestamp, 10)
    current_time_index = data["current_time_index"]
    if interaction: track_index_to_predict = sample["interaction_index"].to(torch.long)
    else: track_index_to_predict = torch.tensor([sample["ego_index"]]).to(torch.long)
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
        "scenario_id": [scenario_id] * num_ego,
        "center_objects_world": center_objects_world,
        "center_gt_trajs_src": agent_trajs[track_index_to_predict],
        "center_objects_id": torch.tensor(data['center_objects_id'], dtype=torch.int32).view(-1, 1),
        "center_objects_type": [data['center_objects_type']] if isinstance(data['center_objects_type'], (str, bytes)) else data['center_objects_type'],
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