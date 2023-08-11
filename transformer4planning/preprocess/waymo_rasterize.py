import numpy as np
import pickle
import math
import cv2
import shapely
import os
import torch
from functools import partial
from transformer4planning.preprocess.utils import (
    transform_trajs_to_center_coords, create_map_data_for_center_objects, 
)

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

def waymo_collate_func_offline(batch, **encode_kwargs): 
    # process as data dictionary
    result = dict()
    
    for key in batch[0].keys():
        if key == "agent_trajs":
            assert len(batch[0][key].shape) == 4
            _, T, _, H = batch[0][key].shape
            ego_length, agent_length = [], []
            for d in batch:
                ego_length.append(d[key].shape[0])
                agent_length.append(d[key].shape[2])
                
            ego_length_total = sum(ego_length)
            max_length = max(agent_length)

            agent_trajs = torch.zeros((ego_length_total, T, max_length, H))
            last_ego = 0
            for i, d in enumerate(batch): 
                agent_trajs[last_ego:last_ego + ego_length[i], :, :agent_length[i], :] = d[key]
                last_ego = last_ego + ego_length[i]
            
            result[key] = agent_trajs
            
        else: result[key] = torch.cat([d[key] for d in batch], dim=0)
    return result

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

