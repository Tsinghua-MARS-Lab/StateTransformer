import os
import gc
import pickle
import numpy as np
import torch
from copy import deepcopy
from shapely.geometry import Point
import time
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from transformer4planning.preprocess.pdm_vectorize import (get_discrete_centerline, 
                                                           get_drivable_area_map,
                                                           load_route_dicts,
                                                           route_roadblock_correction,
                                                           get_starting_lane,
                                                           convert_absolute_to_relative_se2_array,
                                                           PDMPath,
                                                           )

MAP_API=dict()
for map in ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']:
    MAP_API[map] = get_maps_api(
                        map_root="/public/MARS/datasets/nuPlan/nuplan-maps-v1.1",
                        # map_root="/localdata_ssd/nuplan/nuplan-maps-v1.1",
                        map_version="nuplan-maps-v1.0",
                        map_name=map
                        )
    
def get_centerline(sample, split, data_path, 
                   map_radius=50, centerline_samples=120, centerline_interval=1.0,
                   frame_rate=20, past_seconds=2, frame_frequency_rate=2):
    """
    Args:
        sample: the data unit in datasets, include file_name, frame_id and map etc.
        
    """
    filename = sample["file_name"]
    map = sample["map"]
    frame_id = sample["frame_id"]
    route_ids = sample["route_ids"]
    if isinstance(frame_id, torch.Tensor):
        frame_id = frame_id.item()
    if isinstance(route_ids, torch.Tensor):
        route_ids = route_ids.tolist()
    pickle_path = os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            data_dic = pickle.load(f)
            if 'agent_dic' in data_dic:
                agent_dic = data_dic["agent_dic"]
            elif 'agent' in data_dic:
                agent_dic = data_dic['agent']
            else:
                raise ValueError(f'cannot find agent_dic or agent in pickle file, keys: {data_dic.keys()}')
    else:
        print(f"Error: cannot load {filename} from {data_path} with {map}")
        return None
    
    map_api = MAP_API[map]
    # convert ego poses to nuplan format (x, y, heading)
    ego_poses = deepcopy(agent_dic["ego"]["pose"][(frame_id - past_seconds * frame_rate) // frame_frequency_rate:frame_id // frame_frequency_rate, :])
    ego_shape = agent_dic["ego"]["shape"][0]
    nuplan_ego_poses = [StateSE2(x=ego_pose[0], y=ego_pose[1], heading=ego_pose[-1]) for ego_pose in ego_poses]
    anchor_ego_pose = nuplan_ego_poses[-1]
    # build drivable area map and extract centerline    
    drivable_area_map = get_drivable_area_map(map_api, ego_poses[-1], map_radius=map_radius)
    # compute centerlines
    # _, init_route_dict = load_route_dicts(route_ids, map_api)
    # gc.collect()
    # gc.disable()
    # route_ids = route_roadblock_correction(ego_poses[-1], map_api, init_route_dict)
    route_lane_dict, route_block_dict = load_route_dicts(route_ids, map_api)
    # e2_time = time.time()
    # print("time to load corrected routes:", e2_time - e1_time)
    current_lane = get_starting_lane(ego_poses[-1], drivable_area_map, route_lane_dict, ego_shape)
    # e5_time = time.time()
    # print("time to get lane:", e5_time - e4_time)
    centerline = PDMPath(get_discrete_centerline(current_lane, route_block_dict, route_lane_dict))
    current_progress = centerline.project(Point(*anchor_ego_pose.array))
    centerline_progress_values = (
        np.arange(centerline_samples, dtype=np.float64) * centerline_interval + current_progress
    )
    planner_centerline = convert_absolute_to_relative_se2_array(
        anchor_ego_pose,
        centerline.interpolate(centerline_progress_values, as_array=True),
    )

    return planner_centerline

def centerline_map(sample, split, data_path):
    try:
        centerline = get_centerline(sample, split, data_path)
        sample["centerline"] = centerline
    except:
        print("Error: routes is incorrect")
        sample["centerline"] = None
    return sample

if __name__ == "__main__":
    import multiprocessing as mp
    import datasets
    from datasets.arrow_dataset import _concatenate_map_style_datasets
    from datasets import Dataset
    from functools import partial
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/localdata_ssd/nuplan/online_float32_opt")
    # parser.add_argument("--data_path", type=str, default="/public/MARS/datasets/nuPlanCache/online_float32_opt")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="/localdata_ssd/nuplan/centerline")
    parser.add_argument("--dataset_name", type=str, default="train")
    parser.add_argument("--num_proc", type=int, default=40)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=1)
    parser.add_argument("--method", type=str, default="map", help="choose from ['map', 'mp', 'sequential'], map is recommended")

    
    args = parser.parse_args()
        
    data_path = args.data_path
    # root = "/localdata_ssd/nuplan/online_float32_opt/index/val/"
    root = os.path.join(data_path, "index", args.split)
    subset_dirs = os.listdir(root)
    alldatasets = list()
    for i, subset_dir in enumerate(subset_dirs):
        if i >= args.start_id and i < args.end_id:
            print(f"loading {subset_dir}")
            dataset = datasets.load_from_disk(os.path.join(root, subset_dir))
            alldatasets.append(dataset)
        
    dataset = _concatenate_map_style_datasets(alldatasets)
    print(dataset)
    
    
    def yield_centerline(shards):
        for shard in shards:
            # filename, map, frame_id, route_id = filenames[shard], maps[shard], frame_ids[shard], route_ids[shard]
            # centerline_dic = get_centerline(filename, map, frame_id, route_id, args.split, data_path)
            sample = dataset[shard]
            centerline_dic = get_centerline(sample, args.split, data_path)
            yield centerline_dic 

    indices = range(len(dataset))
    print("begin to generate dataset, length is", len(indices))
    
    # dataset map
    if args.method == "map":
        func = partial(centerline_map, split=args.split, data_path=data_path)
        dataset = dataset.map(func, num_proc=args.num_proc)
        dataset.save_to_disk(os.path.join(args.cache_dir, args.dataset_name))
    
    # multiprocessing
    elif args.method == "mp":
        func = partial(get_centerline, split="train", data_path=data_path)
        with mp.Pool(processes=40) as pool:
            result = list(tqdm(pool.imap(func, dataset), total=len(indices)))
    
    elif args.method == "sequential":
        for i in tqdm(indices):
            try:
                centerline = get_centerline(dataset[i], args.split, data_path)
            except:
                print(f"Error: cannot load from {data_path} with {map}")
        
        