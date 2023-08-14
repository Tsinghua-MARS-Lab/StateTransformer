from datasets import Dataset
from dataset_gen.DataLoaderWaymo import WaymoDL

import os
import logging
import argparse

import pickle

import numpy as np
import shapely

import torch
from waymo_open_dataset.protos import scenario_pb2
from waymo_process_to_pickles.datapreprocess import decode_tracks_from_proto, decode_map_features_from_proto

def create_map_data(map_infos):
    map_polylines = []
    last_polyline_index = 0
    polyline_index = []
    for i, info in enumerate(map_infos['all_polylines']):
        if len(info) == 0: continue
        elif len(info) == 1:
            map_polylines.append(info[:, :2])
            polyline_index.append(last_polyline_index)
            last_polyline_index += 1
            continue 

        pts = list(zip(info[:, 0], info[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy

        assert len(simplified_x) == len(simplified_y) and len(simplified_x) > 0, len(simplified_x)

        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y

        map_polylines.append(simplified_xyz)
        polyline_index.append(last_polyline_index)
        last_polyline_index += len(simplified_xyz)
    
    if len(map_polylines) == 0:
        return np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=bool), [0]
    
    map_polylines = np.concatenate(map_polylines, axis=0, dtype=np.float32)
    mask = np.ones_like(map_polylines, dtype=bool)

    return map_polylines, mask, polyline_index

def main(args):
    data_path = args.data_path

    def yield_data(shards, dl, save_dict, output_path, interaction):
        for shard in shards:
            tf_dataset, file_name = dl.get_next_file(specify_file_index=shard)
            if tf_dataset is None:
                continue
            
            file_name = file_name + ".pkl"
            dicts_to_save = {}
            for data in tf_dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytearray(data.numpy()))

                if save_dict:
                    track_infos = decode_tracks_from_proto(scenario.tracks)
                    track_index_to_predict = torch.tensor([cur_pred.track_index for cur_pred in scenario.tracks_to_predict])
                    map_infos = decode_map_features_from_proto(scenario.map_features)
                    
                    agent_trajs = torch.from_numpy(track_infos['trajs'])  # (num_objects, num_timestamp, 10)
                    current_time_index = scenario.current_time_index
                    ego_index = torch.tensor([idx for idx in track_index_to_predict if agent_trajs[idx, current_time_index, -1] == 1], dtype=torch.int32)
                    # agent_dict = process_agents(agent_trajs=agent_trajs)
                    map_polylines_data, map_polylines_mask, polyline_index = create_map_data(map_infos=map_infos)
                        
                    dicts_to_save[scenario.scenario_id] = {
                        "agent_trajs": agent_trajs,
                        "track_index_to_predict": ego_index,
                        "map_polyline": map_polylines_data, 
                        "map_polylines_mask": map_polylines_mask,
                        "polyline_index": polyline_index,
                        "current_time_index": current_time_index,
                        # for evaluation
                        'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
                        'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
                        }
                
                if interaction:
                    interaction_agents = scenario.objects_of_interest
                    if len(interaction_agents) == 0:
                        continue
                    
                    assert len(interaction_agents) == 2
                    interaction_index = [i for i, cur_data in enumerate(scenario.tracks) if cur_data.id in interaction_agents]

                    if len(interaction_index) != 2: 
                        continue

                    yield {
                            "file_name": file_name,
                            "scenario_id": scenario.scenario_id,
                            "interaction_index": interaction_index,
                        }
                else:
                    yield {
                        "file_name": file_name,
                        "scenario_id": scenario.scenario_id,
                    }

            if len(dicts_to_save.keys()) > 0:
                with open(os.path.join(output_path, file_name), "wb") as f:
                    pickle.dump(dicts_to_save, f)
                    f.close()
    
    data_loader = WaymoDL(data_path=data_path, mode=args.mode, interaction=args.interaction)
    file_indices = []
    for i in range(args.num_proc):
        file_indices += range(data_loader.total_file_num)[i::args.num_proc]

    total_file_number = len(file_indices)
    print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}')

    os.makedirs(args.output_path, exist_ok=True)
    waymo_dataset = Dataset.from_generator(yield_data,
                                            gen_kwargs={'shards': file_indices, 'dl': data_loader, 'save_dict':
                                                        args.save_dict, 'output_path': args.output_path, 'interaction': args.interaction},
                                            writer_batch_size=10, cache_dir=args.cache_folder,
                                            num_proc=args.num_proc)
    print('Saving dataset')
    waymo_dataset.set_format(type="torch")
    waymo_dataset.save_to_disk(os.path.join(args.cache_folder, args.dataset_name), num_proc=args.num_proc)
    print('Dataset saved')

if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--data_path", type=dict, default={
            "WAYMO_DATA_ROOT": "/public/MARS/datasets/waymo_prediction_v1.2.0/scenario/",
            "SPLIT_DIR": {
                    'train': "training", 
                    'val': "validation",
                    'test': "testing",
                },
        })
    
    parser.add_argument('--mode', type=str, default="train")  
    parser.add_argument('--save_dict', default=False, action='store_true')
    parser.add_argument('--interaction', default=False, action='store_true')
    
    parser.add_argument('--output_path', type=str, default='/public/MARS/datasets/waymo_motion_cache/t4p_testing/data_dict')
    parser.add_argument('--cache_folder', type=str, default='/public/MARS/datasets/waymo_motion_cache/t4p_testing/waymo_cache')
    parser.add_argument('--dataset_name', type=str, default='t4p_waymo')

    parser.add_argument('--num_proc', type=int, default=20)

    args_p = parser.parse_args()
    main(args_p)