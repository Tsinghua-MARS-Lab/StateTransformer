from datasets import Dataset
from dataset_gen.DataLoaderWaymo import WaymoDL

import os
import argparse
import pickle
from waymo_open_dataset.protos import scenario_pb2
from waymo_process_to_pickles.datapreprocess import decode_tracks_from_proto, decode_map_features_from_proto, decode_dynamic_map_states_from_proto

def main(args):
    data_path = args.data_path

    def yield_data(shards, dl, save_dict, output_path):
        for shard in shards:
            tf_dataset, file_name = dl.get_next_file(specify_file_index=shard)
            if tf_dataset is None:
                continue
            file_name = file_name + ".pkl"
            dicts_to_save = {}
            for data in tf_dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytearray(data.numpy()))
                track_infos = decode_tracks_from_proto(scenario.tracks)

                object_type_to_predict, track_index_to_predict, difficulty_to_predict = [], [], []
                for cur_pred in scenario.tracks_to_predict:
                    cur_idx = cur_pred.track_index
                    if track_infos['object_type'][cur_idx] in args.agent_type:
                        object_type_to_predict.append(track_infos['object_type'][cur_idx])
                        track_index_to_predict.append(cur_idx)
                        difficulty_to_predict.append(cur_pred.difficulty)
                
                if len(track_index_to_predict) == 0: continue
                
                if save_dict:
                    info = {}
                    info['tracks_to_predict'] = {
                        'object_type': object_type_to_predict,
                        'track_index': track_index_to_predict,
                        'difficulty': difficulty_to_predict,
                    }

                    # decode map related data
                    map_infos = decode_map_features_from_proto(scenario.map_features)
                    dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

                    info.update({
                        'track_infos': track_infos,
                        'dynamic_map_infos': dynamic_map_infos,
                        'map_infos': map_infos
                    })

                    info['scenario_id'] = scenario.scenario_id
                    info['timestamps_seconds'] = list(scenario.timestamps_seconds)  # list of int of shape (91)
                    info['current_time_index'] = scenario.current_time_index # int, 10
                    info['sdc_track_index'] = scenario.sdc_track_index
                    info['objects_of_interest'] = list(scenario.objects_of_interest)
                    
                    dicts_to_save[scenario.scenario_id] = info

                for index in track_index_to_predict:
                    yield {
                            "file_name": file_name,
                            "scenario_id": scenario.scenario_id,
                            "track_index_to_predict": index,
                        }
                    
            if len(dicts_to_save.keys()) > 0:
                with open(os.path.join(output_path, file_name), "wb") as f:
                    pickle.dump(dicts_to_save, f)
                    f.close()

    os.makedirs(args.output_path, exist_ok=True)
    data_loader = WaymoDL(data_path=data_path, mode=args.mode)
    file_indices = []
    for i in range(args.num_proc):
        file_indices += range(data_loader.total_file_num)[i::args.num_proc]

    total_file_number = len(file_indices)
    print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}\n Agent type:', args.agent_type)

    waymo_dataset = Dataset.from_generator(yield_data,
                                            gen_kwargs={'shards': file_indices, 'dl': data_loader, 'save_dict':
                                                        args.save_dict, 'output_path': args.output_path},
                                            writer_batch_size=10, cache_dir=args.cache_folder,
                                            num_proc=args.num_proc)
    print('Saving dataset')
    waymo_dataset.set_format(type="torch")
    waymo_dataset.save_to_disk(os.path.join(args.cache_folder, args.dataset_name), num_proc=args.num_proc)
    print('Dataset saved')

if __name__ == '__main__':
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
    parser.add_argument('--agent_type', type=int, nargs="+", default=[3])
    parser.add_argument('--save_dict', default=False, action='store_true')
    parser.add_argument('--output_path', type=str, default='/public/MARS/datasets/waymo_motion_cache/t4p_testing/data_dict')
    parser.add_argument('--cache_folder', type=str, default='/public/MARS/datasets/waymo_motion_cache/t4p_testing')
    parser.add_argument('--dataset_name', type=str, default='t4p_waymo')

    parser.add_argument('--num_proc', type=int, default=20)

    args_p = parser.parse_args()
    main(args_p)