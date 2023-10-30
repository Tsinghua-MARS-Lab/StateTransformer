from datasets import Dataset
from dataset_gen.DataLoaderWaymo import WaymoDL

import os
import argparse
import pickle
import numpy as np
from waymo_open_dataset.protos import scenario_pb2

polyline_type = {
    # for lane
    'TYPE_UNDEFINED': -1,
    'TYPE_FREEWAY': 1,
    'TYPE_SURFACE_STREET': 2,
    'TYPE_BIKE_LANE': 3,

    # for roadline
    'TYPE_UNKNOWN': -1,
    'TYPE_BROKEN_SINGLE_WHITE': 6,
    'TYPE_SOLID_SINGLE_WHITE': 7,
    'TYPE_SOLID_DOUBLE_WHITE': 8,
    'TYPE_BROKEN_SINGLE_YELLOW': 9,
    'TYPE_BROKEN_DOUBLE_YELLOW': 10,
    'TYPE_SOLID_SINGLE_YELLOW': 11,
    'TYPE_SOLID_DOUBLE_YELLOW': 12,
    'TYPE_PASSING_DOUBLE_YELLOW': 13,

    # for roadedge
    'TYPE_ROAD_EDGE_BOUNDARY': 15,
    'TYPE_ROAD_EDGE_MEDIAN': 16,

    # for stopsign
    'TYPE_STOP_SIGN': 17,

    # for crosswalk
    'TYPE_CROSSWALK': 18,

    # for speed bump
    'TYPE_SPEED_BUMP': 19
}

def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [], 
        'object_type': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'trajs': []
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                              x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(cur_data.object_type)
        track_infos['trajs'].append(cur_traj)

    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos

def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    polylines = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = cur_data.lane.type  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.right_boundaries
            ]

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = cur_data.road_line.type

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = cur_data.road_edge.type

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            continue

        polylines.append(cur_polyline.astype(np.float32))
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    map_infos['all_polylines'] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(cur_signal.state)
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])
        
        if len(lane_id) == 0: continue
        
        dynamic_map_infos['lane_id'].append(lane_id)
        dynamic_map_infos['state'].append(state)
        dynamic_map_infos['stop_point'].append(stop_point)

    return dynamic_map_infos

def main(args):
    data_path = args.data_path

    def yield_data(shards, dl, save_dict, output_path):
        for shard in shards:
            tf_dataset, file_name = dl.get_next_file(specify_file_index=shard)
            if tf_dataset is None:
                continue
            
            dict_to_save = {}
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

                    dict_to_save[scenario.scenario_id] = info

                    # with open(os.path.join(output_path, scenario.scenario_id + ".pkl"), "wb") as f:
                    #     pickle.dump(info, f)
                    #     f.close()

                for i, index in enumerate(track_index_to_predict):
                    yield {
                            "scenario_id": scenario.scenario_id,
                            "track_index_to_predict": index,
                            "object_type": object_type_to_predict[i]
                        }
                    
            if len(dict_to_save.keys()) > 0:
                with open(os.path.join(output_path, file_name + ".pkl"), "wb") as f:
                    pickle.dump(dict_to_save, f)
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
    parser.add_argument('--output_path', type=str, default='/public/MARS/datasets/waymo_motion_cache/t4p_tmp/data_dict')
    parser.add_argument('--cache_folder', type=str, default='/public/MARS/datasets/waymo_motion_cache/t4p_tmp')
    parser.add_argument('--dataset_name', type=str, default='t4p_waymo')

    parser.add_argument('--num_proc', type=int, default=50)

    args_p = parser.parse_args()
    main(args_p)