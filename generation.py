from transformers import TrainingArguments, Trainer, TrainerCallback
import torch, pickle

from datasets import Dataset, Features, Value, Array2D, Sequence, Array4D
from dataset_gen.DataLoaderNuPlan import NuPlanDL
from dataset_gen.nuplan_obs import *
from torch.utils.data import DataLoader
import os, time
import importlib.util
import logging
import argparse
import numpy as np

import random

from datasets import Image

# from visulization.checkraster import *
intention_label_data_counter = [0] * 5

def main(args):
    running_mode = args.running_mode
    data_path = {
        'NUPLAN_DATA_ROOT': args.dataset_root,
        'NUPLAN_MAPS_ROOT': os.path.join(args.dataset_root, "maps"),
        'NUPLAN_DB_FILES': os.path.join(args.dataset_root, "nuplan-v1.1", args.data_path),
    }
    road_path = args.road_dic_path

    # print('validating image meta data')
    # meta_path = args.sensor_meta_path
    # sensor_path = args.sensor_blob_path
    # with open(meta_path, 'rb') as f:
    #     meta_folders = f.read()
    #     # split the meta folders string by \n string
    #     folders_from_meta = meta_folders.decode('utf-8').split('\n')[:-1]
    #     for each_folder_path in folders_from_meta:
    #         if 'File group' in each_folder_path:
    #             continue
    #         # fill complete path
    #         each_folder_path_global = os.path.join(sensor_path, each_folder_path)
    #         # check if the folder path is exist
    #         if not os.path.exists(each_folder_path_global):
    #             print(f"folder {each_folder_path_global} is not exist")
    #             exit()
    #         elif len(os.listdir(each_folder_path_global)) == 0:
    #             print(f"folder {each_folder_path_global} is empty")
    #             exit()
    #     exit()

    if args.city is not None:
        with open(args.vehicle_pickle_path, 'rb') as f:
            vehicle_sets = pickle.load(f)
            vehicle_set = vehicle_sets[args.city]
            print(f"{args.city} city vehicle set is {vehicle_set}")

    # check starting or ending number
    starting_file_num = args.starting_file_num if args.starting_file_num != -1 else None
    max_file_num = args.ending_file_num - starting_file_num if args.ending_file_num != -1 and starting_file_num is not None else None
    observation_kwargs = dict(
        max_dis=300,
        high_res_raster_shape=[224, 224], # for high resolution image, we cover 50 meters for delicated short-term actions
        # for high resolution image, we cover 50 meters for delicated short-term actions
        high_res_raster_scale=4.0,
        low_res_raster_shape=[224, 224], # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
        # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
        low_res_raster_scale=0.77,
        past_frame_num=40,
        future_frame_num=160,
        frame_sample_interval=4,
        action_label_scale=100,
    )
    # loaded_dic, _ = dl.get_next(seconds_in_future=9, sample_interval=20)
    # obs = get_observation_for_nsm(observation_kwargs, loaded_dic, 40, 201, nsm_result=None)
    # obs = get_observation_for_autoregression_basedon_previous_coor(observation_kwargs, loaded_dic, 40, 201, nsm_result=None)
    if args.filter_by_scenario_type:
        filter_scenario = ["starting_straight_traffic_light_intersection_traversal", "high_lateral_acceleration",
                           "changing_lane", "high_magnitude_speed", "low_magnitude_speed", "starting_left_turn",
                           "starting_right_turn", "stopping_with_lead", "following_lane_with_lead",
                           "near_multiple_vehicles",
                           "traversing_pickup_dropoff", "behind_long_vehicle", "waiting_for_pedestrian_to_cross",
                           "stationary_in_traffic"]
    else:
        filter_scenario = None

    if args.balance:
        # balance samples
        balance_dic = {"starting_straight_traffic_light_intersection_traversal": 0.4,
                       "high_lateral_acceleration": 2.2,
                       "changing_lane": 0.1,
                       "high_magnitude_speed": 48.6,
                       "low_magnitude_speed": 8.0,
                       "starting_left_turn": 1.6,
                       "starting_right_turn": 1.6,
                       "stopping_with_lead": 0.1,
                       "following_lane_with_lead": 0.1,
                       "near_multiple_vehicles": 3.8,
                       "traversing_pickup_dropoff": 14.6,
                       "behind_long_vehicle": 1.2,
                       "waiting_for_pedestrian_to_cross": 0.5,
                       "stationary_in_traffic": 17.3}

    if args.scenario_filter_yaml_path is not None:
        import yaml
        with open(args.scenario_filter_yaml_path) as file:
            loaded_yaml = yaml.full_load(file)
            scenarios_to_keep = loaded_yaml["scenario_tokens"]
            print('filtering with val14: ', len(scenarios_to_keep))
    else:
        scenarios_to_keep = None

    def yield_data_by_scenario(shards):
        for shard in shards:
            dl = NuPlanDL(scenario_to_start=0,
                          file_to_start=shard,
                          max_file_number=1,
                          data_path=data_path, db=None, gt_relation_path=None,
                          road_dic_path=None,
                          running_mode=running_mode)
            file_name = dl.file_names[0]
            if args.auto_regressive:
                seconds_in_future = 9
            else:
                seconds_in_future = 8
            while not dl.end:
                loaded_dic, _ = dl.get_next(seconds_in_future=seconds_in_future, sample_interval=args.sample_interval)
                if loaded_dic is None:
                    continue
                if loaded_dic["skip"]:
                    continue
                if loaded_dic["agent"]["ego"]["pose"][0][0] == -1:
                    continue
                if len(loaded_dic["route"]) == 0:
                    continue
                # if loaded_dic["type"] not in filter_scenario:
                #     continue
                loaded_dic["agent"]["ego"]["type"] = 7  # Fix Ego Type to 7
                if args.auto_regressive:
                    observation_dic = get_observation_for_autoregression_basedon_previous_coor(
                        observation_kwargs, loaded_dic, 40, 201, nsm_result=None
                    )
                # else:
                #     observation_dic = get_observation_for_nsm(
                #         observation_kwargs, loaded_dic, 40, 201, nsm_result=None)
                other_info = {
                    'file_name':file_name,
                    'scnario_type':loaded_dic["type"],
                    'time_stamp': loaded_dic['lidar_pc_tokens'][40].timestamp,
                    'lidar_token': loaded_dic['lidar_pc_tokens'][40].token,
                }
                observation_dic.update(other_info)
                yield observation_dic
            del dl

    def yield_data_index(shards):
        global intention_label_data_counter
        for shard in shards:
            dl = NuPlanDL(scenario_to_start=0,
                          file_to_start=shard,
                          max_file_number=1,
                          data_path=data_path, db=None, gt_relation_path=None,
                          road_dic_path=None,
                          running_mode=running_mode,
                          filter_scenario=filter_scenario,
                          keep_future_steps=args.keep_future_steps)

            while not dl.end:
                loaded_dic, _ = dl.get_next(seconds_in_future=15, sample_interval=args.sample_interval,
                                            map_name=args.map_name,
                                            scenarios_to_keep=scenarios_to_keep,
                                            filter_still=args.filter_still,
                                            sensor_meta_path=args.sensor_meta_path,
                                            sensor_blob_path=args.sensor_blob_path)
                if loaded_dic is None:
                    continue
                if args.keep_future_steps:
                    # loaded_dic is a list
                    for each_loaded_dic in loaded_dic:
                        if each_loaded_dic["skip"]:
                            continue
                        if each_loaded_dic["agent"]["ego"]["pose"][0][0] == -1:
                            continue
                        if len(each_loaded_dic["route"]) == 0:
                            continue
                        data_to_return = get_scenario_data_index(observation_kwargs, each_loaded_dic)
                        # legitimacy check
                        data_to_return_filtered = {}
                        error = False
                        for each_key in data_to_return:
                            if each_key is None:
                                print("WARNING: None key in data_to_return")
                                error = True
                            if data_to_return[each_key] is not None:
                                # check if none in list
                                if isinstance(data_to_return[each_key], type([])):
                                    filtered_list = []
                                    for each_element in data_to_return[each_key]:
                                        if each_element is None:
                                            print("WARNING: None element in ", each_key)
                                            error = True
                                        else:
                                            filtered_list.append(each_element)
                                    data_to_return_filtered[each_key] = filtered_list
                                else:
                                    data_to_return_filtered[each_key] = data_to_return[each_key]
                            else:
                                error = True
                                print("WARNING: None data in ", each_key)
                        if error:
                            continue
                        for each_intention in each_loaded_dic["intentions"]:
                            intention_label_data_counter[int(each_intention)] += 1
                        # intention_label_data_counter[int(each_loaded_dic["halfs_intention"])] += 1
                        if shard % 200 == 0:
                            print('intention_label_data_counter', intention_label_data_counter)
                        yield data_to_return_filtered
                else:
                    if loaded_dic["skip"]:
                        continue
                    if loaded_dic["agent"]["ego"]["pose"][0][0] == -1:
                        continue
                    if len(loaded_dic["route"]) == 0:
                        continue
                    if args.balance:
                        if 'scenario_type' not in loaded_dic:
                            print("WARNING: no scenario_type in loaded_dic", list(loaded_dic.keys()))
                            continue
                        if random.random() > 1.0 / balance_dic[loaded_dic["scenario_type"]]:
                            continue
                    data_to_return = get_scenario_data_index(observation_kwargs, loaded_dic)
                    # legitimacy check
                    data_to_return_filtered = {}
                    error = False
                    for each_key in data_to_return:
                        if each_key is None:
                            print("WARNING: None key in data_to_return")
                            error = True
                        if data_to_return[each_key] is not None:
                            # check if none in list
                            if isinstance(data_to_return[each_key], type([])):
                                filtered_list = []
                                for each_element in data_to_return[each_key]:
                                    if each_element is None:
                                        print("WARNING: None element in ", each_key)
                                        error = True
                                    else:
                                        filtered_list.append(each_element)
                                data_to_return_filtered[each_key] = filtered_list
                            else:
                                data_to_return_filtered[each_key] = data_to_return[each_key]
                        else:
                            error = True
                            print("WARNING: None data in ", each_key)
                    if error:
                        continue
                    for each_intention in data_to_return["intentions"]:
                        intention_label_data_counter[int(each_intention)] += 1
                    # intention_label_data_counter[int(data_to_return["halfs_intention"])] += 1
                    if shard % 200 == 0:
                        print('intention_label_data_counter', intention_label_data_counter)
                    yield data_to_return_filtered
            del dl

    def yield_data_dic(shards):
        sample_interval = 4
        for shard in shards:
            dl = NuPlanDL(scenario_to_start=0,
                          file_to_start=shard,
                          max_file_number=1,
                          data_path=data_path, db=None, gt_relation_path=None,
                          road_dic_path=road_path,
                          running_mode=running_mode)
            # 5 frames per second, sample every 4 frames
            loaded_dic = dl.get_next_file(specify_file_index=0, map_name=args.map_name, agent_only=True)#, sample_interval=sample_interval)
            if loaded_dic is None:
                continue
            if loaded_dic["skip"]:
                continue
            file_name = dl.file_names[0]
            # result = dict()
            # result["agent_dic"] = loaded_dic["agent"]
            # result["traffic_dic"] = loaded_dic["traffic_light"]
            loaded_dic["file_name"] = file_name
            loaded_dic["sample_interval"] = sample_interval
            # check if folder exists
            store_path = os.path.join(args.cache_folder, args.dataset_name)
            if not os.path.exists(store_path):
                os.makedirs(store_path, exist_ok=True)
            print("Storing at ", os.path.join(store_path, f"{file_name}.pkl"))
            with open(os.path.join(store_path, f"{file_name}.pkl"), "wb") as f:
                pickle.dump(loaded_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Stored at ", os.path.join(store_path, f"{file_name}.pkl"))
            if shard < 2:
                # inspect result
                print("Inspecting result\n**************************\n")
                print("ego pose shape: ", loaded_dic["agent"]["ego"]["pose"].shape, loaded_dic["agent"]["ego"]["speed"].shape, loaded_dic["agent"]["ego"]["starting_frame"], loaded_dic["agent"]["ego"]["ending_frame"])
            yield {'file_name': loaded_dic["file_name"]}
            del dl

    def yield_road_dic(shards):
        for shard in shards:
            dl = NuPlanDL(scenario_to_start=0,
                          file_to_start=shard,
                          max_file_number=1,
                          data_path=data_path, db=None, gt_relation_path=None,
                          road_dic_path=None,
                          running_mode=running_mode,
                          filter_scenario=filter_scenario)

            while not dl.end:
                loaded_dic, _ = dl.get_next(seconds_in_future=9, sample_interval=args.sample_interval,
                                            map_name=args.map_name)
                if loaded_dic is None:
                    continue
                if loaded_dic["skip"]:
                    continue
                if loaded_dic["agent"]["ego"]["pose"][0][0] == -1:
                    continue
                if len(loaded_dic["route"]) == 0:
                    continue
                road_dic = loaded_dic['road']
                map_name = loaded_dic['map']
                # save pickle of map
                store_path = args.cache_folder
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                print("Storing at ", os.path.join(store_path, f"{map_name}.pkl"))
                with open(os.path.join(store_path, f"{map_name}.pkl"), "wb") as f:
                    pickle.dump(road_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("Stored at ", os.path.join(store_path, f"{map_name}.pkl"))
                
                yield {'file_name': dl.file_names[0]}
                break
            del dl
            break

    # dic = yield_data_dic([0])
    starting_scenario = args.starting_scenario if args.starting_scenario != -1 else 0

    NUPLAN_DB_FILES = data_path['NUPLAN_DB_FILES']
    all_file_names = [os.path.join(NUPLAN_DB_FILES, each_path).split('/')[-1].split('.db')[0] for each_path in os.listdir(NUPLAN_DB_FILES) if
                      each_path[0] != '.']
    all_file_names = sorted(all_file_names)
    all_file_path = [os.path.join(NUPLAN_DB_FILES, each_path) for each_path in os.listdir(NUPLAN_DB_FILES) if
                     each_path[0] != '.']
    all_file_path = sorted(all_file_path)

    # file_indices = list(range(data_loader.total_file_num))
    total_file_num = len(os.listdir(data_path['NUPLAN_DB_FILES']))
    if args.ending_file_num == -1 or args.ending_file_num > total_file_num:
        args.ending_file_num = total_file_num
    file_indices = list(range(args.starting_file_num, args.ending_file_num))
    total_file_num = args.ending_file_num - args.starting_file_num
    # load filter pickle file
    if args.filter_pickle_path is not None:
        with open(args.filter_pickle_path, 'rb') as f:
            filter_dic = pickle.load(f)
        # filter file indices for faster loops while genrating dataset
        file_indices_filtered = []
        for idx, each_file_index in enumerate(file_indices):
            each_file = all_file_names[each_file_index]
            # each_file = each_file.split('/')[-1].split('.db')[0]
            if each_file in filter_dic:
                ranks = filter_dic[each_file]['rank']
                for rank in ranks:
                    if rank < args.filter_rank:
                        file_indices_filtered.append(each_file_index)
                        break
            else:
                print(f'file {each_file} not found in evaluation result pkl')
        print(
            f'Filtered {len(file_indices_filtered)} files from {total_file_number} files and {len(list(filter_dic.keys()))} keys')
        file_indices = file_indices_filtered
        print(file_indices)
        total_file_number = len(file_indices)
    else:
        filter_dic = None
    # filter test set by city
    if args.city is not None:
        specific_city_indices = []
        for idx, file_name in enumerate(all_file_names):
            if int(file_name.split('/')[-1][24:26]) in vehicle_set:
                specific_city_indices.append(idx)
        file_indices = specific_city_indices
        print(f'{args.city} city has {len(file_indices)} files in testset! total {len(all_file_names)}')

    # sort by file size
    sorted_file_indices = []
    if args.city is not None:
        sorted_file_names = sorted(all_file_path, key=lambda x: os.stat(x).st_size)
        for i, each_file_name in enumerate(sorted_file_names):
            if int(each_file_name.split('/')[-1][24:26]) in vehicle_set:
                sorted_file_indices.append(all_file_path.index(each_file_name))
        print(f"after sort, {len(sorted_file_indices)} files are chosen")
    else:
        sorted_file_names = sorted(all_file_path, key=lambda x: os.stat(x).st_size)
        for i, each_file_name in enumerate(sorted_file_names):
            if all_file_path.index(each_file_name) in file_indices:
                sorted_file_indices.append(all_file_path.index(each_file_name))
    print(f"Total file num is {total_file_num}")
    sorted_file_indices = sorted_file_indices[:total_file_num]
    # order by processes
    file_indices = []
    for i in range(args.num_proc):
        file_indices += sorted_file_indices[i::args.num_proc]

    total_file_number = len(file_indices)
    print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}')
    # end of sorting
    if args.only_index:
        nuplan_dataset = Dataset.from_generator(yield_data_index,
                                                gen_kwargs={'shards': file_indices},
                                                writer_batch_size=10, cache_dir=args.cache_folder,
                                                num_proc=args.num_proc,
                                                features=Features({"route_ids": Sequence(Value("int64")),
                                                                   "road_ids": Sequence(Value("int64")),
                                                                   "traffic_ids": Sequence(Value("int64")),
                                                                   "traffic_status": Sequence(Value("int64")),
                                                                   "agent_ids": Sequence(Value("string")),
                                                                   "frame_id": Value("int64"),
                                                                   "file_name": Value("string"),
                                                                   "map": Value("string"),
                                                                   "timestamp": Value("int64"),
                                                                   "scenario_type": Value("string"),
                                                                   "t0_frame_id": Value("int64"),
                                                                   "scenario_id": Value("string"),
                                                                   # "halfs_intention": Value("int64"),
                                                                   "intentions": Sequence(Value("int64")),
                                                                   "mission_goal": Sequence(Value("float32")),
                                                                   "expert_goal": Sequence(Value("float32")),
                                                                   "navigation": Sequence(Value("int64")),
                                                                   "images_path": Sequence(Value("string")),
                                                                   }),)
    elif args.only_data_dic:
        nuplan_dataset = Dataset.from_generator(yield_data_dic,
                                                gen_kwargs={'shards': file_indices},
                                                writer_batch_size=10, cache_dir=args.cache_folder,
                                                num_proc=args.num_proc,
                                                features=Features({"file_name": Value("string")})
                                                )
        exit()
    elif args.save_map:
        nuplan_dataset = Dataset.from_generator(yield_road_dic,
                                                gen_kwargs={'shards': file_indices},
                                                writer_batch_size=10, cache_dir=args.cache_folder,
                                                num_proc=args.num_proc,
                                                features=Features({"file_name": Value("string")})
                                                )
        exit()
    elif args.by_scenario:
        nuplan_dataset = Dataset.from_generator(yield_data_by_scenario,
                                                gen_kwargs={'shards': file_indices},
                                                writer_batch_size=2, cache_dir=args.cache_folder,
                                                num_proc=args.num_proc
                                                )
    else:
        nuplan_dataset = Dataset.from_generator(yield_data,
                                                gen_kwargs={'shards': file_indices, 'dl': None,
                                                            'filter_info': filter_dic},
                                                writer_batch_size=10, cache_dir=args.cache_folder,
                                                num_proc=args.num_proc)
    print('Saving dataset with ', args.num_proc)
    nuplan_dataset.set_format(type="torch")
    nuplan_dataset.save_to_disk(os.path.join(args.cache_folder, args.dataset_name), num_proc=args.num_proc)
    print('Dataset saved')
    print('summary intention labels: ', intention_label_data_counter)
    exit()


if __name__ == '__main__':
    from pathlib import Path
    """
    python generation.py  --num_proc 40 --sample_interval 100  
    --dataset_name boston_index_demo  --starting_file_num 0  
    --ending_file_num 10000  --cache_folder /localdata_hdd/nuplan/online_demo/  
    --data_path train_boston  --only_data_dic
    
    python generation.py  --num_proc 40 --sample_interval 100  
    --dataset_name boston_index_interval100  --starting_file_num 0  
    --ending_file_num 10000  --cache_folder /localdata_hdd/nuplan/online_demo/  
    --data_path train_boston  --only_index  
    
    python generation.py  --num_proc 40 --sample_interval 1 --dataset_name pittsburgh_index_full  --starting_file_num 0  --ending_file_num 10000  --cache_folder /localdata_hdd/nuplan/online_pittsburgh_jul  --data_path train_pittsburgh --save_map
    python generation.py  --num_proc 40 --sample_interval 1  --dataset_name vegas2_datadic_float32  --starting_file_num 0  --ending_file_num 10000  --cache_folder /localdata_hdd/nuplan/vegas2_datadic_float32  --data_path train_vegas_2 --save_map
    """

    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--running_mode", type=int, default=None)
    parser.add_argument("--data_path", type=str, default="train_singapore")
    parser.add_argument("--dataset_root", type=str, default="/localdata_hdd/nuplan/dataset")
    parser.add_argument("--sensor_blob_path", type=str, default=None)
    parser.add_argument("--sensor_meta_path", type=str, default=None)
    parser.add_argument("--road_dic_path", type=str, default=str(Path.home()) + "/nuplan/dataset/pickles/road_dic.pkl")
    parser.add_argument("--nsm_label_path", type=str,
                        default="labels/intentions/nuplan_boston/training.wtime.0-100.iter0.pickle")

    parser.add_argument('--starting_file_num', type=int, default=0)
    parser.add_argument('--ending_file_num', type=int, default=10000)
    parser.add_argument('--starting_scenario', type=int, default=-1)
    parser.add_argument('--cache_folder', type=str, default='/localdata_hdd/nuplan_nsm')

    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--balance_rate', type=float, default=1.0,
                        help="balance sample rate of simple scenarios in nsm case")
    parser.add_argument('--sample_interval', type=int, default=200)
    parser.add_argument('--dataset_name', type=str, default='nsm')
    parser.add_argument('--auto_regressive', default=False, action="store_true")
    # pass in filter pickle file path to generate augment dataset
    parser.add_argument('--filter_pickle_path', type=str, default=None)
    parser.add_argument('--filter_rank', type=float, default=0.1,
                        help="keep data with rank lower than this value for dagger")
    parser.add_argument('--scaling_factor_for_dagger', type=float, default=4.0,
                        help="scale up low performance data by Nx for dagger")
    parser.add_argument('--by_scenario', default=False, action='store_true')
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--vehicle_pickle_path', default="vehicle.pkl")
    parser.add_argument('--only_index', default=False, action='store_true')
    parser.add_argument('--only_data_dic', default=False, action='store_true')
    # parser.add_argument('--save_playback', default=True, action='store_true')
    parser.add_argument('--map_name', type=str, default=None)
    parser.add_argument('--save_map', default=False, action='store_true')
    parser.add_argument('--scenario_filter_yaml_path', type=str, default=None)
    parser.add_argument('--filter_by_scenario_type', default=False, action='store_true')
    parser.add_argument('--keep_future_steps', default=False, action='store_true')  # use with scenario_filter_yaml_path for val14
    parser.add_argument('--balance', default=False, action='store_true')
    parser.add_argument('--filter_still', default=False, action='store_true')
    args_p = parser.parse_args()
    main(args_p)
