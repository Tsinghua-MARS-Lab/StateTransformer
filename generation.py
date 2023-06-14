from transformers import TrainingArguments, Trainer, TrainerCallback
import torch, pickle

from datasets import Dataset, Features, Value, Array2D, Sequence, Array4D
from dataset_gen.DataLoaderNuPlan import NuPlanDL
from dataset_gen.nuplan_obs import get_observation_for_nsm
from torch.utils.data import DataLoader
import os, time
import importlib.util
import logging
import argparse
import numpy as np

from visulization.checkraster import *
import pickle

def main(args):
    running_mode = args.running_mode

    # data_path = {
    #     'NUPLAN_DATA_ROOT': "/localdata_ssd" + "/nuplan/dataset",
    #     'NUPLAN_MAPS_ROOT': "/localdata_ssd" + "/nuplan/dataset/maps",
    #     'NUPLAN_DB_FILES': "/localdata_ssd" + "/nuplan/dataset/nuplan-v1.1/{}".format(args.data_path),
    # }
    data_path = {
        'NUPLAN_DATA_ROOT': "/localdata_hdd" + "/nuplan/dataset",
        'NUPLAN_MAPS_ROOT': "/localdata_hdd" + "/nuplan/dataset/maps",
        'NUPLAN_DB_FILES': "/localdata_hdd" + "/nuplan/dataset/nuplan-v1.1/{}".format(args.data_path),
    }
    road_path = args.road_dic_path
    if args.use_nsm:
        nsm_labels = None
        with open(args.nsm_label_path, 'rb') as f:
            # Load the object from the pickle file
            nsm_labels = pickle.load(f)
            print(f'NSM Labels loaded with {len(list(nsm_labels.keys()))} keys')
    
    if args.city is not None:
        with open(args.vehicle_pickle_path, 'rb') as f:
            vehicle_sets = pickle.load(f)
            vehicle_set = vehicle_sets[args.city]
            print(f"{args.city} city vehicle set is {vehicle_set}")

    # check starting or ending number
    starting_file_num = args.starting_file_num if args.starting_file_num != -1 else None
    max_file_num = args.ending_file_num - starting_file_num if args.ending_file_num != -1 and starting_file_num is not None else None
    # dl = NuPlanDL(scenario_to_start=0,
    #             file_to_start=0,
    #             max_file_number=max_file_num,
    #             data_path=data_path, db=None, gt_relation_path=None,
    #             road_dic_path=None,
    #             running_mode=running_mode)
    # scenarios, zero_file = dl.get_scenario_num()
    # print("Total scenario number is", scenarios, "zeros files", zero_file)
    # exit()
    observation_kwargs = dict(
        max_dis=500,
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

    def yield_data(shards, dl, filter_info=None):
        for shard in shards:
            # loaded_dic = dl.get_next_file(specify_file_index=shard)
            # file_name = dl.file_names[shard]
            dl = NuPlanDL(scenario_to_start=starting_scenario,
                          file_to_start=shard,
                          max_file_number=1,
                          data_path=data_path, db=None, gt_relation_path=None,
                          road_dic_path=road_path,
                          running_mode=running_mode)
            loaded_dic = dl.get_next_file(specify_file_index=0)
            file_name = dl.file_names[0]
            if args.use_nsm:
                nsm_result = nsm_labels[file_name] if file_name in nsm_labels else None
                if args.use_nsm and nsm_result is None:
                    print('ERROR: not found, ', file_name, nsm_labels['file_names'])
                    continue
                if loaded_dic is None:
                    print('Ending data loading, No more file to load, current index is: ', shard)
                    break
            else:
                nsm_result = None
            total_frames = len(loaded_dic['lidar_pc_tokens'])

            frames_to_sample = list(range(observation_kwargs['past_frame_num'] + 1,
                                          total_frames - observation_kwargs['future_frame_num'], args.sample_interval))
            if filter_info is not None:
                # only generate data for filtered frames
                frames_to_sample = []

                if file_name not in filter_info:
                    print('ERROR, file name not found after filter, ', file_name, list(filter_info.keys())[:10])
                    continue
                filter_info_this_file = filter_info[file_name]
                frames = filter_info_this_file['frame_index']
                ranks = filter_info_this_file['rank']
                assert len(frames) == len(ranks), f'ERROR, frame and rank length not match, {len(frames)}, {len(ranks)}'
                for idx, frame_idx in enumerate(frames):
                    data_rank = ranks[idx]
                    # loop for current frame
                    if data_rank < args.filter_rank:
                        # augment frames
                        interval = observation_kwargs["frame_sample_interval"]
                        frames_to_sample += list(range(frame_idx - interval, frame_idx + interval,
                                                       int(interval * 2 / args.scaling_factor_for_dagger)))
                        if len(frames_to_sample) == 0:
                            print('test: ', frame_idx, interval, args.scaling_factor_for_dagger)
                if len(frames_to_sample) == 0:
                    print('ERROR, no frames to add, ', frames, ranks)
                frames_to_sample = list(set(frames_to_sample))

            for t in frames_to_sample:

                if args.use_nsm:
                    current_frame_is_valid = nsm_result['valid_frames'][t]
                    target_frame_is_valid = nsm_result['valid_frames'][t + observation_kwargs['frame_sample_interval']]
                    # sample_frames = list(range(t - observation_kwargs["past_frame_num"], t + 1, observation_kwargs["frame_sample_interval"]))
                    # include future frames
                    sample_frames = list(range(t - observation_kwargs["past_frame_num"], t + 1 + 80))
                    skip = False
                    for frame in sample_frames:
                        if len(nsm_result['goal_actions_weights_per_frame'][frame]) == 0:
                            skip = True
                            break
                        if len(nsm_result['current_actions_weights_per_frame'][frame]) == 0:
                            skip = True
                            break
                    if skip:
                        continue
                        # if current_goal_maneuver.value == target_goal_maneuver.value: # downsampling
                    #     if np.random.rand() > args.sample_rate:
                    #         continue
                    if not current_frame_is_valid or not target_frame_is_valid:
                        continue
                    if len(nsm_result['goal_actions_weights_per_frame']) < t - observation_kwargs[
                        'frame_sample_interval'] - 1:
                        continue
                    if len(nsm_result['current_actions_weights_per_frame']) < t - observation_kwargs[
                        'frame_sample_interval'] - 1:
                        continue

                # if args.auto_regressive:
                #     observation_dic = get_observation_for_autoregression_nsm(
                #         observation_kwargs, loaded_dic, t, total_frames, nsm_result=nsm_result)
                # else:
                observation_dic = get_observation_for_nsm(
                    observation_kwargs, loaded_dic, t, total_frames, nsm_result=nsm_result)
                other_info = {
                    'file_name': file_name,
                    'scenario_id': '',  # empty for NuPlan
                    'time_stamp': loaded_dic['lidar_pc_tokens'][t].timestamp,
                    'frame_index': t,
                    'map_name': dl.current_dataset[0].map_name,
                    'lidar_token': loaded_dic['lidar_pc_tokens'][t].token,
                }
                if observation_dic is not None:
                    observation_dic.update(other_info)
                    yield observation_dic
                else:
                    continue
            del dl
        
    filter_scenario = ["starting_straight_traffic_light_intersection_traversal","high_lateral_acceleration",
    "changing_lane", "high_magnitude_speed", "low_magnitude_speed", "starting_left_turn",
    "starting_right_turn", "stopping_with_lead", "following_lane_with_lead","near_multiple_vehicles",
    "traversing_pickup_dropoff", "behind_long_vehicle", "waiting_for_pedestrian_to_cross", "stationary_in_traffic"]  
    def yield_data_by_scenario(shards):
        for shard in shards:
            dl = NuPlanDL(scenario_to_start=0,
                            file_to_start=shard,
                            max_file_number=1,
                            data_path=data_path, db=None, gt_relation_path=None,
                            road_dic_path=None,
                            running_mode=running_mode)
            file_name = dl.file_names[0]
            while not dl.end:
                loaded_dic, _ = dl.get_next(seconds_in_future=8, sample_interval=args.sample_interval)
                if loaded_dic is None:
                    continue
                if loaded_dic["skip"]:
                    continue
                if loaded_dic["agent"]["ego"]["pose"][0][0] == -1:
                    continue
                # if loaded_dic["type"] not in filter_scenario:
                #     continue
                loaded_dic["agent"]["ego"]["type"] = 7 # Fix Ego Type to 7
                observation_dic = get_observation_for_nsm(
                    observation_kwargs, loaded_dic, 40, 201, nsm_result=None)
                other_info = {
                    'file_name':file_name,
                    'scnario_type':loaded_dic["type"],
                    'time_stamp': loaded_dic['lidar_pc_tokens'][40].timestamp,
                    'lidar_token': loaded_dic['lidar_pc_tokens'][40].token,
                }
                observation_dic.update(other_info)
                yield observation_dic
            del dl
    
    starting_scenario = args.starting_scenario if args.starting_scenario != -1 else 0

    NUPLAN_DB_FILES = data_path['NUPLAN_DB_FILES']
    all_file_names = [os.path.join(NUPLAN_DB_FILES, each_path).split('/')[-1].split('.db')[0] for each_path in os.listdir(NUPLAN_DB_FILES) if
                      each_path[0] != '.']
    all_file_names = sorted(all_file_names)
    all_file_path = [os.path.join(NUPLAN_DB_FILES, each_path) for each_path in os.listdir(NUPLAN_DB_FILES) if
                     each_path[0] != '.']
    all_file_path = sorted(all_file_path)

    if args.use_nsm:
        nsm_file_names = nsm_labels['file_names']
        file_indices = []
        for idx, each_file in enumerate(all_file_names):
            if each_file in nsm_file_names:
                # check file is valid?
                if each_file not in nsm_labels:
                    print('Error, file name in names but not in dic?', idx, each_file)
                    continue
                if len(nsm_labels[each_file]['goal_actions_weights_per_frame']) == 0:
                    print('Error, empty goal actions', idx, each_file)
                    continue
                if len(nsm_labels[each_file]['current_actions_weights_per_frame']) == 0:
                    print('Error, empty current actions', idx, each_file)
                    continue
                file_indices.append(idx)
        print(f'loaded {len(file_indices)} from {len(nsm_file_names)} as {file_indices}')
    else:
        # file_indices = list(range(data_loader.total_file_num))
        total_file_num = len(os.listdir(data_path['NUPLAN_DB_FILES']))
        if args.ending_file_num == -1 or args.ending_file_num > total_file_num:
            args.ending_file_num = total_file_num
        file_indices = list(range(args.starting_file_num, args.ending_file_num))
    total_file_number = len(file_indices)
    # load filter pickle file
    if args.filter_pickle_path is not None:
        with open(args.filter_pickle_path, 'rb') as f:
            filter_dic = pickle.load(f)
        assert not args.use_nsm, NotImplementedError
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
    
    total_file_number = len(file_indices)
    print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}')

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
    # end of sorting
    if args.by_scenario:
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
    exit()


if __name__ == '__main__':
    from pathlib import Path

    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--running_mode", type=int, default=None)
    # parser.add_argument("--data_path", type=dict, default={
    #             'NUPLAN_DATA_ROOT': "/media/shiduozhang/My Passport/nuplan",
    #             'NUPLAN_MAPS_ROOT': "/media/shiduozhang/My Passport/nuplan/maps",
    #             'NUPLAN_DB_FILES': "/media/shiduozhang/My Passport/nuplan/train_boston",
    #         })
    parser.add_argument("--data_path", type=str, default="train_singapore")
    parser.add_argument("--road_dic_path", type=str, default=str(Path.home()) + "/nuplan/dataset/pickles/road_dic.pkl")
    parser.add_argument("--nsm_label_path", type=str,
                        default="labels/intentions/nuplan_boston/training.wtime.0-100.iter0.pickle")

    parser.add_argument('--starting_file_num', type=int, default=0)
    parser.add_argument('--ending_file_num', type=int, default=10000)
    parser.add_argument('--starting_scenario', type=int, default=-1)
    parser.add_argument('--cache_folder', type=str, default='/localdata_hdd/nuplan_nsm')

    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--use_nsm', default=False, action='store_true')
    parser.add_argument('--balance_rate', type=float, default=1.0,
                        help="balance sample rate of simple scenarios in nsm case")
    parser.add_argument('--sample_interval', type=int, default=200)
    parser.add_argument('--dataset_name', type=str, default='nsm')
    parser.add_argument('--auto_regressive', default=False)
    # pass in filter pickle file path to generate augment dataset
    parser.add_argument('--filter_pickle_path', type=str, default=None)
    parser.add_argument('--filter_rank', type=float, default=0.1,
                        help="keep data with rank lower than this value for dagger")
    parser.add_argument('--scaling_factor_for_dagger', type=float, default=4.0,
                        help="scale up low performance data by Nx for dagger")
    parser.add_argument('--by_scenario', default=False, action='store_true')
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--vehicle_pickle_path', default="vehicle.pkl")
    # parser.add_argument('--save_playback', default=True, action='store_true')
    args_p = parser.parse_args()
    main(args_p)