# from transformers import Pipeline

# from datasets import load_dataset, load_metric


# from transformers import TransfoXLTokenizer, TransfoXLModel
from transformers import TrainingArguments, Trainer, TrainerCallback
import torch

from datasets import Dataset, Features, Value, Array2D, IterableDataset
from DataLoaderNuPlan import NuPlanDL
import nuplan_obs as obs
from torch.utils.data import DataLoader
import os
import importlib.util
import logging
import argparse
import numpy as np

import pickle

from sklearn.metrics import classification_report

def compute_metrics_nuplan_nsm(model, dataset, batch_size: int):
    """
    Compute accuracy for the following classifications:
    1. intended_maneuver
    2. current_maneuver
    3. pos_x,
    4. pos_y
    """
    print('Computing metrics for classifications')
    intended_m_label = []
    predicted_intended_m_labels = []
    current_m_weights_bias = []
    action_bias_x = []
    action_bias_y = []

    device = model.device
    dataset.shuffle()
    if dataset is None:
        return

    def preprocess_data(examples):
        # take a batch of texts
        for each_key in examples:
            if isinstance(examples[each_key], type(torch.tensor(0))):
                examples[each_key] = examples[each_key].to(device)
        return examples

    # initialize intended maneuver metrics
    for input in dataset.iter(batch_size):
        input = preprocess_data(input)
        output = model(**input)
        intended_m_logits, current_m_logits, pos_x_logits, pos_y_logits = output.all_logits
        intended_m_label.append(input['intended_maneuver_label'])  # tensor
        predicted_intended_m_labels.append(torch.argmax(intended_m_logits, dim=-1))  # tensor
        current_m_weights_bias.append(torch.sum(abs(input['current_maneuver_label'] - current_m_logits), dim=1))

        pos_x = torch.argmax(pos_x_logits, dim=-1)
        pos_y = torch.argmax(pos_y_logits, dim=-1)
        action_label = input['action_label'].clone() + 100
        action_bias_x.append(abs(pos_x - action_label[:, 0]))
        action_bias_y.append(abs(pos_y - action_label[:, 1]))
        # only evaluate one batch
        break

    intended_m_label = torch.stack(intended_m_label, -1).flatten()
    predicted_intended_m_labels = torch.stack(predicted_intended_m_labels, -1).flatten()
    print('Intended Maneuver Classification')
    print(classification_report(predicted_intended_m_labels.cpu().numpy(), intended_m_label.cpu().numpy()))
    current_m_weights_bias = torch.stack(current_m_weights_bias, -1).flatten()
    print('Current Maneuver Classification')
    print(f'{np.average(current_m_weights_bias.cpu().numpy())} over 12')
    action_bias_x = torch.stack(action_bias_x, 0).cpu().numpy()
    print('Pose x offset: ', np.average(action_bias_x))
    action_bias_y = torch.stack(action_bias_y, 0).cpu().numpy()
    print('Pose y offset: ', np.average(action_bias_y))

class NuPlanNSMCallback(TrainerCallback):
    """
    do not use default compute_metrics and prediction_step methods
    as the logits the model returned do not fit into them well
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print('call back log: ', logs)
            with torch.no_grad():
                compute_metrics_nuplan_nsm(kwargs['model'], kwargs['eval_dataloader'],
                                           batch_size=args.per_device_eval_batch_size)

def main(args):

    use_config = args.config is not None
    if use_config:
        spec = importlib.util.spec_from_file_location('config', args.config)
        if spec is None:
            parser.error('Config file not found.')
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        env_config = config.EnvConfig()

        data_path = env_config.env.data_path
        road_path = env_config.env.road_dic_path
        gt_relation_path = env_config.env.relation_gt_path
        running_mode = env_config.env.running_mode
        # use nsm
        use_nsm = env_config.env.nsm
        if use_nsm:
            nsm_labels = None
            with open(env_config.env.nsm_label_path, 'rb') as f:
                # Load the object from the pickle file
                nsm_labels = pickle.load(f)
                print(f'NSM Labels loaded with {len(list(nsm_labels.keys()))} keys')
    else:
        raise NotImplementedError

    # check starting or ending number
    starting_file_num = args.starting_file_num if args.starting_file_num != -1 else None
    max_file_num = args.ending_file_num - starting_file_num if args.ending_file_num != -1 and starting_file_num is not None else None

    observation_kwargs = dict(
        max_dis=500,
        high_res_raster_shape=[224, 224],  # for high resolution image, we cover 50 meters for delicated short-term actions
        high_res_raster_scale=4.0,
        low_res_raster_shape=[224, 224],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
        low_res_raster_scale=0.77,
        past_frame_num=40,
        future_frame_num=160,
        frame_sample_interval=5,
        action_label_scale=100,
    )
    rasterize = True

    def yield_data(shards, dl):
        for shard in shards:
            loaded_dic = dl.get_next_file(specify_file_index=shard)
            file_name = dl.file_names[shard]
            nsm_result = nsm_labels[file_name] if file_name in nsm_labels else None
            if env_config.env.nsm and nsm_result is None:
                print('ERROR: not found, ', file_name, nsm_labels['file_names'])
                continue
            if loaded_dic is None:
                print('Ending data loading, No more file to load, current index is: ', shard)
                break
            
            total_frames = len(loaded_dic['lidar_pc_tokens'])
            for t in range(observation_kwargs['past_frame_num'] + 1,
                           total_frames - observation_kwargs['future_frame_num']):
                if env_config.env.nsm:
                    current_frame_is_valid = nsm_result['valid_frames'][t]
                    target_frame_is_valid = nsm_result['valid_frames'][t+observation_kwargs['frame_sample_interval']]
                    try:
                        current_goal_maneuver = nsm_result['goal_actions_weights_per_frame'][t][0]['action']
                        target_goal_maneuver = nsm_result['goal_actions_weights_per_frame'][t+observation_kwargs['frame_sample_interval']][0]['action']
                    except:
                        continue
                    if current_goal_maneuver.value == target_goal_maneuver.value: # downsampling
                        if np.random.rand() > 1.0/19:
                            continue
                    if not current_frame_is_valid or not target_frame_is_valid:
                        continue
                    if len(nsm_result['goal_actions_weights_per_frame']) < t - observation_kwargs['frame_sample_interval'] - 1:
                        continue
                    if len(nsm_result['current_actions_weights_per_frame']) < t - observation_kwargs['frame_sample_interval'] - 1:
                        continue
                    
                    observation_dic = obs.get_observation_for_nsm(
                        observation_kwargs, loaded_dic, t, total_frames, nsm_result=nsm_result)
                    other_info = {
                        'file_name': file_name,
                        'scenario_id': '',  # empty for NuPlan
                        'time_stamp': loaded_dic['lidar_pc_tokens'][t].timestamp,
                        'frame_index': t,
                        'map_name': 'boston',
                        'lidar_token': loaded_dic['lidar_pc_tokens'][t].token,
                    }
                    if observation_dic is not None:
                        observation_dic.update(other_info)
                        yield observation_dic
                    else:
                        continue
                else:
                    high_res_obs, low_res_obs, agent_vectors, road_vectors = obs.get_observation_per_frame(
                        observation_kwargs, loaded_dic, t, rasterize=rasterize)
                    '''
                    vectors have the following structure in an array with 7*N where N is the number of points:
                    [global_index, instance_index, local_index, instance_types, xs, ys, zs]
                    '''
                    if rasterize:
                        yield {
                            'file_name': file_name,
                            'scenario_id': '',  # empty for NuPlan
                            'time_stamp': loaded_dic['lidar_pc_tokens'][t].timestamp,
                            'frame_index': t,
                            'map_name': 'boston',
                            'lidar_token': loaded_dic['lidar_pc_tokens'][t].token,
                            'high_res_raster': high_res_obs,
                            'low_res_raster': low_res_obs,
                            'agent_vectors': agent_vectors,
                            'road_vectors': road_vectors,
                        }
                    else:
                        yield {
                            'file_name': file_name,
                            'scenario_id': '',  # empty for NuPlan
                            'time_stamp': loaded_dic['lidar_pc_tokens'][t].timestamp,
                            'frame_index': t,
                            'map_name': 'boston',
                            'lidar_token': loaded_dic['lidar_pc_tokens'][t].token,
                            'agent_vectors': agent_vectors,
                            'road_vectors': road_vectors,
                        }

    
        starting_scenario = args.starting_scenario if args.starting_scenario != -1 else 0
        data_loader = NuPlanDL(scenario_to_start=starting_scenario,
                                file_to_start=starting_file_num,
                                max_file_number=max_file_num,
                                data_path=data_path, db=None, gt_relation_path=gt_relation_path,
                                road_dic_path=road_path,
                                running_mode=running_mode)

        if use_nsm:
            nsm_file_names = nsm_labels['file_names']
            file_indices = []
            for idx, each_file in enumerate(data_loader.file_names):
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
            file_indices = list(range(data_loader.total_file_num))
        total_file_number = len(file_indices)
        print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}')

        nuplan_dataset = Dataset.from_generator(yield_data, gen_kwargs={'shards': file_indices, 'dl': data_loader},
                                                writer_batch_size=100, cache_dir=args.cache_folder,
                                                num_proc=args.num_proc)
        print('Saving dataset')
        nuplan_dataset.set_format(type="torch")
        nuplan_dataset.save_to_disk(args.cache_folder+'/nsm_sparse_balance')
        print('Dataset saved')
        exit()

if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/nuplan_training_config.py')
    parser.add_argument('--render', default=False, action='store_false')
    parser.add_argument('--method', type=str, default='unknown')
    parser.add_argument('--log_dir', type=str, default='sim_result')
    parser.add_argument('--overwrite', default=True, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--save_log', default=False, action='store_true')
    parser.add_argument('--starting_file_num', type=int, default=0)
    parser.add_argument('--ending_file_num', type=int, default=100)
    parser.add_argument('--starting_scenario', type=int, default=-1)
    parser.add_argument('--multi_process', default=False, action='store_true')
    parser.add_argument('--file_per_worker', type=int, default=1)
    parser.add_argument('--save_playback_data', default=False, action='store_true')
    parser.add_argument('--max_scenarios', type=int, default=100000)
    parser.add_argument('--cache_folder', type=str, default='/localdata_hdd/nuplan_nsm')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)

    # parser.add_argument('--save_playback', default=True, action='store_true')
    args_p = parser.parse_args()
    main(args_p)