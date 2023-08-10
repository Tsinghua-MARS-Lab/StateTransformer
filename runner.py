# coding=utf-8

"""
Train a Transformer ML Model for Planning
"""

import logging
import os
import sys
import pickle
import copy
from typing import List, Optional, Dict, Any, Tuple, Union
import torch
from torch import nn
from tqdm import tqdm
import copy
import json

import datasets
import numpy as np
import evaluate
import transformers
from datasets import Dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets
from dataclasses import dataclass, field
from functools import partial

from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformer4planning.models.model import build_models
from transformer4planning.preprocess.nuplan_rasterize import nuplan_collate_func
from transformer4planning.utils import ModelArguments
from transformers.trainer_utils import get_last_checkpoint
from transformer4planning.trainer import PlanningTrainer, PlanningTrainingArguments, CustomCallback
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers.trainer_callback import DefaultFlowCallback

from datasets import Dataset, Features, Value, Array2D, Sequence, Array4D

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """
#     model_name: str = field(
#         default="scratch-gpt",
#         metadata={"help": "Name of a planning model backbone"}
#     )
#     model_pretrain_name_or_path: str = field(
#         default=None,
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     predict_result_saving_dir: Optional[str] = field(
#         default=False,
#         metadata={"help": "The target folder to save prediction results."},
#     )
#     predict_trajectory: Optional[bool] = field(
#         default=True,
#     )
#     d_embed: Optional[int] = field(
#         default=256,
#     )
#     d_model: Optional[int] = field(
#         default=256,
#     )
#     d_inner: Optional[int] = field(
#         default=1024,
#     )
#     n_layers: Optional[int] = field(
#         default=4,
#     )
#     n_heads: Optional[int] = field(
#         default=8,
#     )
#     # Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
#     activation_function: Optional[str] = field(
#         default="silu"
#     )
#     loss_fn: Optional[str] = field(
#         default="mse",
#     )
#     task: Optional[str] = field(
#         default="nuplan" # only for mmtransformer
#     )
#     with_traffic_light: Optional[bool] = field(
#         default=True
#     )
#     autoregressive: Optional[bool] = field(
#         default=False
#     )
#     k: Optional[int] = field(
#         default=1,
#         metadata={"help": "Set k for top-k predictions, set to -1 to not use top-k predictions."},
#     )
#     next_token_scorer: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Whether to use next token scorer for prediction."},
#     )
#     past_seq: Optional[int] = field(
#         # 20 frames / 4 = 5 frames per second, 5 * 2 seconds = 10 frames
#         # 20 frames / 10 = 2 frames per second, 2 * 2 seconds = 4 frames
#         default=10,
#         metadata={"help": "past frames to include for prediction/planning."},
#     )
#     x_random_walk: Optional[float] = field(
#         default=0.0
#     )
#     y_random_walk: Optional[float] = field(
#         default=0.0
#     )
#     tokenize_label: Optional[bool] = field(
#         default=True
#     )
#     raster_channels: Optional[int] = field(
#         default=33,
#         metadata={"help": "default is 0, automatically compute. [WARNING] only supports nonauto-gpt now."},
#     )
#     predict_yaw: Optional[bool] = field(
#         default=False
#     )
#     ar_future_interval: Optional[int] = field(
#         default=0,
#         metadata={"help": "default is 0, don't use auturegression. [WARNING] only supports nonauto-gpt now."},
#     )
#     arf_x_random_walk: Optional[float] = field(
#         default=0.0
#     )
#     arf_y_random_walk: Optional[float] = field(
#         default=0.0
#     )
#     trajectory_loss_rescale: Optional[float] = field(
#         default=1.0
#     )
#     visualize_prediction_to_path: Optional[str] = field(
#         default=None
#     )
#     pred_key_points_only: Optional[bool] = field(
#         default=False
#     )
#     specified_key_points: Optional[bool] = field(
#         default=False
#     )
#     forward_specified_key_points: Optional[bool] = field(
#         default=False
#     )
#     token_scenario_tag: Optional[bool] = field(
#         default=False
#     )
#     max_token_len: Optional[int] = field(
#         default=20
#     )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    saved_dataset_folder: Optional[str] = field(
        default=None, metadata={"help": "The path of a pre-saved dataset folder. The dataset should be saved by Dataset.save_to_disk())."}
    )
    saved_valid_dataset_folder: Optional[str] = field(
        default=None, metadata={"help": "The path of a pre-saved validation dataset folder. The dataset should be saved by Dataset.save_to_disk())."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )    

    dataset_scale: Optional[float] = field(
        default=1, metadata={"help":"The dataset size, choose from any float <=1, such as 1, 0.1, 0.01"}
    )
    dagger: Optional[bool] = field(
        default=False, metadata={"help":"Whether to save dagger results"}
    )
    online_preprocess: Optional[bool] = field(
        default=False, metadata={"help":"Whether to generate raster dataset online"}
    )
    datadic_path: Optional[str] = field(
        default=None, metadata={"help":"The root path of data dictionary pickle file"}
    )

@dataclass
class ConfigArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    save_model_config_to_path: Optional[str] = field(
        default=None, metadata={"help": "save current model config to a json file if not None"}
    )
    save_data_config_to_path: Optional[str] = field(
        default=None, metadata={"help": "save current data config to a json file if not None"}
    )
    load_model_config_from_path: Optional[str] = field(
        default=None, metadata={"help": "load model config from a json file if not None"}
    )
    load_data_config_from_path: Optional[str] = field(
        default=None, metadata={"help": "load data config to a json file if not None"}
    )

@dataclass
class DataProcessArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    past_sample_interval: Optional[int] = field(
        default=5
    )
    future_sample_interval: Optional[int] = field(
        default=2
    )
    debug_raster_path: Optional[str] = field(
        default=None
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, DataProcessArguments, PlanningTrainingArguments))
    model_args, data_args, config_args, data_process, training_args = parser.parse_args_into_dataclasses()

    # pre-compute raster channels number
    if model_args.raster_channels == 0:
        road_types = 20
        agent_types = 8
        traffic_types = 4
        past_sample_number = int(2 * 20 / data_process.past_sample_interval)  # past_seconds-2, frame_rate-20
        if 'auto' not in model_args.model_name:
            # will cast into each frame
            if model_args.with_traffic_light:
                model_args.raster_channels = 1 + road_types + traffic_types + agent_types
            else:
                model_args.raster_channels = 1 + road_types + agent_types


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Handle config loading and saving
    # if config_args.load_model_config_from_path is not None:
    #     # Load the data class object from the JSON file
    #     model_parser = HfArgumentParser(ModelArguments)
    #     model_args, = model_parser.parse_json_file(config_args.load_model_config_from_path, allow_extra_keys=True)
    #     print(model_args)
    #     logger.warning("Loading model args, this will overwrite model args from command lines!!!")
    # if config_args.load_data_config_from_path is not None:
    #     # Load the data class object from the JSON file
    #     data_parser = HfArgumentParser(DataTrainingArguments)
    #     data_args, = data_parser.parse_json_file(config_args.load_data_config_from_path, allow_extra_keys=True)
    #     logger.warning("Loading data args, this will overwrite data args from command lines!!!")
    # if config_args.save_model_config_to_path is not None:
    #     with open(config_args.save_model_config_to_path, 'w') as f:
    #         json.dump(model_args.__dict__, f, indent=4)
    # if config_args.save_data_config_to_path is not None:
    #     with open(config_args.save_data_config_to_path, 'w') as f:
    #         json.dump(data_args.__dict__, f, indent=4)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Pass in the directory to load a saved dataset
    # See generation.py to process and save a dataset from the NuPlan Dataset
    """
    Set saved dataset folder to load a saved dataset
    1. Pass None to load from data_args.saved_dataset_folder as the root folder path to load all sub-datasets of each city
    2. Pass the folder of an index files to load one sub-dataset of one city
    """
    if data_args.datadic_path is None:
        from datasets import disable_caching
        disable_caching()
        data_args.datadic_path = data_args.saved_dataset_folder
        # loop all datasets
        logger.info("Loading full set of datasets from {}".format(data_args.datadic_path))
        assert os.path.isdir(data_args.datadic_path)
        index_root = os.path.join(data_args.datadic_path, 'index')
        root_folders = os.listdir(index_root)
        if 'train' in root_folders:
            # load training datasets
            training_datasets = []
            training_index_root_folders = os.path.join(index_root, 'train')
            training_indices = os.listdir(training_index_root_folders)
            for training_index in training_indices:
                training_index_path = os.path.join(training_index_root_folders, training_index)
                if os.path.isdir(training_index_path):
                    # load training dataset
                    logger.info("Loading training dataset {}".format(training_index_path))
                    dataset = Dataset.load_from_disk(training_index_path)
                    if dataset is not None:
                        training_datasets.append(dataset)
            train_dataset = _concatenate_map_style_datasets(training_datasets)
            # add split column
            train_dataset.features.update({'split': Value('string')})
            train_dataset = train_dataset.add_column(name='split', column=['train'] * len(train_dataset))
            train_dataset.set_format(type='torch')
            train_samples = int(len(train_dataset) * float(data_args.dataset_scale))
            train_dataset = train_dataset.select(range(train_samples))
        else:
            raise ValueError("No training dataset found in {}, must include at least one city in /train".format(index_root))
        
        if training_args.do_eval and 'test' in root_folders:
            # load test datasets
            test_datasets = []
            test_index_root_folders = os.path.join(index_root, 'test')
            test_indices = os.listdir(test_index_root_folders)
            for test_index in test_indices:
                test_index_path = os.path.join(test_index_root_folders, test_index)
                if os.path.isdir(test_index_path):
                    # load test dataset
                    logger.info("Loading test dataset {}".format(test_index_path))
                    dataset = Dataset.load_from_disk(test_index_path)
                    if dataset is not None:
                        test_datasets.append(dataset)
            test_dataset = _concatenate_map_style_datasets(test_datasets)
            # add additional column for flagging test set
            test_dataset.features.update({'split': Value('string')})
            test_dataset = test_dataset.add_column('split', column=['test'] * len(test_dataset))
            test_dataset.set_format(type='torch')
        else:
            test_dataset = train_dataset
        all_maps_dic = {}
        all_pickles_dic = {}
        map_folder = os.path.join(data_args.datadic_path, 'map')
        for each_map in os.listdir(map_folder):
            if each_map.endswith('.pkl'):
                map_path = os.path.join(map_folder, each_map)
                with open(map_path, 'rb') as f:
                    map_dic = pickle.load(f)
                map_name = each_map.split('.')[0]
                all_maps_dic[map_name] = map_dic
    else:
        all_maps_dic = None
        all_pickles_dic = None
        if os.path.isdir(data_args.saved_dataset_folder):
            logger.info("loading dataset...")
            dataset = Dataset.load_from_disk(data_args.saved_dataset_folder)
            dataset.features.update({'split': Value('string')})
            dataset = dataset.add_column(name='split', column=['train'] * len(dataset))
            dataset.set_format(type='torch')
            dataset.shuffle(seed=training_args.seed)
            train_samples = int(len(dataset) * float(data_args.dataset_scale))
            train_dataset = dataset.select(range(train_samples))

            if training_args.do_eval:
                test_dataset = Dataset.load_from_disk(data_args.saved_valid_dataset_folder)
                # add additional column for flagging test set
                test_dataset.features.update({'split': Value('string')})
                test_dataset = test_dataset.add_column('split', column=['test'] * len(test_dataset))
            else:
                test_dataset = dataset.select(range(train_samples))
            test_dataset.set_format(type='torch')
        else:
            raise ValueError(f'Dataset directory ({data_args.saved_dataset_folder}) does not exist. Use save_to_disk() to save a dataset first.')

    # loop split info and update for test set
    print('TrainingSet: ', train_dataset, '\nTestSet', test_dataset)

    nuplan_dataset = dict(
        train=train_dataset.shuffle(seed=training_args.seed),
        validation=test_dataset.shuffle(seed=training_args.seed),
        test=test_dataset.shuffle(seed=training_args.seed),
    )

    # Load a model's pretrained weights from a path or from hugging face's model base
    model = build_models(model_args)
    clf_metrics = dict(
        accuracy=evaluate.load("accuracy"),
        f1=evaluate.load("f1"),
        precision=evaluate.load("precision"),
        recall=evaluate.load("recall")
    )
    if 'auto' in model_args.model_name and model_args.k == -1:  # for the case action label as token 
        model.clf_metrics = clf_metrics
    elif model_args.next_token_scorer:
        assert model_args.k > 1 and model_args.ar_future_interval > 0, "ar_future_interval must be greater than 0 and k must be greater than 1"
        model.clf_metrics = clf_metrics

    if training_args.do_train:
        import multiprocessing
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count() / 8))
        train_dataset = nuplan_dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = nuplan_dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        predict_dataset = nuplan_dataset["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Initialize our Trainer
    collate_fn = partial(nuplan_collate_func, autoregressive=model_args.autoregressive,
                         dic_path=data_args.datadic_path,
                         all_maps_dic=all_maps_dic,
                         all_pickles_dic=all_pickles_dic,
                         **data_process.__dict__) if data_args.online_preprocess else None
    trainer = PlanningTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=[CustomCallback,],
        data_collator=collate_fn
    )
    
    trainer.pop_callback(DefaultFlowCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        if model_args.autoregressive:
            result = trainer.evaluate()
            logger.info("***** Final Eval results *****")
            logger.info(f"  {result}")
            hyperparams = {"model": model_args.model_name, "dataset": data_args.saved_dataset_folder, "seed": training_args.seed}
            evaluate.save("./results/", ** result, ** hyperparams)
            logger.info(f" fde: {trainer.fde} ade: {trainer.ade}")

    if training_args.do_predict:
        # Currently only supports single GPU predict outputs
        """
        Will save prediction results, and dagger results if dagger is enabled
        """
        # TODO: fit new online process pipeline to save dagger and prediction results
        logger.info("*** Predict ***")
        with torch.no_grad():
            dagger_results = {
                'file_name':[],
                'frame_id':[],
                'rank':[],
                'ADE':[],
                'FDE':[],
                'y_bias':[]
            }
            prediction_results = {
                'file_names': [],
                'current_frame': [],
                'next_step_action': [],
                'predicted_trajectory': [],
            }
            test_dataloader = DataLoader(
                dataset=predict_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                num_workers=training_args.per_device_eval_batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True
            )


            if model_args.predict_trajectory:
                end_bias_x = []
                end_bias_y = []
                all_bias_x = []
                all_bias_y = []
                losses = []
                loss_fn = torch.nn.MSELoss(reduction="mean")
            
            # def waymo_collate_fn(batch):
            #     import collections
            #     expect_keys = expect_keys = ["high_res_raster", "low_res_raster", "context_actions", "trajectory_label"]
            #
            #     elem = batch[0]
            #     if isinstance(elem, collections.abc.Mapping):
            #         return {key: default_collate([d[key] for d in batch]) for key in expect_keys}
            #
            # if 'mmtransformer' in model_args.model_name and model_args.task == 'waymo':
            #     # Todo: test waymo collate fn
            #     collate_fn = waymo_collate_fn


            for itr, input in enumerate(tqdm(test_dataloader)):
                # move batch to device
                for each_key in input:
                    if isinstance(input[each_key], type(torch.tensor(0))):
                        input[each_key] = input[each_key].to("cuda")

                eval_batch_size = training_args.per_device_eval_batch_size
                if model_args.autoregressive or model_args.ar_future_interval > 0:
                    # Todo: add autoregressive predict
                    traj_pred = model.generate(**input)
                else:
                    output = model(**copy.deepcopy(input))
                    traj_pred = output.logits                   
                    try:
                        file_name = input['file_name']
                        current_frame_idx = input['frame_id']
                    except:
                        file_name = ["null"] * eval_batch_size
                        current_frame_idx = -1 * torch.ones(eval_batch_size)
                    prediction_results['file_names'].extend(file_name)
                    prediction_results['current_frame'].extend(current_frame_idx.cpu().numpy())
                    if data_args.dagger:
                        dagger_results['file_name'].extend(file_name)
                        dagger_results['frame_id'].extend(list(current_frame_idx.cpu().numpy()))
                
                if model_args.predict_trajectory:
                    if model_args.autoregressive:# trajectory label as token case
                        trajectory_label = model.compute_normalized_points(input["trajectory"][:, 10:, :])
                        traj_pred = model.compute_normalized_points(traj_pred)
                        
                    else:
                        if 'mmtransformer' in model_args.model_name and model_args.task == 'waymo':
                            trajectory_label = input["trajectory_label"][:, :, :2]
                            trajectory_label = torch.where(trajectory_label != -1, trajectory_label, traj_pred)
                        else:
                            trajectory_label = input["trajectory_label"][:, 1::2, :]

                    loss = loss_fn(trajectory_label[:, :, :2], traj_pred[:, -trajectory_label.shape[1]:, :2])
                    end_trajectory_label = trajectory_label[:, -1, :]
                    end_point = traj_pred[:, -1, :]
                    end_bias_x.append(end_trajectory_label[:, 0] - end_point[:, 0])
                    end_bias_y.append(end_trajectory_label[:, 1] - end_point[:, 1])
                    all_bias_x.append(trajectory_label[:, :, 0] - traj_pred[:, -trajectory_label.shape[1]:, 0])
                    all_bias_y.append(trajectory_label[:, :, 1] - traj_pred[:, -trajectory_label.shape[1]:, 1])
                    losses.append(loss)

            if model_args.predict_trajectory:
                end_bias_x = torch.stack(end_bias_x, 0).cpu().numpy()
                end_bias_y = torch.stack(end_bias_y, 0).cpu().numpy()
                all_bias_x = torch.stack(all_bias_x, 0).reshape(-1).cpu().numpy()
                all_bias_y = torch.stack(all_bias_y, 0).reshape(-1).cpu().numpy()
                final_loss = torch.mean(torch.stack(losses, 0)).item()
                print('Mean L2 loss: ', final_loss)
                print('End point x offset: ', np.average(np.abs(end_bias_x)))
                print('End point y offset: ', np.average(np.abs(end_bias_y)))
                distance_error = np.sqrt(np.abs(all_bias_x)**2 + np.abs(all_bias_y)**2).reshape(-1, 80)
                final_distance_error = np.sqrt(np.abs(end_bias_x)**2 + np.abs(end_bias_y)**2)
                if data_args.dagger:
                    dagger_results['ADE'].extend(list(np.average(distance_error, axis=1).reshape(-1)))
                    dagger_results['FDE'].extend(list(final_distance_error.reshape(-1)))
                    dagger_results['y_bias'].extend(list(np.average(all_bias_y.reshape(-1, 80), axis=1).reshape(-1)))
                print('ADE', np.average(distance_error))
                print('FDE', np.average(final_distance_error))
            
            # print(dagger_results)
            def compute_dagger_dict(dic):
                tuple_list = list()
                fde_result_list = dict()
                y_bias_result_list = dict()
                for filename, id, ade, fde, y_bias in zip(dic["file_name"], dic["frame_id"], dic["ADE"], dic["FDE"], dic["y_bias"]):
                    if filename == "null":
                        continue
                    tuple_list.append((filename, id, ade, fde, abs(y_bias)))
    
                fde_sorted_list = sorted(tuple_list, key=lambda x:x[3], reverse=True)
                for idx, tp in enumerate(fde_sorted_list): 
                    if tp[0] in fde_result_list.keys():
                        fde_result_list[tp[0]]["frame_id"].append(tp[1])
                        fde_result_list[tp[0]]["ade"].append(tp[2])
                        fde_result_list[tp[0]]["fde"].append(tp[3])
                        fde_result_list[tp[0]]["y_bias"].append(tp[4])
                        fde_result_list[tp[0]]["rank"].append((idx+1)/len(fde_sorted_list))
                        
                    else:
                        fde_result_list[tp[0]] = dict(
                            frame_id=[tp[1]], ade=[tp[2]], fde=[tp[3]], y_bias=[tp[4]], rank=[(idx+1)/len(fde_sorted_list)]
                        )
                y_bias_sorted_list = sorted(tuple_list, key=lambda x:x[-1], reverse=True)
                for idx, tp in enumerate(y_bias_sorted_list): 
                    if tp[0] in y_bias_result_list.keys():
                        y_bias_result_list[tp[0]]["frame_id"].append(tp[1])
                        y_bias_result_list[tp[0]]["ade"].append(tp[2])
                        y_bias_result_list[tp[0]]["fde"].append(tp[3])
                        y_bias_result_list[tp[0]]["y_bias"].append(tp[4])
                        y_bias_result_list[tp[0]]["rank"].append((idx+1)/len(y_bias_sorted_list))
                    else:
                        y_bias_result_list[tp[0]] = dict(
                            frame_id=[tp[1]], ade=[tp[2]], fde=[tp[3]], y_bias=[tp[4]], rank=[(idx+1)/len(y_bias_sorted_list)]
                        )
                return fde_result_list, y_bias_result_list
            
            def draw_histogram_graph(data, title, savepath):
                import matplotlib.pyplot as plt
                plt.hist(data, bins=range(20), edgecolor='black')
                plt.title(title)
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(savepath, "{}.png".format(title)))
            if data_args.dagger:
                draw_histogram_graph(dagger_results["FDE"], title="FDE-distributions", savepath=training_args.output_dir)
                draw_histogram_graph(dagger_results["ADE"], title="ADE-distributions", savepath=training_args.output_dir)
                draw_histogram_graph(dagger_results["y_bias"], title="ybias-distribution", savepath=training_args.output_dir)
                fde_dagger_dic, y_bias_dagger_dic = compute_dagger_dict(dagger_results)


            if training_args.output_dir is not None:
                # save results
                output_file_path = os.path.join(training_args.output_dir, 'generated_predictions.pickle')
                with open(output_file_path, 'wb') as handle:
                    pickle.dump(prediction_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if data_args.dagger:
                    dagger_result_path = os.path.join(training_args.output_dir, "fde_dagger.pkl")
                    with open(dagger_result_path, 'wb') as handle:
                        pickle.dump(fde_dagger_dic, handle)
                    dagger_result_path = os.path.join(training_args.output_dir, "ybias_dagger.pkl")
                    with open(dagger_result_path, 'wb') as handle:
                        pickle.dump(y_bias_dagger_dic, handle)
                    print("dagger results save to {}".format(dagger_result_path))

        # predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # metrics = predict_results.metrics
        # max_predict_samples = (
        #     data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        # )
        # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)

        # if trainer.is_world_process_zero():
        #     if training_args.predict_with_generate:
        #         predictions = tokenizer.batch_decode(
        #             predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #         )
        #         predictions = [pred.strip() for pred in predictions]
        #         output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        #         with open(output_prediction_file, "w") as writer:
        #             writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_pretrain_name_or_path, "tasks": "NuPlanPlanning"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # Automatically saving all args into a json file.
    # TODO: Add this into Trainer class to save config while saving other logs
    # all_args_dic = {**model_args.__dict__, **data_args.__dict__, **config_args.__dict__, **training_args.__dict__}
    # if training_args.do_train:
    #     with open(os.path.join(training_args.output_dir, "training_args.json"), 'w') as f:
    #         json.dump(all_args_dic, f, indent=4)
    # elif training_args.do_eval:
    #     with open(os.path.join(training_args.output_dir, "eval_args.json"), 'w') as f:
    #         json.dump(all_args_dic, f, indent=4)

    return results


if __name__ == "__main__":
    main()
