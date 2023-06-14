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

import datasets
import numpy as np
from datasets import Dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets
from dataclasses import dataclass, field

import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformer4planning.models.model import build_models
from transformers.trainer_utils import get_last_checkpoint
from transformer4planning.trainer import PlanningTrainer, PlanningTrainingArguments, CustomCallback
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import random_split
from transformers.trainer_callback import DefaultFlowCallback
import evaluate


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: str = field(
        default="non-auto-gpt",
        metadata={"help": "Name of a planning model backbone"}
    )
    model_pretrain_name_or_path: str = field(
        default="transfo-xl-wt103",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    predict_result_saving_dir: Optional[str] = field(
        default=False,
        metadata={"help": "The target folder to save prediction results."},
    )
    predict_trajectory: Optional[bool] = field(
        default=True,
    )
    recover_obs: Optional[bool] = field(
        default=False,
    )
    d_embed: Optional[int] = field(
        default=256,
    )
    d_model: Optional[int] = field(
        default=256,
    )
    d_inner: Optional[int] = field(
        default=1024,
    )
    n_layers: Optional[int] = field(
        default=4,
    )
    n_heads: Optional[int] = field(
        default=8,
    )
    # Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    activation_function: Optional[str] = field(
        default = "gelu_new"
    )
    loss_fn: Optional[str] = field(
        default="mse",
    )
    task: Optional[str] = field(
        default="waymo" # only for mmtransformer
    )

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
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The dataset name from hugging face used to push the model."}
    )
    dataset_scale: Optional[str] = field(
        default=1, metadata={"help":"The dataset size, choose from any float <=1, such as 1, 0.1, 0.01"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PlanningTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up pytorch backend
    # if training_args.deepspeed is None:
    #     torch.distributed.init_process_group(backend='nccl')

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
    if os.path.isdir(data_args.saved_dataset_folder):
        items = os.listdir(data_args.saved_dataset_folder)
        if os.path.isdir(os.path.join(data_args.saved_dataset_folder, items[0])): #sub-datasets
            print("concating datasets..")
            concatdatasets = list()
            for i, item in enumerate(items):
                print(os.path.join(data_args.saved_dataset_folder, items[i]))
                tmp = os.listdir(os.path.join(data_args.saved_dataset_folder, items[i]))
                if os.path.isdir(os.path.join(data_args.saved_dataset_folder, items[i], tmp[0])): # for vegas datasets and dagger
                    for sub_item in os.listdir(os.path.join(data_args.saved_dataset_folder, item)):
                        dataset_path = os.path.join(data_args.saved_dataset_folder, item, sub_item)
                        dataset = Dataset.load_from_disk(dataset_path)
                        print(dataset)
                        concatdatasets.append(dataset)
                else: # for boston, pittsburgh and singapore datasets
                    dataset_path = os.path.join(data_args.saved_dataset_folder, item)
                    dataset = Dataset.load_from_disk(dataset_path)
                    # dataset.set_format(type='torch', columns=["intended_maneuver_vector", "current_maneuver_vector", "high_res_raster", "low_res_raster",\
                    #                                         "trajectory_label", "context_actions", "intended_maneuver_label", "current_maneuver_label"])
                    
                    print(dataset)
                    concatdatasets.append(dataset)
          
            concat_dataset = _concatenate_map_style_datasets(concatdatasets)
            concat_dataset.set_format(type='torch')
            concat_dataset.shuffle(seed=training_args.seed)
            train_samples = int(len(concat_dataset) * float(data_args.dataset_scale))
            train_dataset = concat_dataset.select(range(train_samples))
            if training_args.do_eval:
                test_dataset = Dataset.load_from_disk(data_args.saved_valid_dataset_folder)
                test_dataset.set_format(type='torch')
                print(test_dataset)
            nuplan_dataset = dict(
                train=train_dataset,
                validation=test_dataset.shuffle(seed=training_args.seed),
                test=test_dataset.shuffle(seed=training_args.seed)
            )

        else: # whole hugging face dataset   
            print("loading dataset...")
            dataset = Dataset.load_from_disk(data_args.saved_dataset_folder)
            dataset.set_format(type='torch')
            dataset.shuffle(seed=training_args.seed)
            train_samples = int(len(dataset) * float(data_args.dataset_scale))
            train_dataset = dataset.select(range(train_samples))
            print('Dataset Loaded: ', dataset)
            
            if training_args.do_eval:
                test_dataset = Dataset.load_from_disk(data_args.saved_valid_dataset_folder)
                test_dataset.set_format(type='torch')
                print(test_dataset)
            else:
                test_dataset = dataset.select(range(train_samples, len(dataset)))
                test_dataset.set_format(type='torch')
                print(test_dataset)
            
            nuplan_dataset = dict(
                train=train_dataset,
                validation=test_dataset.shuffle(seed=training_args.seed),
                test=test_dataset.shuffle(seed=training_args.seed),
            )
    else:
        raise ValueError(f'Dataset directory ({data_args.saved_dataset_folder}) does not exist. Use save_to_disk() to save a dataset first.')

    # Load a model's pretrained weights from a path or from hugging face's model base
    model = build_models(model_args)
    if 'auto' in model_args.model_name:
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
    trainer = PlanningTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=[CustomCallback,],
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

    # to run eval one time without the trainner
    # if not training_args.do_train and training_args.do_eval:
    #     trainer.evaluate()

    # Evaluation
    results = {}
    if training_args.do_eval:
        if 'auto' in model_args.model_name:
            result = clf_metrics.compute()
            logger.info("***** Final Eval results *****")
            logger.info(f"  {result}")
            hyperparams = {"model": model_args.model_name, "dataset": data_args.saved_dataset_folder, "seed": training_args.seed}
            evaluate.save("./results/", ** result, ** hyperparams)

    if training_args.do_predict:
        from sklearn.metrics import classification_report
        # Currently only supports single GPU predict outputs
        logger.info("*** Predict ***")
        """
        Compute accuracy for the following classifications:
        1. intended_maneuver
        2. current_maneuver
        3. pos_x,
        4. pos_y
        """
        model.eval()
        with torch.no_grad():
            dagger_results = {
                'file_name':[],
                'frame_index':[],
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
            prediction_metrics = {
                'next_step_action': None,
                'predicted_trajectory': None,
            }        
            device = model.device
            def preprocess_data(examples):
                # take a batch of texts
                for each_key in examples:
                    if isinstance(examples[each_key], type(torch.tensor(0))):
                        examples[each_key] = examples[each_key].to(device)
                return examples
                
            if model_args.predict_trajectory:
                end_bias_x = []
                end_bias_y = []
                all_bias_x = []
                all_bias_y = []
                losses = []
                loss_fn = torch.nn.MSELoss(reduction="mean")
    
            # initialize intended maneuver metrics
            def nuplan_collate_fn(batch):
                import collections
                expect_keys = ["file_name", "frame_index", "high_res_raster", "low_res_raster", "context_actions", "trajectory_label"]
                
                elem = batch[0]
                if isinstance(elem, collections.abc.Mapping):
                    return {key: default_collate([d[key] for d in batch]) for key in expect_keys}
            
            def waymo_collate_fn(batch):
                import collections
                expect_keys = expect_keys = ["high_res_raster", "low_res_raster", "context_actions", "trajectory_label"]
                
                elem = batch[0]
                if isinstance(elem, collections.abc.Mapping):
                    return {key: default_collate([d[key] for d in batch]) for key in expect_keys}
            
            if 'mmtransformer' in model_args.model_name and model_args.task == 'waymo':
                collate_fn = waymo_collate_fn
            else:
                collate_fn = nuplan_collate_fn

            test_dataloader = DataLoader(
                dataset=predict_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                num_workers=training_args.per_device_eval_batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True
            )
            for itr, input in enumerate(tqdm(test_dataloader)):
                input = preprocess_data(input)
                input_length = training_args.per_device_eval_batch_size
                if "autogpt" in model_args.model_name:
                    actual_input = dict()
                    actual_input["trajectory"] = input["trajectory"][:, :8]
                    actual_input["high_res_raster"] = input["high_res_raster"].reshape(input_length, 224, 224, -1, 29)[:, :, :, :9, :]
                    actual_input["low_res_raster"] = input["low_res_raster"].reshape(input_length, 224, 224, -1, 29)[:, :, :, :9, :]
                    output = model.generate(**copy.deepcopy(actual_input))
                    traj_pred = output["trajectory"]
                else:
                    output = model(**copy.deepcopy(input))
                    try:
                        intended_m_logits, current_m_logits, traj_pred = output.all_logits
                    except:
                        traj_pred = output.logits                   
                    try:
                        file_name = input['file_name']
                        current_frame_idx = input['frame_index']
                    except:
                        file_name = ["null"] * input_length
                        current_frame_idx = -1 * torch.ones(input_length)
                    prediction_results['file_names'].extend(file_name)
                    prediction_results['current_frame'].extend(current_frame_idx.cpu().numpy())
                    dagger_results['file_name'].extend(file_name)
                    dagger_results['frame_index'].extend(list(current_frame_idx.cpu().numpy()))
                
                if model_args.predict_trajectory:
                    if "autogpt" in model_args.model_name:
                        trajectory_label = model.compute_normalized_points(input["trajectory"][:, 8:, :])
                        traj_pred = model.compute_normalized_points(traj_pred)
                    else:
                        if 'mmtransformer' in model_args.model_name and model_args.task == 'waymo':
                            trajectory_label = input["trajectory_label"][:, :, :2]
                            trajectory_label = torch.where(trajectory_label != -1, trajectory_label, traj_pred)
                        else:
                            trajectory_label = input["trajectory_label"][:, 1::2, :]

                    # print("trajectory_label", trajectory_label[0, :, :2])
                    # print("traj_pred", traj_pred[0, :, :2])
                    loss = loss_fn(trajectory_label[:, :, :2], traj_pred[:, :, :2])
                    end_trajectory_label = trajectory_label[:, -1, :]
                    end_point = traj_pred[:, -1, :]
                    end_bias_x.append(end_trajectory_label[:, 0] - end_point[:, 0])
                    end_bias_y.append(end_trajectory_label[:, 1] - end_point[:, 1])
                    all_bias_x.append(trajectory_label[:, :, 0] - traj_pred[:, :, 0])
                    all_bias_y.append(trajectory_label[:, :, 1] - traj_pred[:, :, 1])
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
                for filename, id, ade, fde, y_bias in zip(dic["file_name"], dic["frame_index"], dic["ADE"], dic["FDE"], dic["y_bias"]):
                    if filename == "null":
                        continue
                    tuple_list.append((filename, id, ade, fde, abs(y_bias)))
    
                fde_sorted_list = sorted(tuple_list, key=lambda x:x[3], reverse=True)
                for idx, tp in enumerate(fde_sorted_list): 
                    if tp[0] in fde_result_list.keys():
                        fde_result_list[tp[0]]["frame_index"].append(tp[1])
                        fde_result_list[tp[0]]["ade"].append(tp[2])
                        fde_result_list[tp[0]]["fde"].append(tp[3])
                        fde_result_list[tp[0]]["y_bias"].append(tp[4])
                        fde_result_list[tp[0]]["rank"].append((idx+1)/len(fde_sorted_list))
                        
                    else:
                        fde_result_list[tp[0]] = dict(
                            frame_index=[tp[1]], ade=[tp[2]], fde=[tp[3]], y_bias=[tp[4]], rank=[(idx+1)/len(fde_sorted_list)]
                        )
                y_bias_sorted_list = sorted(tuple_list, key=lambda x:x[-1], reverse=True)
                for idx, tp in enumerate(y_bias_sorted_list): 
                    if tp[0] in y_bias_result_list.keys():
                        y_bias_result_list[tp[0]]["frame_index"].append(tp[1])
                        y_bias_result_list[tp[0]]["ade"].append(tp[2])
                        y_bias_result_list[tp[0]]["fde"].append(tp[3])
                        y_bias_result_list[tp[0]]["y_bias"].append(tp[4])
                        y_bias_result_list[tp[0]]["rank"].append((idx+1)/len(y_bias_sorted_list))
                    else:
                        y_bias_result_list[tp[0]] = dict(
                            frame_index=[tp[1]], ade=[tp[2]], fde=[tp[3]], y_bias=[tp[4]], rank=[(idx+1)/len(y_bias_sorted_list)]
                        )
                return fde_result_list, y_bias_result_list
            
            def draw_histogram_graph(data, title, savepath):
                import matplotlib.pyplot as plt
                plt.hist(data, bins=range(20), edgecolor='black')
                plt.title(title)
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(savepath, "{}.png".format(title)))

            draw_histogram_graph(dagger_results["FDE"], title="FDE-distributions", savepath=training_args.output_dir)
            draw_histogram_graph(dagger_results["ADE"], title="ADE-distributions", savepath=training_args.output_dir)
            draw_histogram_graph(dagger_results["y_bias"], title="ybias-distribution", savepath=training_args.output_dir)
            fde_dagger_dic, y_bias_dagger_dic = compute_dagger_dict(dagger_results)
            # print(fde_dagger_dic)
            # print(y_bias_dagger_dic)

            if training_args.output_dir is not None:
                # save results
                output_file_path = os.path.join(training_args.output_dir, 'generated_predictions.pickle')
                with open(output_file_path, 'wb') as handle:
                    pickle.dump(prediction_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                dagger_result_path = os.path.join(training_args.output_dir, "fde_dagger.pkl")
                with open(dagger_result_path, 'wb') as handle:
                    pickle.dump(fde_dagger_dic, handle)
                dagger_result_path = os.path.join(training_args.output_dir, "ybias_dagger.pkl")
                with open(dagger_result_path, 'wb') as handle:
                    pickle.dump(y_bias_dagger_dic, handle)
                print("dagger results save to {}".format(dagger_result_path))

            # additionally use waymo metrics to test
            # if model_args.task == "waymo":
            #     from waymo_open_dataset.metrics.ops import py_metrics_ops
            #     from waymo_open_dataset.metrics.python import config_util_py as config_util
            #     from waymo_open_dataset.protos import motion_metrics_pb2
            #     def _default_metrics_config():
            #         config = motion_metrics_pb2.MotionMetricsConfig()
            #         config_text = """
            #           track_steps_per_second: 10
            #           prediction_steps_per_second: 2
            #           track_history_samples: 10
            #           track_future_samples: 80
            #           speed_lower_bound: 1.4
            #           speed_upper_bound: 11.0
            #           speed_scale_lower: 0.5
            #           speed_scale_upper: 1.0
            #           step_configurations {
            #             measurement_step: 5
            #             lateral_miss_threshold: 1.0
            #             longitudinal_miss_threshold: 2.0
            #           }
            #           step_configurations {
            #             measurement_step: 9
            #             lateral_miss_threshold: 1.8
            #             longitudinal_miss_threshold: 3.6
            #           }
            #           step_configurations {
            #             measurement_step: 15
            #             lateral_miss_threshold: 3.0
            #             longitudinal_miss_threshold: 6.0
            #           }
            #           max_predictions: 6
            #           """
            #         text_format.Parse(config_text, config)
            #         return config
            #     class MotionMetrics:
            #         """Wrapper for motion metrics computation."""
            #
            #         def __init__(self, config, is_short=False):
            #             # super().__init__()
            #             self._ground_truth_trajectory = []
            #             self._ground_truth_is_valid = []
            #             self._prediction_trajectory = []
            #             self._prediction_score = []
            #             self._object_type = []
            #             self._scenario_id = []
            #             self._object_id = []
            #             self._metrics_config = config
            #             self.is_short = is_short
            #             self.not_compute = False
            #
            #             self.args = None
            #
            #         def reset_state(self):
            #             self._ground_truth_trajectory = []
            #             self._ground_truth_is_valid = []
            #             self._prediction_trajectory = []
            #             self._prediction_score = []
            #             self._object_type = []
            #             self._scenario_id = []
            #             self._object_id = []
            #
            #         def update_state(self, prediction_trajectory, prediction_score,
            #                          ground_truth_trajectory, ground_truth_is_valid, object_type, scenario_id, object_id):
            #             if self.is_short:
            #                 interval = (
            #                         self._metrics_config.track_steps_per_second //
            #                         self._metrics_config.prediction_steps_per_second)
            #                 assert len(prediction_trajectory.shape) == 4, prediction_trajectory.shape
            #                 # warning: numpy, not tf.tensor
            #                 if not isinstance(prediction_trajectory, np.ndarray):
            #                     prediction_trajectory = prediction_trajectory.numpy()
            #                 if prediction_trajectory.shape[2] == self.args.future_frame_num:
            #                     prediction_trajectory = prediction_trajectory[:, :, (interval - 1)::interval, :].copy()
            #                 else:
            #                     assert prediction_trajectory.shape[2] == 16
            #                 ground_truth_trajectory = None
            #                 ground_truth_is_valid = None
            #
            #             self._prediction_trajectory.append(prediction_trajectory)
            #             self._prediction_score.append(prediction_score)
            #             self._ground_truth_trajectory.append(ground_truth_trajectory)
            #             self._ground_truth_is_valid.append(ground_truth_is_valid)
            #             self._object_type.append(object_type)
            #             self._scenario_id.append(scenario_id)
            #             self._object_id.append(object_id)
            #
            #         def get_all(self):
            #             return (
            #                 self._prediction_trajectory,
            #                 self._prediction_score,
            #                 self._ground_truth_trajectory,
            #                 self._ground_truth_is_valid,
            #                 self._object_type,
            #                 self._scenario_id,
            #                 self._object_id,
            #             )
            #
            #         def result(self):
            #             # [batch_size, steps, 2].
            #             if self.is_short or self.not_compute:
            #                 return None
            #             if len(self._prediction_trajectory) == 0:
            #                 return None
            #             prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
            #             # [batch_size].
            #             prediction_score = tf.concat(self._prediction_score, 0)
            #             # [batch_size, gt_steps, 7].
            #             ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
            #             # [batch_size, gt_steps].
            #             ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
            #             # [batch_size].
            #             object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)
            #
            #             # We are predicting more steps than needed by the eval code. Subsample.
            #             interval = (
            #                     self._metrics_config.track_steps_per_second //
            #                     self._metrics_config.prediction_steps_per_second)
            #             # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
            #             if len(prediction_trajectory.shape) == 5:
            #                 prediction_trajectory = prediction_trajectory[:, :, :, (interval - 1)::interval, :]
            #             else:
            #                 assert len(prediction_trajectory.shape) == 4, prediction_trajectory.shape
            #                 prediction_trajectory = prediction_trajectory[:, :, tf.newaxis, (interval - 1)::interval, :]
            #
            #             # Prepare these into shapes expected by the metrics computation.
            #             #
            #             # num_agents_per_joint_prediction is also 1 here.
            #             # [batch_size, top_k].
            #             assert len(prediction_score.shape) == 2
            #             prediction_score = prediction_score[:, :]
            #             # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].
            #             if len(ground_truth_trajectory.shape) == 4:
            #                 pass
            #             else:
            #                 ground_truth_trajectory = ground_truth_trajectory[:, tf.newaxis]
            #             # # SQ: change to hard checking, adding a new axis at the end to fit target dimension does not make sense ->>>>>
            #             # assert len(ground_truth_trajectory.shape) == 4, ground_truth_trajectory.shape
            #
            #             # [batch_size, num_agents_per_joint_prediction, gt_steps].
            #             if len(ground_truth_is_valid.shape) == 3:
            #                 pass
            #             else:
            #                 ground_truth_is_valid = ground_truth_is_valid[:, tf.newaxis]
            #             # [batch_size, num_agents_per_joint_prediction].
            #             if len(object_type.shape) == 2:
            #                 pass
            #             else:
            #                 object_type = object_type[:, tf.newaxis]
            #
            #             return py_metrics_ops.motion_metrics(
            #                 config=self._metrics_config.SerializeToString(),
            #                 prediction_trajectory=prediction_trajectory,
            #                 prediction_score=prediction_score,
            #                 ground_truth_trajectory=ground_truth_trajectory,
            #                 ground_truth_is_valid=ground_truth_is_valid,
            #                 object_type=object_type)
            #
            #     metrics_config = _default_metrics_config()
            #     motion_metrics = MotionMetrics(metrics_config)
            #     metric_names = config_util.get_breakdown_names_from_motion_config(
            #         metrics_config)
            #
            #     for i in range(args.distributed_training - 1):
            #         motion_metrics_ = queue.get()
            #         assert isinstance(motion_metrics_, MotionMetrics), type(motion_metrics_)
            #         motion_metrics_.args = args
            #         for each in zip(*motion_metrics_.get_all()):
            #             motion_metrics.update_state(*each)
            #     print('all metric_values', len(motion_metrics.get_all()[0]))
            #
            #     score_file = utils.get_eval_identifier()
            #
            #     utils.logging(utils.metric_values_to_string(motion_metrics.result(), metric_names),
            #                   type=score_file, to_screen=True, append_time=True)
            #
            #     gather_and_output_motion_metrics(args, device, queue, motion_metrics, metric_names, MotionMetrics)
            #     gather_and_output_others(args, device, queue, motion_metrics)

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
    
    # push to hub?
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
