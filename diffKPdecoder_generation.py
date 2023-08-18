# coding=utf-8

"""
Generate Training & Validation Feature for Diffusion Key Point Decoder used in class TrajGPTDiffusionKPDecoder
"""

import logging
import os
import sys
import pickle
import copy
from typing import Optional
import torch
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
import wandb
# wandb.login(key='3cb4a5ee4aefb4f4e25dae6a16db1f59568ac603')

from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformer4planning.models.model import build_models
from transformer4planning.utils import ModelArguments
from transformers.trainer_utils import get_last_checkpoint
from transformer4planning.trainer import PlanningTrainer, PlanningTrainingArguments, CustomCallback
from torch.utils.data import DataLoader
from transformers.trainer_callback import DefaultFlowCallback

from datasets import Dataset, Value

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)
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
    map_path: Optional[str] = field(
        default=None, metadata={"help":"The root path of map file"}
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
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
# Maybe can be removed into some utils.py? The same function as in runner.py
def load_dataset(root, split='train', dataset_scale=1, select=False):
    datasets = []
    index_root_folders = os.path.join(root, split)
    indices = os.listdir(index_root_folders)
 
    for index in indices:
        index_path = os.path.join(index_root_folders, index)
        if os.path.isdir(index_path):
            # load training dataset
            logger.info("Loading training dataset {}".format(index_path))
            dataset = Dataset.load_from_disk(index_path)
            if dataset is not None:
                datasets.append(dataset)
        else:
            continue
    # For nuplan dataset directory structure, each split obtains multi cities directories, so concat is required;
    # But for waymo dataset, index directory is just the datset, so load directory directly to build dataset. 
    if len(datasets) > 0: 
        dataset = _concatenate_map_style_datasets(datasets)
    else: 
        dataset = Dataset.load_from_disk(index_root_folders)
    # add split column
    dataset.features.update({'split': Value('string')})
    dataset = dataset.add_column(name='split', column=[split] * len(dataset))
    dataset.set_format(type='torch')
    if select:
        samples = int(len(dataset) * float(dataset_scale))
        dataset = dataset.select(range(samples))
    return dataset



def generate_feature_for_dataset(loaded_dataset, collate_fn, current_model, training_args):
    current_model.eval()
    dataset = loaded_dataset
    
    rank = int(os.environ['LOCAL_RANK'])
    sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_eval_batch_size, sampler=sampler, collate_fn=collate_fn, num_workers = training_args.dataloader_num_workers)
    device = torch.device(f"cuda:{rank}")
    model = current_model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    
    pbar = dataloader
    if rank == 0:
        pbar = tqdm(dataloader, dynamic_ncols=True)
        
    model.eval()
    with torch.no_grad():
        for counter, data in enumerate(pbar):
            for each_key in data:
                if isinstance(data[each_key],type(torch.tensor(0))):
                    data[each_key] = data[each_key].to(device)    
            dummy_output = model(**data)  # The model's forward method will handle saving
            # Optionally, if rank == 0, you can set post-fix info for tqdm
            if rank == 0:
                pbar.set_postfix_str(f"Counter: {counter}")
    

    
    # current_dataloader = torch.utils.data.dataloader(
    #     loaded_dataset,
    #     batch_size = training_args.per_device_eval_batch_size,
    #     num_workers = training_args.per_device_eval_batch_size,
    #     collate_fn = collate_fn,
    #     pin_memory = True,
    #     drop_last = True
    # )
    # for itr, input in enumerate(tqdm(test_dataloader)):
    #     for each_key in input:
    #         if isinstance(input[each_key], type(torch.tensor(0))):
    #             input[each_key] = input[each_key].to("cuda")
    #     dummy_forward_result = current_model(**copy.deepcopy(input))
    #     # then do nothing, simply going through the forward method of our current_model
    #     pass

def main():
    pl.seed_everything(42)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, PlanningTrainingArguments))
    model_args, data_args, _, training_args = parser.parse_args_into_dataclasses()
    # pre-compute raster channels number
    if model_args.raster_channels == 0:
        road_types = 20
        agent_types = 8
        traffic_types = 4
        past_sample_number = int(2 * 20 / model_args.past_sample_interval)  # past_seconds-2, frame_rate-20
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

    from datasets import disable_caching
    disable_caching()
    # loop all datasets
    logger.info("Loading full set of datasets from {}".format(data_args.saved_dataset_folder))
    assert os.path.isdir(data_args.saved_dataset_folder)
    index_root = os.path.join(data_args.saved_dataset_folder, 'index')
    root_folders = os.listdir(index_root)
    GEN_FOR_TRAIN = False
    GEN_FOR_EVAL = False
    GEN_FOR_TEST = False
        
    if 'train' in root_folders:
        train_dataset = load_dataset(index_root, "train", data_args.dataset_scale, True)
        GEN_FOR_TRAIN = True
        print("Found Trainset, will generate feature for Trainset.")
    else:
        train_dataset = None
        print("Trainset not found, will not generate feature for Trainset.")
    
    if training_args.do_eval and 'test' in root_folders:
        test_dataset = load_dataset(index_root, "test", data_args.dataset_scale, False)
        GEN_FOR_TEST = True
        print("Found Testset, will generate feature for Testset.")
    else:
        test_dataset = None
        print('Testset not found, will not generate feature for Testset.')
    
    if (training_args.do_eval or training_args.do_predict) and 'val' in root_folders:
        val_dataset = load_dataset(index_root, "val", data_args.dataset_scale, False)
        GEN_FOR_EVAL = True
        print("Found Evalset, will generate feature for Evalset.")
    else:
        val_dataset = None
        print('Validation set not found, will not generate feature for Evalset.')

    if model_args.task == "nuplan":
        all_maps_dic = {}
        map_folder = os.path.join(data_args.saved_dataset_folder, 'map')
        for each_map in os.listdir(map_folder):
            if each_map.endswith('.pkl'):
                map_path = os.path.join(map_folder, each_map)
                with open(map_path, 'rb') as f:
                    map_dic = pickle.load(f)
                map_name = each_map.split('.')[0]
                all_maps_dic[map_name] = map_dic

    # loop split info and update for test set
    print('TrainingSet: ', train_dataset, '\nValidationSet', val_dataset, '\nTestingSet', test_dataset)

    dataset_dict = dict(
        train=train_dataset.shuffle(seed=training_args.seed) if GEN_FOR_TRAIN else None,
        validation=val_dataset.shuffle(seed=training_args.seed) if GEN_FOR_EVAL else None,
        test=test_dataset.shuffle(seed=training_args.seed) if GEN_FOR_TEST else None,
    )

    # Load a model's pretrained weights from a path or from hugging face's model base
    model = build_models(model_args)

    
    import multiprocessing
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count() / 8))
    
    if GEN_FOR_TRAIN:
        train_dataset = dataset_dict["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if GEN_FOR_EVAL:
        eval_dataset = dataset_dict["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    if GEN_FOR_TEST:
        predict_dataset = dataset_dict["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    
    # Initialize our Trainer
    if model_args.task == "nuplan":
        if model_args.encoder_type == "raster":
            from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
            collate_fn = partial(nuplan_rasterize_collate_func,
                                dic_path=data_args.saved_dataset_folder,
                                all_maps_dic=all_maps_dic,
                                **model_args.__dict__)
        elif model_args.encoder_type == "vector":
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            from transformer4planning.preprocess.pdm_vectorize import nuplan_vector_collate_func
            map_api = dict()
            for map in ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']:
                map_api[map] = get_maps_api(map_root=data_args.nuplan_map_path,
                                    map_version="nuplan-maps-v1.0",
                                    map_name=map
                                    )
            collate_fn = partial(nuplan_vector_collate_func, 
                                 dic_path=data_args.saved_dataset_folder, 
                                 map_api=map_api)
    elif model_args.task == "waymo":
        from transformer4planning.preprocess.waymo_vectorize import waymo_collate_func
        if model_args.encoder_type == "vector":
            collate_fn = partial(waymo_collate_func, 
                                 data_path=data_args.saved_dataset_folder, 
                                 interaction=model_args.interaction)
        elif model_args.encoder_type == "raster":
            raise NotImplementedError
    else:
        raise AttributeError("task must be nuplan or waymo")
    
    root_dir = model.save_testing_diffusion_dataset_dir[:-5]
    if GEN_FOR_TRAIN:
        if not os.path.exists(root_dir + 'train/'):
            os.makedirs(root_dir + 'train/')
        model.save_testing_diffusion_dataset_dir = root_dir + 'train/'
        print("Generating feature for Trainset...")
        generate_feature_for_dataset(train_dataset, collate_fn, model, training_args)
    else:
        print("Trainset not found.")
    if GEN_FOR_EVAL:
        if not os.path.exists(root_dir + 'val/'):
            os.makedirs(root_dir + 'val/')
        model.save_testing_diffusion_dataset_dir = root_dir + 'val/'
        print("Generating feature for Evalset...")
        generate_feature_for_dataset(eval_dataset, collate_fn, model, training_args)
    else:
        print("Evalset not found.")
    if GEN_FOR_TEST:
        if not os.path.exists(root_dir + 'test/'):
            os.makedirs(root_dir + 'test/')
        model.save_testing_diffusion_dataset_dir = root_dir + 'test/'
        print("Generating feature for Testset...")
        generate_feature_for_dataset(test_dataset, collate_fn, model, training_args)
    else:
        print("Testset not found.")

if __name__ == "__main__":
    main()