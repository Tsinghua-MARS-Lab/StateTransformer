# coding=utf-8

"""
Train a Transformer ML Model for Planning
"""

import logging
import os
import sys
import pickle
import copy
import torch
from tqdm import tqdm
import copy
import json
import multiprocessing as mp
import datasets
import numpy as np
import evaluate
import transformers
from datasets import Dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets
from functools import partial

from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformer4planning.models.model import build_models
from transformer4planning.utils.args import (
    ModelArguments, 
    DataTrainingArguments, 
    ConfigArguments, 
    PlanningTrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from transformer4planning.trainer import (PlanningTrainer, CustomCallback)
from torch.utils.data import DataLoader
from transformers.trainer_callback import DefaultFlowCallback
from transformer4planning.trainer import compute_metrics

from datasets import Dataset, Value

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

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
    try:
        # for some new dataset, split column is already added
        if split == 'train_alltype':
            dataset = dataset.add_column(name='split', column=['train'] * len(dataset))
        else:
            dataset = dataset.add_column(name='split', column=[split] * len(dataset))
    except:
        pass
    dataset.set_format(type='torch')
    if "centerline" in dataset.column_names:
        dataset = dataset.filter(lambda example: np.sum(np.array(example["centerline"])) != 0, num_proc=mp.cpu_count())
    if select:
        samples = int(len(dataset) * float(dataset_scale))
        dataset = dataset.select(range(samples))
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, PlanningTrainingArguments))
    model_args, data_args, _, training_args = parser.parse_args_into_dataclasses()
    model_args.generate_diffusion_dataset_for_key_points_decoder = True
    # set default label names
    training_args.label_names = ['trajectory_label']

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
    if model_args.task == "nuplan" or model_args.task == "waymo": # nuplan datasets are stored in index format
        index_root = os.path.join(data_args.saved_dataset_folder, 'index')
    elif model_args.task == "train_diffusion_decoder":
        index_root = data_args.saved_dataset_folder
    
    
    root_folders = os.listdir(index_root)

    if data_args.use_full_training_set:
        if 'train_alltype' in root_folders:
            train_dataset = load_dataset(index_root, "train_alltype", data_args.dataset_scale, True)
        else:
            raise ValueError("No training dataset found in {}, must include at least one city in /train_alltype".format(index_root))
    else:
        if 'train' in root_folders:
            train_dataset = load_dataset(index_root, "train", data_args.dataset_scale, True)
        else:
            raise ValueError("No training dataset found in {}, must include at least one city in /train".format(index_root))
    
    if 'test' in root_folders:
        test_dataset = load_dataset(index_root, "test", data_args.dataset_scale, False)
    else:
        print('Testset not found, using training set as test set')
        test_dataset = train_dataset
    
    if 'val' in root_folders:
        val_dataset = load_dataset(index_root, "val", data_args.dataset_scale, False)
    else:
        print('Validation set not found, using training set as val set')
        val_dataset = test_dataset
        # val_dataset = train_dataset

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
        train=train_dataset.shuffle(seed=training_args.seed),
        validation=val_dataset.shuffle(seed=training_args.seed),
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

    
    import multiprocessing
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count() / 8))
    train_dataset = dataset_dict["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    
    eval_dataset = dataset_dict["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    
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
                                            map_name=map)
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
        raise AttributeError("task for diffusion feature generation must be nuplan or waymo, not train_diffusion_decoder or others")

    trainer = PlanningTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[CustomCallback,],
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.pop_callback(DefaultFlowCallback)
    
    
    
    # First we generate the testing set for our diffusion decoder.
    print("We skip generating diff feats for eval set.")
    result = trainer.evaluate()
    logger.info(f"during eval set generation: {result}")
    
    trainer.model.key_points_decoder.save_testing_diffusion_feature_dir = model.key_points_decoder.save_testing_diffusion_feature_dir[:-4] + 'train/'
    # print("Now generating the other 40%.")
    trainer.eval_dataset = train_dataset.select(range(int(len(train_dataset)*0),len(train_dataset)))
    result = trainer.evaluate()
    logger.info(f"during training set generation: {result}")
    # try:
    #     if model_args.autoregressive or True:
    #         result = trainer.evaluate()
    # except Exception as e:
    #     # The code would throw an exception at the end of evaluation loop since we return None in evaluation step
    #     # But this is not a big deal since we have just saved everything we need in the model's forward method.
    #     print(e)
    #     pass
    
    # Then we generate the training set for our diffusion decoder.
    # Since it's way more faster to run an evaluation iter than a training iter (because no back-propagation is needed), we do this by substituting the testing set with our training set.
    
    
    
    trainer.model.key_points_decoder.save_testing_diffusion_feature_dir = model.key_points_decoder.save_testing_diffusion_feature_dir[:-6] + 'test/'
    trainer.eval_dataset = test_dataset
    result = trainer.evaluate()
    logger.info(f"during testing set generation: {result}")
        


if __name__ == "__main__":
    main()
