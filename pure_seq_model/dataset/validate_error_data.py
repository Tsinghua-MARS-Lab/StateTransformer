import torch
import pickle

from dataset.waymo_dataset_checker import WaymoDatasetChecker
from omegaconf import OmegaConf

from utils.logger import log

config = OmegaConf.load('configs/config_v1_aug.yaml')

dataset = WaymoDatasetChecker(config.DATA_CONFIG, True, log)

dataset[3425]


