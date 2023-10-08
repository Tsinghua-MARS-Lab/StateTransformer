import torch
import pickle
import os
import pathlib

from typing import Dict, List, Tuple, cast
import numpy as np
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame
from utils.base_math import angle_to_range

import time

class WaymoDatasetChecker(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg: Dict, training: bool, logger=None):
        if training:
            self.mode = 'train'
        else:
            self.mode = 'val'

        self.dataset_config = dataset_cfg
        self.logger = logger

        self.data_root = self.dataset_config.dataset_info.data_root
        self.data_path = os.path.join(self.data_root, self.dataset_config.dataset_info.split_dir[self.mode])
        self.infos = self.get_all_infos(os.path.join(self.data_root, self.dataset_config.dataset_info.info_file[self.mode]))

        self.data_cache_path = os.path.join(self.data_root, self.dataset_config.dataset_info.cache_dir[self.mode])
        pathlib.Path(self.data_cache_path).mkdir(parents=True, exist_ok=True) 

        self.logger.info(f'Total scenes after filters: {len(self.infos)}')

    def get_all_infos(self, info_path: str):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_config.dataset_info.sample_interval[self.mode]]
        self.logger.info(f'Total scenes before filters: {len(infos)}')

        return infos
        
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        self.print_max_diff(index)
        return 
    
    def print_max_diff(self, index):
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(os.path.join(self.data_path, f'sample_{scene_id}.pkl'), 'rb') as f:
            info = pickle.load(f)

        trajs = torch.from_numpy(info['track_infos']['trajs']).float()
        trajs_diff = trajs[:, 1:, :] - trajs[:, :-1, :]

        trajs_diff_valid = ((trajs[:, 1:, -1] + trajs[:, :-1, -1]) == 2).float()

        trajs_diff = trajs_diff * trajs_diff_valid.unsqueeze(-1)

        print('============================================================= x ==========================================')
        max_diff, max_row, max_column = self.find_max_and_index(trajs_diff[:, :, 0])
        print("max x diff: ", max_diff, ", agent: ", max_row, ', time: ', max_column)
        print("trajs info: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), 0])
        print("trajs valid: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), -1])

        print('============================================================= y ==========================================')
        max_diff, max_row, max_column = self.find_max_and_index(trajs_diff[:, :, 1])
        print("max y diff: ", max_diff, ", agent: ", max_row, ', time: ', max_column)
        print("trajs info: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), 1])
        print("trajs valid: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), -1])

        print('============================================================= heading ==========================================')
        max_diff, max_row, max_column = self.find_max_and_index(trajs_diff[:, :, 6])
        print("max heading diff: ", max_diff, ", agent: ", max_row, ', time: ', max_column)
        print("trajs info: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), 6])
        print("trajs valid: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), -1])

        print('============================================================= vel x ==========================================')
        max_diff, max_row, max_column = self.find_max_and_index(trajs_diff[:, :, 7])
        print("max vel x diff: ", max_diff, ", agent: ", max_row, ', time: ', max_column)
        print("trajs info: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), 7])
        print("trajs valid: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), -1])

        print('============================================================= vel y ==========================================')
        max_diff, max_row, max_column = self.find_max_and_index(trajs_diff[:, :, 8])
        print("max vel y diff: ", max_diff, ", agent: ", max_row, ', time: ', max_column)
        print("trajs info: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), 8])
        print("trajs valid: ", trajs[max_row, max(0, max_column-10):min(90, max_column+10), -1])

        return

    def find_max_and_index(self, diff_tensor):
        # diff_tensor: [agent_num, 90]
        max_in_column, max_column_index = torch.max(diff_tensor, dim=1)
        max_in_row, max_row_index = torch.max(max_in_column, dim=0)

        return max_in_row, max_row_index.item(), max_column_index[max_row_index.item()].item()

