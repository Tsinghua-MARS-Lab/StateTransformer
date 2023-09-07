# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import os
import math
import numpy as np
from pathlib import Path
import pickle
import torch

from torch.utils.data._utils.collate import default_collate
import dataset_gen.waymo.common_util as common_utils
from dataset_gen.waymo.config import cfg



class WaymoDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, training=True, logger=None, saved_dataset_folder=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.logger = logger
        self.saved_dataset_folder = saved_dataset_folder
        
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]

        self.infos = self.get_all_infos(self.data_root / self.dataset_cfg.INFO_FILE[self.mode])
        self.logger.info(f'Total scenes after filters: {len(self.infos)}')

    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        self.logger.info(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        ret_infos = self.load_feature_data(index)
    
        return ret_infos

    def load_feature_data(self, index):
        """
        Args:
            index (index):

        Returns:

        """
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(self.saved_dataset_folder + f'/opt_res_{scene_id}.pkl', 'rb') as f:
            feature_data = pickle.load(f)
            
        info['hidden_state'] = feature_data['hidden_state']
        info['label'] = feature_data['label']
        info['label_cls'] = feature_data['label_cls']
        info['label_mask'] = feature_data['label_mask']
        
        return info

    def merge_batch_by_padding_1st_dim(self, tensor_list, return_pad_mask=False):

        maxt_feat0 = max([x.shape[0] for x in tensor_list])

        tensor_shape = list(tensor_list[0].shape)
        tensor_shape[0] = maxt_feat0
        
        ret_tensor_list = []
        ret_mask_list = []
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]

            new_tensor = cur_tensor.new_zeros(tensor_shape)
            new_tensor[:cur_tensor.shape[0], ...] = cur_tensor
            ret_tensor_list.append(new_tensor)

            new_mask_tensor = cur_tensor.new_zeros(maxt_feat0)
            new_mask_tensor[:cur_tensor.shape[0]] = 1
            ret_mask_list.append(new_mask_tensor.bool())

        ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
        ret_mask = torch.cat(ret_mask_list, dim=0)

        if return_pad_mask:
            return ret_tensor, ret_mask
        return ret_tensor

    def collate_batch(self, batch_list):
        excepted_keys = ['label', 'hidden_state', 'label_cls', 'label_mask']
        batch_size = len(batch_list)
        key_to_list = {}
        
        for key in excepted_keys:
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]
            
        result = dict()
        for key, val_list in key_to_list.items():
            val_list = [torch.from_numpy(x) for x in val_list]
            # result[key] = self.merge_batch_by_padding_1st_dim(val_list)
            result[key] = torch.concat(val_list, dim=0)
            
            if key == 'label':
                result[key] = result[key].unsqueeze(1)

        return result


