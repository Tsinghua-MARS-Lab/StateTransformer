import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from transformer4planning.models.encoder.augmentation import DataAugmentation

class TrajectoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_proposal = config.use_proposal
        self.use_key_points = config.use_key_points
        self.config = config

        self.augmentation = DataAugmentation()

        # config selected indices
        if 'specified' in self.use_key_points:
            # 80, 40, 20, 10, 5
            if self.use_key_points == 'specified_forward':
                self.selected_indices = [4, 9, 19, 39, 79]
            elif self.use_key_points == 'specified_backward':
                self.selected_indices = [79, 39, 19, 9, 4]
            elif self.use_key_points == 'specified_two_backward':
                self.selected_indices = [79, 4]
            else:
                assert False, f"specified key points should be either specified_forward or specified_backward {self.use_key_points}"
        elif 'universal' in self.use_key_points:
            self.selected_indices = [15, 31, 47, 63, 79]
        elif 'denoise_kp' in self.use_key_points:
            self.selected_indices = [79] * 10 + [39, 19, 9, 4]
        else:
            self.selected_indices = []

    def forward(self, **kwargs):  
        """
        Abstract forward function for all encoder classes;
        Encoders obtain input attributes from kwargs in need and return torch.Tensor: input_embeds and Dict: info_dict,
        where info_dict include all the information that will be used in backbone and decoder.          
        """
        raise NotImplementedError

    def select_keypoints(self, info_dict):
        """
        Universal keypoints selection function.
        return  `future_key_points`: torch.Tensor, the key points selected
                `selected_indices`: List
                `indices`: List
        """
        trajectory_label = info_dict['trajectory_label']
        device = trajectory_label.device

        # use autoregressive future interval
        if 'specified' in self.use_key_points:
            # 80, 40, 20, 10, 5
            future_key_points = trajectory_label[:, self.selected_indices, :]
            # indices = torch.tensor(self.selected_indices, device=device, dtype=float) / 80.0
        elif 'universal' in self.use_key_points:
            ar_future_interval = 20
            future_key_points = trajectory_label[:, ar_future_interval - 1::ar_future_interval, :]
            # indices = torch.arange(future_key_points.shape[1], device=device) / future_key_points.shape[1]
            # self.selected_indices = [15, 31, 47, 63, 79]
        elif 'denoise_kp' in self.use_key_points:
            # WIP
            future_key_points = trajectory_label[:, self.selected_indices, :2]
        else:
            assert False, "key points should be either specified or universal"
        
        return future_key_points