import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from transformer4planning.models.encoder.augmentation import DataAugmentation

class TrajectoryEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.use_proposal = model_args.use_proposal
        self.use_key_points = model_args.use_key_points
        self.model_args = model_args

        self.augmentation = DataAugmentation()
        self.selected_indices = []

    def forward(self, **kwargs):  
        """
        Abstract forward function for all encoder classes;
        Encoders obtain input attributes from kwargs in need and return torch.Tensor: input_embeds and Dict: info_dict,
        where info_dict include all the information that will be used in backbone and decoder.          
        """
        raise NotImplementedError

    def select_keypoints(self, trajectory_label):
        """
        Universal keypoints selection function.
        return  `future_key_points`: torch.Tensor, the key points selected
                `selected_indices`: List
                `indices`: List
        """
        device = trajectory_label.device
        # use autoregressive future interval
        if 'specified' in self.use_key_points:
            # 80, 40, 20, 10, 5
            if self.use_key_points == 'specified_forward':
                self.selected_indices = [4, 9, 19, 39, 79]
            elif self.use_key_points == 'specified_backward':
                self.selected_indices = [79, 39, 19, 9, 4]
            else:
                assert False, "specified key points should be either specified_forward or specified_backward"
            future_key_points = trajectory_label[:, self.selected_indices, :]
            indices = torch.tensor(self.selected_indices, device=device, dtype=float) / 80.0
        elif 'universal' in self.use_key_points:
            ar_future_interval = 20
            future_key_points = trajectory_label[:, ar_future_interval - 1::ar_future_interval, :]
            indices = torch.arange(future_key_points.shape[1], device=device) / future_key_points.shape[1]
            self.selected_indices = []
        else:
            assert False, "key points should be either specified or universal"
        
        return future_key_points, self.selected_indices, indices