from typing import Dict
import torch
from torch import nn
from transformer4planning.models.utils import *
from transformer4planning.models.encoder.base import TrajectoryEncoder

class PDMEncoder(TrajectoryEncoder):
    def __init__(self, 
                 pdm_kwargs:Dict,
                 tokenizer_kwargs:Dict = None,
                 model_args = None):
        super(PDMEncoder, self).__init__(model_args, tokenizer_kwargs)
        # 3*3 means 3d pos, 3d vel & 3d acc concat
        self.state_embed = nn.Sequential(
            nn.Linear(
                3 * 3, pdm_kwargs.get("hidden_dim", None)
            ),
            nn.ReLU(),
        )

        self.centerline_embed = nn.Sequential(
            nn.Linear(pdm_kwargs.get("centerline_dim") * 3, pdm_kwargs.get("hidden_dim")),
            nn.ReLU(),
        )

        self.action_m_embed = nn.Sequential(
            nn.Linear(4, pdm_kwargs.get("hidden_dim")),
            nn.Tanh()
        )
    
    def forward(self, **kwargs):
        """
        PDM feature includes 
         `ego history states`: including `ego_position`, `ego_velocity`, `ego_acceleration`,
                               each with shape of (batchsize, history_dim, 3)
         `planner centerline`: torch.Tensor, (batchsize, centerline_num, 3) and scenario type
        """
        batch_size, context_length = kwargs.get("ego_position").shape[:2]
        ego_position = kwargs.get("ego_position").float() # shape (bsz, history_dim, 3)
        ego_velocity = kwargs.get("ego_velocity", torch.zeros_like(ego_position)).float() # shape (bsz, history_dim, 3)
        ego_acceleration = kwargs.get("ego_acceleration", torch.zeros_like(ego_velocity)).float() # shape (bsz, history_dim, 3)
        planner_centerline = kwargs.get("planner_centerline", None)
        if planner_centerline is not None:
            planner_centerline = planner_centerline.reshape(batch_size, -1).float() # (bsz, centerline_num, 3) -> (bsz, centerline_dim * 3)
        scenario_type = kwargs.get("scenario_type")
        trajectory_label = kwargs.get("trajectory_label", None)
        pred_length = kwargs.get("pred_length", trajectory_label.shape[1])
        device = ego_position.device if ego_position is not None else None
        
        assert trajectory_label is not None, "trajectory_label should not be None"
        _, pred_length = trajectory_label.shape[:2]
        
        # encode ego history states, with shape of (bsz, history_dim, 9)
        state_features = torch.cat(
            [ego_position, ego_velocity, ego_acceleration], dim=-1
        )
        state_encodings = self.state_embed(state_features)

        if planner_centerline is not None:
            # encode planner centerline
            centerline_encodings = self.centerline_embed(planner_centerline)
            planner_embed = torch.cat(
                [state_encodings, centerline_encodings.unsqueeze(1)], dim=1
            )
            context_length += 1
        else:
            planner_embed = state_encodings

        if self.token_scenario_tag:
            scenario_tag_ids = torch.tensor(self.tokenizer(text=scenario_type, max_length=self.model_args.max_token_len, padding='max_length')["input_ids"])
            scenario_tag_embeds = self.tag_embedding(scenario_tag_ids.to(device)).squeeze(1)
            assert scenario_tag_embeds.shape[1] == self.model_args.max_token_len, f'{scenario_tag_embeds.shape} vs {self.model_args.max_token_len}'
            planner_embed = torch.cat([scenario_tag_embeds, planner_embed], dim=1)
        
        # use trajectory label to build keypoints
        if self.use_key_points is not None:
            future_key_points, selected_indices, indices = self.select_keypoints(trajectory_label)
            assert future_key_points.shape[1] != 0, 'future points not enough to sample'
            expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
            # argument future trajectory
            future_key_points_aug = self.augmentation.trajectory_augmentation(future_key_points.clone(), self.model_args.arf_x_random_walk, self.model_args.arf_y_random_walk, expanded_indices)
            if not self.model_args.predict_yaw:
                # keep the same information when generating future points
                future_key_points_aug[:, :, 2:] = 0

            future_key_embeds = self.action_m_embed(future_key_points_aug)
            planner_embed = torch.cat([planner_embed, future_key_embeds, 
                                       torch.zeros((batch_size, pred_length, planner_embed.shape[-1]), device=device)], dim=1)

        else:
            planner_embed = torch.cat([planner_embed, 
                                       torch.zeros((batch_size, pred_length, planner_embed.shape[-1]), device=device)], dim=1)
            future_key_points, selected_indices = None, []
        
        info_dict = {
            "future_key_points": future_key_points,
            "selected_indices": selected_indices,
            "trajectory_label": trajectory_label,
            "pred_length": pred_length,
            "context_length": context_length + self.scenario_type_len,
        }
        
        return planner_embed, info_dict