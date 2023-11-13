from typing import Dict
import torch
from torch import nn
from transformer4planning.models.utils import *
from transformer4planning.models.encoder.base import TrajectoryEncoder

class CNNDownSamplingResNet(nn.Module):
    def __init__(self, d_embed, in_channels, resnet_type='resnet18', pretrain=False):
        super(CNNDownSamplingResNet, self).__init__()
        import torchvision.models as models
        if resnet_type == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 512
        elif resnet_type == 'resnet34':
            self.cnn = models.resnet34(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 512
        elif resnet_type == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 2048
        elif resnet_type == 'resnet101':
            self.cnn = models.resnet101(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 2048
        elif resnet_type == 'resnet152':
            self.cnn = models.resnet152(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 2048
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[1:-1]))
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=cls_feature_dim, out_features=d_embed, bias=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.cnn(x)
        output = self.classifier(x.squeeze(-1).squeeze(-1))
        return output


class NuplanRasterizeEncoder(TrajectoryEncoder):
    def __init__(self, 
                 cnn_kwargs:Dict, 
                 action_kwargs:Dict,
                 config = None
                 ):

        super().__init__(config)
        self.cnn_downsample = CNNDownSamplingResNet(d_embed=cnn_kwargs.get("d_embed", None), 
                                                    in_channels=cnn_kwargs.get("in_channels", None), 
                                                    resnet_type=cnn_kwargs.get("resnet_type", "resnet18"),
                                                    pretrain=cnn_kwargs.get("pretrain", False))
        self.action_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh())
        self.kps_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh())
        
    def forward(self, **kwargs):
        """
        Nuplan raster encoder require inputs:
        `high_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `low_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `context_actions`: torch.Tensor, shape (batch_size, seq, 4)
        `trajectory_label`: torch.Tensor, shape (batch_size, seq, 2/4), depend on whether pred yaw value
        `pred_length`: int, the length of prediction trajectory
        `context_length`: int, the length of context actions
        """
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        context_actions = kwargs.get("context_actions", None)
        trajectory_label = kwargs.get("trajectory_label", None)
        aug_current = kwargs.get("aug_current", None)

        assert trajectory_label is not None, "trajectory_label should not be None"
        device = trajectory_label.device
        _, pred_length = trajectory_label.shape[:2]
        context_length = context_actions.shape[1] if context_actions is not None else -1  # -1 in case of pdm encoder

        # add noise to context actions
        context_actions = self.augmentation.trajectory_linear_augmentation(context_actions, self.config.x_random_walk, self.config.y_random_walk)
        # raster observation encoding & context action ecoding
        action_embeds = self.action_m_embed(context_actions)
        
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), context_length, self.config.with_traffic_light, self.config.use_centerline)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), context_length, self.config.with_traffic_light, self.config.use_centerline)
        # casted channel number: 33 - 1 goal, 20 raod types, 3 traffic light, 9 agent types for each time frame
        # context_length: 8, 40 frames / 5
        batch_size, context_length, c, h, w = high_res_seq.shape

        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)

        state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds  # index: 0, 2, 4, .., 18
        input_embeds[:, 1::2, :] = action_embeds  # index: 1, 3, 5, .., 19

        # add keypoints encoded embedding
        if self.use_key_points == 'no':
            input_embeds = torch.cat([input_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)
            future_key_points = None
        else:
            future_key_points, selected_indices, indices = self.select_keypoints(trajectory_label)
            assert future_key_points.shape[1] != 0, 'future points not enough to sample'
            # expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
            # argument future trajectory
            future_key_points_aug = self.augmentation.trajectory_linear_augmentation(future_key_points.clone(), self.config.arf_x_random_walk, self.config.arf_y_random_walk)
            if not self.config.predict_yaw:
                # keep the same information when generating future points
                future_key_points_aug[:, :, 2:] = 0

            future_key_embeds = self.kps_m_embed(future_key_points_aug)
            input_embeds = torch.cat([input_embeds, future_key_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)

        info_dict = {
            "future_key_points": future_key_points,
            "selected_indices": self.selected_indices,
            "trajectory_label": trajectory_label,
            "pred_length": pred_length,
            "context_length": context_length * 2,
            "aug_current": aug_current,
        }

        return input_embeds, info_dict
