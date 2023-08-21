from typing import Dict
import torch
from torch import nn, Tensor
from transformers import GPT2Tokenizer
from transformer4planning.models.utils import *

class AugmentationMixin(nn.Module):
    def __init__(self):
        super().__init__()
    
    def trajectory_augmentation(self, target_traj, x_noise_scale, y_noise_scale, expanded_indices=1):
        if self.training and x_noise_scale > 0:
            x_noise = (torch.rand(target_traj.shape, device=target_traj.device) * x_noise_scale * 2 - x_noise_scale) * expanded_indices
            target_traj[..., 0] += x_noise[..., 0] 
        if self.training and y_noise_scale > 0:
            y_noise = (torch.rand(target_traj.shape, device=target_traj.device) * y_noise_scale * 2 - y_noise_scale) * expanded_indices
            target_traj[..., 1] += y_noise[..., 1]
        return target_traj

    def raster_augmentation(self, raster):
        raise NotImplementedError

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
    

class NuplanRasterizeEncoder(AugmentationMixin):
    def __init__(self, 
                 cnn_kwargs:Dict, 
                 action_kwargs:Dict,
                 tokenizer_kwargs:Dict = None,
                 model_args = None
                 ):
        super().__init__()
        self.cnn_downsample = CNNDownSamplingResNet(d_embed=cnn_kwargs.get("d_embed", None), 
                                                    in_channels=cnn_kwargs.get("in_channels", None), 
                                                    resnet_type=cnn_kwargs.get("resnet_type", "resnet18"),
                                                    pretrain=cnn_kwargs.get("pretrain", False))
        self.action_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh()) # 2*embed is because the cnn downsample will be apply to both high&low resolution rasters
        self.token_scenario_tag = model_args.token_scenario_tag
        self.ar_future_interval = model_args.ar_future_interval
        self.model_args = model_args
        if self.token_scenario_tag:
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_kwargs.get("dirpath", None))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tag_embedding = nn.Embedding(self.tokenizer.vocab_size, tokenizer_kwargs.get("d_embed", None))    
 
    
    def forward(self, **kwargs):
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        context_actions = kwargs.get("context_actions", None)
        trajectory_label = kwargs.get("trajectory_label", None)
        scenario_type = kwargs.get("scenario_type", None)
        context_length = kwargs.get("context_length", None)
        pred_length = kwargs.get("pred_length", None)
        device = high_res_raster.device if high_res_raster is not None else None
        # add noise to context actions
        context_actions = self.trajectory_augmentation(context_actions, self.model_args.x_random_walk, self.model_args.y_random_walk)
        
        # raster observation encoding & context action ecoding
        action_embeds = self.action_m_embed(context_actions)
        
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), context_length, self.model_args.with_traffic_light)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), context_length, self.model_args.with_traffic_light)
        
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
        
        # scenario tag encoding
        if self.token_scenario_tag:
            assert scenario_type is not None, "scenario_type is None for token_scenario_tag"
            assert scenario_type[0] != 'Unknown', f"scenario_type is Unknown for token_scenario_tag"
            scenario_tag_ids = torch.tensor(self.tokenizer(text=scenario_type, max_length=self.model_args.max_token_len, padding='max_length')["input_ids"])
            scenario_tag_embeds = self.tag_embedding(scenario_tag_ids.to(device)).squeeze(1)
            assert scenario_tag_embeds.shape[1] == self.model_args.max_token_len, f'{scenario_tag_embeds.shape} vs {self.model_args.max_token_len}'
            input_embeds = torch.cat([scenario_tag_embeds, input_embeds], dim=1)

        # add keypoints encoded embedding
        if self.ar_future_interval == 0:
            input_embeds = torch.cat([input_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)

        elif self.ar_future_interval > 0:
            # use autoregressive future interval
            if self.model_args.specified_key_points:
                # 80, 40, 20, 10, 5
                if self.model_args.forward_specified_key_points:
                    selected_indices = [4, 9, 19, 39, 79]
                else:
                    selected_indices = [79, 39, 19, 9, 4]
                future_key_points = trajectory_label[:, selected_indices, :]
                indices = torch.tensor(selected_indices, device=device, dtype=float) / 80.0                   
            else:
                future_key_points = trajectory_label[:, self.ar_future_interval - 1::self.ar_future_interval, :]
                indices = torch.arange(future_key_points.shape[1], device=device) / future_key_points.shape[1]
                selected_indices = None
            assert future_key_points.shape[1] != 0, 'future points not enough to sample'
            expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
            # argument future trajectory
            future_key_points_aug = self.trajectory_augmentation(future_key_points.clone(), self.model_args.arf_x_random_walk, self.model_args.arf_y_random_walk, expanded_indices)
            if not self.model_args.predict_yaw:
                # keep the same information when generating future points
                future_key_points_aug[:, :, 2:] = 0

            future_key_embeds = self.action_m_embed(future_key_points_aug)
            input_embeds = torch.cat([input_embeds, future_key_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)
        else:
            raise ValueError("ar_future_interval should be non-negative", self.ar_future_interval)

        return input_embeds, future_key_points, selected_indices

class PDMEncoder(nn.Module):
    def __init__(self, history_dim, centerline_dim, hidden_dim):
        super(PDMEncoder, self).__init__()
        self.state_encoding = nn.Sequential(
            nn.Linear(
                history_dim * 3 * 3, hidden_dim
            ),
            nn.ReLU(),
        )

        self.centerline_encoding = nn.Sequential(
            nn.Linear(centerline_dim * 3, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, input):
        # TODO: fix feature input& complete the forward function
        batch_size = input["ego_position"].shape[0]

        ego_position = input["ego_position"].reshape(batch_size, -1).float()
        ego_velocity = input["ego_velocity"].reshape(batch_size, -1).float()
        ego_acceleration = input["ego_acceleration"].reshape(batch_size, -1).float()

        # encode ego history states
        state_features = torch.cat(
            [ego_position, ego_velocity, ego_acceleration], dim=-1
        )
        state_encodings = self.state_encoding(state_features)

        # encode planner centerline
        planner_centerline = input["planner_centerline"].reshape(batch_size, -1).float()
        centerline_encodings = self.centerline_encoding(planner_centerline)

        # decode future trajectory
        planner_features = torch.cat(
            [state_encodings, centerline_encodings], dim=-1
        )
        return planner_features