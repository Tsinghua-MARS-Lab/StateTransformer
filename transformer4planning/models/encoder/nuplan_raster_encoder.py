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
        else:
            assert False, f'Unknown resnet type: {resnet_type}'
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


from transformers.activations import ACT2FN
class STRMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        projector_hidden_act = 'gelu'
        vision_hidden_size = 768
        model_hidden_size = config.get("d_embed")
        self.linear_1 = nn.Linear(vision_hidden_size, model_hidden_size, bias=True)
        self.act = ACT2FN[projector_hidden_act]
        self.linear_2 = nn.Linear(model_hidden_size, model_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NuplanRasterizeEncoder(TrajectoryEncoder):
    def __init__(self,  config=None):
        super().__init__(config)
        action_kwargs = dict(
            d_embed=self.config.n_embd
        )
        # if 'resnet' in self.config.raster_encoder_type:
        if isinstance(self.config.raster_encoder_type, str) and 'vit' in self.config.raster_encoder_type:
            from transformers import ViTModel, ViTConfig
            vit_config = ViTConfig()
            vit_config.hidden_size = self.config.n_embd // 2
            vit_config.num_channels = self.config.raster_channels
            vit_config.intermediate_size = self.config.vit_intermediate_size  # must be multiplier of 12 (number of the head)
            vit_config.num_attention_heads = self.config.n_head
            vit_config.return_dict = True
            self.image_downsample = ViTModel(vit_config)
            print('Building ViT encoder')
        else:
            try:
                cnn_kwargs = dict(
                    d_embed=self.config.n_embd // 2,
                    in_channels=self.config.raster_channels,
                    resnet_type=self.config.resnet_type,
                    pretrain=self.config.pretrain_encoder
                )
            except:
                cnn_kwargs = dict(
                    d_embed=self.config.n_embd // 2,
                    in_channels=self.config.raster_channels,
                    resnet_type=self.config.raster_encoder_type,
                    pretrain=self.config.pretrain_encoder
                )
            self.cnn_downsample = CNNDownSamplingResNet(d_embed=cnn_kwargs.get("d_embed", None),
                                                        in_channels=cnn_kwargs.get("in_channels", None),
                                                        resnet_type=cnn_kwargs.get("resnet_type", "resnet18"),
                                                        pretrain=cnn_kwargs.get("pretrain", False))
            print('Building ResNet encoder')
        # separate key point encoder is hard to train with larger models due to sparse signals
        if self.config.use_speed:
            self.action_m_embed = nn.Sequential(nn.Linear(7, action_kwargs.get("d_embed")), nn.Tanh())
        else:
            self.action_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh())

        if self.config.separate_kp_encoder:
            self.kps_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh())
        if 'denoise_kp' in self.use_key_points:
            self.kps_m_embed = nn.Sequential(nn.Linear(2, action_kwargs.get("d_embed")), nn.Tanh())

        if self.use_proposal:
            self.proposal_m_embed = nn.Sequential(nn.Linear(1, action_kwargs.get("d_embed")), nn.Tanh())

        self.image_processor = None
        self.camera_image_encoder = None
        self.image_feature_connector = None
        if config.camera_image_encoder == 'dinov2':
            # WIP
            from transformers import AutoImageProcessor, Dinov2Model
            try:
                self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                self.camera_image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
            except:
                # using local checkpoints due to the GFW blocking of China
                self.image_processor = AutoImageProcessor.from_pretrained("/public/MARS/t4p/dinov2", local_files_only=True)
                self.camera_image_encoder = Dinov2Model.from_pretrained("/public/MARS/t4p/dinov2", local_files_only=True)
            # self.camera_image_m_embed = nn.Sequential(nn.Linear(257*768, action_kwargs.get("d_embed")), nn.Tanh())
            # self.camera_image_m_embed = nn.Sequential(nn.Linear(768, action_kwargs.get("d_embed")), nn.Tanh())
            # self.camera_image_m_embed = nn.Sequential(nn.Linear(768, action_kwargs.get("d_embed"), bias=False))
            self.camera_image_m_embed = STRMultiModalProjector(action_kwargs)
            for param in self.camera_image_encoder.parameters():
                param.requires_grad = False
        
    def forward(self, **kwargs):
        """
        Nuplan raster encoder require inputs:
        `high_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `low_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `context_actions`: torch.Tensor, shape (batch_size, seq, 4 / 6)
        `trajectory_label`: torch.Tensor, shape (batch_size, seq, 2/4), depend on whether pred yaw value
        `pred_length`: int, the length of prediction trajectory
        `context_length`: int, the length of context actions

        To use camera image encoder, the input should also contain:
        `camera_image`: torch.Tensor, shape (batch_size, 8(cameras), 1080, 1920, 3)
        """
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        context_actions = kwargs.get("context_actions", None)
        trajectory_label = kwargs.get("trajectory_label", None)
        aug_current = kwargs.get("aug_current", None)

        is_training = kwargs.get("is_training", None)
        assert is_training is not None, "is_training should not be None"
        self.augmentation.training = is_training

        assert trajectory_label is not None, "trajectory_label should not be None"
        device = trajectory_label.device
        _, pred_length = trajectory_label.shape[:2]
        action_seq_length = context_actions.shape[1] if context_actions is not None else -1  # -1 in case of pdm encoder

        # add noise to context actions
        context_actions = self.augmentation.trajectory_linear_augmentation(context_actions, self.config.x_random_walk, self.config.y_random_walk)
        # raster observation encoding & context action ecoding
        action_embeds = self.action_m_embed(context_actions)
        
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), action_seq_length, self.config.with_traffic_light)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), action_seq_length, self.config.with_traffic_light)
        # casted channel number: 33 - 1 goal, 20 raod types, 3 traffic light, 9 agent types for each time frame
        # context_length: 8, 40 frames / 5
        batch_size, action_seq_length, c, h, w = high_res_seq.shape
        assert c == self.config.raster_channels, "raster channel number should be {}, but got {}".format(self.config.raster_channels, c)

        if self.config.raster_encoder_type == 'vit':
            high_res_embed = self.image_downsample(pixel_values=high_res_seq.to(torch.float32).reshape(batch_size * action_seq_length, c, h, w)).last_hidden_state[:, 1:, :]
            low_res_embed = self.image_downsample(pixel_values=low_res_seq.to(torch.float32).reshape(batch_size * action_seq_length, c, h, w)).last_hidden_state[:, 1:, :]
            # batch_size * context_length, 196 (14*14), embed_dim//2
            _, sequence_length, half_embed = high_res_embed.shape
            high_res_embed = high_res_embed.reshape(batch_size, action_seq_length, sequence_length, half_embed)
            low_res_embed = low_res_embed.reshape(batch_size, action_seq_length, sequence_length, half_embed)

            state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)  # batch_size, action_seq_length, sequence_length, embed_dim
            n_embed = action_embeds.shape[-1]
            context_length = action_seq_length + action_seq_length * sequence_length
            input_embeds = torch.zeros(
                (batch_size, context_length, n_embed),
                dtype=torch.float32,
                device=device
            )
            for j in range(action_seq_length):
                input_embeds[:, j * (1 + sequence_length): j * (1 + sequence_length) + sequence_length, :] = state_embeds[:, j, :, :]
            input_embeds[:, sequence_length::1 + sequence_length, :] = action_embeds
        else:
            high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * action_seq_length, c, h, w))
            low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * action_seq_length, c, h, w))
            high_res_embed = high_res_embed.reshape(batch_size, action_seq_length, -1)
            low_res_embed = low_res_embed.reshape(batch_size, action_seq_length, -1)

            state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)
            n_embed = action_embeds.shape[-1]
            context_length = action_seq_length * 2
            input_embeds = torch.zeros(
                (batch_size, context_length, n_embed),
                dtype=torch.float32,
                device=device
            )
            input_embeds[:, ::2, :] = state_embeds  # index: 0, 2, 4, .., 18
            input_embeds[:, 1::2, :] = action_embeds  # index: 1, 3, 5, .., 19

        if self.camera_image_encoder is not None:
            camera_images = kwargs.get("camera_images", None)
            assert camera_images is not None, "camera_image should not be None"
            if self.image_processor is not None:
                _, _, image_width, image_height, image_channels = camera_images.shape
                camera_images = camera_images.reshape(batch_size*8, image_width, image_height, image_channels)
                camera_inputs = self.image_processor(camera_images, return_tensors="pt")
            camera_inputs = camera_inputs.to(device)
            camera_image_feature = self.camera_image_encoder(**camera_inputs).last_hidden_state  # batch_size * 8, 257, 768

            # compress all patches into one hidden state
            # camera_image_feature = camera_image_feature.reshape(batch_size, 8, 257 * 768)
            # camera_image_embed = self.camera_image_m_embed(camera_image_feature)  # batch_size, 8 * 257, n_embed
            # input_embeds = torch.cat([input_embeds, camera_image_embed], dim=1)
            # context_length += 8

            # spead all patches into hidden states
            camera_image_feature = camera_image_feature.reshape(batch_size, 8 * 257, 768)
            camera_image_embed = self.camera_image_m_embed(camera_image_feature)  # batch_size, 8 * 257, n_embed
            input_embeds = torch.cat([input_embeds, camera_image_embed], dim=1)
            context_length += 8 * 257

        # add proposal embedding
        if self.use_proposal:
            if self.config.autoregressive_proposals:
                intentions = kwargs.get('intentions', None)  # batch_size, 16
                assert intentions is not None, "intentions should not be None when using proposal"
                proposal_embeds = self.proposal_m_embed(intentions.unsqueeze(-1).float())  # batch_size, 16, 256
                # print('test encoder 1: ', proposal_embeds.shape)
                input_embeds = torch.cat([input_embeds, proposal_embeds], dim=1)
            else:
                halfs_intention = kwargs.get("halfs_intention", None)
                intentions = kwargs.get("intentions", None)
                if halfs_intention is None:
                    if len(intentions.shape) == 1:
                        intentions = intentions.unsqueeze(0)  # add batch dimension
                    halfs_intention = intentions[:, 0]
                assert halfs_intention is not None, "halfs_intention should not be None when using proposal"
                # check if halfs_intention is a scalar
                if len(halfs_intention.shape) == 0:
                    halfs_intention = halfs_intention.unsqueeze(0)  # add batch dimension
                # print('test encoder 1: ', halfs_intention, halfs_intention.shape, halfs_intention.unsqueeze(-1).float().shape)
                proposal_embeds = self.proposal_m_embed(halfs_intention.unsqueeze(-1).float()).unsqueeze(1)  # batch_size, 1, 256
                # print('test encoder 2: ', proposal_embeds.shape)
                input_embeds = torch.cat([input_embeds, proposal_embeds], dim=1)

        info_dict = {
            "trajectory_label": trajectory_label,
            "pred_length": pred_length,
            "context_length": context_length,
            "aug_current": aug_current,
        }

        # add keypoints encoded embedding
        if self.use_key_points == 'no':
            input_embeds = torch.cat([input_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)
            info_dict['future_key_points'] = None
            future_key_points = None
        else:
            future_key_points = self.select_keypoints(info_dict)
            assert future_key_points.shape[1] != 0, 'future points not enough to sample'
            # expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
            if 'denoise_kp' in self.use_key_points:
                future_key_points_aug = self.augmentation.kp_denoising(future_key_points.clone())
                future_key_embeds = self.kps_m_embed(future_key_points_aug)
            else:
                # argument future trajectory
                future_key_points_aug = self.augmentation.trajectory_linear_augmentation(future_key_points.clone(), self.config.arf_x_random_walk, self.config.arf_y_random_walk)
                if not self.config.predict_yaw:
                    # keep the same information when generating future points
                    future_key_points_aug[:, :, 2:] = 0

                if self.config.separate_kp_encoder:
                    future_key_embeds = self.kps_m_embed(future_key_points_aug)
                else:
                    if self.config.use_speed:
                        # padding speed, padding the last dimension from 4 to 7
                        future_key_points_aug = torch.cat([future_key_points_aug, torch.zeros_like(future_key_points_aug)[:, :, :3]], dim=-1)
                        future_key_embeds = self.action_m_embed(future_key_points_aug)
                    else:
                        future_key_embeds = self.action_m_embed(future_key_points_aug)

            input_embeds = torch.cat([input_embeds, future_key_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)
            info_dict['future_key_points'] = future_key_points

        info_dict['selected_indices'] = self.selected_indices
        if self.use_proposal:
            if self.config.autoregressive_proposals:
                info_dict["intentions"] = intentions
            else:
                info_dict["halfs_intention"] = halfs_intention

        return input_embeds, info_dict
