# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved
import torch
import torch.nn as nn
import pickle

from transformer4planning.libs.mtr.transformer import (transformer_encoder_layer, position_encoding_utils)
from transformer4planning.libs.mtr import polyline_encoder
from transformer4planning.libs.mtr.ops.knn import knn_utils
from typing import Dict
from transformer4planning.models.encoder.base import TrajectoryEncoder
from transformer4planning.utils.waymo_utils import nll_loss_gmm_direct, build_mlps, get_batch_offsets


class MTREncoder(nn.Module):
    def __init__(self, encoder_model_dim):
        super().__init__()
        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=30,
            hidden_dim=256,
            num_layers=3,
            out_channels=encoder_model_dim,
            return_multipoints_feature=True
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=9,
            hidden_dim=64,
            num_layers=5,
            num_pre_layers=3,
            out_channels=encoder_model_dim,
            return_multipoints_feature=False
        )

        # build transformer encoder layers
        self.use_local_attn = True
        self_attn_layers = []
        for _ in range(6):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=encoder_model_dim,
                nhead=8,
                dropout=0.1,
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = encoder_model_dim

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None, return_multipoints_feature=False):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels,
            return_multipoints_feature=return_multipoints_feature
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask']
        map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos']
        obj_trajs_pos = input_dict['obj_trajs_pos']
        map_polylines_center = input_dict['map_polylines_center']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, num_timestamp, C)        
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn
        obj_valid_mask = obj_trajs_mask
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)
        n_out_embed = obj_polylines_feature.shape[-1]

        global_token_feature = torch.cat((obj_polylines_feature.view(num_center_objects, num_objects*num_timestamps, n_out_embed), map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask.view(num_center_objects, -1), map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_pos.view(num_center_objects, num_objects*num_timestamps, -1), map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=16
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects*num_timestamps].view(num_center_objects, num_objects, num_timestamps, n_out_embed)
        map_polylines_feature = global_token_feature[:, num_objects*num_timestamps:][:, :, None, :].repeat(1, 1, num_timestamps, 1)
        assert map_polylines_feature.shape[1] == num_polylines

        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos

        return batch_dict

class WaymoVectorizeEncoder(TrajectoryEncoder):
    def __init__(self, 
                 action_kwargs:Dict,
                 config,
                 ):
        super().__init__(config)
        self.config = config
        encoder_model_dim = min(config.d_model, 768)
        self.context_encoder = MTREncoder(encoder_model_dim)
        self.action_m_embed = nn.Sequential(nn.Linear(10, action_kwargs.get("d_embed")), nn.Tanh())
        self.kps_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh())
        self.proposal_m_embed = nn.Sequential(nn.Linear(2, action_kwargs.get("d_embed")), nn.Tanh())

        self.in_proj_obj = nn.Sequential(
            nn.Linear(self.context_encoder.num_out_channels, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        
        self.in_proj_map = nn.Sequential(
            nn.Linear(self.context_encoder.num_out_channels, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.load_intention_proposals(config.proposal_path, 
                                    ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'])
        self.build_dense_future_prediction_layers(config.d_model, 80)
    
    def load_intention_proposals(self, file_path, agent_types):
        with open(file_path, 'rb') as f:
            intention_points_dict = pickle.load(f)
        
        self.intention_points = {}
        for cur_type in agent_types:
            cur_intention_points = intention_points_dict[cur_type]
            cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
            self.intention_points[cur_type] = cur_intention_points

    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames):
        self.obj_pos_encoding_layer = build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True
        )

        self.future_traj_mlps = build_mlps(
            c_in=4 * num_future_frames, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )

    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        
        # num_center_objects, num_objects, n_timestamp, _ = obj_feature.shape
        # obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        # obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, n_timestamp, obj_feature_valid.shape[-1])
        # obj_feature[obj_mask] = obj_feature_valid

        
        obj_feature = obj_feature.max(dim=2)[0]
        obj_mask = (obj_mask.sum(dim=-1) > 0)
        num_center_objects, num_objects,  _ = obj_feature.shape
        num_future_frames = 80

        # dense future prediction
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], num_future_frames, 7)

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2)  # (num_valid_objects, C)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, num_objects, num_future_frames, 7) # pred_dense_trajs
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid

        return ret_obj_feature, ret_pred_dense_future_trajs
    
    def get_dense_future_prediction_loss(self, pred_dense_trajs, obj_trajs_future_state, obj_trajs_future_mask):
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = nn.functional.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1, 1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).cuda().long()  # (num_center_objects * num_objects)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects, num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        loss_reg_gmm, _ = nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs, gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)
        loss_reg = loss_reg.mean()

        return loss_reg

    def forward(self, **kwargs):
        input_dict = kwargs
        batch_size = input_dict['obj_trajs'].shape[0]
        device = input_dict['obj_trajs'].device

        batch_dict = self.context_encoder({'input_dict': input_dict})

        # prepare O (observation)
        obj_feature = batch_dict['obj_feature']
        map_feature = batch_dict['map_feature']
        
        obj_mask = batch_dict['obj_mask']
        num_center_objects, num_objects, n_timestamp, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]
        map_mask = batch_dict['map_mask']

        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, n_timestamp, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, n_timestamp, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid
        
        state_embeds = torch.cat((map_feature, obj_feature), dim=1) # (bs, num_poly+num_obj, num_timestamp, 256)
        state_embeds = state_embeds.max(dim=1)[0]
        
        # traj
        trajectory_label = input_dict['trajectory_label']
        trajectory_label_mask = input_dict['center_gt_trajs_mask'].unsqueeze(-1)
        
        # action context
        context_actions = input_dict['center_objects_past']
        # add noise to context actions
        context_actions = self.augmentation.trajectory_linear_augmentation(context_actions, self.config.x_random_walk, self.config.y_random_walk)

        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1]  # past_interval=10, past_frames=2 * 20, context_length = 40/10=4

        # create OAOAOA..
        input_embeds = torch.zeros(
            (batch_size, context_length * 2, action_embeds.shape[-1]),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds  # index: 0, 2, 4, .., 18
        input_embeds[:, 1::2, :] = action_embeds  # index: 1, 3, 5, .., 19
        
        # future trajectory
        pred_length = trajectory_label.shape[1]
        info_dict = {
            "trajectory_label": trajectory_label,
            "trajectory_label_mask": trajectory_label_mask,
            "pred_length": pred_length,
            "context_length": context_length * 2,
        }

        # dense prediction
        if self.config.dense_pred:
            _, ret_pred_dense_future_trajs = self.apply_dense_future_prediction(obj_feature, obj_mask, batch_dict['obj_pos'])
            loss_pred_future = self.get_dense_future_prediction_loss(ret_pred_dense_future_trajs, input_dict['obj_trajs_future_state'], input_dict['obj_trajs_future_mask'])
        
            info_dict["dense_pred_loss"] = loss_pred_future

        # prepare proposals
        if self.use_proposal:
            gt_proposal_mask = trajectory_label_mask[:, -1, :] # (bs, 1)
            type_idx_str = {
                1: 'TYPE_VEHICLE',
                2: 'TYPE_PEDESTRIAN',
                3: 'TYPE_CYCLIST',
            }
            center_obj_types = input_dict['center_objects_type']
            
            center_obj_proposal_pts = [self.intention_points[type_idx_str[center_obj_types[i]]].unsqueeze(0) for i in range(batch_size)]
            center_obj_proposal_pts = torch.cat(center_obj_proposal_pts, dim=0) # (bs, 64, 2)
            dist2GT = torch.norm(trajectory_label[:, [-1], :2] - center_obj_proposal_pts, dim=2)
            proposal_GT_cls = dist2GT[:, :].argmin(dim = 1) # (bs, )

            proposal_GT_logits = center_obj_proposal_pts[torch.arange(batch_size), proposal_GT_cls, :] * gt_proposal_mask # (bs, 2)
            proposal_embedding = self.proposal_m_embed(proposal_GT_logits).unsqueeze(1)
            input_embeds = torch.cat([input_embeds, proposal_embedding], dim=1)

            info_dict["gt_proposal_cls"] = proposal_GT_cls
            info_dict["gt_proposal_mask"] = gt_proposal_mask
            info_dict["gt_proposal_logits"] = proposal_GT_logits
            info_dict["center_obj_proposal_pts"] = center_obj_proposal_pts

        # prepare keypoints
        n_embed = action_embeds.shape[-1]
        if self.use_key_points == 'no':
            input_embeds = torch.cat([input_embeds,
                                      torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)
        else:
            future_key_points = self.select_keypoints(trajectory_label)
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
            
            info_dict["future_key_points"] = future_key_points
            info_dict["selected_indices"] = self.selected_indices
            info_dict["key_points_mask"] = trajectory_label_mask[:, self.selected_indices, :]

        return input_embeds, info_dict
