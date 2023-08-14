# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved
import torch
import torch.nn as nn


from transformer4planning.models.encoder.utils.transformer import transformer_encoder_layer, position_encoding_utils
from transformer4planning.models.encoder.utils import polyline_encoder
from transformer4planning.models.encoder.utils import common_utils
from transformer4planning.ops.knn import knn_utils


class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL,
            return_multipoints_feature=True
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL,
            return_multipoints_feature=False
        )

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL

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
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
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
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda() 

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda() 
        obj_trajs_pos = input_dict['obj_trajs_pos'].cuda() 
        map_polylines_center = input_dict['map_polylines_center'].cuda() 
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, num_timestamp, C)        
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn
        # obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        obj_valid_mask = obj_trajs_mask
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)
        n_out_embed = obj_polylines_feature.shape[-1]

        global_token_feature = torch.cat((obj_polylines_feature.view(num_center_objects, num_objects*num_timestamps, n_out_embed), map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask.view(num_center_objects, -1), map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_pos.view(num_center_objects, num_objects*num_timestamps, -1), map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects*num_timestamps].view(num_center_objects, num_objects, num_timestamps, n_out_embed)
        map_polylines_feature = global_token_feature[:, num_objects*num_timestamps:][:, :, None, :].repeat(1, 1, num_timestamps, 1)
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict
    
from typing import Dict
from transformer4planning.models.encoder.base import EncoderBase

class WaymoVectorizeEncoder(EncoderBase):
    def __init__(self, 
                 mtr_config,
                 action_kwargs:Dict,
                 tokenizer_kwargs:Dict = None,
                 model_args = None
                 ):
        super().__init__(model_args, tokenizer_kwargs)
        self.model_args = model_args
        self.token_scenario_tag = model_args.token_scenario_tag
        self.ar_future_interval = model_args.ar_future_interval
        self.context_encoder = SimpleEncoder(mtr_config.CONTEXT_ENCODER)
        self.action_m_embed = nn.Sequential(nn.Linear(4, action_kwargs.get("d_embed")), nn.Tanh())
 
    def from_marginal_to_joint(self, hidden_state, info_dict, update_info_dict=False):
        device = hidden_state.device
        agents_num_per_scenario = info_dict["agents_num_per_scenario"]
        max_agents_num = max(agents_num_per_scenario)
        scenario_num = len(agents_num_per_scenario)
        feature_length, hidden_dim = hidden_state.shape[-2:]
        hidden_state_joint = torch.zeros((scenario_num, max_agents_num * feature_length, hidden_dim), dtype=torch.float32, device=device)
        input_embeds_mask = torch.zeros((scenario_num, max_agents_num * feature_length), dtype=torch.bool, device=device)
        agent_index_global = 0
        for i in range(scenario_num):
            agents_num = agents_num_per_scenario[i]
            scenario_embeds = torch.zeros((agents_num * feature_length, hidden_dim), dtype=torch.float32, device=device)
            for j in range(agents_num):
                scenario_embeds[j::agents_num, :] = hidden_state[agent_index_global]
                agent_index_global += 1

            hidden_state_joint[i, :agents_num * feature_length, :] = scenario_embeds
            input_embeds_mask[i, :agents_num * feature_length] = 1
        
        if update_info_dict:
            info_dict.update({
                "input_embeds_mask": input_embeds_mask,
            })
        
        return hidden_state_joint

    def forward(self, **kwargs):
        input_dict = kwargs.get("input_dict")
        agent_trajs = input_dict['agent_trajs']
        batch_size = agent_trajs.shape[0]
        device = agent_trajs.device
        track_index_to_predict = input_dict["track_index_to_predict"]

        state_embeds = self.context_encoder(input_dict)

        ego_trajs = [traj[track_index_to_predict[i], :, :] for i, traj in enumerate(agent_trajs)]
        ego_trajs = torch.stack(ego_trajs, dim=0).to(device).squeeze(1)

        trajectory_label = ego_trajs[:, 11:, [0, 1, 2, 6]]
        pred_length = trajectory_label.shape[1]
        trajectory_label_mask = ego_trajs[:, 11:, -1].unsqueeze(-1)
        context_actions = ego_trajs[:, :11, [0, 1, 2, 6]]

        # add noise to context actions
        context_actions = self.trajectory_augmentation(context_actions, self.model_args.x_random_walk, self.model_args.y_random_walk)
        
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1]

        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds  # index: 0, 2, 4, .., 18
        input_embeds[:, 1::2, :] = action_embeds  # index: 1, 3, 5, .., 19

        future_embeds_shape = (batch_size, pred_length, n_embed)
        # add keypoints encoded embedding
        if self.ar_future_interval == 0:
            input_embeds = torch.cat([input_embeds,
                                      torch.zeros((future_embeds_shape), device=device)], dim=1)

        elif self.ar_future_interval > 0:
            # use autoregressive future interval
            future_key_points, selected_indices, indices = self.select_keypoints(trajectory_label)
            assert future_key_points.shape[1] != 0, 'future points not enough to sample'
            expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
            # argument future trajectory
            future_key_points_aug = self.trajectory_augmentation(future_key_points.clone(), self.model_args.arf_x_random_walk, self.model_args.arf_y_random_walk, expanded_indices)
            if not self.model_args.predict_yaw:
                # keep the same information when generating future points
                future_key_points_aug[:, :, 2:] = 0

            future_key_embeds = self.action_m_embed(future_key_points_aug)
            input_embeds = torch.cat([input_embeds, future_key_embeds,
                                      torch.zeros(future_embeds_shape, device=device)], dim=1)
        else:
            raise ValueError("ar_future_interval should be non-negative", self.ar_future_interval)

        if selected_indices is not None:
            future_key_points_gt_mask = trajectory_label_mask[:, selected_indices, :]
        else:
            future_key_points_gt_mask = trajectory_label_mask[:, self.ar_future_interval - 1::self.ar_future_interval, :]
        
        info_dict = {
            "trajectory_label": trajectory_label,
            "trajectory_label_mask": trajectory_label_mask,
            "context_length": context_length,
            "future_key_points": future_key_points,
            "future_key_points_gt_mask": future_key_points_gt_mask,
            "selected_indices": selected_indices,
        }

        if self.model_args.interaction:
            info_dict.update({
                "agents_num_per_scenario": input_dict["agents_num_per_scenario"],
            })
            input_embeds = self.from_marginal_to_joint(input_embeds, info_dict, update_info_dict=True)

        return input_embeds, info_dict
    
class SimpleEncoder(MTREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.output_dim = config.D_MODEL
        self.map_polyline_encoder = nn.Sequential(nn.Linear(2, 128, bias=False), nn.ReLU(),
                                                  nn.Linear(128, 256, bias=False), nn.ReLU(),
                                                  nn.Linear(256, self.output_dim, bias=True), nn.ReLU(),)

    def forward(self, **kwargs):
        input_dict = kwargs.get("input_dict", None)
        assert input_dict is not None, "input_dict is None, check model inputs"
        past_trajs = input_dict['agent_trajs'][:, :, :11, :]
        map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']

        num_center_objects, num_objects, num_timestamps, _ = past_trajs.shape
        agent_trajs_mask = past_trajs[..., -1] > 0
        map_polylines_mask = map_polylines_mask[..., 0] > 0
        # apply polyline encoder
        obj_polylines_feature = self.agent_polyline_encoder(past_trajs, agent_trajs_mask)  # (num_center_objects, num_objects, num_timestamp, C)   
        map_polylines_feature = self.map_polyline_encoder(map_polylines)  # (num_center_objects, num_polylines, C)
        map_polylines_feature, _ = torch.max(map_polylines_feature, dim=1, keepdim=True)
        map_polylines_feature = map_polylines_feature.repeat(1, num_timestamps, 1)

        agents_attention_mask = agent_trajs_mask.view(num_center_objects, -1)
        map_attention_mask = torch.ones((num_center_objects, num_timestamps), device=past_trajs.device, dtype=torch.bool)

        n_out_embed = obj_polylines_feature.shape[-1]
        global_token_feature = torch.cat((obj_polylines_feature.view(num_center_objects, num_objects*num_timestamps, n_out_embed), map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((agents_attention_mask, map_attention_mask), dim=1)

        obj_trajs_pos = past_trajs[..., :3]
        map_polylines_center = torch.zeros((num_center_objects, num_timestamps, 3), device=past_trajs.device)
        global_token_pos = torch.cat((obj_trajs_pos.contiguous().view(num_center_objects, num_objects*num_timestamps, -1), map_polylines_center), dim=1)

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        global_token_feature_max, _ = torch.max(global_token_feature.view(num_center_objects, -1, num_timestamps, self.output_dim), dim=1)

        return global_token_feature_max.squeeze(1)
