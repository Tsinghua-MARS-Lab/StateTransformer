# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved
import torch
import torch.nn as nn


from transformer4planning.libs.mtr.transformer import (transformer_encoder_layer, position_encoding_utils)
from transformer4planning.libs.mtr import polyline_encoder
from transformer4planning.utils import mtr_utils
from transformer4planning.libs.mtr.ops.knn import knn_utils

AGENT_TYPES = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST', 'TYPE_OTHERS']
MAP_TYPES = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]


class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        self.num_agent_type = 4
        self.num_road_type = 16
        # build polyline encoders
        # self.agent_polyline_encoder = [self.build_polyline_encoder(
        #     in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
        #     hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
        #     num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
        #     out_channels=self.model_cfg.D_MODEL,
        #     return_multipoints_feature=True
        # ) for _ in range(self.num_agent_type)] 
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL,
            return_multipoints_feature=True
        )

        self.map_polyline_encoder = nn.ModuleList([self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL,
            return_multipoints_feature=False
        ) for _ in range(self.num_road_type)])

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
        batch_offsets = mtr_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
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
    
    def organize_by_type(self, input_dict):
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask']
        map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']

        obj_trajs_pos = input_dict['obj_trajs_pos']
        map_polylines_center = input_dict['map_polylines_center']
        
        # num_center_objects, _, num_timestamps, num_attr = obj_trajs.shape
        # print(obj_trajs.shape)
        # obj_types = input_dict["obj_types"][None, :]
        # print(obj_types.shape)
        # obj_trajs_typed, obj_trajs_mask_typed, obj_trajs_pos_typed = {}, {}, {}
        # for i in range(self.num_agent_type):
        #     agent_type = AGENT_TYPES[i]
        #     mask = (obj_types[:, :] == agent_type)
        #     print(mask.shape)
        #     exit()
        #     num = [mask[j].sum() for j in range(num_center_objects)]
        #     max_num = max(num)
        #     if max_num == 0: continue

        #     agent_input = torch.zeros((num_center_objects, max_num, num_timestamps, num_attr), dtype=obj_trajs.dtype, device=obj_trajs.device)
        #     agent_mask = torch.zeros((num_center_objects, max_num, num_timestamps), dtype=obj_trajs.dtype, device=obj_trajs.device)
        #     agent_pos = torch.zeros((num_center_objects, max_num, num_timestamps, 3), dtype=obj_trajs.dtype, device=obj_trajs.device)
        #     for j in range(num_center_objects):
        #         agent_input[j, :num[j], ...] = obj_trajs[j, mask[j, :], ...]
        #         agent_mask[j, :num[j], ...] = obj_trajs_mask[j, mask[j, :], ...]
        #         agent_pos[j, :num[j], ...] = obj_trajs_pos[j, mask[j, :], ...]

        #     obj_trajs_typed[agent_type] = agent_input
        #     obj_trajs_mask_typed[agent_type] = agent_mask > 0
        #     obj_trajs_pos_typed[agent_type] = agent_pos
        
        map_polylines_type = map_polylines[:, :, 1, 6]
        num_center_objects, _, num_polyline, num_attr = map_polylines.shape
        map_polylines_typed, map_polylines_mask_typed, map_polylines_center_typed = {}, {}, {}
        for i in range(self.num_road_type):
            map_type = MAP_TYPES[i]
            mask = (map_polylines_type[:, :] == map_type)
            num = [mask[j].sum() for j in range(num_center_objects)]
            max_num = max(num)
            
            if max_num == 0: continue

            map_input = torch.zeros((num_center_objects, max_num, num_polyline, num_attr), dtype=map_polylines.dtype, device=map_polylines.device)
            map_mask = torch.zeros((num_center_objects, max_num, num_polyline), dtype=map_polylines.dtype, device=map_polylines.device)
            map_pos = torch.zeros((num_center_objects, max_num, 3), dtype=map_polylines.dtype, device=map_polylines.device)
            for j in range(num_center_objects):
                map_input[j, :num[j], ...] = map_polylines[j, mask[j, :], ...]
                map_mask[j, :num[j], ...] = map_polylines_mask[j, mask[j, :], ...]
                map_pos[j, :num[j], ...] = map_polylines_center[j, mask[j, :], ...]

            map_polylines_typed[map_type] = map_input
            map_polylines_mask_typed[map_type] = map_mask > 0
            map_polylines_center_typed[map_type] = map_pos

        input_dict.update({
            # "obj_trajs": obj_trajs_typed,
            # "obj_trajs_mask": obj_trajs_mask_typed,
            # "obj_trajs_pos": obj_trajs_pos_typed,
            "map_polylines": map_polylines_typed,
            "map_polylines_mask": map_polylines_mask_typed,
            "map_polylines_center": map_polylines_center_typed,
        })

        return input_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = self.organize_by_type(batch_dict['input_dict'])
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask']
        map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']

        obj_trajs_pos = input_dict['obj_trajs_pos']
        map_polylines_center = input_dict['map_polylines_center']
        
        # obj_polylines_feature_typed, obj_polylines_mask_typed, obj_polylines_pos_typed = [], [], []
        # for i in range(self.num_agent_type):
        #     agent_type = AGENT_TYPES[i]
        #     if agent_type in obj_trajs.keys():
        #         obj_trajs_per_type = obj_trajs[agent_type]
        #         obj_trajs_mask_per_type = obj_trajs_mask[agent_type]
        #         obj_trajs_pos_per_type = obj_trajs_pos[agent_type]

                # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, num_timestamp, C)        

        #         obj_polylines_feature_typed.append(obj_polylines_feature)
        #         obj_polylines_mask_typed.append(obj_trajs_mask_per_type)
        #         obj_polylines_pos_typed.append(obj_trajs_pos_per_type)

        map_polylines_feature_typed, map_polylines_mask_typed, map_polylines_pos_typed = [], [], []
        for i in range(self.num_road_type):
            map_type = MAP_TYPES[i]      
            if map_type in map_polylines.keys():
                map_polylines_per_type = map_polylines[map_type]
                map_polylines_mask_per_type = map_polylines_mask[map_type]
                map_polylines_center_per_type = map_polylines_center[map_type]

                map_polylines_feature = self.map_polyline_encoder[i](map_polylines_per_type, map_polylines_mask_per_type)  # (num_center_objects, num_polylines, C)

                map_polylines_feature_typed.append(map_polylines_feature)
                map_polylines_mask_typed.append(map_polylines_mask_per_type)
                map_polylines_pos_typed.append(map_polylines_center_per_type)
                
        # obj_polylines_feature = torch.cat(obj_polylines_feature_typed, dim=1)
        obj_valid_mask = obj_trajs_mask
        # obj_trajs_pos = torch.cat(obj_polylines_pos_typed, dim=1)

        map_polylines_feature = torch.cat(map_polylines_feature_typed, dim=1)
        map_valid_mask = (torch.cat(map_polylines_mask_typed, dim=1).sum(dim=-1) > 0)
        map_polylines_center = torch.cat(map_polylines_pos_typed, dim=1)

        num_center_objects, num_objects, num_timestamps, n_out_embed = obj_polylines_feature.shape
        num_polylines = map_polylines_feature.shape[1]

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
        # center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        # batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        # batch_dict['obj_mask'] = obj_valid_mask
        # batch_dict['map_mask'] = map_valid_mask
        # batch_dict['obj_pos'] = obj_trajs_last_pos
        # batch_dict['map_pos'] = map_polylines_center
        
        # lane feature
        # lane_mask = (map_polylines[:, :, 1, 6] == 3) | (map_polylines[:, :, 1, 6] == 2) | (map_polylines[:, :, 1, 6] == 1) # (num_center_objects, num_polylines)
        # lane_num = [lane_mask[i].sum() for i in range(num_center_objects)]
        # max_lane_num = max(lane_num)

        # lane_polyline_feature = torch.zeros((num_center_objects, max_lane_num, num_timestamps, n_out_embed), dtype=map_polylines_feature.dtype, device=map_polylines_feature.device)
        # for i in range(num_center_objects):
        #     lane_polyline_feature[i, :lane_num[i], ...] = map_polylines_feature[i, lane_mask[i, :], ...]
        # batch_dict['map_lane_feature'] = lane_polyline_feature
        
        # not_lane_mask = torch.logical_not(lane_mask)
        # not_lane_num = [not_lane_mask[i].sum() for i in range(num_center_objects)]
        # max_not_lane_num = max(not_lane_num)
        
        # not_lane_polyline_feature = torch.zeros((num_center_objects, max_not_lane_num, num_timestamps, n_out_embed), dtype=map_polylines_feature.dtype, device=map_polylines_feature.device)
        # for i in range(num_center_objects):
        #     not_lane_polyline_feature[i, :not_lane_num[i], ...] = map_polylines_feature[i, not_lane_mask[i, :], ...]
        # batch_dict['map_others_feature'] = not_lane_polyline_feature

        return batch_dict
    
class MTREncoderWithType(MTREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.map_pooling = nn.Sequential(nn.Linear(self.model_cfg.D_MODEL * self.num_road_type, self.model_cfg.D_MODEL), nn.Tanh())

    def forward(self, batch_dict):
        input_dict = self.organize_by_type(batch_dict['input_dict'])
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask']
        map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']

        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask).max(dim=1)[0]  # (num_center_objects, num_objects, num_timestamp, C)        

        num_center_objects, _, num_timesteps, _ = obj_trajs.shape

        map_polylines_feature_typed = []
        for i in range(self.num_road_type):
            map_type = MAP_TYPES[i]      
            if map_type in map_polylines.keys():
                map_polylines_per_type = map_polylines[map_type]
                map_polylines_mask_per_type = map_polylines_mask[map_type]

                map_polylines_feature = self.map_polyline_encoder[i](map_polylines_per_type, map_polylines_mask_per_type).max(dim=1)[0]  # (num_center_objects, num_polylines, C)
            else:
                map_polylines_feature = self.map_polyline_encoder[i](torch.zeros(num_center_objects, 1, 1, self.model_cfg.NUM_INPUT_ATTR_MAP, device=obj_trajs.device), torch.zeros(num_center_objects, 1, 1, dtype=bool, device=obj_trajs.device)).squeeze(1)
            map_polylines_feature_typed.append(map_polylines_feature)
        map_polylines_feature_typed = torch.stack(map_polylines_feature_typed, dim=1).view(num_center_objects, -1)
        map_polylines_feature = self.map_pooling(map_polylines_feature_typed).unsqueeze(1).repeat(1, num_timesteps, 1)
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature

        return batch_dict