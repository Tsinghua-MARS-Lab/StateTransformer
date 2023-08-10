import torch
from torch import nn
from omegaconf import DictConfig
import copy
from typing import Dict, List, Tuple, cast

import models.base_model as base_model
from models.mtr_models.polyline_encoder import PointNetPolylineEncoder

class PureSeqModelV1(nn.Module):
    def __init__(self, model_config: DictConfig) -> None:
        super().__init__()
        self.model_parameter = model_config.model_parameter
        self.data_dim = model_config.data_dim

        # map in polyline encoder
        self.lane_in_polyline_encoder = PointNetPolylineEncoder(self.data_dim.lane, self.model_parameter.in_polyline_dim, 
                                                                self.model_parameter.in_polyline_layer, self.model_parameter.in_polyline_pre_layer)
        self.road_line_in_polyline_encoder = PointNetPolylineEncoder(self.data_dim.road_line, self.model_parameter.in_polyline_dim, 
                                                                self.model_parameter.in_polyline_layer, self.model_parameter.in_polyline_pre_layer)
        self.road_edge_in_polyline_encoder = PointNetPolylineEncoder(self.data_dim.road_edge, self.model_parameter.in_polyline_dim, 
                                                                self.model_parameter.in_polyline_layer, self.model_parameter.in_polyline_pre_layer)
        self.map_others_in_polyline_encoder = PointNetPolylineEncoder(self.data_dim.map_others, self.model_parameter.in_polyline_dim, 
                                                                self.model_parameter.in_polyline_layer, self.model_parameter.in_polyline_pre_layer)

        # map between polyline encoder
        between_polyline_encoder = PointNetPolylineEncoder(self.model_parameter.in_polyline_dim, self.model_parameter.between_polyline_dim, 
                                                           self.model_parameter.between_polyline_layer, self.model_parameter.between_polyline_pre_layer, 
                                                           self.model_parameter.seq_embedding_dim)
        self.lane_between_polyline_encoder = copy.deepcopy(between_polyline_encoder)
        self.road_line_between_polyline_encoder = copy.deepcopy(between_polyline_encoder)
        self.road_edge_between_polyline_encoder = copy.deepcopy(between_polyline_encoder)
        self.map_others_between_polyline_encoder = copy.deepcopy(between_polyline_encoder)

        # ego/agent seq input embedding
        self.ego_seq_embedding = base_model.MLP(self.data_dim.agent, self.model_parameter.seq_embedding_dim)
        self.agent_seq_embedding = base_model.MLP(self.data_dim.agent, self.model_parameter.seq_embedding_dim)

        # pos encoder
        self.pos_encoder = nn.Parameter(torch.zeros(20, self.model_parameter.seq_embedding_dim))

        # seq transformer
        seq_self_attn = base_model.SelfAttention(self.model_parameter.seq_embedding_dim, self.model_parameter.seq_head)
        seq_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.seq_embedding_dim, self.model_parameter.seq_inter_dim)
        seq_layer = base_model.TransformerEncoderLayer(self.model_parameter.seq_embedding_dim, seq_self_attn, seq_p_ff)
        self.seq_layerlist = nn.ModuleList([copy.deepcopy(seq_layer) for _ in range(self.model_parameter.seq_layer)])

        # head
        self.ego_head = base_model.MLP(self.model_parameter.seq_embedding_dim, self.data_dim.agent_target)
        self.agent_head = base_model.MLP(self.model_parameter.seq_embedding_dim, self.data_dim.agent_target)


    def forward(self, input_dict: Dict):
        device = input_dict['ego_feature_list'][0][0].device

        lane_encoded_feature = self.encode_map_feature(input_dict['lane_feature_list'], input_dict['lane_mask_list'],
                                                       self.lane_in_polyline_encoder, self.lane_between_polyline_encoder, device)
        road_line_encoded_feature = self.encode_map_feature(input_dict['road_line_feature_list'], input_dict['road_line_mask_list'],
                                                       self.road_line_in_polyline_encoder, self.road_line_between_polyline_encoder, device)
        road_edge_encoded_feature = self.encode_map_feature(input_dict['road_edge_feature_list'], input_dict['road_edge_mask_list'],
                                                       self.road_edge_in_polyline_encoder, self.road_edge_between_polyline_encoder, device)
        map_others_encoded_feature = self.encode_map_feature(input_dict['map_others_feature_list'], input_dict['map_others_mask_list'],
                                                       self.map_others_in_polyline_encoder, self.map_others_between_polyline_encoder, device)

        ego_feature_input = input_dict['ego_feature_list']
        ego_label_input = input_dict['ego_label_list']
        agent_feature_input = input_dict['agent_feature_list']
        agent_label_input = input_dict['agent_label_list']
        agent_to_predict_num_input = input_dict['agent_to_predict_num']


    def generate(self, x):
        pass

    def encode_map_feature(self, map_data_list: List, map_mask_list: List, in_polyline_encoder, between_polyline_encoder, device):
        batch_size = len(map_data_list)
        time_set_num = self.data_dim.time_set_num
        points_in_polyline = self.data_dim.points_in_polyline
        data_dim = map_data_list[0][0].size(2)

        # find max polyline num
        max_polyline_num = 0
        for i in range(batch_size):
            for j in range(time_set_num):
                if max_polyline_num < map_data_list[i][j].size(0):
                    max_polyline_num = map_data_list[i][j].size(0)

        # build feature tensor
        batch_map_data = torch.zeros((batch_size, time_set_num, max_polyline_num, points_in_polyline, data_dim), device=device)
        batch_map_point_mask = torch.zeros((batch_size, time_set_num, max_polyline_num, points_in_polyline), device=device, dtype=torch.bool)
        batch_map_polyline_mask = torch.zeros((batch_size, time_set_num, max_polyline_num), device=device, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(time_set_num):
                batch_map_data[i, j, :map_data_list[i][j].size(0), :, :] = map_data_list[i][j]
                batch_map_point_mask[i, j, :map_data_list[i][j].size(0), :] = map_mask_list[i][j]
                batch_map_polyline_mask[i, j, :map_data_list[i][j].size(0)] = 1

        # encode
        in_polyline_encoded_data = in_polyline_encoder(batch_map_data.view((batch_size*time_set_num, max_polyline_num, points_in_polyline, data_dim)),
                                                       batch_map_point_mask.view((batch_size*time_set_num, max_polyline_num, points_in_polyline))) 
        between_polyline_encoded_data = between_polyline_encoder(in_polyline_encoded_data.view(batch_size, time_set_num, max_polyline_num, data_dim),
                                                                 batch_map_polyline_mask) # (batch, 8, feature_dim)
        return between_polyline_encoded_data

    def encode_agent_feature(self, agent_data_list: List, agent_embedding, device):
        batch_size = len(agent_data_list)
        time_set_num = self.data_dim.time_set_num
        data_dim = agent_data_list[0][0].size(2)
        
        max_agent_num = 0
        for i in range(batch_size):
            for j in time_set_num:
                if max_agent_num < agent_data_list[i][j].size(0):
                    max_agent_num = agent_data_list[i][j].size(0)

        batch_agent_data = torch.zeros((batch_size, time_set_num, max_agent_num, data_dim), device=device)