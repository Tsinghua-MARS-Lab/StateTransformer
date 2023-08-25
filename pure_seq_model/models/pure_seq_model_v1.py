import torch
from torch import nn
import copy
from typing import Dict, List, Tuple, cast
import os

import models.base_model as base_model
from models.mtr_models.polyline_encoder import PointNetPolylineEncoder

class PureSeqModelV1(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_parameter = config.model_parameter
        self.data_dim = config.data_dim

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

        # loss
        self._fn_mse = torch.nn.MSELoss(reduction='none')


    def forward(self, input_dict: Dict):
        if self.training:
            input_dict = self.normalize_input(input_dict)
            ego_output, agent_output = self.model_forward(input_dict)
            ego_output, agent_output = self.denormalize_output(ego_output, agent_output)

            loss, tb_dict, disp_dict = self.get_loss(ego_output, input_dict['ego_label'].squeeze(2),
                                                     agent_output, input_dict['agent_label'].view(agent_output.size(0), agent_output.size(1), agent_output.size(2)),
                                                     input_dict['agent_valid'].view(agent_output.size(0), agent_output.size(1)))
            return loss, tb_dict, disp_dict
        else:
            ego_output, agent_output = self.traj_generation(input_dict)

            return ego_output, agent_output


    def model_forward(self, input_dict: Dict):
        device = input_dict['ego_feature'].device
        batch_size = input_dict['ego_feature'].size(0)
        time_set_num = self.data_dim.time_set_num

        # map encoder
        lane_encoded_feature = self.encode_map_feature(input_dict['lane_feature'], input_dict['lane_mask'],
                                                       self.lane_in_polyline_encoder, self.lane_between_polyline_encoder, device)
        road_line_encoded_feature = self.encode_map_feature(input_dict['road_line_feature'], input_dict['road_line_mask'],
                                                       self.road_line_in_polyline_encoder, self.road_line_between_polyline_encoder, device)
        road_edge_encoded_feature = self.encode_map_feature(input_dict['road_edge_feature'], input_dict['road_edge_mask'],
                                                       self.road_edge_in_polyline_encoder, self.road_edge_between_polyline_encoder, device)
        map_others_encoded_feature = self.encode_map_feature(input_dict['map_others_feature'], input_dict['map_others_mask'],
                                                       self.map_others_in_polyline_encoder, self.map_others_between_polyline_encoder, device)

        # ego, agent embedding
        ego_data_embeded = self.ego_seq_embedding(input_dict['ego_feature'])
        agent_data_embeded = self.agent_seq_embedding(input_dict['agent_feature'])

        # construct input seq and input mask
        seq_tensor, seq_mask = self.construct_input_seq(lane_encoded_feature, road_line_encoded_feature, road_edge_encoded_feature, map_others_encoded_feature,
                                                                    ego_data_embeded, agent_data_embeded, input_dict['agent_valid'])

        # pos encoder. each time set use same pos encoder
        len_per_time_set = 4+1+agent_data_embeded.size(2)
        for i in range(time_set_num):
            seq_tensor[:, i*len_per_time_set:(i+1)*len_per_time_set, :] = seq_tensor[:, i*len_per_time_set:(i+1)*len_per_time_set, :] + self.pos_encoder[i, :]
            
        # transformer
        for layer in self.seq_layerlist:
            seq_tensor = layer(seq_tensor, attn_mask=seq_mask)

        # ego heads
        ego_index = [i*len_per_time_set+4 for i in range(time_set_num)]
        ego_output = self.ego_head(seq_tensor[:, ego_index, :])

        # agent heads
        agent_index = []
        for i in range(time_set_num):
            agent_index += list(range(i*len_per_time_set+5, (i+1)*len_per_time_set))
        agent_output = self.agent_head(seq_tensor[:, agent_index, :])
        # agent_valid_reshape = input_dict['agent_valid'].view(batch_size, agent_output.size(1))

        return ego_output, agent_output


    def encode_map_feature(self, map_data: torch.Tensor, map_mask: torch.Tensor, in_polyline_encoder, between_polyline_encoder, device):
        in_polyline_encoded_data = in_polyline_encoder(map_data.view((map_data.size(0)*map_data.size(1), map_data.size(2), map_data.size(3), map_data.size(4))),
                                                       map_mask.view((map_data.size(0)*map_data.size(1), map_data.size(2), map_data.size(3)))) 
        polyline_valid = (map_mask.sum(dim=-1) > 0)
        between_polyline_encoded_data = between_polyline_encoder(in_polyline_encoded_data.view(map_data.size(0), map_data.size(1), map_data.size(2), self.model_parameter.in_polyline_dim),
                                                                 polyline_valid) # (batch, 8, feature_dim)
        return between_polyline_encoded_data

    def construct_input_seq(self, lane: torch.Tensor, road_line: torch.Tensor, road_edge: torch.Tensor, map_others: torch.Tensor,
                            ego: torch.Tensor, agent: torch.Tensor, agent_valid: torch.Tensor):
        len_per_time_set = 4+1+agent.size(2)
        seq_len = agent.size(1)*len_per_time_set
        input_seq_tensor = torch.zeros(agent.size(0), seq_len, agent.size(3), device=agent.device)
        input_seq_mask_base = torch.ones(agent.size(0), seq_len, seq_len, dtype=torch.bool, device=agent.device)

        for i in range(agent.size(1)):
            # input seq
            input_seq_tensor[:, i*len_per_time_set, :] = lane[:, i, :]
            input_seq_tensor[:, i*len_per_time_set+1, :] = road_line[:, i, :]
            input_seq_tensor[:, i*len_per_time_set+2, :] = road_edge[:, i, :]
            input_seq_tensor[:, i*len_per_time_set+3, :] = map_others[:, i, :]
            input_seq_tensor[:, i*len_per_time_set+4, :] = ego[:, i, 0, :]
            input_seq_tensor[:, i*len_per_time_set+5:i*len_per_time_set+len_per_time_set, :] = agent[:, i, :, :]

            # input mask
            input_seq_mask_base[:, i*len_per_time_set:, i*len_per_time_set:i*len_per_time_set+5] = False
            agent_valid_one_time_matrix = (agent_valid[:, i, :].unsqueeze(1).repeat(1, seq_len-i*len_per_time_set, 1) == 0)
            input_seq_mask_base[:, i*len_per_time_set:, i*len_per_time_set+5:i*len_per_time_set+len_per_time_set] = agent_valid_one_time_matrix

        input_seq_mask = input_seq_mask_base.repeat_interleave(self.model_parameter.seq_head, dim=0)

        return input_seq_tensor, input_seq_mask
            

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch

    def get_loss(self, ego_output, ego_label, agent_output, agent_label, agent_valid):
        # ego_output, ego_label: [batch, 8, 5*time_sample_num]
        # agent_output, agent_label: [batch, 8*max_agent_num, 5*time_sample_num]
        # agent_valid: [batch, 8*max_agent_num]

        # get index and weight
        xy_index = list(range(0, ego_output.size(-1), 5)) + list(range(1, ego_output.size(-1), 5))
        heading_index = list(range(2, ego_output.size(-1), 5))
        vel_index = list(range(3, ego_output.size(-1), 5)) + list(range(4, ego_output.size(-1), 5))

        xy_weight = self.model_parameter.loss_weight['xy']
        heading_weight = self.model_parameter.loss_weight['heading']
        vel_weight = self.model_parameter.loss_weight['vel']

        # ego loss
        ego_diff = self._fn_mse(ego_output, ego_label)
        ego_xy_loss = xy_weight * torch.mean(ego_diff[:, :, xy_index])
        ego_heading_loss = heading_weight * torch.mean(ego_diff[:, :, heading_index])
        ego_vel_loss = vel_weight * torch.mean(ego_diff[:, :, vel_index])

        ego_loss = ego_xy_loss + ego_heading_loss + ego_vel_loss

        # agent loss
        agent_diff = self._fn_mse(agent_output, agent_label)
        agent_xy_loss = xy_weight * torch.mean(torch.mean(agent_diff[:, :, xy_index], dim=-1) * agent_valid)
        agent_heading_loss = heading_weight * torch.mean(torch.mean(agent_diff[:, :, heading_index], dim=-1) * agent_valid)
        agent_vel_loss = vel_weight * torch.mean(torch.mean(agent_diff[:, :, vel_index], dim=-1) * agent_valid)

        agent_loss = agent_xy_loss + agent_heading_loss + agent_vel_loss

        num_valid_agent = torch.sum(agent_valid)/(agent_valid.size(0)*ego_output.size(1))

        # total loss
        loss = ego_loss + num_valid_agent * agent_loss

        # update info
        tb_dict = {}
        disp_dict = {}
        tb_dict['loss'] = loss.item()
        disp_dict['loss'] = loss.item()

        tb_dict['ego_xy_loss'] = ego_xy_loss.item()
        tb_dict['ego_heading_loss'] = ego_heading_loss.item()
        tb_dict['ego_vel_loss'] = ego_vel_loss.item()
        tb_dict['ego_loss'] = ego_loss.item()
        tb_dict['agent_xy_loss'] = agent_xy_loss.item()
        tb_dict['agent_heading_loss'] = agent_heading_loss.item()
        tb_dict['agent_vel_loss'] = agent_vel_loss.item()
        tb_dict['agent_loss'] = agent_loss.item()
        tb_dict['num_valid_agent'] = num_valid_agent.item()

        # print(ego_output)
        # print(ego_label)
        # print(tb_dict)
        # assert(0)

        return loss, tb_dict, disp_dict

    def normalize_input(self, input_dict):
        # warning: this function will modify input_dict!
        xy_norm_para = self.model_parameter.normalization_para['xy']
        heading_norm_para = self.model_parameter.normalization_para['heading']
        vel_norm_para = self.model_parameter.normalization_para['vel']

        # map feature
        map_keys = ['lane_feature', 'road_line_feature', 'road_edge_feature', 'map_others_feature']
        for key in map_keys:
            map_feature_tensor = input_dict[key]
            map_feature_tensor[..., [0, 1, 4, 5]] = map_feature_tensor[..., [0, 1, 4, 5]] / xy_norm_para
            input_dict[key] = map_feature_tensor

        # ego, agent feature
        ego_agent_keys = ['ego_feature', 'agent_feature']
        xy_index = list(range(0, input_dict['ego_label'].size(-1), 5)) + list(range(1, input_dict['ego_label'].size(-1), 5))
        heading_index = list(range(2, input_dict['ego_label'].size(-1), 5))
        vel_index = list(range(3, input_dict['ego_label'].size(-1), 5)) + list(range(4, input_dict['ego_label'].size(-1), 5))
        for key in ego_agent_keys:
            feature_tensor = input_dict[key]
            feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
            feature_tensor[..., heading_index] = feature_tensor[..., heading_index] / heading_norm_para
            feature_tensor[..., vel_index] = feature_tensor[..., vel_index] / vel_norm_para
            input_dict[key] = feature_tensor

        return input_dict

    def denormalize_output(self, ego_output, agent_output):
        xy_norm_para = self.model_parameter.normalization_para['xy']
        heading_norm_para = self.model_parameter.normalization_para['heading']
        vel_norm_para = self.model_parameter.normalization_para['vel']

        xy_index = list(range(0, ego_output.size(-1), 5)) + list(range(1, ego_output.size(-1), 5))
        heading_index = list(range(2, ego_output.size(-1), 5))
        vel_index = list(range(3, ego_output.size(-1), 5)) + list(range(4, ego_output.size(-1), 5))

        denormed_ego_output = torch.zeros(ego_output.shape, device=ego_output.device)
        denormed_ego_output[..., xy_index] = ego_output[..., xy_index] * xy_norm_para
        denormed_ego_output[..., heading_index] = ego_output[..., heading_index] * heading_norm_para
        denormed_ego_output[..., vel_index] = ego_output[..., vel_index] * vel_norm_para

        denormed_agent_output = torch.zeros(agent_output.shape, device=agent_output.device)
        denormed_agent_output[..., xy_index] = agent_output[..., xy_index] * xy_norm_para
        denormed_agent_output[..., heading_index] = agent_output[..., heading_index] * heading_norm_para
        denormed_agent_output[..., vel_index] = agent_output[..., vel_index] * vel_norm_para
        
        return denormed_ego_output, denormed_agent_output


    # auto-regessive generating future traj
    def traj_generation(self, input_dict):
        # get first step input and output
        for key in input_dict.keys():
            input_dict[key] = input_dict[key][:, 0:1, ...]

        normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
        normed_ego_output, normed_agent_output = self.model_forward(normed_input_dict)
        ego_output, agent_output = self.denormalize_output(normed_ego_output, normed_agent_output)

        # generate next input based on output
        
        
        

    def extend_input_based_on_last_output(self, input_dict, ego_output, agent_output):
        # get current ego pose relative to last ego pose
        ego_pose = ego_output[:, -1, 0:2]
        ego_frame = ego_pose
        rotation_frame = torch.tensor([0, 0, ego_pose[2]])

        # transform map features
        map_keys = ['lane', 'road_line', 'road_edge', 'map_others']
        for key in map_keys:
            last_map_feature = input_dict[key][:, -1, ...].unsqueeze(1)

        pass