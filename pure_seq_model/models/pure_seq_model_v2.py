import torch
from torch import nn
import copy
from typing import Dict, List, Tuple, cast
import os

import models.base_model as base_model
from models.mtr_models.polyline_encoder import PointNetPolylineEncoder
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame
from utils.build_anchor import build_anchor_tensor

class PureSeqModelV2(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_parameter = config.MODEL.model_parameter
        self.data_dim = config.MODEL.data_dim
        self.data_config = config.DATA_CONFIG

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
        self.ego_seq_embedding = base_model.MLP(self.data_dim.ego, self.model_parameter.seq_embedding_dim)
        self.agent_seq_embedding = base_model.MLP(self.data_dim.agent, self.model_parameter.seq_embedding_dim)

        # ego/agent anchor embedding
        self.ego_anchor_embedding = base_model.MLP(self.data_dim.ego_anchor, self.model_parameter.seq_embedding_dim)
        self.agent_anchor_embedding = base_model.MLP(self.data_dim.agent_anchor, self.model_parameter.seq_embedding_dim)

        # pos encoder
        # self.pos_encoder = nn.Parameter(torch.zeros(20, self.model_parameter.seq_embedding_dim))
        self.pos_encoder = base_model.PositionalEncoding(self.model_parameter.seq_embedding_dim)

        # seq transformer
        seq_self_attn = base_model.SelfAttention(self.model_parameter.seq_embedding_dim, self.model_parameter.seq_head)
        seq_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.seq_embedding_dim, self.model_parameter.seq_inter_dim)
        seq_layer = base_model.TransformerEncoderLayer(self.model_parameter.seq_embedding_dim, seq_self_attn, seq_p_ff)
        self.seq_layerlist = nn.ModuleList([copy.deepcopy(seq_layer) for _ in range(self.model_parameter.seq_layer)])

        # head
        self.ego_head = nn.Linear(self.model_parameter.seq_embedding_dim, self.data_dim.ego_target)
        self.agent_head = nn.Linear(self.model_parameter.seq_embedding_dim, self.data_dim.agent_target)

        self.ego_decision_head = nn.Linear(self.model_parameter.seq_embedding_dim, self.data_dim.anchor_num)
        self.agent_decision_head = nn.Linear(self.model_parameter.seq_embedding_dim, self.data_dim.anchor_num)

        # loss
        if self.model_parameter.loss_type == 'L1':
            self._norm = torch.nn.L1Loss(reduction='none')
        elif self.model_parameter.loss_type == 'L2':
            self._norm = torch.nn.MSELoss(reduction='none')
        else:
            assert(0)
        self._fn_cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        # anchor
        time_sample_num = self.data_config.time_info.time_sample_num
        step_time = self.data_config.time_info.time_period_per_data * self.data_config.time_info.time_sample_interval
        anchor_tensor = build_anchor_tensor(self.data_config.anchor_param.speed, self.data_config.anchor_param.angle, step_time, time_sample_num)
        self.register_buffer("anchor_tensor", anchor_tensor)


    def forward(self, input_dict: Dict):
        if self.training:
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            ego_output, agent_output, ego_decision_output, agent_decision_output = self.model_forward(normed_input_dict)
            ego_output, agent_output = self.denormalize_output(ego_output, agent_output)

            loss, tb_dict, disp_dict = self.get_loss(ego_output, input_dict['ego_label'].squeeze(2),
                                                     agent_output, input_dict['agent_label'].view(agent_output.size(0), agent_output.size(1), agent_output.size(2)),
                                                     input_dict['agent_valid'].view(agent_output.size(0), agent_output.size(1)),
                                                     ego_decision_output, input_dict['ego_decision_label'].squeeze(2),
                                                     agent_decision_output, input_dict['agent_decision_label'].view(agent_output.size(0), agent_output.size(1)))
            return loss, tb_dict, disp_dict
        else:
            traj_list, log_prob_list = self.traj_generation(input_dict, self.model_parameter.traj_num)
            input_dict['traj_list'] = traj_list
            input_dict['log_prob_list'] = log_prob_list

            return input_dict

        # normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
        # # normed_input_dict = self.normalize_input(input_dict)
        # ego_output, agent_output = self.model_forward(normed_input_dict)
        # ego_output, agent_output = self.denormalize_output(ego_output, agent_output)

        # loss, tb_dict, disp_dict = self.get_loss(ego_output, input_dict['ego_label'].squeeze(2),
        #                                             agent_output, input_dict['agent_label'].view(agent_output.size(0), agent_output.size(1), agent_output.size(2)),
        #                                             input_dict['agent_valid'].view(agent_output.size(0), agent_output.size(1)))
        # return loss, tb_dict, disp_dict

    def model_forward(self, input_dict: Dict, is_decision_forward: bool = False):
        # Decision_forward is used in inference. In decision_forward:
        # input_dict['non-anchor-feature'].size(1) = input_dict['anchor-feature'].size(1) + 1
        # input_dict['anchor-feature'] is 0 tensor for the first run

        device = input_dict['ego_feature'].device
        batch_size = input_dict['ego_feature'].size(0)
        time_set_num = input_dict['ego_feature'].size(1)

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
        ego_data_embedded = self.ego_seq_embedding(input_dict['ego_feature'])
        agent_data_embedded = self.agent_seq_embedding(input_dict['agent_feature'])

        if is_decision_forward and time_set_num==1:
            ego_anchor_embedded = None
            agent_anchor_embedded = None
        else:
            ego_anchor_embedded = self.ego_anchor_embedding(input_dict['ego_anchor_feature'])
            agent_anchor_embedded = self.agent_anchor_embedding(input_dict['agent_anchor_feature'])

        # construct input seq and input mask
        seq_tensor, seq_mask = self.construct_input_seq(lane_encoded_feature, road_line_encoded_feature, road_edge_encoded_feature, map_others_encoded_feature,
                                                        ego_data_embedded, agent_data_embedded, input_dict['agent_valid'], ego_anchor_embedded, agent_anchor_embedded,
                                                        is_decision_forward)

        # pos encoder. each time set use same pos encoder
        len_per_time_set = 4+2+2*agent_data_embedded.size(2) # map:4, ego:1, agent:64
        # for i in range(time_set_num):
        #     seq_tensor[:, i*len_per_time_set:(i+1)*len_per_time_set, :] = seq_tensor[:, i*len_per_time_set:(i+1)*len_per_time_set, :] + self.pos_encoder[i, :]
        seq_tensor = self.pos_encoder(seq_tensor)

        # transformer
        for layer in self.seq_layerlist:
            # seq_tensor = layer(seq_tensor, attn_mask=seq_mask)
            seq_tensor = layer(seq_tensor, attn_mask=seq_mask)

        # ego decision head
        ego_decision_index = [i*len_per_time_set+4 for i in range(time_set_num)]
        ego_decision_output = self.ego_decision_head(seq_tensor[:, ego_decision_index, :])

        # agent decision head
        agent_decision_index = []
        for i in range(time_set_num):
            agent_decision_index += list(range(i*len_per_time_set+5, i*len_per_time_set+5+agent_data_embedded.size(2)))
        agent_decision_output = self.agent_decision_head(seq_tensor[:, agent_decision_index, :])


        # ego, agent head
        time_set_num_for_traj = time_set_num
        if is_decision_forward:
            time_set_num_for_traj = time_set_num-1

        if time_set_num_for_traj == 0:
            ego_output, agent_output = None, None
        else:
            ego_index = [i*len_per_time_set+5+agent_data_embedded.size(2) for i in range(time_set_num_for_traj)]
            ego_output = self.ego_head(seq_tensor[:, ego_index, :])

            agent_index = []
            for i in range(time_set_num_for_traj):
                agent_index += list(range(i*len_per_time_set+6+agent_data_embedded.size(2), (i+1)*len_per_time_set))
            agent_output = self.agent_head(seq_tensor[:, agent_index, :])

        return ego_output, agent_output, ego_decision_output, agent_decision_output


    def encode_map_feature(self, map_data: torch.Tensor, map_mask: torch.Tensor, in_polyline_encoder, between_polyline_encoder, device):
        in_polyline_encoded_data = in_polyline_encoder(map_data.view((map_data.size(0)*map_data.size(1), map_data.size(2), map_data.size(3), map_data.size(4))),
                                                       map_mask.view((map_data.size(0)*map_data.size(1), map_data.size(2), map_data.size(3)))) 
        polyline_valid = (map_mask.sum(dim=-1) > 0)
        between_polyline_encoded_data = between_polyline_encoder(in_polyline_encoded_data.view(map_data.size(0), map_data.size(1), map_data.size(2), self.model_parameter.in_polyline_dim),
                                                                 polyline_valid) # (batch, 8, feature_dim)
        return between_polyline_encoded_data

    def construct_input_seq(self, lane: torch.Tensor, road_line: torch.Tensor, road_edge: torch.Tensor, map_others: torch.Tensor,
                            ego: torch.Tensor, agent: torch.Tensor, agent_valid: torch.Tensor, ego_anchor: torch.Tensor, agent_anchor: torch.Tensor,
                            is_decision_forward: bool = False):
        len_per_time_set = 4+2+agent.size(2)*2
        if is_decision_forward:
            seq_len = agent.size(1)*len_per_time_set-1-agent.size(2)
        else:
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
            input_seq_tensor[:, i*len_per_time_set+5:i*len_per_time_set+5+agent.size(2), :] = agent[:, i, :, :]

            if not(is_decision_forward and (i==(agent.size(1)-1))):
                input_seq_tensor[:, i*len_per_time_set+5+agent.size(2), :] = ego_anchor[:, i, 0, :]
                input_seq_tensor[:, i*len_per_time_set+5+agent.size(2)+1:i*len_per_time_set+len_per_time_set, :] = agent_anchor[:, i, :, :]

            # input mask
            input_seq_mask_base[:, i*len_per_time_set:, i*len_per_time_set:i*len_per_time_set+5] = False
            agent_valid_matrix = (agent_valid[:, i, :].unsqueeze(1).repeat(1, seq_len-i*len_per_time_set, 1) == 0)
            input_seq_mask_base[:, i*len_per_time_set:, i*len_per_time_set+5:i*len_per_time_set+5+agent.size(2)] = agent_valid_matrix

            if not(is_decision_forward and (i==(agent.size(1)-1))):
                input_seq_mask_base[:, i*len_per_time_set+5+agent.size(2):, i*len_per_time_set+5+agent.size(2)] = False
                agent_anchor_valid_matrix = (agent_valid[:, i, :].unsqueeze(1).repeat(1, seq_len-i*len_per_time_set-5-agent.size(2), 1) == 0)
                input_seq_mask_base[:, i*len_per_time_set+5+agent.size(2):, i*len_per_time_set+5+agent.size(2)+1:i*len_per_time_set+len_per_time_set] = agent_anchor_valid_matrix

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

    def get_loss(self, ego_output, ego_label, agent_output, agent_label, agent_valid, ego_decision_output, ego_decision_label, agent_decision_output, agent_decision_label):
        # ego_output, ego_label: [batch, 8, 3*time_sample_num]
        # agent_output, agent_label: [batch, 8*max_agent_num, 3*time_sample_num]
        # agent_valid: [batch, 8*max_agent_num]
        # ego_decision_output, ego_decision_label: [batch, 8, 16], [batch, 8]
        # agent_decision_output, agent_decision_label: [batch, 8*max_agent_num, 16], [batch, 8*max_agent_num]

        # get index and weight
        x_index = list(range(0, ego_output.size(-1), 3))
        y_index = list(range(1, ego_output.size(-1), 3))
        heading_index = list(range(2, ego_output.size(-1), 3))

        x_weight = self.model_parameter.loss_weight['x']
        y_weight = self.model_parameter.loss_weight['y']
        heading_weight = self.model_parameter.loss_weight['heading']

        regression_weight = self.model_parameter.regression_weight
        classification_weight = self.model_parameter.classification_weight

        # ego loss
        ego_diff = self._norm(ego_output, ego_label)
        ego_x_loss = x_weight * torch.mean(ego_diff[:, :, x_index])
        ego_y_loss = y_weight * torch.mean(ego_diff[:, :, y_index])
        ego_heading_loss = heading_weight * torch.mean(ego_diff[:, :, heading_index])

        ego_loss = ego_x_loss + ego_y_loss + ego_heading_loss

        # agent loss
        agent_diff = self._norm(agent_output, agent_label)
        agent_x_loss = x_weight * torch.mean(torch.mean(agent_diff[:, :, x_index], dim=-1) * agent_valid)
        agent_y_loss = y_weight * torch.mean(torch.mean(agent_diff[:, :, y_index], dim=-1) * agent_valid)
        agent_heading_loss = heading_weight * torch.mean(torch.mean(agent_diff[:, :, heading_index], dim=-1) * agent_valid)
        agent_loss = agent_x_loss + agent_y_loss + agent_heading_loss

        num_valid_agent = torch.sum(agent_valid)/(agent_valid.size(0)*ego_output.size(1))
        agent_num = agent_output.size(1)/ego_output.size(1)
        
        # ego/agent decision loss
        ego_decision_loss = torch.mean(self._fn_cross_entropy(ego_decision_output.view(-1, self.data_dim.anchor_num), ego_decision_label.view(-1)))

        agent_decision_loss = self._fn_cross_entropy(agent_decision_output.view(-1, self.data_dim.anchor_num), agent_decision_label.view(-1))
        agent_decision_loss = torch.mean(agent_decision_loss.view(agent_decision_output.size(0), agent_decision_output.size(1)) * agent_valid)

        # total loss
        # loss = ego_loss + num_valid_agent * agent_loss
        regression_loss = regression_weight * (ego_loss + agent_num * agent_loss)
        classification_loss = classification_weight * (ego_decision_loss + agent_num * agent_decision_loss)
        loss = regression_loss + classification_loss

        # update info
        tb_dict = {}
        disp_dict = {}
        tb_dict['loss'] = loss.item()
        disp_dict['loss'] = loss.item()

        tb_dict['ego_x_loss'] = ego_x_loss.item()
        tb_dict['ego_y_loss'] = ego_y_loss.item()
        tb_dict['ego_heading_loss'] = ego_heading_loss.item()
        tb_dict['ego_loss'] = ego_loss.item()
        tb_dict['agent_x_loss'] = agent_x_loss.item()
        tb_dict['agent_y_loss'] = agent_y_loss.item()
        tb_dict['agent_heading_loss'] = agent_heading_loss.item()
        tb_dict['agent_loss'] = agent_loss.item()
        tb_dict['num_valid_agent'] = num_valid_agent.item()

        tb_dict['ego_decision_loss'] = ego_decision_loss.item()
        tb_dict['agent_decision_loss'] = agent_decision_loss.item()
        tb_dict['classification_loss'] = classification_loss.item()
        tb_dict['regression_loss'] = regression_loss.item()

        # for i in range(8):
        #     print('==========================================', i)
        #     print('ego_output x: ', ego_output[i, :, x_index])
        #     print('ego_label x: ', ego_label[i, :, x_index])
        #     print('ego_output y: ', ego_output[i, :, y_index])
        #     print('ego_label y: ', ego_label[i, :, y_index])
        #     print('ego_output h: ', ego_output[i, :, heading_index])
        #     print('ego_label h: ', ego_label[i, :, heading_index])


        # for i in range(8):
        #     print('===========================================', i)
        #     print("agent output x: ", agent_output[i, 0, x_index])
        #     print("agent label x: ", agent_label[i, 0, x_index])
        #     print("agent output y: ", agent_output[i, 0, y_index])
        #     print("agent label y: ", agent_label[i, 0, y_index])
        #     print("agent output h: ", agent_output[i, 0, heading_index])
        #     print("agent label h: ", agent_label[i, 0, heading_index])

        # # for i in range(8):
        # print((torch.mean(agent_diff[:, :, x_index], dim=-1) * agent_valid)[0, list(range(0, 512, 64))])
        # print((torch.mean(agent_diff[:, :, x_index], dim=-1) * agent_valid)[1, list(range(0, 512, 64))])
        # print((torch.mean(agent_diff[:, :, x_index], dim=-1) * agent_valid)[2, list(range(0, 512, 64))])
        # print(tb_dict)
        # assert(0)
        return loss, tb_dict, disp_dict

    def normalize_input(self, input_dict):
        # warning: this function will modify input_dict!
        xy_norm_para = self.model_parameter.normalization_para['xy']

        # map feature
        map_keys = ['lane_feature', 'road_line_feature', 'road_edge_feature', 'map_others_feature']
        for key in map_keys:
            map_feature_tensor = input_dict[key]
            map_feature_tensor[..., [0, 1, 4, 5]] = map_feature_tensor[..., [0, 1, 4, 5]] / xy_norm_para
            input_dict[key] = map_feature_tensor

        # ego feature
        xy_index = list(range(0, input_dict['ego_feature'].size(-1)-5, 5)) + list(range(1, input_dict['ego_feature'].size(-1)-5, 5)) + [-5, -4]

        feature_tensor = input_dict['ego_feature']
        feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
        input_dict['ego_feature'] = feature_tensor

        # agent feature
        xy_index = list(range(0, input_dict['agent_feature'].size(-1)-5, 10)) + list(range(1, input_dict['agent_feature'].size(-1)-5, 10)) + \
                   list(range(5, input_dict['agent_feature'].size(-1)-5, 10)) + list(range(6, input_dict['agent_feature'].size(-1)-5, 10)) + [-5, -4]

        feature_tensor = input_dict['agent_feature']
        feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
        input_dict['agent_feature'] = feature_tensor

        # ego, agent anchor feature
        if input_dict['ego_anchor_feature'] is not None:
            xy_index = list(range(0, input_dict['ego_anchor_feature'].size(-1), 3)) + list(range(1, input_dict['ego_anchor_feature'].size(-1), 3))

            feature_tensor = input_dict['ego_anchor_feature']
            feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
            input_dict['ego_anchor_feature'] = feature_tensor

            feature_tensor = input_dict['agent_anchor_feature']
            feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
            input_dict['agent_anchor_feature'] = feature_tensor

        return input_dict

    def denormalize_output(self, ego_output, agent_output):
        xy_norm_para = self.model_parameter.normalization_para['xy']
        heading_norm_para = self.model_parameter.normalization_para['heading']

        xy_index = list(range(0, ego_output.size(-1), 3)) + list(range(1, ego_output.size(-1), 3))
        heading_index = list(range(2, ego_output.size(-1), 3))

        denormed_ego_output = torch.zeros(ego_output.shape, device=ego_output.device)
        denormed_ego_output[..., xy_index] = ego_output[..., xy_index] * xy_norm_para
        denormed_ego_output[..., heading_index] = ego_output[..., heading_index] * heading_norm_para

        denormed_agent_output = torch.zeros(agent_output.shape, device=agent_output.device)
        denormed_agent_output[..., xy_index] = agent_output[..., xy_index] * xy_norm_para
        denormed_agent_output[..., heading_index] = agent_output[..., heading_index] * heading_norm_para
        
        return denormed_ego_output, denormed_agent_output

    def traj_generation(self, input_dict, traj_num):
        # traj_list_tensor: [batch, traj_num, max_agent_num, 80, 3]

        traj_list = []
        log_prob_list = []

        # generate one deterministic traj
        one_traj, log_prob_sum = self.generate_one_traj(copy.deepcopy(input_dict), True)
        traj_list.append(one_traj.unsqueeze(1))
        log_prob_list.append(log_prob_sum.unsqueeze(1))

        # generate (traj_num-1) sampled trajs
        for _ in range(traj_num-1):
            one_traj, log_prob_sum = self.generate_one_traj(copy.deepcopy(input_dict), False)
            traj_list.append(one_traj.unsqueeze(1))
            log_prob_list.append(log_prob_sum.unsqueeze(1))
            # traj_list.append(one_traj.unsqueeze(1))
            # log_prob_list.append(log_prob_sum.unsqueeze(1))

        traj_list_tensor = torch.cat(traj_list, dim=1)
        log_prob_list_tensor = torch.cat(log_prob_list, dim=1)

        return traj_list_tensor, log_prob_list_tensor

    def generate_one_traj(self, input_dict, is_deterministic):
        # get first step input and output
        for key in ['lane_feature', 'lane_mask', 'road_line_feature', 'road_line_mask', 'road_edge_feature', 'road_edge_mask', 
                    'map_others_feature', 'map_others_mask', 'ego_feature', 'agent_feature', 'agent_valid']:
            input_dict[key] = input_dict[key][:, 0, ...].unsqueeze(1)
        for key in ['ego_anchor_feature', 'agent_anchor_feature']:
            input_dict[key] = None

        log_prob_sum = torch.zeros(input_dict['ego_feature'].size(0), device=input_dict['ego_feature'].device)

        normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
        normed_ego_output, normed_agent_output, ego_decision_output, agent_decision_output = self.model_forward(normed_input_dict, is_decision_forward=True)
        input_dict, log_prob_sum = self.extend_input_by_sample(input_dict, ego_decision_output, agent_decision_output, is_deterministic, log_prob_sum)
        normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
        normed_ego_output, normed_agent_output, ego_decision_output, agent_decision_output = self.model_forward(normed_input_dict, is_decision_forward=False)
        ego_output, agent_output = self.denormalize_output(normed_ego_output, normed_agent_output)

        # generate next input based on output
        time_set_num = self.data_config.time_info.time_set_num
        for i in range(time_set_num-1):
            input_dict = self.extend_input_based_on_last_output(input_dict, ego_output, agent_output)
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            normed_ego_output, normed_agent_output, ego_decision_output, agent_decision_output = self.model_forward(normed_input_dict, is_decision_forward=True)
            input_dict, log_prob_sum = self.extend_input_by_sample(input_dict, ego_decision_output, agent_decision_output, is_deterministic, log_prob_sum)
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            normed_ego_output, normed_agent_output, ego_decision_output, agent_decision_output = self.model_forward(normed_input_dict, is_decision_forward=False)
            ego_output, agent_output = self.denormalize_output(normed_ego_output, normed_agent_output)
        
        # get trajs in agent own frame
        ego_traj, agent_traj = self.build_traj_from_output(ego_output, agent_output, input_dict['agent_to_predict_num'])

        # transform agent trajs to ego frame
        agent_traj = self.transform_agent_traj_to_origin_frame(input_dict['agent_feature'][:, 0, :, 0:3], agent_traj, input_dict['agent_to_predict_num'])

        # transform agent
        ego_current_frame = input_dict['ego_current_pose'][:, 0, :].unsqueeze(1).repeat(1, agent_traj.size(1), 1)
        agent_global_traj = self.transform_agent_traj_to_origin_frame(ego_current_frame, agent_traj, input_dict['agent_to_predict_num'])

        return agent_global_traj, log_prob_sum


    def extend_input_by_sample(self, input_dict, ego_decision_output, agent_decision_output, is_deterministic = False, log_prob_sum = 0):
        # extend ego_anchor_feature, agent_anchor_feature by sample from decision
        # ego_anchor_feature: [batch, t, 1, time_sample_num*3] -> [batch, t+1, 1, time_sample_num*3]
        # agent_anchor_feature: [batch, t, n, time_sample_num*3] -> [batch, t+1, n, 3*time_sample_num*3]
        # ego_decision_output: [batch, t*1, 16], agent_decision_output: [batch, t*n, 16]

        # sample decision
        agent_num = input_dict['agent_feature'].size(2)
        if is_deterministic:
            ego_anchor_index = torch.argmax(ego_decision_output[:, -1, :], dim=1)
            agent_anchor_index = torch.argmax(agent_decision_output[:, -agent_num:, :], dim=2)
        else:
            ego_sampler = torch.distributions.categorical.Categorical(logits=ego_decision_output[:, -1, :])
            ego_anchor_index = ego_sampler.sample()

            agent_sampler = torch.distributions.categorical.Categorical(logits=agent_decision_output[:, -agent_num:, :])
            agent_anchor_index = agent_sampler.sample()
        
        # get anchor and log prob sum
        batch = ego_decision_output.size(0)
        next_ego_anchor = torch.zeros(batch, 1, 1, self.anchor_tensor.size(1), device=ego_decision_output.device)
        next_agent_anchor = torch.zeros(batch, 1, agent_num, self.anchor_tensor.size(1), device=ego_decision_output.device)
        for i in range(batch):
            next_ego_anchor[i, 0, 0, :] = self.anchor_tensor[ego_anchor_index[i], :]
            log_prob_sum[i] = log_prob_sum[i] + ego_decision_output[i, -1, ego_anchor_index[i]]
            for j in range(agent_num):
                next_agent_anchor[i, 0, j, :] = self.anchor_tensor[agent_anchor_index[i, j], :]
                log_prob_sum += agent_decision_output[i, -agent_num+j, agent_anchor_index[i, j]]

        # concat feature
        if input_dict['ego_anchor_feature'] is None:
            input_dict['ego_anchor_feature'] = next_ego_anchor
            input_dict['agent_anchor_feature'] = next_agent_anchor
        else:
            input_dict['ego_anchor_feature'] = torch.cat((input_dict['ego_anchor_feature'], next_ego_anchor), dim=1)
            input_dict['agent_anchor_feature'] = torch.cat((input_dict['agent_anchor_feature'], next_agent_anchor), dim=1)

        return input_dict, log_prob_sum


    def extend_input_based_on_last_output(self, input_dict, ego_output, agent_output):
        # get current ego pose relative to last ego pose
        ego_pose_rel_to_last = ego_output[:, -1, 0:3]

        # transform map features
        map_keys = ['lane', 'road_line', 'road_edge', 'map_others']
        for key in map_keys:
            last_map_feature = input_dict[key+'_feature'][:, -1, ...].unsqueeze(1)
            next_map_feature = self.transform_map_feature_to_frame(last_map_feature, ego_pose_rel_to_last)
            input_dict[key+'_feature'] = torch.cat((input_dict[key+'_feature'], next_map_feature), dim=1)

            last_map_mask = input_dict[key+'_mask'][:, -1, ...].unsqueeze(1)
            input_dict[key+'_mask'] = torch.cat((input_dict[key+'_mask'], last_map_mask), dim=1)
        
        # transform ego output to ego feature
        last_ego_feature = input_dict['ego_feature'][:, -1, ...].unsqueeze(1)
        next_ego_feature = self.transform_ego_feature_from_output(ego_output[:, -1, :].unsqueeze(1), last_ego_feature)
        input_dict['ego_feature'] = torch.cat((input_dict['ego_feature'], next_ego_feature), dim=1)

        # transform agent output to agent feature
        agent_num = input_dict['agent_feature'].size(2)
        last_agent_feature = input_dict['agent_feature'][:, -1, ...].unsqueeze(1)
        next_agent_feature = self.transform_agent_feature_from_output(agent_output[:, -agent_num:, :], last_agent_feature, ego_output[:, -1, :].unsqueeze(1))
        input_dict['agent_feature'] = torch.cat((input_dict['agent_feature'], next_agent_feature), dim=1)

        last_agent_valid = input_dict['agent_valid'][:, -1, ...].unsqueeze(1)
        input_dict['agent_valid'] = torch.cat((input_dict['agent_valid'], last_agent_valid), dim=1)
        
        return input_dict
    

    def transform_map_feature_to_frame(self, map_feature: torch.Tensor, ego_pose: torch.Tensor):
        batch_size = ego_pose.size(0)

        for i in range(batch_size):
            ego_frame = ego_pose[i, :]
            rotation_frame = torch.tensor([0, 0, ego_frame[2]], device=ego_pose.device)

            # reshape map_feature [1, 1, n, 20, d] to [1*1*n*20, d]
            reshaped_map_feature = map_feature[i, ...].view(-1, map_feature.size(-1))

            # transform
            reshaped_map_feature[:, 0:2] = coordinates_to_local_frame(reshaped_map_feature[:, 0:2], ego_frame)
            reshaped_map_feature[:, 2:4] = coordinates_to_local_frame(reshaped_map_feature[:, 2:4], rotation_frame)
            reshaped_map_feature[:, 4:6] = coordinates_to_local_frame(reshaped_map_feature[:, 4:6], ego_frame)

        return map_feature

    def transform_ego_feature_from_output(self, last_ego_output: torch.Tensor, last_ego_feature: torch.Tensor):
        # last_ego_output: [batch, 1, 3*time_sample_num]
        # last_ego_feature: [batch, 1, 1, 5*time_sample_num+5]
        batch_size = last_ego_output.size(0)
        time_sample_num = self.data_config.time_info.time_sample_num

        next_ego_feature = last_ego_feature.clone().detach()
        for i in range(batch_size):
            ego_frame = last_ego_output[i, 0, 0:3]

            reshaped_ego_output = last_ego_output[i, 0, :].view(time_sample_num, 3)
            reshaped_next_ego_feature = next_ego_feature[i, 0, 0, 0:5*time_sample_num].view(time_sample_num, 5)
            reshaped_next_ego_feature[:, 0:3] = global_state_se2_tensor_to_local(reshaped_ego_output[:, 0:3], ego_frame)
            reshaped_next_ego_feature[:, 3] = torch.cos(reshaped_next_ego_feature[:, 2])
            reshaped_next_ego_feature[:, 4] = torch.sin(reshaped_next_ego_feature[:, 2])

        return next_ego_feature

    def transform_agent_feature_from_output(self, last_agent_output: torch.Tensor, last_agent_feature: torch.Tensor, last_ego_output: torch.Tensor):
        # last_agent_output: [batch, 1*agent_num, 3*time_sample_num]
        # last_agent_feature: [batch, 1, agent_num, 10*time_sample_num]
        batch_size = last_ego_output.size(0)
        agent_num = last_agent_feature.size(2)
        time_sample_num = self.data_config.time_info.time_sample_num

        next_agent_feature = last_agent_feature.clone().detach()
        for i in range(batch_size):
            ego_frame = last_ego_output[i, 0, 0:3]
            rotation_frame = torch.tensor([0, 0, ego_frame[2]], device=last_ego_output.device)

            reshaped_agent_output = last_agent_output[i, :, :].view(agent_num, time_sample_num, 3)
            reshaped_agent_feature = next_agent_feature[i, 0, :, 0:10*time_sample_num].view(agent_num, time_sample_num, 10)

            for j in range(agent_num):
                agent_frame = last_agent_feature[i, 0, j, 0:3]
                agent_rotation_frame = torch.tensor([0, 0, agent_frame[2]], device=last_agent_output.device)

                agent_next_frame = last_agent_output[i, j, 0:3]
                agent_next_rotation_frame = torch.tensor([0, 0, agent_next_frame[2]], device=last_agent_output.device)

                reshaped_agent_feature[j, :, 5:8] = global_state_se2_tensor_to_local(reshaped_agent_output[j, :, 0:3], agent_next_frame)
                reshaped_agent_feature[j, :, 8] = torch.cos(reshaped_agent_feature[j, :, 9])
                reshaped_agent_feature[j, :, 9] = torch.sin(reshaped_agent_feature[j, :, 9])

                last_agent_in_last_ego_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=last_agent_output.device), agent_frame)
                pose_in_last_ego_frame = global_state_se2_tensor_to_local(reshaped_agent_output[j, :, 0:3], last_agent_in_last_ego_frame[0, :])
                reshaped_agent_feature[j, :, 0:3] = global_state_se2_tensor_to_local(pose_in_last_ego_frame, ego_frame)
                reshaped_agent_feature[j, :, 3] = torch.cos(reshaped_agent_feature[j, :, 2])
                reshaped_agent_feature[j, :, 4] = torch.sin(reshaped_agent_feature[j, :, 2])

        return next_agent_feature

    def build_traj_from_output(self, ego_output, agent_output, agent_to_predict_num):
        # input: 
        # ego_output, ego_label: [batch, 8, 3*time_sample_num]
        # agent_output, agent_label: [batch, 8*max_agent_num, 3*time_sample_num]
        # agent_to_predict_num: [batch, 1]
        # output: 
        # ego_traj: [batch, 1, 80, 3],
        # agent_traj: [batch, max_agent_num, 80, 3],
        # 3 is x, y, heading

        time_set_num = self.data_config.time_info.time_set_num
        total_time_step = self.data_config.time_info.total_time_step
        time_sample_num = self.data_config.time_info.time_sample_num
        time_set_interval = self.data_config.time_info.time_set_interval
        max_agent_num = self.data_config.max_agent_num
        batch = ego_output.size(0)

        ego_traj = torch.zeros(batch, 1, total_time_step, 3, device=ego_output.device)
        agent_traj = torch.zeros(batch, max_agent_num, total_time_step, 3, device=ego_output.device)

        agent_output_reshaped = agent_output.view(batch, 8, max_agent_num, time_sample_num, 3)

        for i in range(batch):
            ego_frame = torch.zeros(3, device=ego_output.device)
            agent_frame = torch.zeros(int(agent_to_predict_num[i, 0].item()), 3, device=ego_output.device)

            for j in range(time_set_num):
                # generate ego traj
                ego_traj[i, 0, j*time_set_interval:(j+1)*time_set_interval, :] = self.output_to_traj_in_one_time_set(
                    base_frame=ego_frame,
                    output=ego_output[i, j, :].view(time_sample_num, 3)
                )
                ego_frame = ego_traj[i, 0, (j+1)*time_set_interval-1, 0:3]

                # generate agent traj
                for k in range(int(agent_to_predict_num[i, 0].item())):
                    agent_traj[i, k, j*time_set_interval:(j+1)*time_set_interval, :] = self.output_to_traj_in_one_time_set(
                        base_frame=agent_frame[k, :],
                        output=agent_output_reshaped[i, j, k, :, :]
                    )
                    agent_frame[k, :] = agent_traj[i, k, (j+1)*time_set_interval-1, 0:3]

        return ego_traj, agent_traj

    def output_to_traj_in_one_time_set(self, base_frame, output):
        # input:
        # base_frame: [3], outputs are expressed in this frame
        # output: [time_sample_num, 3]
        # output:
        # traj: [10, 3]

        # interpolate output. Notice that, the output index is reversed
        time_sample_num = self.data_config.time_info.time_sample_num
        time_sample_interval = self.data_config.time_info.time_sample_interval
        if time_sample_num == 5 and time_sample_interval == 2:
            interpolated_output = torch.zeros(10, 3, device=output.device)

            interpolated_output[[1, 3, 5, 7, 9], :] = output[[4, 3, 2, 1, 0], :]
            interpolated_output[[2, 4, 6, 8], :] = 0.5*(output[[4, 3, 2, 1], :] + output[[3, 2, 1, 0], :])
            interpolated_output[0, 0:3] = 0.5*output[-1, 0:3]
            interpolated_output[0, 3:5] = output[-1, 3:5] # this is not accurate, but no effect on metric
        else:
            assert 0

        # transform output to origin frame
        origin_in_base_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=output.device), base_frame)
        interpolated_output[:, 0:3] = global_state_se2_tensor_to_local(interpolated_output[:, 0:3], origin_in_base_frame[0, :])

        return interpolated_output

    def transform_agent_traj_to_origin_frame(self, current_frame, agent_traj, agent_to_predict_num):
        # input:
        # current_frame: [batch, max_agent_num, 3]
        # agent_traj: [batch, max_agent_num, 80, 3]
        # agent_to_predict_num: [batch, 1]
        # output:
        # agent_traj: [batch, max_agent_num, 80, 3]
        batch = agent_traj.size(0)
        for i in range(batch):
            for j in range(int(agent_to_predict_num[i, 0].item())):
                origin_in_current_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=agent_traj.device), current_frame[i, j, :])

                agent_traj[i, j, :, 0:3] = global_state_se2_tensor_to_local(agent_traj[i, j, :, 0:3], origin_in_current_frame[0, :])

        return agent_traj

    def traj_generation_by_aggregation(self, input_dict):
        traj_num = self.model_parameter.traj_num
        traj_trial_num = self.model_parameter.traj_trial_num
        traj_init_min_dis = self.model_parameter.traj_init_min_dis
        traj_iteration = self.model_parameter.traj_iteration

        # generate traj candidate
        traj_candidate_list, candidtae_log_prob_list = self.traj_generation(input_dict, traj_trial_num)

        # traj_aggregation



        return 

    def traj_aggregation(self, candidate_traj, candidate_log_prob):
        # candidate_traj: [batch, trial_num, max_agent_num, 80, 3]
        # candidate_traj: [batch, trial_num]




        pass

