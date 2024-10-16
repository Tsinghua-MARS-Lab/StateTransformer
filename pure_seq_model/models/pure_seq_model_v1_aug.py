import torch
from torch import nn
import copy
from typing import Dict, List, Tuple, cast
import os

import models.base_model as base_model
from models.mtr_models.polyline_encoder import PointNetPolylineEncoder
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame

class PureSeqModelV1Aug(nn.Module):
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

        # pos encoder
        # self.pos_encoder = nn.Parameter(torch.zeros(20, self.model_parameter.seq_embedding_dim))
        self.pos_encoder = base_model.PositionalEncoding(self.model_parameter.seq_embedding_dim)

        # seq transformer
        seq_self_attn = base_model.SelfAttention(self.model_parameter.seq_embedding_dim, self.model_parameter.seq_head)
        seq_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.seq_embedding_dim, self.model_parameter.seq_inter_dim)
        seq_layer = base_model.TransformerEncoderLayer(self.model_parameter.seq_embedding_dim, seq_self_attn, seq_p_ff)
        self.seq_layerlist = nn.ModuleList([copy.deepcopy(seq_layer) for _ in range(self.model_parameter.seq_layer)])

        # head
        self.ego_head = base_model.MLP(self.model_parameter.seq_embedding_dim, self.data_dim.ego_target)
        self.agent_head = base_model.MLP(self.model_parameter.seq_embedding_dim, self.data_dim.agent_target)

        # loss
        if self.model_parameter.loss_type == 'L1':
            self._norm = torch.nn.L1Loss(reduction='none')
        elif self.model_parameter.loss_type == 'L2':
            self._norm = torch.nn.MSELoss(reduction='none')
        else:
            assert(0)


    def forward(self, input_dict: Dict):
        if self.training:
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            # normed_input_dict = self.normalize_input(input_dict)
            ego_output, agent_output = self.model_forward(normed_input_dict)
            ego_output, agent_output = self.denormalize_output(ego_output, agent_output)

            # ego_output, agent_output = self.vel_constant_test(input_dict)

            loss, tb_dict, disp_dict = self.get_loss(ego_output, input_dict['ego_label'].squeeze(2),
                                                     agent_output, input_dict['agent_label'].view(agent_output.size(0), agent_output.size(1), agent_output.size(2)),
                                                     input_dict['agent_valid'].view(agent_output.size(0), agent_output.size(1)))
            return loss, tb_dict, disp_dict
        else:
            agent_global_traj = self.traj_generation(input_dict)
            input_dict['agent_global_traj'] = agent_global_traj

            return input_dict

        # normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
        # # normed_input_dict = self.normalize_input(input_dict)
        # ego_output, agent_output = self.model_forward(normed_input_dict)
        # ego_output, agent_output = self.denormalize_output(ego_output, agent_output)

        # # ego_output, agent_output = self.vel_constant_test(input_dict)

        # loss, tb_dict, disp_dict = self.get_loss(ego_output, input_dict['ego_label'].squeeze(2),
        #                                             agent_output, input_dict['agent_label'].view(agent_output.size(0), agent_output.size(1), agent_output.size(2)),
        #                                             input_dict['agent_valid'].view(agent_output.size(0), agent_output.size(1)))
        # return loss, tb_dict, disp_dict

    def model_forward(self, input_dict: Dict):
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
        ego_data_embeded = self.ego_seq_embedding(input_dict['ego_feature'])
        agent_data_embeded = self.agent_seq_embedding(input_dict['agent_feature'])

        # construct input seq and input mask
        seq_tensor, seq_mask = self.construct_input_seq(lane_encoded_feature, road_line_encoded_feature, road_edge_encoded_feature, map_others_encoded_feature,
                                                                    ego_data_embeded, agent_data_embeded, input_dict['agent_valid'])

        # pos encoder. each time set use same pos encoder
        len_per_time_set = 4+1+agent_data_embeded.size(2) # map:4, ego:1, agent:64
        # for i in range(time_set_num):
        #     seq_tensor[:, i*len_per_time_set:(i+1)*len_per_time_set, :] = seq_tensor[:, i*len_per_time_set:(i+1)*len_per_time_set, :] + self.pos_encoder[i, :]
        seq_tensor = self.pos_encoder(seq_tensor)

        # transformer
        for layer in self.seq_layerlist:
            # seq_tensor = layer(seq_tensor, attn_mask=seq_mask)
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
        x_index = list(range(0, ego_output.size(-1), 5))
        y_index = list(range(1, ego_output.size(-1), 5))
        heading_index = list(range(2, ego_output.size(-1), 5))
        vel_x_index = list(range(3, ego_output.size(-1), 5))
        vel_y_index = list(range(4, ego_output.size(-1), 5))

        x_weight = self.model_parameter.loss_weight['x']
        y_weight = self.model_parameter.loss_weight['y']
        heading_weight = self.model_parameter.loss_weight['heading']
        vel_x_weight = self.model_parameter.loss_weight['vel_x']
        vel_y_weight = self.model_parameter.loss_weight['vel_y']

        # ego loss
        ego_diff = self._norm(ego_output, ego_label)
        ego_x_loss = x_weight * torch.mean(ego_diff[:, :, x_index])
        ego_y_loss = y_weight * torch.mean(ego_diff[:, :, y_index])
        ego_heading_loss = heading_weight * torch.mean(ego_diff[:, :, heading_index])
        ego_vel_x_loss = vel_x_weight * torch.mean(ego_diff[:, :, vel_x_index])
        ego_vel_y_loss = vel_y_weight * torch.mean(ego_diff[:, :, vel_y_index])

        ego_loss = ego_x_loss + ego_y_loss + ego_heading_loss + ego_vel_x_loss + ego_vel_y_loss

        # agent loss
        agent_diff = self._norm(agent_output, agent_label)
        agent_x_loss = x_weight * torch.mean(torch.mean(agent_diff[:, :, x_index], dim=-1) * agent_valid)
        agent_y_loss = y_weight * torch.mean(torch.mean(agent_diff[:, :, y_index], dim=-1) * agent_valid)
        agent_heading_loss = heading_weight * torch.mean(torch.mean(agent_diff[:, :, heading_index], dim=-1) * agent_valid)
        agent_vel_x_loss = vel_x_weight * torch.mean(torch.mean(agent_diff[:, :, vel_x_index], dim=-1) * agent_valid)
        agent_vel_y_loss = vel_y_weight * torch.mean(torch.mean(agent_diff[:, :, vel_y_index], dim=-1) * agent_valid)

        agent_loss = agent_x_loss + agent_y_loss + agent_heading_loss + agent_vel_x_loss + agent_vel_y_loss

        num_valid_agent = torch.sum(agent_valid)/(agent_valid.size(0)*ego_output.size(1))
        agent_num = agent_output.size(1)/ego_output.size(1)
        
        # total loss
        # loss = ego_loss + num_valid_agent * agent_loss
        loss = ego_loss + agent_num * agent_loss

        # update info
        tb_dict = {}
        disp_dict = {}
        tb_dict['loss'] = loss.item()
        disp_dict['loss'] = loss.item()

        tb_dict['ego_x_loss'] = ego_x_loss.item()
        tb_dict['ego_y_loss'] = ego_y_loss.item()
        tb_dict['ego_heading_loss'] = ego_heading_loss.item()
        tb_dict['ego_vel_x_loss'] = ego_vel_x_loss.item()
        tb_dict['ego_vel_y_loss'] = ego_vel_y_loss.item()
        tb_dict['ego_loss'] = ego_loss.item()
        tb_dict['agent_x_loss'] = agent_x_loss.item()
        tb_dict['agent_y_loss'] = agent_y_loss.item()
        tb_dict['agent_heading_loss'] = agent_heading_loss.item()
        tb_dict['agent_vel_x_loss'] = agent_vel_x_loss.item()
        tb_dict['agent_vel_y_loss'] = agent_vel_y_loss.item()
        tb_dict['agent_loss'] = agent_loss.item()
        tb_dict['num_valid_agent'] = num_valid_agent.item()

        return loss, tb_dict, disp_dict

    def normalize_input(self, input_dict):
        # warning: this function will modify input_dict!
        xy_norm_para = self.model_parameter.normalization_para['xy']
        vel_norm_para = self.model_parameter.normalization_para['vel']

        # map feature
        map_keys = ['lane_feature', 'road_line_feature', 'road_edge_feature', 'map_others_feature']
        for key in map_keys:
            map_feature_tensor = input_dict[key]
            map_feature_tensor[..., [0, 1, 4, 5]] = map_feature_tensor[..., [0, 1, 4, 5]] / xy_norm_para
            input_dict[key] = map_feature_tensor

        # ego feature
        xy_index = list(range(0, input_dict['ego_feature'].size(-1)-5, 7)) + list(range(1, input_dict['ego_feature'].size(-1)-5, 7)) + [-5, -4]
        vel_index = list(range(5, input_dict['ego_feature'].size(-1)-5, 7)) + list(range(6, input_dict['ego_feature'].size(-1)-5, 7))

        feature_tensor = input_dict['ego_feature']
        feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
        feature_tensor[..., vel_index] = feature_tensor[..., vel_index] / vel_norm_para
        input_dict['ego_feature'] = feature_tensor

        # agent feature
        xy_index = list(range(0, input_dict['agent_feature'].size(-1)-5, 14)) + list(range(1, input_dict['agent_feature'].size(-1)-5, 14)) + \
                   list(range(7, input_dict['agent_feature'].size(-1)-5, 14)) + list(range(8, input_dict['agent_feature'].size(-1)-5, 14)) + [-5, -4]
        vel_index = list(range(5, input_dict['agent_feature'].size(-1)-5, 14)) + list(range(6, input_dict['agent_feature'].size(-1)-5, 14)) + \
                    list(range(12, input_dict['agent_feature'].size(-1)-5, 14)) + list(range(13, input_dict['agent_feature'].size(-1)-5, 14))

        feature_tensor = input_dict['agent_feature']
        feature_tensor[..., xy_index] = feature_tensor[..., xy_index] / xy_norm_para
        feature_tensor[..., vel_index] = feature_tensor[..., vel_index] / vel_norm_para
        input_dict['agent_feature'] = feature_tensor

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
        for key in ['lane_feature', 'lane_mask', 'road_line_feature', 'road_line_mask', 'road_edge_feature', 'road_edge_mask', 
                    'map_others_feature', 'map_others_mask', 'ego_feature', 'agent_feature', 'agent_valid']:
            input_dict[key] = input_dict[key][:, 0, ...].unsqueeze(1)

        normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
        normed_ego_output, normed_agent_output = self.model_forward(normed_input_dict)
        ego_output, agent_output = self.denormalize_output(normed_ego_output, normed_agent_output)

        # generate next input based on output
        time_set_num = self.data_config.time_info.time_set_num
        for i in range(time_set_num-1):
            input_dict = self.extend_input_based_on_last_output(input_dict, ego_output, agent_output)
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            normed_ego_output, normed_agent_output = self.model_forward(normed_input_dict)
            ego_output, agent_output = self.denormalize_output(normed_ego_output, normed_agent_output)
        
        # get trajs in agent own frame
        ego_traj, agent_traj = self.build_traj_from_output(ego_output, agent_output, input_dict['agent_to_predict_num'])

        print_index = list(range(0, 80, 5))
        # print('=================================================================================')
        # for i in range(ego_traj.size(0)):
        #     print('--------------------------------------')
        #     print('agent_traj x: ', agent_traj[i, 0, print_index, 0])
        #     print('agent_traj y: ', agent_traj[i, 0, print_index, 1])
        #     print('agent_traj heading: ', agent_traj[i, 0, print_index, 2])
        #     print('agent_traj vx: ', agent_traj[i, 0, print_index, 3])
        #     print('agent_label x: ', input_dict['agent_label'][i, :, 0, 0])
        #     print('agent_label y: ', input_dict['agent_label'][i, :, 0, 1])
        #     print('agent_label heading: ', input_dict['agent_label'][i, :, 0, 2])
        #     print('agent_label vx: ', input_dict['agent_label'][i, :, 0, 3])

        # print('=================================================================================')
        # print('=================================================================================')
        # print('=================================================================================')

        # transform agent trajs to ego frame
        agent_traj = self.transform_agent_traj_to_origin_frame(input_dict['agent_feature'][:, 0, :, 0:3], agent_traj, input_dict['agent_to_predict_num'])

        # transform agent
        ego_current_frame = input_dict['ego_current_pose'][:, 0, :].unsqueeze(1).repeat(1, agent_traj.size(1), 1)

        # print('=================================================================================')
        # for i in range(ego_traj.size(0)):
        #     print('--------------------------------------')
        #     print('agent_traj x: ', agent_traj[i, 0, print_index, 0])

        # print('=================================================================================')
        # print('=================================================================================')
        # print('=================================================================================')
        agent_global_traj = self.transform_agent_traj_to_origin_frame(ego_current_frame, agent_traj, input_dict['agent_to_predict_num'])

        # for i in range(ego_traj.size(0)):
        #     print('--------------------------------------')
        #     print('ego_traj x: ', ego_traj[i, 0, print_index, 0])
        #     print('ego_traj y: ', ego_traj[i, 0, print_index, 1])
        #     print('ego_traj vx: ', ego_traj[i, 0, print_index, 3])
        #     print('ego_label x: ', input_dict['ego_label'][i, :, 0, 0])
        #     print('ego_label y: ', input_dict['ego_label'][i, :, 0, 1])
        #     print('ego_label vx: ', input_dict['ego_label'][i, :, 0, 3])

        # print('=================================================================================')
        # for i in range(ego_traj.size(0)):
        #     print('--------------------------------------')
        #     print('agent_traj x: ', agent_traj[i, 0, print_index, 0])
            # print('agent_traj y: ', agent_traj[i, 0, print_index, 1])
            # print('agent_traj heading: ', agent_traj[i, 0, print_index, 2])
            # print('agent_traj vx: ', agent_traj[i, 0, print_index, 3])
            # print('agent_label x: ', input_dict['agent_label'][i, :, 0, 0])
            # print('agent_label y: ', input_dict['agent_label'][i, :, 0, 1])
            # print('agent_label vx: ', input_dict['agent_label'][i, :, 0, 3])
            # print('ego_current_pos: ', input_dict['ego_current_pose'][i, 0, :])
            # print('traj: ', input_dict['center_gt_trajs_src'][i, 0, 0, :, 0])

        # assert(0)

        return agent_global_traj


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
        # last_ego_output: [batch, 1, 5*time_sample_num]
        # last_ego_feature: [batch, 1, 1, 7*time_sample_num+5]
        batch_size = last_ego_output.size(0)
        time_sample_num = self.data_config.time_info.time_sample_num

        next_ego_feature = last_ego_feature.clone().detach()
        for i in range(batch_size):
            ego_frame = last_ego_output[i, 0, 0:3]
            rotation_frame = torch.tensor([0, 0, ego_frame[2]], device=last_ego_output.device)

            reshaped_ego_output = last_ego_output[i, 0, :].view(time_sample_num, 5)
            reshaped_next_ego_feature = next_ego_feature[i, 0, 0, 0:7*time_sample_num].view(time_sample_num, 7)
            reshaped_next_ego_feature[:, 0:3] = global_state_se2_tensor_to_local(reshaped_ego_output[:, 0:3], ego_frame)
            reshaped_next_ego_feature[:, 3] = torch.cos(reshaped_next_ego_feature[:, 2])
            reshaped_next_ego_feature[:, 4] = torch.sin(reshaped_next_ego_feature[:, 2])
            reshaped_next_ego_feature[:, 5:7] = coordinates_to_local_frame(reshaped_ego_output[:, 3:5], rotation_frame)

        return next_ego_feature

    def transform_agent_feature_from_output(self, last_agent_output: torch.Tensor, last_agent_feature: torch.Tensor, last_ego_output: torch.Tensor):
        # last_agent_output: [batch, 1*agent_num, 5*time_sample_num]
        # last_agent_feature: [batch, 1, agent_num, 14*time_sample_num]
        batch_size = last_ego_output.size(0)
        agent_num = last_agent_feature.size(2)
        time_sample_num = self.data_config.time_info.time_sample_num

        next_agent_feature = last_agent_feature.clone().detach()
        for i in range(batch_size):
            ego_frame = last_ego_output[i, 0, 0:3]
            rotation_frame = torch.tensor([0, 0, ego_frame[2]], device=last_ego_output.device)

            reshaped_agent_output = last_agent_output[i, :, :].view(agent_num, time_sample_num, 5)
            reshaped_agent_feature = next_agent_feature[i, 0, :, 0:14*time_sample_num].view(agent_num, time_sample_num, 14)

            for j in range(agent_num):
                agent_frame = last_agent_feature[i, 0, j, 0:3]
                agent_rotation_frame = torch.tensor([0, 0, agent_frame[2]], device=last_agent_output.device)

                agent_next_frame = last_agent_output[i, j, 0:3]
                agent_next_rotation_frame = torch.tensor([0, 0, agent_next_frame[2]], device=last_agent_output.device)

                reshaped_agent_feature[j, :, 7:10] = global_state_se2_tensor_to_local(reshaped_agent_output[j, :, 0:3], agent_next_frame)
                reshaped_agent_feature[j, :, 10] = torch.cos(reshaped_agent_feature[j, :, 9])
                reshaped_agent_feature[j, :, 11] = torch.sin(reshaped_agent_feature[j, :, 9])
                reshaped_agent_feature[j, :, 12:14] = coordinates_to_local_frame(reshaped_agent_output[j, :, 3:5], agent_next_rotation_frame)

                last_agent_in_last_ego_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=last_agent_output.device), agent_frame)
                pose_in_last_ego_frame = global_state_se2_tensor_to_local(reshaped_agent_output[j, :, 0:3], last_agent_in_last_ego_frame[0, :])
                reshaped_agent_feature[j, :, 0:3] = global_state_se2_tensor_to_local(pose_in_last_ego_frame, ego_frame)
                reshaped_agent_feature[j, :, 3] = torch.cos(reshaped_agent_feature[j, :, 2])
                reshaped_agent_feature[j, :, 4] = torch.sin(reshaped_agent_feature[j, :, 2])

                vel_in_last_ego_frame = coordinates_to_local_frame(reshaped_agent_output[j, :, 3:5], -agent_rotation_frame)
                reshaped_agent_feature[j, :, 5:7] = coordinates_to_local_frame(vel_in_last_ego_frame, rotation_frame)

        return next_agent_feature

    def build_traj_from_output(self, ego_output, agent_output, agent_to_predict_num):
        # input: 
        # ego_output, ego_label: [batch, 8, 5*time_sample_num]
        # agent_output, agent_label: [batch, 8*max_agent_num, 5*time_sample_num]
        # agent_to_predict_num: [batch, 1]
        # output: 
        # ego_traj: [batch, 1, 80, 5],
        # agent_traj: [batch, max_agent_num, 80, 5],
        # 5 is x, y, heading, vel_x, vel_y

        time_set_num = self.data_config.time_info.time_set_num
        total_time_step = self.data_config.time_info.total_time_step
        time_sample_num = self.data_config.time_info.time_sample_num
        time_set_interval = self.data_config.time_info.time_set_interval
        max_agent_num = self.data_config.max_agent_num
        batch = ego_output.size(0)

        ego_traj = torch.zeros(batch, 1, total_time_step, 5, device=ego_output.device)
        agent_traj = torch.zeros(batch, max_agent_num, total_time_step, 5, device=ego_output.device)

        agent_output_reshaped = agent_output.view(batch, 8, max_agent_num, time_sample_num, 5)

        for i in range(batch):
            ego_frame = torch.zeros(3, device=ego_output.device)
            agent_frame = torch.zeros(int(agent_to_predict_num[i, 0].item()), 3, device=ego_output.device)

            for j in range(time_set_num):
                # generate ego traj
                ego_traj[i, 0, j*time_set_interval:(j+1)*time_set_interval, :] = self.output_to_traj_in_one_time_set(
                    base_frame=ego_frame,
                    output=ego_output[i, j, :].view(time_sample_num, 5)
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
        # output: [time_sample_num, 5]
        # output:
        # traj: [10, 5]

        # interpolate output. Notice that, the output index is reversed
        time_sample_num = self.data_config.time_info.time_sample_num
        time_sample_interval = self.data_config.time_info.time_sample_interval
        if time_sample_num == 5 and time_sample_interval == 2:
            interpolated_output = torch.zeros(10, 5, device=output.device)

            interpolated_output[[1, 3, 5, 7, 9], :] = output[[4, 3, 2, 1, 0], :]
            interpolated_output[[2, 4, 6, 8], :] = 0.5*(output[[4, 3, 2, 1], :] + output[[3, 2, 1, 0], :])
            interpolated_output[0, 0:3] = 0.5*output[-1, 0:3]
            interpolated_output[0, 3:5] = output[-1, 3:5] # this is not accurate, but no effect on metric
        else:
            assert 0

        # transform output to origin frame
        origin_in_base_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=output.device), base_frame)
        interpolated_output[:, 0:3] = global_state_se2_tensor_to_local(interpolated_output[:, 0:3], origin_in_base_frame[0, :])
        interpolated_output[:, 3:5] = coordinates_to_local_frame(interpolated_output[:, 3:5], torch.tensor([0, 0, origin_in_base_frame[0, 2]], device=base_frame.device))

        return interpolated_output

    def transform_agent_traj_to_origin_frame(self, current_frame, agent_traj, agent_to_predict_num):
        # input:
        # current_frame: [batch, max_agent_num, 3]
        # agent_traj: [batch, max_agent_num, 80, 5]
        # agent_to_predict_num: [batch, 1]
        # output:
        # agent_traj: [batch, max_agent_num, 80, 5]
        batch = agent_traj.size(0)
        for i in range(batch):
            for j in range(int(agent_to_predict_num[i, 0].item())):
                origin_in_current_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=agent_traj.device), current_frame[i, j, :])

                agent_traj[i, j, :, 0:3] = global_state_se2_tensor_to_local(agent_traj[i, j, :, 0:3], origin_in_current_frame[0, :])
                agent_traj[i, j, :, 3:5] = coordinates_to_local_frame(agent_traj[i, j, :, 3:5], torch.tensor([0, 0, origin_in_current_frame[0, 2]], device=current_frame.device))

        return agent_traj


    def vel_constant_test(self, input_dict):
        # ego_output, ego_label: [batch, 8, 5*time_sample_num]
        # agent_output, agent_label: [batch, 8*max_agent_num, 5*time_sample_num]
        ego_feature = input_dict['ego_feature']
        agent_feature = input_dict['agent_feature']

        ego_result = torch.zeros(ego_feature.size(0), ego_feature.size(1), ego_feature.size(2), 5*5, device=ego_feature.device)
        agent_result = torch.zeros(agent_feature.size(0), agent_feature.size(1), agent_feature.size(2), 5*5, device=ego_feature.device)

        time_intervel = torch.tensor([[1.0, 0.8, 0.6, 0.4, 0.2]], device=ego_feature.device)
        current_ego_vel_x = ego_feature[..., 5]
        current_agent_vel_x = agent_feature[..., 12]
        ego_rel_x = current_ego_vel_x.unsqueeze(-1).matmul(time_intervel)
        agent_rel_x = current_agent_vel_x.unsqueeze(-1).matmul(time_intervel)

        # vel_x is current vel_x, x is x, others are zeros
        x_index = list(range(0, 25, 5))
        vel_x_index = list(range(3, 25, 5))
        ego_result[..., x_index] = ego_rel_x
        ego_result[..., vel_x_index] = current_ego_vel_x.unsqueeze(-1).repeat(1, 1, 1, 5)
        agent_result[..., x_index] = agent_rel_x
        agent_result[..., vel_x_index] = current_agent_vel_x.unsqueeze(-1).repeat(1, 1, 1, 5)

        return ego_result.view(ego_feature.size(0), ego_feature.size(1)*ego_feature.size(2), 5*5), agent_result.view(agent_feature.size(0), agent_feature.size(1)*agent_feature.size(2), 5*5)
    