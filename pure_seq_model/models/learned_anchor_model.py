import torch
from torch import nn
import copy
from typing import Dict, List, Tuple, cast
import os
import math

import models.base_model as base_model
from models.mtr_models.polyline_encoder import PointNetPolylineEncoder
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame

class LearnedAnchorModel(nn.Module):
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

        # ego/agent seq input embedding
        self.target_embedding = base_model.MLP(self.data_dim.target, self.model_parameter.encoder_embedding_dim)
        self.agent_embedding = base_model.MLP(self.data_dim.agent, self.model_parameter.encoder_embedding_dim)

        # pos encoder
        self.pos_encoder = base_model.PositionalEncoding(self.model_parameter.encoder_embedding_dim)

        # encoder transformer
        encoder_self_attn = base_model.SelfAttention(self.model_parameter.encoder_embedding_dim, self.model_parameter.encoder_head)
        encoder_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.encoder_embedding_dim, self.model_parameter.encoder_inter_dim)
        encoder_layer = base_model.TransformerEncoderLayer(self.model_parameter.encoder_embedding_dim, encoder_self_attn, encoder_p_ff)
        self.encoder_layerlist = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.model_parameter.encoder_layer)])

        # decoder transformer
        self.encoder_decoder_embed = base_model.MLP(self.model_parameter.encoder_embedding_dim, self.model_parameter.decoder_embedding_dim)
        self.anchor_embed = base_model.MLP(self.model_parameter.anchor_dim*self.data_config.time_info.target_time_step, self.model_parameter.decoder_embedding_dim)

        decoder_self_attn = base_model.SelfAttention(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_head)
        decoder_cross_attn = base_model.CrossAttention(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_head)
        decoder_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_inter_dim)
        decoder_layer = base_model.TransformerDecoderLayer(self.model_parameter.decoder_embedding_dim, decoder_self_attn, decoder_cross_attn, decoder_p_ff)
        self.decoder_layerlist = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.model_parameter.decoder_layer)])
        
        # head
        self.prob_head = nn.Linear(self.model_parameter.decoder_embedding_dim, 1)
        self.traj_head = nn.Linear(self.model_parameter.decoder_embedding_dim, self.data_dim.label*self.data_config.time_info.target_time_step)

        # anchor
        anchor_seeds, self.num_anchor, anchor_seeds_normed = self.build_anchor_tensor()
        self.anchor_seeds_normed = nn.Parameter(anchor_seeds_normed)

        # loss
        if self.model_parameter.loss_type == 'L1':
            self._norm = torch.nn.L1Loss(reduction='none')
        elif self.model_parameter.loss_type == 'L2':
            self._norm = torch.nn.MSELoss(reduction='none')
        else:
            assert(0)
        self._fn_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, input_dict: Dict):
        if self.training:
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            output_prob, output_traj = self.model_forward(normed_input_dict)
            output_traj = self.denormalize_output(output_traj)
            anchor_seeds = self.denormalize_anchor_seeds(self.anchor_seeds_normed)

            loss, tb_dict, disp_dict = self.get_loss(output_prob, output_traj, input_dict['target_label'], anchor_seeds)
            return loss, tb_dict, disp_dict
        else:
            normed_input_dict = self.normalize_input(copy.deepcopy(input_dict))
            output_prob, output_traj = self.model_forward(normed_input_dict)
            output_traj = self.denormalize_output(output_traj)

            _, sorted_prob_index = torch.sort(output_prob, descending=True, dim=1)
            # print("top 1 traj: ", output_traj[0, sorted_prob_index[0, 0], :, 0])
            # print("label: ", input_dict['target_label'][0, 0, :, 0])

            transformed_output_traj = self.transformer_traj_for_eval(output_traj, input_dict['target_current_pose'])

            # sort depends on prob
            traj_list = [transformed_output_traj[i, sorted_prob_index[i, 0:6], :, :] for i in range(output_prob.size(0))]
            prob_list = [output_prob[i, sorted_prob_index[i, 0:6]] for i in range(output_prob.size(0))]

            input_dict['traj_list'] = torch.stack(traj_list)
            input_dict['log_prob_list'] = torch.softmax(torch.stack(prob_list), dim=1)
            
            # print("transformed_output_traj: ", traj_list[0][0, :, 0])
            # print("gt: ", input_dict['center_gt_trajs_src'][0, 11:, 0])
            # print('======================================================')

            return input_dict

    def model_forward(self, input_dict: Dict):
        batch_size = input_dict['target_feature'].size(0)

        # map encoder
        lane_encoded_feature, lane_valid = self.encode_map_feature(input_dict['lane_feature'], input_dict['lane_mask'],
                                                       self.lane_in_polyline_encoder)
        road_line_encoded_feature, road_line_valid = self.encode_map_feature(input_dict['road_line_feature'], input_dict['road_line_mask'],
                                                       self.road_line_in_polyline_encoder)
        road_edge_encoded_feature, road_edge_valid = self.encode_map_feature(input_dict['road_edge_feature'], input_dict['road_edge_mask'],
                                                       self.road_edge_in_polyline_encoder)
        map_others_encoded_feature, map_others_valid = self.encode_map_feature(input_dict['map_others_feature'], input_dict['map_others_mask'],
                                                       self.map_others_in_polyline_encoder)

        # target, agent embedding
        target_data_embeded = self.target_embedding(input_dict['target_feature'])
        agent_data_embeded = self.agent_embedding(input_dict['agent_feature'])

        # construct input seq and input mask
        encoder_tensor, encoder_mask = self.construct_input_seq(lane_encoded_feature, lane_valid, road_line_encoded_feature, road_line_valid, 
                                                        road_edge_encoded_feature, road_edge_valid, map_others_encoded_feature, map_others_valid,
                                                        target_data_embeded, agent_data_embeded, input_dict['agent_valid'])
        encoder_tensor = self.pos_encoder(encoder_tensor)

        # encoder
        for layer in self.encoder_layerlist:
            encoder_tensor = layer(encoder_tensor, key_padding_mask = encoder_mask)

        # decoder
        encoder_tensor = self.encoder_decoder_embed(encoder_tensor)
        anchor_embed = self.anchor_embed(self.anchor_seeds_normed)
        anchor_embed = anchor_embed.repeat([batch_size, 1, 1])

        for layer in self.decoder_layerlist:
            anchor_embed = layer(anchor_embed, encoder_tensor, key_padding_mask = encoder_mask)

        # get prob, traj and max prob traj
        output_prob = self.prob_head(anchor_embed).squeeze(2)

        output_traj = self.traj_head(anchor_embed)

        target_time_step = self.data_config.time_info.target_time_step
        anchor_dim = self.model_parameter.anchor_dim
        output_traj_reshaped = output_traj.view(output_traj.size(0), output_traj.size(1), target_time_step, anchor_dim)

        return output_prob, output_traj_reshaped


    def encode_map_feature(self, map_data: torch.Tensor, map_mask: torch.Tensor, in_polyline_encoder):
        in_polyline_encoded_data = in_polyline_encoder(map_data, map_mask) 
        polyline_valid = map_mask.any(dim=-1)
        return in_polyline_encoded_data, polyline_valid


    def construct_input_seq(self, lane, lane_valid, road_line, road_line_valid, 
                            road_edge, road_edge_valid, map_others, map_others_valid,
                            target, agent, agent_valid):
        target_reshaped = target.view(target.size(0), target.size(1)*target.size(2), target.size(3))
        agent_reshaped = agent.view(agent.size(0), agent.size(1)*agent.size(2), agent.size(3))

        encoder_seq = torch.cat([target_reshaped, agent_reshaped, lane, road_line, road_edge, map_others], dim=1)

        target_valid = torch.ones(encoder_seq.size(0), self.data_config.time_info.feature_time_step, device=agent.device)
        agent_valid_extend = agent_valid.unsqueeze(2).repeat(1, 1, self.data_config.time_info.feature_time_step).view(agent_valid.size(0), -1)
        encoder_valid = torch.cat([target_valid, agent_valid_extend, lane_valid, road_line_valid, road_edge_valid, map_others_valid], dim=1)

        return encoder_seq, (encoder_valid==0)

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

    def get_loss(self, output_prob, output_traj, label_traj, anchor_traj):
        # output_prob: [batch, anchor_num]
        # output_traj: [batch, anchor_num, 16, 3]
        # label_traj: [batch, 1, 16, 3]
        # anchor_traj: [1, anchor_num, 16*3]
        target_time_step = self.data_config.time_info.target_time_step
        anchor_dim = self.model_parameter.anchor_dim
        
        anchor_traj_reshaped = anchor_traj.view(anchor_traj.size(0), anchor_traj.size(1), target_time_step, anchor_dim)

        # find closest traj index based on anchor or output
        if self.model_parameter.loss_anchor_for_choice:
            base_traj = anchor_traj_reshaped.repeat(output_traj.size(0), 1, 1, 1)
        else:
            base_traj = output_traj

        # get diff
        label_traj_reshaped = label_traj.repeat(1, output_traj.size(1), 1, 1)
        base_diff = self._norm(label_traj_reshaped, base_traj)
        base_x_diff = torch.mean(base_diff[..., 0], dim=2)
        base_y_diff = torch.mean(base_diff[..., 1], dim=2)
        base_heading_diff = torch.mean(base_diff[..., 2], dim=2)
        base_weighted_diff = self.model_parameter.loss_weight['x'] * base_x_diff + \
                        self.model_parameter.loss_weight['y'] * base_y_diff + \
                        self.model_parameter.loss_weight['heading'] * base_heading_diff
        
        # get index and corresponding traj
        _, min_traj_index = torch.min(base_weighted_diff, dim=1)
        min_traj_list = [output_traj[i, min_traj_index[i], :, :] for i in range(output_traj.size(0))]
        min_traj = torch.stack(min_traj_list).unsqueeze(1)

        # get regression loss
        min_diff = self._norm(min_traj, label_traj)
        weighted_x_diff = self.model_parameter.loss_weight['x'] * torch.mean(min_diff[..., 0])
        weighted_y_diff = self.model_parameter.loss_weight['y'] * torch.mean(min_diff[..., 1])
        weighted_heading_diff = self.model_parameter.loss_weight['heading'] * torch.mean(min_diff[..., 2])
        regression_loss = weighted_x_diff + weighted_y_diff + weighted_heading_diff

        # get classification loss
        classification_loss = self._fn_cross_entropy(output_prob, min_traj_index)

        # get loss
        loss = self.model_parameter.regression_weight * regression_loss + self.model_parameter.classification_weight * classification_loss

        # update info
        tb_dict = {}
        disp_dict = {}
        tb_dict['loss'] = loss.item()
        disp_dict['loss'] = loss.item()

        tb_dict['weighted_x_diff'] = weighted_x_diff.item()
        tb_dict['weighted_y_diff'] = weighted_y_diff.item()
        tb_dict['weighted_heading_diff'] = weighted_heading_diff.item()
        tb_dict['regression_loss'] = regression_loss.item()
        tb_dict['classification_loss'] = classification_loss.item()

        return loss, tb_dict, disp_dict

    def normalize_input(self, input_dict):
        # warning: this function will modify input_dict!
        xy_norm_para = self.model_parameter.normalization_para['xy']
        heading_norm_para = self.model_parameter.normalization_para['heading']
        vel_norm_para = self.model_parameter.normalization_para['vel']
        shape_norm_para = self.model_parameter.normalization_para['shape']

        # map feature
        map_keys = ['lane_feature', 'road_line_feature', 'road_edge_feature', 'map_others_feature']
        for key in map_keys:
            map_feature_tensor = input_dict[key]
            map_feature_tensor[..., [0, 1, 4, 5]] = map_feature_tensor[..., [0, 1, 4, 5]] / xy_norm_para
            input_dict[key] = map_feature_tensor

        # target feature
        feature_tensor = input_dict['target_feature']
        feature_tensor[..., [0, 1]] = feature_tensor[..., [0, 1]] / xy_norm_para
        feature_tensor[..., [5, 6]] = feature_tensor[..., [5, 6]] / vel_norm_para
        feature_tensor[..., [7, 8]] = feature_tensor[..., [7, 8]] / shape_norm_para
        input_dict['target_feature'] = feature_tensor

        # agent feature
        feature_tensor = input_dict['agent_feature']
        feature_tensor[..., [0, 1]] = feature_tensor[..., [0, 1]] / xy_norm_para
        feature_tensor[..., [5, 6]] = feature_tensor[..., [5, 6]] / vel_norm_para
        feature_tensor[..., [7, 8]] = feature_tensor[..., [7, 8]] / shape_norm_para
        input_dict['agent_feature'] = feature_tensor

        return input_dict

    def denormalize_output(self, output_traj):
        xy_norm_para = self.model_parameter.normalization_para['xy']
        heading_norm_para = self.model_parameter.normalization_para['heading']

        denormed_output_traj = torch.zeros(output_traj.shape, device=output_traj.device)
        denormed_output_traj[..., 0:2] = output_traj[..., 0:2] * xy_norm_para
        denormed_output_traj[..., 2] = output_traj[..., 2] * heading_norm_para
        
        return denormed_output_traj
    
    def denormalize_anchor_seeds(self, anchor_seeds_normed):
        xy_norm_para = self.model_parameter.normalization_para['xy']
        heading_norm_para = self.model_parameter.normalization_para['heading']
        
        anchor_seeds = torch.zeros(anchor_seeds_normed.shape, device=anchor_seeds_normed.device)
        x_index = list(range(0, anchor_seeds_normed.size(-1), 3))
        y_index = list(range(1, anchor_seeds_normed.size(-1), 3))
        heading_index = list(range(2, anchor_seeds_normed.size(-1), 3))
        anchor_seeds[..., x_index] = anchor_seeds_normed[..., x_index] * xy_norm_para
        anchor_seeds[..., y_index] = anchor_seeds_normed[..., y_index] * xy_norm_para
        anchor_seeds[..., heading_index] = anchor_seeds_normed[..., heading_index] * heading_norm_para

        return anchor_seeds

    def build_anchor_tensor(self)-> torch.Tensor:
        step_time = self.data_config.time_info.time_interval_time*self.data_config.time_info.target_time_interval
        step_num = self.data_config.time_info.target_time_step
        angle_list = self.model_parameter.anchor_angle_list
        speed_list = self.model_parameter.anchor_speed_list
        num_anchor = len(angle_list)*len(speed_list)+1
        
        anchor_tensor = torch.zeros(1, num_anchor, self.model_parameter.anchor_dim*step_num)
        anchor_tensor_normed = torch.zeros(1, num_anchor, self.model_parameter.anchor_dim*step_num)
        for a_index in range(len(angle_list)):
            for s_index in range(len(speed_list)):
                angle = angle_list[a_index]
                distance = speed_list[s_index]*step_time
                current_x = 0
                current_y = 0
                current_yaw = 0
                for i in range(step_num):
                    current_yaw = (i+1)*angle/step_num
                    current_x += distance*math.cos(current_yaw)
                    current_y += distance*math.sin(current_yaw)
                    anchor_tensor[0, 1+a_index*len(speed_list)+s_index, 0+self.model_parameter.anchor_dim*i] = current_x
                    anchor_tensor[0, 1+a_index*len(speed_list)+s_index, 1+self.model_parameter.anchor_dim*i] = current_y
                    anchor_tensor[0, 1+a_index*len(speed_list)+s_index, 2+self.model_parameter.anchor_dim*i] = current_yaw
                    anchor_tensor_normed[0, 1+a_index*len(speed_list)+s_index, 0+self.model_parameter.anchor_dim*i] = current_x/self.model_parameter.normalization_para['xy']
                    anchor_tensor_normed[0, 1+a_index*len(speed_list)+s_index, 1+self.model_parameter.anchor_dim*i] = current_y/self.model_parameter.normalization_para['xy']
                    anchor_tensor_normed[0, 1+a_index*len(speed_list)+s_index, 2+self.model_parameter.anchor_dim*i] = current_yaw/self.model_parameter.normalization_para['heading']
        return anchor_tensor, num_anchor, anchor_tensor_normed
    

    def transformer_traj_for_eval(self, output_traj, target_current_pose):
        # output_traj: [batch, anchor_num, 16, 3]
        # target_current_pose: [batch, 3]
        # transformed_traj: [batch, anchor_num, 80, 3]

        # transform to origin
        output_traj_reshape = output_traj.view(output_traj.size(0), output_traj.size(1)*output_traj.size(2), output_traj.size(3))

        for i in range(output_traj.size(0)):
            current_frame = target_current_pose[i, :]
            origin_in_current_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=output_traj.device), current_frame)
            output_traj_reshape[i, :, :] = global_state_se2_tensor_to_local(output_traj_reshape[i, :, :], origin_in_current_frame[0, :])
        output_traj_reshape = output_traj_reshape.view(output_traj.size(0), output_traj.size(1), output_traj.size(2), output_traj.size(3))

        # transform to 80 point
        transformed_traj = output_traj_reshape.repeat_interleave(5, dim=2)

        return transformed_traj


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