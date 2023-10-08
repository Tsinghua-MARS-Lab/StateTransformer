import torch
from torch import nn
import copy
from typing import Dict, List, Tuple, cast
import os
import math

import models.base_model as base_model
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame
from mtr_trainer.mtr_utils.loss_utils import nll_loss_gmm_direct

class WayformerGMMModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_parameter = config.MODEL.model_parameter
        self.data_dim = config.MODEL.data_dim
        self.data_config = config.DATA_CONFIG

        # map input embedding
        self.lane_embedding = base_model.MLP(self.data_dim.lane, self.model_parameter.encoder_embedding_dim)
        self.road_line_embedding = base_model.MLP(self.data_dim.road_line, self.model_parameter.encoder_embedding_dim)
        self.road_edge_embedding = base_model.MLP(self.data_dim.road_edge, self.model_parameter.encoder_embedding_dim)
        self.map_others_embedding = base_model.MLP(self.data_dim.map_others, self.model_parameter.encoder_embedding_dim)

        # target/agent seq input embedding
        self.target_embedding = base_model.MLP(self.data_dim.target, self.model_parameter.encoder_embedding_dim)
        self.agent_embedding = base_model.MLP(self.data_dim.agent, self.model_parameter.encoder_embedding_dim)

        # pos encoder
        self.pos_encoder = base_model.PositionalEncoding(self.model_parameter.encoder_embedding_dim)

        # encoder transformer
        self.init_latent_vector = nn.Parameter(torch.zeros(self.model_parameter.latent_seq_length, self.model_parameter.encoder_embedding_dim))
        encoder_self_attn = base_model.SelfAttention(self.model_parameter.encoder_embedding_dim, self.model_parameter.encoder_head, dropout=self.model_parameter.dropout_rate)
        encoder_cross_attn = base_model.CrossAttention(self.model_parameter.encoder_embedding_dim, self.model_parameter.encoder_head, dropout=self.model_parameter.dropout_rate)
        encoder_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.encoder_embedding_dim, self.model_parameter.encoder_inter_dim, dropout=self.model_parameter.dropout_rate)
        encoder_latent_query_layer = base_model.TransformerDecoderLayer(self.model_parameter.encoder_embedding_dim, encoder_self_attn, encoder_cross_attn, encoder_p_ff, dropout=self.model_parameter.dropout_rate)
        self.encoder_layerlist = nn.ModuleList([copy.deepcopy(encoder_latent_query_layer) for _ in range(self.model_parameter.encoder_layer)])

        # decoder transformer
        self.encoder_decoder_embed = base_model.MLP(self.model_parameter.encoder_embedding_dim, self.model_parameter.decoder_embedding_dim)
        self.anchor_embed = base_model.MLP(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_embedding_dim)

        decoder_self_attn = base_model.SelfAttention(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_head, dropout=self.model_parameter.dropout_rate)
        decoder_cross_attn = base_model.CrossAttention(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_head, dropout=self.model_parameter.dropout_rate)
        decoder_p_ff = base_model.PositionwiseFeedForward(self.model_parameter.decoder_embedding_dim, self.model_parameter.decoder_inter_dim, dropout=self.model_parameter.dropout_rate)
        decoder_layer = base_model.TransformerDecoderLayer(self.model_parameter.decoder_embedding_dim, decoder_self_attn, decoder_cross_attn, decoder_p_ff, dropout=self.model_parameter.dropout_rate)
        self.decoder_layerlist = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.model_parameter.decoder_layer)])
        
        # head
        self.prob_head = nn.Linear(self.model_parameter.decoder_embedding_dim, 1)
        self.traj_head = nn.Linear(self.model_parameter.decoder_embedding_dim, self.data_dim.label*self.data_config.time_info.target_time_step)

        # anchor
        self.anchor_seeds = nn.Parameter(torch.Tensor(1, self.model_parameter.anchor_num, self.model_parameter.decoder_embedding_dim))

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
            output_prob, output_traj = self.model_forward(input_dict)

            loss, tb_dict, disp_dict = self.get_loss(output_prob, output_traj, input_dict['target_label'], self.anchor_seeds)
            return loss, tb_dict, disp_dict
        else:
            output_prob, output_traj = self.model_forward(input_dict)

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

        # map embedding
        lane_data_embeded = self.lane_embedding(input_dict['lane_feature'])
        road_line_data_embeded = self.road_line_embedding(input_dict['road_line_feature'])
        road_edge_data_embeded = self.road_edge_embedding(input_dict['road_edge_feature'])
        map_others_data_embeded = self.map_others_embedding(input_dict['map_others_feature'])

        # target, agent embedding
        target_data_embeded = self.target_embedding(input_dict['target_feature'])
        agent_data_embeded = self.agent_embedding(input_dict['agent_feature'])

        # construct input seq and input mask
        encoder_tensor, encoder_mask = self.construct_input_seq(lane_data_embeded, input_dict['lane_valid'], road_line_data_embeded, input_dict['road_line_valid'], 
                                                        road_edge_data_embeded, input_dict['road_edge_valid'], map_others_data_embeded, input_dict['map_others_valid'],
                                                        target_data_embeded, agent_data_embeded, input_dict['agent_valid'])
        encoder_tensor = self.pos_encoder(encoder_tensor)

        
        # encoder
        latent_vector = self.init_latent_vector.repeat(batch_size, 1, 1)
        for layer in self.encoder_layerlist:
            latent_vector = layer(latent_vector, encoder_tensor, encoder_mask)

        # decoder
        latent_vector = self.encoder_decoder_embed(latent_vector)
        anchor_embed = self.anchor_embed(self.anchor_seeds)
        anchor_embed = anchor_embed.repeat([batch_size, 1, 1])

        for layer in self.decoder_layerlist:
            anchor_embed = layer(anchor_embed, latent_vector)

        # get prob, traj and max prob traj
        output_prob = self.prob_head(anchor_embed).squeeze(2)

        output_traj = self.traj_head(anchor_embed)

        target_time_step = self.data_config.time_info.target_time_step
        output_traj_reshaped = output_traj.view(output_traj.size(0), output_traj.size(1), target_time_step, self.data_dim.label)

        return output_prob, output_traj_reshaped

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
        # output_traj: [batch, anchor_num, 16, 5]
        # label_traj: [batch, 1, 16, 3]
        # anchor_traj: [1, anchor_num, 16*3]
        target_time_step = self.data_config.time_info.target_time_step
        
        # find closest traj index based on anchor or output
        base_traj = output_traj

        # get diff
        label_traj_reshaped = label_traj.repeat(1, output_traj.size(1), 1, 1)
        base_diff = self._norm(label_traj_reshaped[..., 0:2], base_traj[..., 0:2])
        base_x_diff = torch.mean(base_diff[..., 0], dim=2)
        base_y_diff = torch.mean(base_diff[..., 1], dim=2)
        base_weighted_diff = self.model_parameter.loss_weight['x'] * base_x_diff + \
                        self.model_parameter.loss_weight['y'] * base_y_diff
        
        # get index and corresponding traj
        _, min_traj_index = torch.min(base_weighted_diff, dim=1)
        min_traj_list = [output_traj[i, min_traj_index[i], :, :] for i in range(output_traj.size(0))]
        min_traj = torch.stack(min_traj_list).unsqueeze(1)

        # get regression loss
        gmm_loss, _ = nll_loss_gmm_direct(pred_scores=output_prob,
                                          pred_trajs=output_traj,
                                          gt_trajs=label_traj[:, 0, :, 0:2], 
                                          gt_valid_mask=torch.ones(label_traj.size(0), label_traj.size(2), device=label_traj.device),
                                          pre_nearest_mode_idxs=min_traj_index)

        # get classification loss
        classification_loss = self._fn_cross_entropy(output_prob, min_traj_index)

        # get loss
        loss = self.model_parameter.regression_weight * gmm_loss.mean() + self.model_parameter.classification_weight * classification_loss

        # update info
        tb_dict = {}
        disp_dict = {}
        tb_dict['loss'] = loss.item()
        disp_dict['loss'] = loss.item()

        tb_dict['regression_loss'] = gmm_loss.mean().item()
        tb_dict['classification_loss'] = classification_loss.item()

        return loss, tb_dict, disp_dict


    def transformer_traj_for_eval(self, output_traj, target_current_pose):
        # output_traj: [batch, anchor_num, 16, 5]
        # target_current_pose: [batch, 3]
        # transformed_traj: [batch, anchor_num, 80, 3]

        # transform to origin
        output_traj_reshape = output_traj.view(output_traj.size(0), output_traj.size(1)*output_traj.size(2), output_traj.size(3))

        for i in range(output_traj.size(0)):
            current_frame = target_current_pose[i, :]
            origin_in_current_frame = global_state_se2_tensor_to_local(torch.zeros(1, 3, device=output_traj.device), current_frame)
            output_traj_reshape[i, :, 0:2] = coordinates_to_local_frame(output_traj_reshape[i, :, 0:2], origin_in_current_frame[0, :])
        output_traj_reshape = output_traj_reshape.view(output_traj.size(0), output_traj.size(1), output_traj.size(2), output_traj.size(3))

        # transform to 80 point
        transformed_traj = output_traj_reshape.repeat_interleave(5, dim=2)

        return transformed_traj
