import torch
from torch import nn
import copy
from typing import Dict, List, Tuple, cast
import os

import models.base_model as base_model
from models.mtr_models.polyline_encoder import PointNetPolylineEncoder
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame

class ConstantVelModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_parameter = config.MODEL.model_parameter
        self.data_dim = config.MODEL.data_dim
        self.data_config = config.DATA_CONFIG

        # loss
        if self.model_parameter.loss_type == 'L1':
            self._norm = torch.nn.L1Loss(reduction='none')
        elif self.model_parameter.loss_type == 'L2':
            self._norm = torch.nn.MSELoss(reduction='none')
        else:
            assert(0)


    def forward(self, input_dict: Dict):
        ego_output, agent_output = self.vel_constant_test(input_dict)

        loss, tb_dict, disp_dict = self.get_loss(ego_output, input_dict['ego_label'].squeeze(2),
                                                    agent_output, input_dict['agent_label'].view(agent_output.size(0), agent_output.size(1), agent_output.size(2)),
                                                    input_dict['agent_valid'].view(agent_output.size(0), agent_output.size(1)))
        return loss, tb_dict, disp_dict


    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        return 0, 0

    def load_params_from_file(self, filename, logger, to_cpu=False):
        return 0, 0

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

        # for data filter
        disp_dict['max_ego_x'] = torch.max(ego_diff[:, :, x_index])
        disp_dict['max_ego_y'] = torch.max(ego_diff[:, :, y_index])
        disp_dict['max_ego_heading'] = torch.max(ego_diff[:, :, heading_index])
        disp_dict['max_ego_vel_x'] = torch.max(ego_diff[:, :, vel_x_index])
        disp_dict['max_ego_vel_y'] = torch.max(ego_diff[:, :, vel_y_index])

        disp_dict['max_agent_x'] = torch.max(agent_diff[:, :, x_index]*agent_valid.unsqueeze(-1))
        disp_dict['max_agent_y'] = torch.max(agent_diff[:, :, y_index]*agent_valid.unsqueeze(-1))
        disp_dict['max_agent_heading'] = torch.max(agent_diff[:, :, heading_index]*agent_valid.unsqueeze(-1))
        disp_dict['max_agent_vel_x'] = torch.max(agent_diff[:, :, vel_x_index]*agent_valid.unsqueeze(-1))
        disp_dict['max_agent_vel_y'] = torch.max(agent_diff[:, :, vel_y_index]*agent_valid.unsqueeze(-1))

        return loss, tb_dict, disp_dict


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
    