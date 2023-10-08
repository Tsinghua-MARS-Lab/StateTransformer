import torch
import pickle

from dataset.waymo_dataset_v1_aug import WaymoDatasetV1Aug
from models.constant_vel_model import ConstantVelModel
from omegaconf import OmegaConf

from utils.logger import log

config = OmegaConf.load('configs/config_v1_aug.yaml')

dataset = WaymoDatasetV1Aug(config.DATA_CONFIG, True, log)
model = ConstantVelModel(config)
print('dataset len: ', len(dataset))


error_data_list = []
with torch.no_grad():
    sum_loss = 0
    for i in range(len(dataset)):
        loss, tb_dict, disp_dict = model(dataset[i])

        is_false = False
        if disp_dict['max_ego_x'] > 1000:
            is_false = True
            print("max_ego_x: ", disp_dict['max_ego_x'], 'index: ', i)
        if disp_dict['max_ego_y'] > 1000:
            is_false = True
            print("max_ego_y: ", disp_dict['max_ego_y'], 'index: ', i)
        if disp_dict['max_ego_heading'] > 1:
            is_false = True
            print("max_ego_heading: ", disp_dict['max_ego_heading'], 'index: ', i)
        if disp_dict['max_ego_vel_x'] > 1000:
            is_false = True
            print("max_ego_vel_x: ", disp_dict['max_ego_vel_x'], 'index: ', i)
        if disp_dict['max_ego_vel_y'] > 1000:
            is_false = True
            print("max_ego_vel_y: ", disp_dict['max_ego_vel_y'], 'index: ', i)

        if disp_dict['max_agent_x'] > 1000:
            is_false = True
            print("max_agent_x: ", disp_dict['max_agent_x'], 'index: ', i)
        if disp_dict['max_agent_y'] > 1000:
            is_false = True
            print("max_agent_y: ", disp_dict['max_agent_y'], 'index: ', i)
        if disp_dict['max_agent_heading'] > 1:
            is_false = True
            print("max_agent_heading: ", disp_dict['max_agent_heading'], 'index: ', i)
        if disp_dict['max_agent_vel_x'] > 100:
            is_false = True
            print("max_agent_vel_x: ", disp_dict['max_agent_vel_x'], 'index: ', i)
        if disp_dict['max_agent_vel_y'] > 100:
            is_false = True
            print("max_agent_vel_y: ", disp_dict['max_agent_vel_y'], 'index: ', i)

        if is_false is True:
            error_data_list.append(i)

        sum_loss += loss
        if i % 500 == 0:
            print("finish: ", i, ", avg_loss: ", sum_loss/100.0)
            sum_loss = 0

# print("save error list")
# file_path = 'error_data_list.pkl'

# with open(file_path, 'wb') as f:
#     pickle.dump(error_data_list, f)