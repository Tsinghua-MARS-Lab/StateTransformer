import torch
from omegaconf import OmegaConf
from utils.logger import log

dataset_config = OmegaConf.load('configs/config_v1.yaml')

from dataset.waymo_dataset_v1 import WaymoDatasetV1

val_dataset = WaymoDatasetV1(dataset_config.DATA_CONFIG, False, log)

data_dict = val_dataset[1234]
# 1234 agent 14

for key in data_dict.keys():
    print('key: ', key, ', type: ', type(data_dict[key]))

# print('===========================ego_feature_list===============================')

# print("ego_feature_list length:, ", len(data_dict['ego_feature_list'][0]))
# print("ego_feature shape:, ", data_dict['ego_feature_list'][0][0].shape)
# print("ego_feature:, ", data_dict['ego_feature_list'][0][0])

print('===========================ego_label===============================')

print("ego_label shape:, ", data_dict['ego_label'].shape)
print("ego_label:, ", data_dict['ego_label'][0, 0, :])

print("agent_label shape:, ", data_dict['agent_label'].shape)
# print("agent_label:, ", data_dict['agent_label'][0, 0, 10, :])
print("agent_label:, ", data_dict['agent_label'][0, 0, 14, :])

# print("agent_feature:, ", data_dict['agent_feature'][0, 0, 10, :])
print("agent_feature:, ", data_dict['agent_feature'][0, 0, 14, :])
print("agent_feature:, ", data_dict['agent_feature'][0, 1, 14, :])
print("agent_feature:, ", data_dict['agent_feature'][0, 2, 14, :])
print("agent_valid:, ", data_dict['agent_valid'][0, 0, 14])

# print('===========================agent_feature_list===============================')
# print('valid_agent_to_predict: ', data_dict['agent_to_predict_num'][0])
# print("agent_feature_list length:, ", len(data_dict['agent_feature_list'][0]))
# print("agent_feature shape:, ", data_dict['agent_feature_list'][0][0].shape)
# for i in range(len(data_dict['agent_feature_list'][0])):
#     print("agent_feature_num:, ", data_dict['agent_feature_list'][0][i].shape)

# print('===========================lane_feature_mask===============================')
# print("lane_mask shape: ", data_dict['lane_mask'].shape)
# print(data_dict['lane_mask'][0, 0, 0, :])
# print(data_dict['lane_mask'][0, 0, 1, :])
# print(data_dict['lane_mask'][0, 0, 2, :])
# print(data_dict['lane_mask'][0, 0, 511, :])
# print(data_dict['lane_mask'][0, 0, 250, :])