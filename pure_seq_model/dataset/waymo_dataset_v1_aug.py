import torch
import pickle
import os
import pathlib

from typing import Dict, List, Tuple, cast
import numpy as np
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame
from utils.base_math import angle_to_range

import time


class WaymoDatasetV1Aug(torch.utils.data.Dataset):
    # A scenario contains 9s data. 
    # For the input sequence, each 1 second responds to one input set.
    # To use denser agent infos, each agent feature in one input set can contains infos in multiple time instance within this one second period

    # Map infos are grouped into several categories same as MTR: lane, road_line, road_edge, stop_sign+crosswalk+speed_bump
    # Traffic light infos are contained in lane infos

    # x, y, heading, vel infos are expressed in current ego coordinates.
    # Target x, y, heading are relative movement expressed in current ego coordinate. Target vel are expressed in current ego coordinate
    # So for auto-regressing generation, we need tranform coordinate from current to next ego cood for next input generation

    # for each time set (time_set_num is 9-1)

    # map:
    # lane: [n(512), 20, 19], feature info:[x, y, dir_x, dir_y, next_x, next_y, speed_limit, Freeway, SurfaceStreet, BikeLane,
    #        Unknown, Arrow_Stop, Arrow_Caution, Arrow_Go, Stop, Caution, Go, Flashing_Stop, Flashing_Caution]
    # road_line: [n(128), 20, 14], feature info:[x, y, dir_x, dir_y, next_x, next_y,
    #        BrokenSingleWhite, SolidSingleWhite, SolidDoubleWhite, BrokenSingleYellow, BrokenDoubleYellow, SolidSingleYellow, SolidDoubleYellow, PassingDoubleYellow]
    # road_edge: [n(128), 20, 8], feature info:[x, y, dir_x, dir_y, next_x, next_y, RoadEdgeBoundary, RoadEdgeMedian]
    # stop_sign+crosswalk+speed_bump: [n(64), 20, 9], feature info:[x, y, dir_x, dir_y, next_x, next_y, StopSign, Crosswalk, SpeedBump]
    # mask*4: [n, 20]

    # agent, agent will valid only if current step and next step are valid:
    # ego: [1, 7*time_sample_num + 5], feature info:[x, y, heading, cos(heading), sin(heading), v_x, v_y, ... , length, width, TYPE_VEHICLE, TYPE_PEDESTRAIN, TYPE_CYCLIST]
    # agent: [n, 14*time_sample_num + 5], feature info:[x_ego, y_ego, heading_ego, cos(heading_ego), sin(heading_ego), v_x_ego, v_y_ego,
    #                                                  x_self, y_self, heading_self, cos(heading_self), sin(heading_self), v_x_self, v_y_self,
    #                                                  ... ,
    #                                                  length, width, TYPE_VEHICLE, TYPE_PEDESTRAIN, TYPE_CYCLIST]
    # agent_valid: [n(64)]

    # label:
    # ego: [1, 5*time_sample_num], feature info:[delta_x, delta_y, delta_heading, v_x, v_y, ... ,]
    # agent: [n(64), 5*time_sample_num], feature info:[delta_x_self, delta_y_self, delta_heading_self, v_x_self, v_y_self, ... ,]
    # agent_to_predict_num: int

    # for the whole scenario, the output data is:
    # Dict[lane: torch.Tensor(batch_size, 8, n, 20, 18), road_line: ..., ..., agent: torch.Tensor(batch_size, 8, n, 14*time_sample_num+5), ...]

    def __init__(self, dataset_cfg: Dict, training: bool, logger=None):
        if training:
            self.mode = 'train'
        else:
            self.mode = 'val'

        self.dataset_config = dataset_cfg
        self.logger = logger

        self.data_root = self.dataset_config.dataset_info.data_root
        self.data_path = os.path.join(self.data_root, self.dataset_config.dataset_info.split_dir[self.mode])
        self.infos = self.get_all_infos(os.path.join(self.data_root, self.dataset_config.dataset_info.info_file[self.mode]))

        self.data_cache_path = os.path.join(self.data_root, self.dataset_config.dataset_info.cache_dir[self.mode])
        pathlib.Path(self.data_cache_path).mkdir(parents=True, exist_ok=True) 

        self.logger.info(f'Total scenes after filters: {len(self.infos)}')

    def get_all_infos(self, info_path: str):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_config.dataset_info.sample_interval[self.mode]]
        self.logger.info(f'Total scenes before filters: {len(infos)}')

        return infos
        
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        # use cache if exist
        cache_file_name = os.path.join(self.data_cache_path, f'cache_{index}.pkl')
        if os.path.exists(cache_file_name):
            with open(cache_file_name, 'rb') as f:
                try:
                    feature_dict = pickle.load(f)
                except:
                    feature_dict = self.generate_from_raw_data(index)
        else:
            feature_dict = self.generate_from_raw_data(index)
            with open(cache_file_name, 'wb') as f:
                pickle.dump(feature_dict, f)
        return feature_dict
    
        # feature_dict = self.generate_from_raw_data(index)
        # return feature_dict

    def generate_from_raw_data(self, index):
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(os.path.join(self.data_path, f'sample_{scene_id}.pkl'), 'rb') as f:
            info = pickle.load(f)

        # get current time index, time set num, first time index. 

        # # For standard waymo data, current_time_index=10, time_set_num=8, first_time_index=10
        # current_time_index = info['current_time_index']
        # total_time_steps = len(info['timestamps_seconds'])
        # time_set_interval = self.dataset_config.time_info.time_set_interval
        # past_time_set = current_time_index // time_set_interval
        # future_time_set = (total_time_steps - current_time_index - 1) // time_set_interval
        # time_set_num = past_time_set + future_time_set - 1
        # first_time_index = current_time_index - (past_time_set-1) * time_set_interval

        time_set_interval = self.dataset_config.time_info.time_set_interval
        time_set_num = self.dataset_config.time_info.time_set_num
        first_time_index = time_set_interval
        
        # generate time_set_num sets samples
        lane_feature_list, lane_mask_list = [], []
        road_line_feature_list, road_line_mask_list = [], []
        road_edge_feature_list, road_edge_mask_list = [], []
        map_others_feature_list, map_others_mask_list = [], []

        ego_feature_list, ego_label_list = [], []
        agent_feature_list, agent_label_list, agent_valid_list = [], [], []

        sdc_track_index = info['sdc_track_index']
        valid_agent_to_predict = self.check_valid_agent_to_predict(info['track_infos']['trajs'], sdc_track_index, info['tracks_to_predict']['track_index'])

        trajs = torch.from_numpy(info['track_infos']['trajs']).float()
        polyline_tensor = torch.from_numpy(info['map_infos']['all_polylines']).float()

        agent_index = self.get_proper_agent_index(trajs, sdc_track_index, valid_agent_to_predict)

        for i in range(time_set_num):
            # get ego, agent feature
            ego_feature, ego_label = self.get_ego_feature_and_label_in_one_time_set(trajs, sdc_track_index, first_time_index+i*time_set_interval)
            agent_feature, agent_label, agent_valid = self.get_agent_feature_and_label_in_one_time_set(trajs=trajs, 
                                                                                          object_type=info['track_infos']['object_type'], 
                                                                                          ego_index=sdc_track_index, 
                                                                                          valid_agent_to_predict=valid_agent_to_predict, 
                                                                                          time_index=first_time_index+i*time_set_interval,
                                                                                          agent_index=agent_index)
            ego_feature_list.append(ego_feature)
            ego_label_list.append(ego_label)
            agent_feature_list.append(agent_feature)
            agent_label_list.append(agent_label)
            agent_valid_list.append(agent_valid)

            # tranform map feature coodinates to ego frame
            ego_info = trajs[sdc_track_index, first_time_index+i*time_set_interval, :]
            ego_frame = torch.tensor([ego_info[0], ego_info[1], ego_info[6]])
            rotation_frame = torch.tensor([0, 0, ego_info[6]])
            transformed_polyline_tensor = torch.clone(polyline_tensor)
            transformed_polyline_tensor[:, 0:2] = coordinates_to_local_frame(transformed_polyline_tensor[:, 0:2], ego_frame) # x, y
            transformed_polyline_tensor[:, 3:5] = coordinates_to_local_frame(transformed_polyline_tensor[:, 3:5], rotation_frame) # dir_x, dir_y

            # build map features
            lane_feature, lane_mask = self.get_lane_feature(transformed_polyline_tensor, 
                                                            info['map_infos']['lane'], 
                                                            info['dynamic_map_infos']['lane_id'][first_time_index+i*time_set_interval],
                                                            info['dynamic_map_infos']['state'][first_time_index+i*time_set_interval])
            lane_feature_list.append(lane_feature)
            lane_mask_list.append(lane_mask)

            road_line_feature, road_line_mask = self.get_map_feature_except_lane(transformed_polyline_tensor, info['map_infos']['road_line'], 'road_line')
            road_line_feature_list.append(road_line_feature)
            road_line_mask_list.append(road_line_mask)

            road_edge_feature, road_edge_mask = self.get_map_feature_except_lane(transformed_polyline_tensor, info['map_infos']['road_edge'], 'road_edge')
            road_edge_feature_list.append(road_edge_feature)
            road_edge_mask_list.append(road_edge_mask)

            map_others_info = info['map_infos']['stop_sign'] + info['map_infos']['crosswalk'] + info['map_infos']['speed_bump']
            map_others_feature, map_others_mask = self.get_map_feature_except_lane(transformed_polyline_tensor, map_others_info, 'map_others')
            map_others_feature_list.append(map_others_feature)
            map_others_mask_list.append(map_others_mask)

        feature_dict = {
            'lane_feature': torch.stack(lane_feature_list).unsqueeze(0),
            'lane_mask': torch.stack(lane_mask_list).unsqueeze(0),
            'road_line_feature': torch.stack(road_line_feature_list).unsqueeze(0),
            'road_line_mask': torch.stack(road_line_mask_list).unsqueeze(0),
            'road_edge_feature': torch.stack(road_edge_feature_list).unsqueeze(0),
            'road_edge_mask': torch.stack(road_edge_mask_list).unsqueeze(0),
            'map_others_feature': torch.stack(map_others_feature_list).unsqueeze(0),
            'map_others_mask': torch.stack(map_others_mask_list).unsqueeze(0),

            'ego_feature': torch.stack(ego_feature_list).unsqueeze(0),
            'ego_label': torch.stack(ego_label_list).unsqueeze(0),
            'agent_feature': torch.stack(agent_feature_list).unsqueeze(0),
            'agent_label': torch.stack(agent_label_list).unsqueeze(0),
            'agent_valid': torch.stack(agent_valid_list).unsqueeze(0),
            'agent_to_predict_num': torch.zeros(1, 1) + len(valid_agent_to_predict)
        }
        feature_dict = self.generate_input_dict_for_eval(info, feature_dict, valid_agent_to_predict, trajs, sdc_track_index)

        return feature_dict

    def check_valid_agent_to_predict(self, trajs, ego_index: int, agent_to_predict_index: List):
        assert trajs[ego_index, :, -1].all()

        valid_agent_to_predict = []
        for index in agent_to_predict_index:
            if trajs[index, :, -1].all():
                valid_agent_to_predict.append(index)

        return valid_agent_to_predict

    def get_ego_feature_and_label_in_one_time_set(self, trajs: torch.Tensor, ego_index: int, time_index: int):
        time_sample_num = self.dataset_config.time_info.time_sample_num
        time_sample_interval = self.dataset_config.time_info.time_sample_interval
        time_set_interval = self.dataset_config.time_info.time_set_interval

        # build ego feature
        ego_feature = torch.zeros([1, 7*time_sample_num + 5])
        for i in range(time_sample_num):
            ego_feature[0, 0+i*7] = trajs[ego_index, time_index - i*time_sample_interval, 0] # x
            ego_feature[0, 1+i*7] = trajs[ego_index, time_index - i*time_sample_interval, 1] # y
            ego_feature[0, 2+i*7] = trajs[ego_index, time_index - i*time_sample_interval, 6] # heading

            ego_feature[0, 5+i*7] = trajs[ego_index, time_index - i*time_sample_interval, 7] # v_x
            ego_feature[0, 6+i*7] = trajs[ego_index, time_index - i*time_sample_interval, 8] # v_y
        ego_feature[0, 7*time_sample_num+0] = trajs[ego_index, time_index, 3] # length
        ego_feature[0, 7*time_sample_num+1] = trajs[ego_index, time_index, 4] # width
        ego_feature[0, 7*time_sample_num+2] = 1 # ego type, vehicle

        # build ego label
        ego_label = torch.zeros([1, 5*time_sample_num])
        for i in range(time_sample_num):
            ego_label[0, 0+i*5] = trajs[ego_index, time_index + time_set_interval - i*time_sample_interval, 0]-trajs[ego_index, time_index, 0]  # x
            ego_label[0, 1+i*5] = trajs[ego_index, time_index + time_set_interval - i*time_sample_interval, 1]-trajs[ego_index, time_index, 1] # y
            ego_label[0, 2+i*5] = angle_to_range(
                trajs[ego_index, time_index + time_set_interval - i*time_sample_interval, 6]-trajs[ego_index, time_index, 6]) # heading
            ego_label[0, 3+i*5] = trajs[ego_index, time_index + time_set_interval - i*time_sample_interval, 7] # v_x
            ego_label[0, 4+i*5] = trajs[ego_index, time_index + time_set_interval - i*time_sample_interval, 8] # v_y

        # transfer to ego coordinate
        ego_info = trajs[ego_index, time_index, :]
        ego_frame = torch.tensor([ego_info[0], ego_info[1], ego_info[6]])
        rotation_frame = torch.tensor([0, 0, ego_info[6]])
        for i in range(time_sample_num):
            ego_feature[:, 0+i*7:3+i*7] = global_state_se2_tensor_to_local(ego_feature[:, 0+i*7:3+i*7], ego_frame)
            ego_feature[:, 3+i*7] = torch.cos(ego_feature[:, 2+i*7]) # cos(heading)
            ego_feature[:, 4+i*7] = torch.sin(ego_feature[:, 2+i*7]) # sin(heading)
            ego_feature[:, 5+i*7:7+i*7] = coordinates_to_local_frame(ego_feature[:, 5+i*7:7+i*7], rotation_frame)

            ego_label[:, 0+i*5:2+i*5] = coordinates_to_local_frame(ego_label[:, 0+i*5:2+i*5], rotation_frame)
            ego_label[:, 3+i*5:5+i*5] = coordinates_to_local_frame(ego_label[:, 3+i*5:5+i*5], rotation_frame)
            
        return ego_feature, ego_label

    def get_proper_agent_index(self, trajs: torch.Tensor, ego_index: int, valid_agent_to_predict: List):
        # we perpare agent index by:
        # 1. ignore ego
        # 2. put valid_agent_to_predict at front
        # 3. filter out agent invalid for all times
        # 3. if agent num is less than max_agent_num, return
        # 4. if agent num is larger than max_agent_num, 
        # 4. sort agent based on the distance to ego at time 1.0(index 10)

        max_agent_num = self.dataset_config.max_agent_num
        time_set_num = self.dataset_config.time_info.time_set_num
        time_set_interval = self.dataset_config.time_info.time_set_interval

        # get agents except ego and predict agent
        other_agent_index = list(range(trajs.size(0)))
        other_agent_index = [i for i in other_agent_index if i != ego_index]
        for predict_index in valid_agent_to_predict:
            other_agent_index = [i for i in other_agent_index if i != predict_index]

        # filter out agents invalid for all times
        valid_tensor = torch.zeros(trajs.size(0), time_set_num).bool()
        for i in range(time_set_num):
            valid_tensor[:, i] = trajs[:, i*time_set_interval+1:(i+2)*time_set_interval+1, -1].all(dim=1)
        invalid_index_tensor = (valid_tensor==False).all(dim=1)
        valid_other_agent_index = [i for i in other_agent_index if invalid_index_tensor[i]==False]

        # return if agent num <= max_agent_num
        if len(valid_other_agent_index) + len(valid_agent_to_predict) <= max_agent_num:
            return valid_agent_to_predict + valid_other_agent_index

        # sort agent based on distance to ego
        dis_to_ego = torch.hypot(trajs[:, time_set_interval, 0]-trajs[ego_index, time_set_interval, 0],
                                 trajs[:, time_set_interval, 1]-trajs[ego_index, time_set_interval, 1])
        _, sorted_index = torch.sort(dis_to_ego)
        sorted_valid_other_index = []
        for i in sorted_index:
            for j in valid_other_agent_index:
                if i == j:
                    sorted_valid_other_index.append(i)
                    break

        other_agent_num = max_agent_num - len(valid_agent_to_predict)
        assert(len(sorted_valid_other_index) > other_agent_num)
        filter_other_index = sorted_valid_other_index[:other_agent_num]
        filter_other_index.sort()

        return valid_agent_to_predict + filter_other_index

    def get_agent_feature_and_label_in_one_time_set(self, trajs: torch.Tensor, object_type: List, ego_index: int, valid_agent_to_predict: List, time_index: int, agent_index: List):
        # ego state is not include in agent part
        # valid_agent_to_predict are at the beginning of all agents
        # only contains valid agents. if agent num is large than max, agents are filtered by distance to ego

        time_sample_num = self.dataset_config.time_info.time_sample_num
        time_sample_interval = self.dataset_config.time_info.time_sample_interval
        time_set_interval = self.dataset_config.time_info.time_set_interval
        max_agent_num = self.dataset_config.max_agent_num

        # move valid_agent_to_predict to the front of agnets
        agent_tensor = trajs[agent_index, :, :]
        agent_type = [object_type[i] for i in agent_index]
        
        # build agent feature tensor
        agent_feature = torch.zeros([agent_tensor.size(0), 14*time_sample_num + 5])
        for i in range(time_sample_num):
            agent_feature[:, 0+i*14] = agent_tensor[:, time_index - i*time_sample_interval, 0] # x
            agent_feature[:, 1+i*14] = agent_tensor[:, time_index - i*time_sample_interval, 1] # y
            agent_feature[:, 2+i*14] = agent_tensor[:, time_index - i*time_sample_interval, 6] # heading
            agent_feature[:, 5+i*14] = agent_tensor[:, time_index - i*time_sample_interval, 7] # v_x
            agent_feature[:, 6+i*14] = agent_tensor[:, time_index - i*time_sample_interval, 8] # v_y
        agent_feature[:, 14*time_sample_num+0] = agent_tensor[:, time_index, 3] # length
        agent_feature[:, 14*time_sample_num+1] = agent_tensor[:, time_index, 4] # width

        for i in range(agent_tensor.size(0)):
            if agent_type[i] == 'TYPE_VEHICLE':
                agent_feature[i, 14*time_sample_num+2] = 1
            elif agent_type[i] == 'TYPE_PEDESTRIAN':
                agent_feature[i, 14*time_sample_num+3] = 1
            elif agent_type[i] == 'TYPE_CYCLIST':
                agent_feature[i, 14*time_sample_num+4] = 1

        # build agent valid tensor
        agent_valid = agent_tensor[:, time_index-time_set_interval+1:time_index+time_set_interval+1, -1].all(dim=1).float()

        # build agent label tensor
        agent_label = torch.zeros([agent_tensor.size(0), 5*time_sample_num])
        for i in range(time_sample_num):
            agent_label[:, 0+i*5] = agent_tensor[:, time_index + time_set_interval - i*time_sample_interval, 0]-agent_tensor[:, time_index, 0] # x
            agent_label[:, 1+i*5] = agent_tensor[:, time_index + time_set_interval - i*time_sample_interval, 1]-agent_tensor[:, time_index, 1] # y
            agent_label[:, 2+i*5] = angle_to_range(
                agent_tensor[:, time_index + time_set_interval - i*time_sample_interval, 6]-agent_tensor[:, time_index, 6]) # heading
            agent_label[:, 3+i*5] = agent_tensor[:, time_index + time_set_interval - i*time_sample_interval, 7] # v_x
            agent_label[:, 4+i*5] = agent_tensor[:, time_index + time_set_interval - i*time_sample_interval, 8] # v_y

        # tranfer to ego and agent coordinate frame
        ego_info = trajs[ego_index, time_index, :]
        ego_frame = torch.tensor([ego_info[0], ego_info[1], ego_info[6]])
        rotation_frame = torch.tensor([0, 0, ego_info[6]])

        for i in range(time_sample_num):
            # agent feature
            agent_feature[:, 0+i*14:3+i*14] = global_state_se2_tensor_to_local(agent_feature[:, 0+i*14:3+i*14], ego_frame) # x, y, heading in ego frame
            agent_feature[:, 3+i*14] = torch.cos(agent_feature[:, 2+i*14]) # cos(heading) in ego frame
            agent_feature[:, 4+i*14] = torch.sin(agent_feature[:, 2+i*14]) # sin(heading) in ego frame
            agent_feature[:, 5+i*14:7+i*14] = coordinates_to_local_frame(agent_feature[:, 5+i*14:7+i*14], rotation_frame) # vel_x, vel_y in ego frame

            for j in range(agent_feature.size(0)):
                agent_frame = torch.tensor([agent_feature[j, 0], agent_feature[j, 1], agent_feature[j, 2]])
                agent_rotation_frame = torch.tensor([0, 0, agent_feature[j, 2]])

                agent_feature[j, 7+i*14:10+i*14] = global_state_se2_tensor_to_local(agent_feature[j, 0+i*14:3+i*14].unsqueeze(0), agent_frame).squeeze(0) # x, y, heading in agent frame
                agent_feature[j, 12+i*14:14+i*14] = coordinates_to_local_frame(agent_feature[j, 5+i*14:7+i*14].unsqueeze(0), agent_rotation_frame).squeeze(0) # vel_x, vel_y in agent frame
            agent_feature[:, 10+i*14] = torch.cos(agent_feature[:, 9+i*14]) # cos(heading) in agent frame
            agent_feature[:, 11+i*14] = torch.sin(agent_feature[:, 9+i*14]) # sin(heading) in agent frame

            # agent_label
            agent_label[:, 0+i*5:2+i*5] = coordinates_to_local_frame(agent_label[:, 0+i*5:2+i*5], rotation_frame)
            agent_label[:, 3+i*5:5+i*5] = coordinates_to_local_frame(agent_label[:, 3+i*5:5+i*5], rotation_frame)

            for j in range(agent_feature.size(0)):
                agent_rotation_frame = torch.tensor([0, 0, agent_feature[j, 2]])
                agent_label[j, 0+i*5:2+i*5] = coordinates_to_local_frame(agent_label[j, 0+i*5:2+i*5].unsqueeze(0), agent_rotation_frame).squeeze(0) # x,y in agent frame
                agent_label[j, 3+i*5:5+i*5] = coordinates_to_local_frame(agent_label[j, 3+i*5:5+i*5].unsqueeze(0), agent_rotation_frame).squeeze(0) # vel_x, vel_y in agent frame
                
        # pad to max_agent_num
        agent_feature_padded = torch.zeros((max_agent_num, agent_feature.size(1)))
        agent_label_padded = torch.zeros((max_agent_num, agent_label.size(1)))
        agent_valid_padded = torch.zeros(max_agent_num)

        agent_feature_padded[:agent_feature.size(0), :] = agent_feature
        agent_label_padded[:agent_feature.size(0), :] = agent_label
        agent_valid_padded[:agent_feature.size(0)] = agent_valid

        return agent_feature_padded, agent_label_padded, agent_valid_padded

    def get_map_feature_except_lane(self, polylines_tensor: torch.Tensor, map_feature_info: List, feature_type: str):
        polyline_point_num = self.dataset_config.map_feature.polyline_point_num
        polyline_subsample_interval = self.dataset_config.map_feature.polyline_subsample_interval

        map_feature_list = []
        map_feature_mask_list = []
        for i in range(len(map_feature_info)):
            polyline_index = map_feature_info[i]['polyline_index']
            for index in range(polyline_index[0], polyline_index[1], polyline_point_num*polyline_subsample_interval):
                one_map_feature_tensor, one_map_feature_mask = self.build_one_map_feature_tensor_except_lane(
                    polylines_tensor[index:min(index+polyline_point_num*polyline_subsample_interval, polyline_index[1]), :],
                    feature_type
                    )
                map_feature_list.append(one_map_feature_tensor)
                map_feature_mask_list.append(one_map_feature_mask)

        # reshape to max_agent_num
        max_polyline_num = self.dataset_config.map_feature.max_polyline_num[feature_type]
        map_feature_padded = torch.zeros((max_polyline_num, self.dataset_config.map_feature.polyline_point_num, self.dataset_config.map_feature.feature_dim[feature_type]))
        map_feature_mask_padded = torch.zeros((max_polyline_num, self.dataset_config.map_feature.polyline_point_num))

        if max_polyline_num >= len(map_feature_list):
            # pad zeros
            for i in range(len(map_feature_list)):
                map_feature_padded[i, :, :] = map_feature_list[i]
                map_feature_mask_padded[i, :] = map_feature_mask_list[i]
        else:
            # filtered by closest point to ego
            map_feature_feature = torch.stack(map_feature_list)
            map_feature_mask = torch.stack(map_feature_mask_list)

            point_dis = torch.hypot(map_feature_feature[:, :, 0], map_feature_feature[:, :, 1])
            min_dis_in_polyline, _ = torch.min(point_dis, dim=1)
            _, sorted_index = torch.sort(min_dis_in_polyline)

            map_feature_padded = map_feature_feature[sorted_index[:max_polyline_num], :, :]
            map_feature_mask_padded = map_feature_mask[sorted_index[:max_polyline_num], :]

        return map_feature_padded, (map_feature_mask_padded>0)

    def build_one_map_feature_tensor_except_lane(self, polylines: torch.Tensor, feature_type: str):
        polyline_point_num = self.dataset_config.map_feature.polyline_point_num
        polyline_subsample_interval = self.dataset_config.map_feature.polyline_subsample_interval
        polyline_valid_points = (polylines.size(0)-1) // polyline_subsample_interval + 1

        pos_feature_tensor = torch.zeros(polyline_point_num, 6)
        map_feature_mask = torch.zeros(polyline_point_num)

        pos_feature_tensor[:polyline_valid_points, 0:2] = polylines[::polyline_subsample_interval, 0:2] # x, y
        pos_feature_tensor[:polyline_valid_points, 2:4] = polylines[::polyline_subsample_interval, 3:5] # dir_x, dir_y
        pos_feature_tensor[:polyline_valid_points-1, 4:6] = polylines[polyline_subsample_interval::polyline_subsample_interval, 0:2] # next_x, next_y
        pos_feature_tensor[polyline_valid_points-1, 4:6] = pos_feature_tensor[polyline_valid_points-1, 0:2]
        map_feature_mask[:polyline_valid_points] = 1

        if feature_type == 'road_line':
            one_hot_tensor = torch.zeros(polyline_point_num, 8)
            if polylines[0, 6] == 6: # TYPE_BROKEN_SINGLE_WHITE
                one_hot_tensor[:polyline_valid_points, 0] = 1
            elif polylines[0, 6] == 7: # TYPE_SOLID_SINGLE_WHITE
                one_hot_tensor[:polyline_valid_points, 1] = 1
            elif polylines[0, 6] == 8: # TYPE_SOLID_DOUBLE_WHITE
                one_hot_tensor[:polyline_valid_points, 2] = 1
            elif polylines[0, 6] == 9: # TYPE_BROKEN_SINGLE_YELLOW
                one_hot_tensor[:polyline_valid_points, 3] = 1
            elif polylines[0, 6] == 10: # TYPE_BROKEN_DOUBLE_YELLOW
                one_hot_tensor[:polyline_valid_points, 4] = 1
            elif polylines[0, 6] == 11: # TYPE_SOLID_SINGLE_YELLOW
                one_hot_tensor[:polyline_valid_points, 5] = 1
            elif polylines[0, 6] == 12: # TYPE_SOLID_DOUBLE_YELLOW
                one_hot_tensor[:polyline_valid_points, 6] = 1
            elif polylines[0, 6] == 13: # TYPE_PASSING_DOUBLE_YELLOW
                one_hot_tensor[:polyline_valid_points, 7] = 1
        elif feature_type == 'road_edge':
            one_hot_tensor = torch.zeros(polyline_point_num, 2)
            if polylines[0, 6] == 15: # TYPE_ROAD_EDGE_BOUNDARY
                one_hot_tensor[:polyline_valid_points, 0] = 1
            elif polylines[0, 6] == 16: # TYPE_ROAD_EDGE_MEDIAN
                one_hot_tensor[:polyline_valid_points, 1] = 1
        elif feature_type == 'map_others':
            one_hot_tensor = torch.zeros(polyline_point_num, 3)
            if polylines[0, 6] == 17: # TYPE_STOP_SIGN
                one_hot_tensor[:polyline_valid_points, 0] = 1
            elif polylines[0, 6] == 18: # TYPE_CROSSWALK
                one_hot_tensor[:polyline_valid_points, 1] = 1
            elif polylines[0, 6] == 19: # TYPE_SPEED_BUMP
                one_hot_tensor[:polyline_valid_points, 2] = 1

        map_feature_tensor = torch.cat([pos_feature_tensor, one_hot_tensor], dim=1)

        return map_feature_tensor, map_feature_mask

    def get_lane_feature(self, polylines_tensor: torch.Tensor, lane_info: List, traffic_light_index, traffic_light_state):
        
        polyline_point_num = self.dataset_config.map_feature.polyline_point_num
        polyline_subsample_interval = self.dataset_config.map_feature.polyline_subsample_interval

        lane_feature_list = []
        lane_feature_mask_list = []
        for i in range(len(lane_info)):
            # get traffic light info
            lane_traffic_light_state = 'LANE_STATE_UNKNOWN'
            for index in range(traffic_light_index.shape[1]):
                if traffic_light_index[0, index] == lane_info[i]['id']:
                    lane_traffic_light_state = traffic_light_state[0, index]
                    break

            polyline_index = lane_info[i]['polyline_index']
            for index in range(polyline_index[0], polyline_index[1], polyline_point_num*polyline_subsample_interval):
                one_lane_feature_tensor, one_lane_feature_mask = self.build_one_lane_feature_tensor(
                    polylines_tensor[index:min(index+polyline_point_num*polyline_subsample_interval, polyline_index[1]), :],
                    lane_traffic_light_state,
                    lane_info[i]['speed_limit_mph']
                    )
                lane_feature_list.append(one_lane_feature_tensor)
                lane_feature_mask_list.append(one_lane_feature_mask)

        # reshape to max_agent_num
        max_polyline_num = self.dataset_config.map_feature.max_polyline_num['lane']
        map_feature_padded = torch.zeros((max_polyline_num, self.dataset_config.map_feature.polyline_point_num, self.dataset_config.map_feature.feature_dim['lane']))
        map_feature_mask_padded = torch.zeros((max_polyline_num, self.dataset_config.map_feature.polyline_point_num))

        if max_polyline_num >= len(lane_feature_list):
            # pad zeros
            for i in range(len(lane_feature_list)):
                map_feature_padded[i, :, :] = lane_feature_list[i]
                map_feature_mask_padded[i, :] = lane_feature_mask_list[i]
        else:
            # filtered by closest point to ego
            map_feature_feature = torch.stack(lane_feature_list)
            map_feature_mask = torch.stack(lane_feature_mask_list)

            point_dis = torch.hypot(map_feature_feature[:, :, 0], map_feature_feature[:, :, 1])
            min_dis_in_polyline, _ = torch.min(point_dis, dim=1)
            _, sorted_index = torch.sort(min_dis_in_polyline)

            map_feature_padded = map_feature_feature[sorted_index[:max_polyline_num], :, :]
            map_feature_mask_padded = map_feature_mask[sorted_index[:max_polyline_num], :]

        return map_feature_padded, (map_feature_mask_padded>0)

    def build_one_lane_feature_tensor(self, polylines: torch.Tensor, lane_traffic_light_state: str, speed_limit: float):
        polyline_point_num = self.dataset_config.map_feature.polyline_point_num
        polyline_subsample_interval = self.dataset_config.map_feature.polyline_subsample_interval
        polyline_valid_points = (polylines.size(0)-1) // polyline_subsample_interval + 1

        pos_feature_tensor = torch.zeros(polyline_point_num, 7)
        map_feature_mask = torch.zeros(polyline_point_num)

        pos_feature_tensor[:polyline_valid_points, 0:2] = polylines[::polyline_subsample_interval, 0:2] # x, y
        pos_feature_tensor[:polyline_valid_points, 2:4] = polylines[::polyline_subsample_interval, 3:5] # dir_x, dir_y
        pos_feature_tensor[:polyline_valid_points-1, 4:6] = polylines[polyline_subsample_interval::polyline_subsample_interval, 0:2] # next_x, next_y
        pos_feature_tensor[polyline_valid_points-1, 4:6] = pos_feature_tensor[polyline_valid_points-1, 0:2]
        pos_feature_tensor[:polyline_valid_points-1, 6] = speed_limit # speed_limit
        map_feature_mask[:polyline_valid_points] = 1

        one_hot_tensor = torch.zeros(polyline_point_num, 3)
        if polylines[0, 6] == 1: # TYPE_FREEWAY
            one_hot_tensor[:polyline_valid_points, 0] = 1
        elif polylines[0, 6] == 2: # TYPE_SURFACE_STREET
            one_hot_tensor[:polyline_valid_points, 1] = 1
        elif polylines[0, 6] == 3: # TYPE_BIKE_LANE
            one_hot_tensor[:polyline_valid_points, 2] = 1

        traffic_light_tensor = torch.zeros(polyline_point_num, 9)
        if lane_traffic_light_state == 'LANE_STATE_UNKNOWN':
            traffic_light_tensor[:polyline_valid_points, 0] = 1
        elif lane_traffic_light_state == 'LANE_STATE_ARROW_STOP':
            traffic_light_tensor[:polyline_valid_points, 1] = 1
        elif lane_traffic_light_state == 'LANE_STATE_ARROW_CAUTION':
            traffic_light_tensor[:polyline_valid_points, 2] = 1
        elif lane_traffic_light_state == 'LANE_STATE_ARROW_GO':
            traffic_light_tensor[:polyline_valid_points, 3] = 1
        elif lane_traffic_light_state == 'LANE_STATE_STOP':
            traffic_light_tensor[:polyline_valid_points, 4] = 1
        elif lane_traffic_light_state == 'LANE_STATE_CAUTION':
            traffic_light_tensor[:polyline_valid_points, 5] = 1
        elif lane_traffic_light_state == 'LANE_STATE_GO':
            traffic_light_tensor[:polyline_valid_points, 6] = 1
        elif lane_traffic_light_state == 'LANE_STATE_FLASHING_STOP':
            traffic_light_tensor[:polyline_valid_points, 7] = 1
        elif lane_traffic_light_state == 'LANE_STATE_FLASHING_CAUTION':
            traffic_light_tensor[:polyline_valid_points, 8] = 1

        map_feature_tensor = torch.cat([pos_feature_tensor, one_hot_tensor, traffic_light_tensor], dim=1)

        return map_feature_tensor, map_feature_mask
    
    def collate_batch(self, batch_list):
        feature_dict = {}
        for key in batch_list[0].keys():
            if key in ['scenario_id', 'center_objects_id', 'center_objects_type']:
                temp_list = []
                for i in range(len(batch_list)):
                    temp_list.append(batch_list[i][key][0])
                feature_dict[key] = temp_list
            else:
                temp_list = []
                for i in range(len(batch_list)):
                    temp_list.append(batch_list[i][key])
                feature_dict[key] = torch.cat(temp_list, dim=0)

        return feature_dict
    
    def generate_input_dict_for_eval(self, info, ret_dict, valid_agent_to_predict, trajs, ego_index):
        # scenario_id: list[str]
        # center_objects_id: list[list[str]]
        # center_objects_type: list[list[]]
        # center_gt_trajs_src: [batch, 1, max_agent_num, 91, 10]
        # track_index_to_predict: [batch, 1, max_agent_num]
        # ego_current_pose: [batch, 1, 3]
        max_agent_num = self.dataset_config.max_agent_num

        # scenario_id
        scene_id = info['scenario_id']
        ret_dict['scenario_id'] = [scene_id]

        # track_index_to_predict, center_objects_id, center_objects_type, center_gt_trajs_src
        object_id = info['track_infos']['object_id']
        object_type = info['track_infos']['object_type']

        track_index_to_predict = torch.zeros(1, 1, max_agent_num)
        center_objects_id_single = []
        center_objects_type_single = []
        center_gt_trajs_src = torch.zeros(1, 1, max_agent_num, 91, 10)

        for i in range(len(valid_agent_to_predict)):
            track_index_to_predict[0, 0, i] = valid_agent_to_predict[i]
            center_objects_id_single.append(object_id[valid_agent_to_predict[i]])
            center_objects_type_single.append(object_type[valid_agent_to_predict[i]])
            center_gt_trajs_src[0, 0, i, :, :] = trajs[valid_agent_to_predict[i], :, :]

        ret_dict['track_index_to_predict'] = track_index_to_predict
        ret_dict['center_objects_id'] = [center_objects_id_single]
        ret_dict['center_objects_type'] = [center_objects_type_single]
        ret_dict['center_gt_trajs_src'] = center_gt_trajs_src

        # ego pose at 1s
        ego_current_pose = torch.zeros(1, 1, 3)
        ego_current_pose[0, 0, 0:2] = trajs[ego_index, 10, 0:2]
        ego_current_pose[0, 0, 2] = trajs[ego_index, 10, 6]
        ret_dict['ego_current_pose'] = ego_current_pose

        return ret_dict
    

    def generate_prediction_dicts(self, input_dict, output_path=None):
        agent_to_predict_num = input_dict['agent_to_predict_num']

        batch = agent_to_predict_num.size(0)
        mode_num = 6
        agent_global_traj = input_dict['agent_global_traj']
        agent_global_traj = agent_global_traj.unsqueeze(2).repeat(1, 1, mode_num, 1, 1)

        pred_dict_list = []
        for i in range(batch):
            for j in range(int(agent_to_predict_num[i, 0].item())):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][i],
                    'pred_trajs': agent_global_traj[i, j, :, :, 0:2].cpu().numpy(),
                    'pred_scores': torch.ones(mode_num).cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][i][j],
                    'object_type': input_dict['center_objects_type'][i][j],
                    'gt_trajs': input_dict['center_gt_trajs_src'][i, 0, j, ...].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][i, 0, j].item(),
                }
                pred_dict_list.append(single_pred_dict)

        return pred_dict_list
    

    def evaluation(self, pred_dicts, output_path=None, eval_method='waymo', **kwargs):
        if eval_method == 'waymo':
            from mtr_trainer.mtr_datasets.waymo_eval import waymo_evaluation
            try:
                num_modes_for_eval = pred_dicts[0][0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=num_modes_for_eval)

            metric_result_str = '\n'
            for key in metric_results:
                metric_results[key] = metric_results[key]
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str
        else:
            raise NotImplementedError

        return metric_result_str, metric_results