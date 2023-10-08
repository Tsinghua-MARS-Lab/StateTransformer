import torch
import pickle
import os
import pathlib

from typing import Dict, List, Tuple, cast
import numpy as np
from utils.torch_geometry import global_state_se2_tensor_to_local, coordinates_to_local_frame
from utils.base_math import angle_to_range

import time


class WaymoDatasetWayformer(torch.utils.data.Dataset):
    # all info are expressed in target frame
    # targets num may vary in different scenes.
    
    # map:
    # lane: [target_num, n(512), 19], feature info:[x, y, dir_x, dir_y, next_x, next_y, speed_limit, Freeway, SurfaceStreet, BikeLane,
    #        Unknown, Arrow_Stop, Arrow_Caution, Arrow_Go, Stop, Caution, Go, Flashing_Stop, Flashing_Caution]
    # road_line: [target_num, n(256), 14], feature info:[x, y, dir_x, dir_y, next_x, next_y,
    #        BrokenSingleWhite, SolidSingleWhite, SolidDoubleWhite, BrokenSingleYellow, BrokenDoubleYellow, SolidSingleYellow, SolidDoubleYellow, PassingDoubleYellow]
    # road_edge: [target_num, n(128), 8], feature info:[x, y, dir_x, dir_y, next_x, next_y, RoadEdgeBoundary, RoadEdgeMedian]
    # stop_sign+crosswalk+speed_bump: [target_num, n(128), 9], feature info:[x, y, dir_x, dir_y, next_x, next_y, StopSign, Crosswalk, SpeedBump]
    # valid*4: [target_num, n]

    # agent
    # target: [target_num, 1, 5, 14], feature info:[last_x, last_y, x, y, heading, cos(heading), sin(heading), v_x, v_y, length, width, TYPE_VEHICLE, TYPE_PEDESTRAIN, TYPE_CYCLIST]
    # agent: [target_num, n(64), 5, 14], feature info:[last_x, last_y, x, y, heading, cos(heading), sin(heading), v_x, v_y, length, width, TYPE_VEHICLE, TYPE_PEDESTRAIN, TYPE_CYCLIST]
    # agent_valid: [target_num, n(64)]
    # time_step is in reversed order

    # label:
    # target_label: [target_num, 1, 16, 3], feature info:[x, y, heading]
    # time_step is in order
    
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
        # # use cache if exist
        # cache_file_name = os.path.join(self.data_cache_path, f'cache_{index}.pkl')
        # if os.path.exists(cache_file_name):
        #     with open(cache_file_name, 'rb') as f:
        #         try:
        #             feature_dict = pickle.load(f)
        #         except:
        #             feature_dict = self.generate_from_raw_data(index)
        # else:
        #     feature_dict = self.generate_from_raw_data(index)
        #     while feature_dict is None:
        #         index = (index+1)%len(self.infos)
        #         feature_dict = self.generate_from_raw_data(index)
        #     with open(cache_file_name, 'wb') as f:
        #         pickle.dump(feature_dict, f)
        # return feature_dict
    
        feature_dict = self.generate_from_raw_data(index)
        while feature_dict is None:
            index = (index+1)%len(self.infos)
            feature_dict = self.generate_from_raw_data(index)
        return feature_dict

    def generate_from_raw_data(self, index):
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(os.path.join(self.data_path, f'sample_{scene_id}.pkl'), 'rb') as f:
            info = pickle.load(f)
        
        # generate time_set_num sets samples
        lane_feature_list, lane_valid_list = [], []
        road_line_feature_list, road_line_valid_list = [], []
        road_edge_feature_list, road_edge_valid_list = [], []
        map_others_feature_list, map_others_valid_list = [], []

        target_feature_list, target_label_list = [], []
        agent_feature_list, agent_valid_list = [], []

        sdc_track_index = info['sdc_track_index']
        trajs = torch.from_numpy(info['track_infos']['trajs']).float()
        polyline_tensor = torch.from_numpy(info['map_infos']['all_polylines']).float()

        trajs = self.filter_data_by_diff(trajs)
        valid_agent_to_predict = self.check_valid_agent_to_predict(trajs, sdc_track_index, info['tracks_to_predict']['track_index'], 
                                                                   info['track_infos']['object_type'], self.dataset_config.dataset_info.object_type)
        if len(valid_agent_to_predict) == 0:
            return None
        
        for i in range(len(valid_agent_to_predict)):
            # get ego, agent feature
            target_type = info['track_infos']['object_type'][valid_agent_to_predict[i]]
            target_feature, target_label = self.get_target_feature_and_label(trajs, valid_agent_to_predict[i], target_type, 10)
            agent_feature, agent_valid = self.get_agent_feature_and_label(trajs=trajs, 
                                                                          object_type=info['track_infos']['object_type'], 
                                                                          target_index=valid_agent_to_predict[i], 
                                                                          time_index=10)
            target_feature_list.append(target_feature)
            target_label_list.append(target_label)
            agent_feature_list.append(agent_feature)
            agent_valid_list.append(agent_valid)

            # tranform map feature coodinates to ego frame
            target_info = trajs[valid_agent_to_predict[i], 10, :]
            target_frame = torch.tensor([target_info[0], target_info[1], target_info[6]])
            rotation_frame = torch.tensor([0, 0, target_info[6]])
            transformed_polyline_tensor = torch.clone(polyline_tensor)
            transformed_polyline_tensor[:, 0:2] = coordinates_to_local_frame(transformed_polyline_tensor[:, 0:2], target_frame) # x, y
            transformed_polyline_tensor[:, 3:5] = coordinates_to_local_frame(transformed_polyline_tensor[:, 3:5], rotation_frame) # dir_x, dir_y

            # build map features
            lane_feature, lane_valid = self.get_lane_feature(transformed_polyline_tensor, 
                                                            info['map_infos']['lane'], 
                                                            info['dynamic_map_infos']['lane_id'][10],
                                                            info['dynamic_map_infos']['state'][10])
            lane_feature_list.append(lane_feature)
            lane_valid_list.append(lane_valid)

            road_line_feature, road_line_valid = self.get_map_feature_except_lane(transformed_polyline_tensor, info['map_infos']['road_line'], 'road_line')
            road_line_feature_list.append(road_line_feature)
            road_line_valid_list.append(road_line_valid)

            road_edge_feature, road_edge_valid = self.get_map_feature_except_lane(transformed_polyline_tensor, info['map_infos']['road_edge'], 'road_edge')
            road_edge_feature_list.append(road_edge_feature)
            road_edge_valid_list.append(road_edge_valid)

            map_others_info = info['map_infos']['stop_sign'] + info['map_infos']['crosswalk'] + info['map_infos']['speed_bump']
            map_others_feature, map_others_valid = self.get_map_feature_except_lane(transformed_polyline_tensor, map_others_info, 'map_others')
            map_others_feature_list.append(map_others_feature)
            map_others_valid_list.append(map_others_valid)

        feature_dict = {
            'lane_feature': torch.stack(lane_feature_list),
            'lane_valid': torch.stack(lane_valid_list),
            'road_line_feature': torch.stack(road_line_feature_list),
            'road_line_valid': torch.stack(road_line_valid_list),
            'road_edge_feature': torch.stack(road_edge_feature_list),
            'road_edge_valid': torch.stack(road_edge_valid_list),
            'map_others_feature': torch.stack(map_others_feature_list),
            'map_others_valid': torch.stack(map_others_valid_list),

            'target_feature': torch.stack(target_feature_list),
            'target_label': torch.stack(target_label_list),
            'agent_feature': torch.stack(agent_feature_list),
            'agent_valid': torch.stack(agent_valid_list),
        }
        feature_dict = self.generate_input_dict_for_eval(info, feature_dict, valid_agent_to_predict, trajs)

        return feature_dict

    def check_valid_agent_to_predict(self, trajs, ego_index: int, agent_to_predict_index: List, data_object_type: List, target_type: List):
        # assert trajs[ego_index, :, -1].all()
        valid_agent_to_predict = []
        for index in agent_to_predict_index:
            if trajs[index, :, -1].all():
                object_type = data_object_type[index]
                if object_type in target_type:
                    valid_agent_to_predict.append(index)

        return valid_agent_to_predict

    def filter_data_by_diff(self, trajs):
        max_xy_diff = self.dataset_config.filter_param.max_xy_diff
        max_heading_diff = self.dataset_config.filter_param.max_heading_diff

        trajs_diff = trajs[:, 1:, :] - trajs[:, :-1, :]

        x_invalid_index = torch.abs(trajs_diff[:, :, 0]) > max_xy_diff
        y_invalid_index = torch.abs(trajs_diff[:, :, 1]) > max_xy_diff
        heading_invalid_index = torch.abs(angle_to_range(trajs_diff[:, :, 2])) > max_heading_diff

        trajs[:, :-1, 9][x_invalid_index] = 0
        trajs[:, :-1, 9][y_invalid_index] = 0
        trajs[:, :-1, 9][heading_invalid_index] = 0

        return trajs

    def get_target_feature_and_label(self, trajs: torch.Tensor, target_index: int, target_type: str, time_index: int):
        feature_time_step = self.dataset_config.time_info.feature_time_step
        feature_time_interval = self.dataset_config.time_info.feature_time_interval
        target_time_step = self.dataset_config.time_info.target_time_step
        target_time_interval = self.dataset_config.time_info.target_time_interval

        # build target feature
        target_feature = torch.zeros([1, feature_time_step, 9+5])
        for i in range(feature_time_step):
            target_feature[0, i, 0] = trajs[target_index, time_index - (i+1)*feature_time_interval, 0] # last_x
            target_feature[0, i, 1] = trajs[target_index, time_index - (i+1)*feature_time_interval, 1] # last_y

            target_feature[0, i, 2] = trajs[target_index, time_index - i*feature_time_interval, 0] # x
            target_feature[0, i, 3] = trajs[target_index, time_index - i*feature_time_interval, 1] # y
            target_feature[0, i, 4] = trajs[target_index, time_index - i*feature_time_interval, 6] # heading

            target_feature[0, i, 7] = trajs[target_index, time_index - i*feature_time_interval, 7] # v_x
            target_feature[0, i, 8] = trajs[target_index, time_index - i*feature_time_interval, 8] # v_y
        target_feature[0, :, 9] = trajs[target_index, time_index, 3] # length
        target_feature[0, :, 10] = trajs[target_index, time_index, 4] # width
        if target_type == 'TYPE_VEHICLE':
            target_feature[0, :, 11] = 1
        elif target_type == 'TYPE_PEDESTRIAN':
            target_feature[0, :, 12] = 1
        elif target_type == 'TYPE_CYCLIST':
            target_feature[0, :, 13] = 1
        
        # build target label
        target_label = torch.zeros([1, target_time_step, 3])
        for i in range(target_time_step):
            target_label[0, i, 0] = trajs[target_index, time_index + (i+1)*target_time_interval, 0]-trajs[target_index, time_index, 0]  # x
            target_label[0, i, 1] = trajs[target_index, time_index + (i+1)*target_time_interval, 1]-trajs[target_index, time_index, 1]  # y
            target_label[0, i, 2] = angle_to_range(
                trajs[target_index, time_index + (i+1)*target_time_interval, 6]-trajs[target_index, time_index, 6]) # heading
        
        # transfer to target coordinate
        target_info = trajs[target_index, time_index, :]
        target_frame = torch.tensor([target_info[0], target_info[1], target_info[6]])
        rotation_frame = torch.tensor([0, 0, target_info[6]])
        for i in range(feature_time_step):
            target_feature[:, i, 0:2] = coordinates_to_local_frame(target_feature[:, i, 0:2], target_frame)
            target_feature[:, i, 2:5] = global_state_se2_tensor_to_local(target_feature[:, i, 2:5], target_frame)
            target_feature[:, i, 5] = torch.cos(target_feature[:, i, 4]) # cos(heading)
            target_feature[:, i, 6] = torch.sin(target_feature[:, i, 4]) # sin(heading)
            target_feature[:, i, 7:9] = coordinates_to_local_frame(target_feature[:, i, 7:9], rotation_frame)

        for i in range(target_time_step):
            target_label[:, i, 0:2] = coordinates_to_local_frame(target_label[:, i, 0:2], rotation_frame)
        
        return target_feature, target_label

    def get_agent_feature_and_label(self, trajs: torch.Tensor, object_type: List, target_index: int, time_index: int):
        max_agent_num = self.dataset_config.max_agent_num
        feature_time_step = self.dataset_config.time_info.feature_time_step
        feature_time_interval = self.dataset_config.time_info.feature_time_interval

        # filter agent index by:
        # 1. valid from 0~10
        # 2. if agent num is larger than max_agent_num, sort agent based on the distance to target at time 1.0(index 10)

        # filter out invalid agents 
        valid_tensor = torch.zeros(trajs.size(0)).bool()
        valid_tensor[:] = trajs[:, 0:time_index+1, -1].all(dim=1)
        valid_agent_index = [i for i in range(trajs.size(0)) if valid_tensor[i]==True]

        if len(valid_agent_index) > max_agent_num:

            # sort agent based on distance to ego
            dis_to_ego = torch.hypot(trajs[:, time_index, 0]-trajs[target_index, time_index, 0],
                                    trajs[:, time_index, 1]-trajs[target_index, time_index, 1])
            _, sorted_index = torch.sort(dis_to_ego)
            sorted_valid_index = []
            for i in sorted_index:
                for j in valid_agent_index:
                    if i == j:
                        sorted_valid_index.append(i)
                        break

            valid_agent_index = sorted_valid_index[:max_agent_num]
            valid_agent_index.sort()

        # get agent_tensor
        agent_tensor = trajs[valid_agent_index, :, :]
        agent_type = [object_type[i] for i in valid_agent_index]
        
        # build agent feature tensor
        agent_feature = torch.zeros([agent_tensor.size(0), feature_time_step, 9 + 5])
        for i in range(feature_time_step):
            agent_feature[:, i, 0] = agent_tensor[:, time_index - (i+1)*feature_time_interval, 0] # last_x
            agent_feature[:, i, 1] = agent_tensor[:, time_index - (i+1)*feature_time_interval, 1] # last_y

            agent_feature[:, i, 2] = agent_tensor[:, time_index - i*feature_time_interval, 0] # x
            agent_feature[:, i, 3] = agent_tensor[:, time_index - i*feature_time_interval, 1] # y
            agent_feature[:, i, 4] = agent_tensor[:, time_index - i*feature_time_interval, 6] # heading
            agent_feature[:, i, 7] = agent_tensor[:, time_index - i*feature_time_interval, 7] # v_x
            agent_feature[:, i, 8] = agent_tensor[:, time_index - i*feature_time_interval, 8] # v_y
        agent_feature[:, :, 9] = agent_tensor[:, time_index, 3].unsqueeze(1) # length
        agent_feature[:, :, 10] = agent_tensor[:, time_index, 4].unsqueeze(1) # width

        for i in range(agent_tensor.size(0)):
            if agent_type[i] == 'TYPE_VEHICLE':
                agent_feature[i, :, 11] = 1
            elif agent_type[i] == 'TYPE_PEDESTRIAN':
                agent_feature[i, :, 12] = 1
            elif agent_type[i] == 'TYPE_CYCLIST':
                agent_feature[i, :, 13] = 1

        # build agent valid tensor
        agent_valid = torch.ones(agent_tensor.size(0))

        # tranfer to ego and agent coordinate frame
        target_info = trajs[target_index, time_index, :]
        target_frame = torch.tensor([target_info[0], target_info[1], target_info[6]])
        rotation_frame = torch.tensor([0, 0, target_info[6]])

        for i in range(feature_time_step):
            # agent feature
            agent_feature[:, i, 0:2] = coordinates_to_local_frame(agent_feature[:, i, 0:2], target_frame)
            agent_feature[:, i, 2:5] = global_state_se2_tensor_to_local(agent_feature[:, i, 2:5], target_frame)
            agent_feature[:, i, 5] = torch.cos(agent_feature[:, i, 4]) # cos(heading)
            agent_feature[:, i, 6] = torch.sin(agent_feature[:, i, 4]) # sin(heading)
            agent_feature[:, i, 7:9] = coordinates_to_local_frame(agent_feature[:, i, 7:9], rotation_frame)

        # pad to max_agent_num
        agent_feature_padded = torch.zeros((max_agent_num, agent_feature.size(1), agent_feature.size(2)))
        agent_valid_padded = torch.zeros(max_agent_num)

        agent_feature_padded[:agent_feature.size(0), :, :] = agent_feature
        agent_valid_padded[:agent_feature.size(0)] = agent_valid

        return agent_feature_padded, agent_valid_padded

    def get_map_feature_except_lane(self, polylines_tensor: torch.Tensor, map_feature_info: List, feature_type: str):
        map_feature_list = []
        for i in range(len(map_feature_info)):
            polyline_index = map_feature_info[i]['polyline_index']

            one_map_feature_tensor = self.build_one_map_feature_tensor_except_lane(polylines_tensor[polyline_index[0]:polyline_index[1], :],
                                                                                   feature_type)
            map_feature_list.append(one_map_feature_tensor)

        # reshape to max_agent_num
        max_polyline_num = self.dataset_config.map_feature.max_polyline_num[feature_type]
        map_feature_padded = torch.zeros(max_polyline_num, self.dataset_config.map_feature.feature_dim[feature_type])
        map_feature_valid_padded = torch.zeros(max_polyline_num)

        if len(map_feature_list) == 0:
            return map_feature_padded, map_feature_valid_padded

        map_feature = torch.cat(map_feature_list, dim=0)
        polyline_dis = torch.hypot(map_feature[:, 0], map_feature[:, 1])
        _, sorted_index = torch.sort(polyline_dis)
        if max_polyline_num >= map_feature.size(0):
            # pad zeros
            map_feature_padded[:map_feature.size(0), :] = map_feature[sorted_index[:], :]
            map_feature_valid_padded[:map_feature.size(0)] = 1
        else:
            # filtered by closest point to ego
            map_feature_padded = map_feature[sorted_index[:max_polyline_num], :]
            map_feature_valid_padded[:] = 1

        return map_feature_padded, map_feature_valid_padded


    def build_one_map_feature_tensor_except_lane(self, polylines: torch.Tensor, feature_type: str):
        polyline_subsample_interval = self.dataset_config.map_feature.polyline_subsample_interval
        polyline_valid_points = (polylines.size(0)-1) // polyline_subsample_interval + 1

        pos_feature_tensor = torch.zeros(polyline_valid_points, 6)

        pos_feature_tensor[:, 0:2] = polylines[::polyline_subsample_interval, 0:2] # x, y
        pos_feature_tensor[:, 2:4] = polylines[::polyline_subsample_interval, 3:5] # dir_x, dir_y
        pos_feature_tensor[:polyline_valid_points-1, 4:6] = polylines[polyline_subsample_interval::polyline_subsample_interval, 0:2] # next_x, next_y
        pos_feature_tensor[-1, 4:6] = pos_feature_tensor[-1, 0:2]

        if feature_type == 'road_line':
            one_hot_tensor = torch.zeros(polyline_valid_points, 8)
            if polylines[0, 6] == 6: # TYPE_BROKEN_SINGLE_WHITE
                one_hot_tensor[:, 0] = 1
            elif polylines[0, 6] == 7: # TYPE_SOLID_SINGLE_WHITE
                one_hot_tensor[:, 1] = 1
            elif polylines[0, 6] == 8: # TYPE_SOLID_DOUBLE_WHITE
                one_hot_tensor[:, 2] = 1
            elif polylines[0, 6] == 9: # TYPE_BROKEN_SINGLE_YELLOW
                one_hot_tensor[:, 3] = 1
            elif polylines[0, 6] == 10: # TYPE_BROKEN_DOUBLE_YELLOW
                one_hot_tensor[:, 4] = 1
            elif polylines[0, 6] == 11: # TYPE_SOLID_SINGLE_YELLOW
                one_hot_tensor[:, 5] = 1
            elif polylines[0, 6] == 12: # TYPE_SOLID_DOUBLE_YELLOW
                one_hot_tensor[:, 6] = 1
            elif polylines[0, 6] == 13: # TYPE_PASSING_DOUBLE_YELLOW
                one_hot_tensor[:, 7] = 1
        elif feature_type == 'road_edge':
            one_hot_tensor = torch.zeros(polyline_valid_points, 2)
            if polylines[0, 6] == 15: # TYPE_ROAD_EDGE_BOUNDARY
                one_hot_tensor[:, 0] = 1
            elif polylines[0, 6] == 16: # TYPE_ROAD_EDGE_MEDIAN
                one_hot_tensor[:, 1] = 1
        elif feature_type == 'map_others':
            one_hot_tensor = torch.zeros(polyline_valid_points, 3)
            if polylines[0, 6] == 17: # TYPE_STOP_SIGN
                one_hot_tensor[:, 0] = 1
            elif polylines[0, 6] == 18: # TYPE_CROSSWALK
                one_hot_tensor[:, 1] = 1
            elif polylines[0, 6] == 19: # TYPE_SPEED_BUMP
                one_hot_tensor[:, 2] = 1

        map_feature_tensor = torch.cat([pos_feature_tensor, one_hot_tensor], dim=1)

        return map_feature_tensor

    def get_lane_feature(self, polylines_tensor: torch.Tensor, lane_info: List, traffic_light_index, traffic_light_state):
        lane_feature_list = []
        for i in range(len(lane_info)):
            # get traffic light info
            lane_traffic_light_state = 'LANE_STATE_UNKNOWN'
            for index in range(traffic_light_index.shape[1]):
                if traffic_light_index[0, index] == lane_info[i]['id']:
                    lane_traffic_light_state = traffic_light_state[0, index]
                    break

            polyline_index = lane_info[i]['polyline_index']

            one_lane_feature_tensor = self.build_one_lane_feature_tensor(polylines_tensor[polyline_index[0]:polyline_index[1], :], 
                                                                         lane_traffic_light_state, 
                                                                         lane_info[i]['speed_limit_mph'])
            lane_feature_list.append(one_lane_feature_tensor)
        lane_feature = torch.cat(lane_feature_list, dim=0)

        # reshape to max_agent_num
        max_polyline_num = self.dataset_config.map_feature.max_polyline_num['lane']
        lane_feature_padded = torch.zeros((max_polyline_num, lane_feature.size(1)))
        lane_feature_valid_padded = torch.zeros(max_polyline_num)

        polyline_dis = torch.hypot(lane_feature[:, 0], lane_feature[:, 1])
        _, sorted_index = torch.sort(polyline_dis)
        if max_polyline_num >= lane_feature.size(0):
            # pad zeros
            lane_feature_padded[:lane_feature.size(0), :] = lane_feature[sorted_index[:], :]
            lane_feature_valid_padded[:lane_feature.size(0)] = 1
        else:
            # filtered by closest point to target
            lane_feature_padded = lane_feature[sorted_index[:max_polyline_num], :]
            lane_feature_valid_padded[:] = 1

        return lane_feature_padded, lane_feature_valid_padded

    def build_one_lane_feature_tensor(self, polylines: torch.Tensor, lane_traffic_light_state: str, speed_limit: float):
        polyline_subsample_interval = self.dataset_config.map_feature.polyline_subsample_interval
        polyline_valid_points = (polylines.size(0)-1) // polyline_subsample_interval + 1

        pos_feature_tensor = torch.zeros(polyline_valid_points, 7)

        pos_feature_tensor[:, 0:2] = polylines[::polyline_subsample_interval, 0:2] # x, y
        pos_feature_tensor[:, 2:4] = polylines[::polyline_subsample_interval, 3:5] # dir_x, dir_y
        pos_feature_tensor[:polyline_valid_points-1, 4:6] = polylines[polyline_subsample_interval::polyline_subsample_interval, 0:2] # next_x, next_y
        pos_feature_tensor[-1, 4:6] = pos_feature_tensor[-1, 0:2]
        pos_feature_tensor[:polyline_valid_points-1, 6] = speed_limit # speed_limit

        one_hot_tensor = torch.zeros(polyline_valid_points, 3)
        if polylines[0, 6] == 1: # TYPE_FREEWAY
            one_hot_tensor[:, 0] = 1
        elif polylines[0, 6] == 2: # TYPE_SURFACE_STREET
            one_hot_tensor[:, 1] = 1
        elif polylines[0, 6] == 3: # TYPE_BIKE_LANE
            one_hot_tensor[:, 2] = 1

        traffic_light_tensor = torch.zeros(polyline_valid_points, 9)
        if lane_traffic_light_state == 'LANE_STATE_UNKNOWN':
            traffic_light_tensor[:, 0] = 1
        elif lane_traffic_light_state == 'LANE_STATE_ARROW_STOP':
            traffic_light_tensor[:, 1] = 1
        elif lane_traffic_light_state == 'LANE_STATE_ARROW_CAUTION':
            traffic_light_tensor[:, 2] = 1
        elif lane_traffic_light_state == 'LANE_STATE_ARROW_GO':
            traffic_light_tensor[:, 3] = 1
        elif lane_traffic_light_state == 'LANE_STATE_STOP':
            traffic_light_tensor[:, 4] = 1
        elif lane_traffic_light_state == 'LANE_STATE_CAUTION':
            traffic_light_tensor[:, 5] = 1
        elif lane_traffic_light_state == 'LANE_STATE_GO':
            traffic_light_tensor[:, 6] = 1
        elif lane_traffic_light_state == 'LANE_STATE_FLASHING_STOP':
            traffic_light_tensor[:, 7] = 1
        elif lane_traffic_light_state == 'LANE_STATE_FLASHING_CAUTION':
            traffic_light_tensor[:, 8] = 1

        map_feature_tensor = torch.cat([pos_feature_tensor, one_hot_tensor, traffic_light_tensor], dim=1)

        return map_feature_tensor
    
    def collate_batch(self, batch_list):
        feature_dict = {}
        for key in batch_list[0].keys():
            if key in ['scenario_id', 'center_objects_id', 'center_objects_type']:
                temp_list = []
                for i in range(len(batch_list)):
                    temp_list = temp_list + batch_list[i][key]
                feature_dict[key] = temp_list
            else:
                temp_list = []
                for i in range(len(batch_list)):
                    temp_list.append(batch_list[i][key])
                feature_dict[key] = torch.cat(temp_list, dim=0)

        return feature_dict
    
    def generate_input_dict_for_eval(self, info, ret_dict, valid_agent_to_predict, trajs):
        # scenario_id: list[str], len=target_num
        # center_objects_id: list[str], len=target_num
        # center_objects_type: list[str], len=target_num
        # center_gt_trajs_src: [target_num, 91, 10]
        # track_index_to_predict: [target_num]
        # target_current_pose: [target_num, 3]
        target_num = len(valid_agent_to_predict)

        # scenario_id
        scene_id = info['scenario_id']
        ret_dict['scenario_id'] = [scene_id] * target_num

        # track_index_to_predict, center_objects_id, center_objects_type, center_gt_trajs_src
        object_id = info['track_infos']['object_id']
        object_type = info['track_infos']['object_type']

        track_index_to_predict = torch.zeros(target_num)
        center_objects_id = []
        center_objects_type = []
        center_gt_trajs_src = torch.zeros(target_num, 91, 10)

        for i in range(target_num):
            track_index_to_predict[i] = valid_agent_to_predict[i]
            center_objects_id.append(object_id[valid_agent_to_predict[i]])
            center_objects_type.append(object_type[valid_agent_to_predict[i]])
            center_gt_trajs_src[i, :, :] = trajs[valid_agent_to_predict[i], :, :]

        ret_dict['track_index_to_predict'] = track_index_to_predict
        ret_dict['center_objects_id'] = center_objects_id
        ret_dict['center_objects_type'] = center_objects_type
        ret_dict['center_gt_trajs_src'] = center_gt_trajs_src

        # target pose at 1s
        target_current_pose = torch.zeros(target_num, 3)
        for i in range(target_num):
            target_current_pose[i, 0:2] = trajs[valid_agent_to_predict[i], 10, 0:2]
            target_current_pose[i, 2] = trajs[valid_agent_to_predict[i], 10, 6]
        ret_dict['target_current_pose'] = target_current_pose

        return ret_dict
    
    def generate_prediction_dicts(self, input_dict, output_path=None):
        batch = input_dict['traj_list'].size(0)

        pred_dict_list = []
        for i in range(batch):
            single_pred_dict = {
                'scenario_id': input_dict['scenario_id'][i],
                'pred_trajs': input_dict['traj_list'][i, :, :, 0:2].cpu().numpy(),
                'pred_scores': input_dict['log_prob_list'][i, :].cpu().numpy(),
                'object_id': input_dict['center_objects_id'][i],
                'object_type': input_dict['center_objects_type'][i],
                'gt_trajs': input_dict['center_gt_trajs_src'][i, ...].cpu().numpy(),
                'track_index_to_predict': input_dict['track_index_to_predict'][i].item(),
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