import copy
import os
from pathlib import Path
import tempfile
import hydra
import time
import time

"""
This code is currently tested on nuPlan devkit v1.0.0
"""

# Location of path with all training configs
# CONFIG_PATH = '../../../../codes_on_git/nuplan-devkit/nuplan/planning/script/config/training'
# CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifacts
# SAVE_DIR = Path(tempfile.gettempdir()) / 'tutorial_nuplan_framework'  # optionally replace with persistent dir
# EXPERIMENT = 'training_raster_experiment'
# LOG_DIR = str(SAVE_DIR / EXPERIMENT)


# os.environ['NUPLAN_DATA_ROOT'] = "/Users/qiaosun/nuplan/dataset"
# os.environ['NUPLAN_MAPS_ROOT'] = "/Users/qiaosun/nuplan/dataset/maps"
# # os.environ['NUPLAN_DB_FILES'] = "/Users/qiaosun/nuplan/dataset/nuplan-v1.0/mini/"
# os.environ['NUPLAN_DB_FILES'] = "/Users/qiaosun/nuplan/dataset/nuplan-v1.0/public_set_boston_train/"

# for SH server
# os.environ['NUPLAN_DATA_ROOT'] = "/public/MARS/datasets/nuPlan/data"
# os.environ['NUPLAN_MAPS_ROOT'] = "/public/MARS/datasets/nuPlan/nuplan-maps-v1.0"
# os.environ['NUPLAN_DB_FILES'] = "/public/MARS/datasets/nuPlan/data/nuplan-v1.0/data/public_set_boston_train"



# set to 0 to iterate all files
FILE_TO_START = 0
# interesting scenes:
# 185 see 4 ways stop line, 201 roundabout, 223 & 230 for a huge intersection
# interesting scenes in the visual file:
# 54 challenging heterogeneous turnings, 53 highway cut-in against dense traffic
# 52
# 59 failure case while turning with an ending point not finishing the turning
# 78 for a good two vehicle interaction demo
# for relationship flip: 134, 138, 139, 226
# for simulation
# failure cases: 13
SCENE_TO_START = 0  # nuplan 1-17 unreasonable stuck by ped nearby  2-62 wrong route in dataset
# 107 for nudging  # 38 for turning large intersection failure
SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD = 20
SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD = 0.1

TOTAL_FRAMES_IN_FUTURE = 7
FREQUENCY = 0.05

MAP_RADIUS = 100



import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Type, cast

from hydra._internal.utils import _locate
from omegaconf import DictConfig

from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map
from nuplan.planning.script.builders.worker_pool_builder import build_worker

from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import extract_tracked_objects

import interactive_sim.envs.util as util

import math
import os
import numpy as np
import random
from interactions.detect_relations import get_relation_on_crossing, form_tree_from_edges
import pickle

# set to 0 to iterate all files
# interesting scenes:
# 185 see 4 ways stop line, 201 roundabout, 223 & 230 for a huge intersection
# interesting scenes in the visual file:
# 54 challenging heterogeneous turnings, 53 highway cut-in against dense traffic
# 52
# 59 failure case while turning with an ending point not finishing the turning
# 78 for a good two vehicle interaction demo
# for relationship flip: 134, 138, 139, 226
# for simulation
# failure cases: 13

def vector2radical(vector):
    x = float(vector[0])
    y = float(vector[1])
    return math.atan2(y, x + 0.01)


def velvector2value(vector_np_x, vector_np_y):
    rst = []
    for i in range(len(vector_np_x)):
        x = float(vector_np_x[i])
        y = float(vector_np_y[i])
        rst.append(math.sqrt(x * x + y * y))
    return np.array(rst)


def miniPiPi_to_zeroTwoPie(direction):
    if direction < 0:
        return direction + math.pi * 2
    else:
        return direction


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def is_turning_agent(dirs):
    # turning right=1 turning left=2
    if len(dirs.shape) < 1:
        return False
    dir_n = dirs.shape[0]
    start_dir = 0
    end_dir = 0
    for j in range(int(dir_n / 3)):
        angel = normalize_angle(dirs[j])
        if not (angel == -1 or angel == 0):
            start_dir = angel
            break

    for j in range(int(dir_n / 3)):
        angel = normalize_angle(dirs[-j - 1])
        if not (angel == -1 or angel == 0):
            end_dir = angel
            break

    if start_dir == 0 or end_dir == 0:
        return False
    new_start_dir = miniPiPi_to_zeroTwoPie(start_dir)
    new_end_dir = miniPiPi_to_zeroTwoPie(end_dir)

    if abs(new_start_dir - new_end_dir) > (math.pi / 180 * 30):
        return True
    return False


def search_same_way_lanes(one_inbound_lane_id, road_dic, in_or_out=0, marking=0):
    # in_or_out: 0=inbound provided, 1=outbound provided
    outbound_lanes = []
    inbound_lanes = []
    # search from these inbound lanes
    xy_np = road_dic[one_inbound_lane_id]["xyz"][:, :2]
    dir_np = road_dic[one_inbound_lane_id]["dir"]
    if len(xy_np.shape) < 1 or len(dir_np.shape) < 1:
        return None
    if in_or_out:
        # outbound given
        entry_pt = xy_np[-1]
        entry_dir = dir_np[-2]
    else:
        # inbound given
        entry_pt = xy_np[0]
        entry_dir = dir_np[0]

    entry_pts_list = [entry_pt]
    out_pts_list = [entry_pt]
    pt_dist_threshold = SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD
    dir_threshold = SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD
    if in_or_out:
        # outbound given
        entry_dir = normalize_angle(entry_dir + math.pi)

    for road_seg_key in road_dic.keys():
        # if road_seg_key == tl_key:
        #     continue
        if road_dic[road_seg_key]["type"] not in [1, 2]:
            continue
        target_xy_np = road_dic[road_seg_key]["xyz"][:, :2]
        target_dir_np = road_dic[road_seg_key]["dir"]
        if len(target_xy_np.shape) < 1 or len(target_dir_np.shape) < 1:
            continue
        if target_xy_np.shape[0] < 2:
            print("ERROR: lane target_xy_np size too short. ", road_seg_key, target_xy_np)
            continue
        if target_dir_np.shape[0] < 3:
            # print("ERROR: lane target_dir_np size too short. ", road_seg_key, target_dir_np)
            # [[-1.71647068], [0.]]
            continue
        target_seg_entry_pt = target_xy_np[0]
        target_starting_dir = target_dir_np[0]
        if abs(normalize_angle(float(target_starting_dir) - float(entry_dir))) < dir_threshold:
            for one_entry_pt in entry_pts_list:
                # disth, distv = handvdistance(one_entry_pt, target_seg_entry_pt, entry_dir)
                dist = util.euclidean_distance(one_entry_pt, target_seg_entry_pt)
                if dist < pt_dist_threshold:
                    # if abs(disth) < pt_dist_threshold and abs(distv) < 5:
                    inbound_lanes.append(road_seg_key)
                    entry_pts_list.append(target_seg_entry_pt)
                    if marking:
                        road_dic[road_seg_key]["marking"] = 4
                    break
        target_seg_ending_pt = target_xy_np[-1]
        target_ending_dir = target_dir_np[-2]
        if abs(normalize_angle(float(target_ending_dir) - float(normalize_angle(entry_dir + math.pi)))) < dir_threshold:
            for one_entry_pt in out_pts_list:
                dist = util.euclidean_distance(one_entry_pt, target_seg_ending_pt)
                if dist < pt_dist_threshold:
                    outbound_lanes.append(road_seg_key)
                    out_pts_list.append(target_seg_ending_pt)
                    if marking:
                        road_dic[road_seg_key]["marking"] = 5
                    break

    return [outbound_lanes, inbound_lanes]


def handvdistance(pt1, pt2, direction):
    new_pt2_x, new_pt2_y = rotate(pt1, pt2, -direction)
    return pt1[0] - new_pt2_x, pt1[1] - new_pt2_y


import lzma
import random
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Tuple

# import msgpack
# from bokeh.document.document import Document
# from bokeh.io import show
# from bokeh.layouts import column

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)
from nuplan.planning.simulation.callback.serialization_callback import convert_sample_to_scene
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.database.nuplan_db import nuplan_scenario_queries

def get_default_scenario_extraction(
        scenario_duration: float = 15.0,
        extraction_offset: float = -2.0,
        subsample_ratio: float = 0.5,
) -> ScenarioExtractionInfo:
    """
    Get default scenario extraction instructions used in visualization.
    :param scenario_duration: [s] Duration of scenario.
    :param extraction_offset: [s] Offset of scenario (e.g. -2 means start scenario 2s before it starts).
    :param subsample_ratio: Scenario resolution.
    :return: Scenario extraction info object.
    """
    return ScenarioExtractionInfo(DEFAULT_SCENARIO_NAME, scenario_duration, extraction_offset, subsample_ratio)

def get_default_scenario_from_token(log_db: NuPlanDB, token: str, token_timestamp: int) -> NuPlanScenario:
    """
    Build a scenario with default parameters for visualization.
    :param log_db: Log database object that the token belongs to.
    :param token: Lidar pc token to be used as anchor for the scenario.
    :return: Instantiated scenario object.
    """
    args = [DEFAULT_SCENARIO_NAME, get_default_scenario_extraction(), get_pacifica_parameters()]
    return NuPlanScenario(
        data_root=log_db._data_root,
        log_file_load_path=log_db.load_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=token_timestamp,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root=log_db.maps_db._map_root,
        map_version=log_db.maps_db._map_version,
        map_name=log_db.map_name,
        scenario_extraction_info=get_default_scenario_extraction(),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )
    # return NuPlanScenario(log_db, log_db.log_name, token, *args)

class NuPlanDL:
    def __init__(self, file_to_start=None, scenario_to_start=None, max_file_number=None,
                 gt_relation_path=None, cpus=10, db=None, data_path=None, road_dic_path=None, running_mode=None):

        NUPLAN_MAP_VERSION = 'nuplan-maps-v1.0'
        if data_path is None:
            NUPLAN_DATA_ROOT = "/Users/qiaosun/nuplan/dataset"
            NUPLAN_MAPS_ROOT = "/Users/qiaosun/nuplan/dataset/maps"
            NUPLAN_DB_FILES = "/Users/qiaosun/nuplan/dataset/nuplan-v1.0/public_set_boston_train/"
        else:
            NUPLAN_DATA_ROOT = data_path['NUPLAN_DATA_ROOT']
            NUPLAN_MAPS_ROOT = data_path['NUPLAN_MAPS_ROOT']
            NUPLAN_DB_FILES = data_path['NUPLAN_DB_FILES']

        files_names = [os.path.join(NUPLAN_DB_FILES, each_path) for each_path in os.listdir(NUPLAN_DB_FILES) if each_path[0] != '.']


        for each_path in files_names:
            if '.db' not in each_path:
                print('ERROR', each_path)

        files_names = sorted(files_names)

        # MAX_FILE = 70
        # files_names = files_names[50:MAX_FILE]
        # MAX_FILE = 70
        files_names = files_names[file_to_start:file_to_start+max_file_number]

        self.global_file_names = files_names

        if db is None:
            db = NuPlanDBWrapper(
                data_root=NUPLAN_DATA_ROOT,
                map_root=NUPLAN_MAPS_ROOT,
                db_files=files_names,
                map_version=NUPLAN_MAP_VERSION,
                max_workers=cpus
            )
        
        
        self.current_dataset = db
        # available_scenario_types = defaultdict(list)
        # for log_db in db.log_dbs:
        #     for tag in log_db.scenario_tag:
        #         available_scenario_types[tag.type].append((log_db, tag.lidar_pc_token))

        self.total_file_num = len(self.current_dataset.log_dbs)
        self.current_file_index = FILE_TO_START
        if file_to_start is not None and file_to_start >= 0:
            self.current_file_index = file_to_start

        self.file_names = [nuplanDB.name for nuplanDB in self.current_dataset.log_dbs]
        if self.current_file_index >= self.total_file_num:
            self.current_file_total_scenario = 0
            self.end = True
            print("Init with index out of max file number")
        else:
            self.current_file_total_scenario = len(db.log_dbs[self.current_file_index].scenario_tag)
            self.end = False

        self.max_file_number = max_file_number
        self.start_file_number = self.current_file_index
        if scenario_to_start is not None and scenario_to_start >= 0:
            self.current_scenario_index = scenario_to_start - 1
        else:
            self.current_scenario_index = SCENE_TO_START - 1
        self.map_api = None

        self.total_frames = None
        # self.loaded_playback = None
        self.running_mode = running_mode
        self.road_dic_path = road_dic_path
        #self.traffic_dic_path = traffic_dic_path
        self.running_mode = running_mode
        self.road_dic_path = road_dic_path
        #self.traffic_dic_path = traffic_dic_path
        self.gt_relation_path = gt_relation_path
        self.timestamp = None

        print("Data Loader Initialized NuPlan: ", self.file_names[0],
              self.start_file_number, FILE_TO_START, self.current_file_index, file_to_start, self.current_scenario_index, self.current_file_total_scenario, self.max_file_number, self.total_file_num)

    def get_map(self):
        log_db = self.current_dataset.log_dbs[self.current_file_index]
        lidar_token = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[
            self.current_scenario_index].lidar_pc_token
        scenario = get_default_scenario_from_token(log_db, lidar_token, 0)
        road_dic, traffic_dic = self.pack_scenario_to_roaddic(scenario=scenario, map_radius=999999)
        road_dic = self.generate_parking_lots(road_dic)
        return road_dic

    def generate_parking_lots(self, road_dic):
        new_dic = {}
        for each_id in road_dic:
            if road_dic[each_id]['type'] not in [14]:
                continue
            if road_dic[each_id]['xyz'].shape[0] in [4, 5]:
                dist01 = util.euclidean_distance(road_dic[each_id]['xyz'][0, :2], road_dic[each_id]['xyz'][1, :2])
                dist12 = util.euclidean_distance(road_dic[each_id]['xyz'][1, :2], road_dic[each_id]['xyz'][2, :2])
                narrow = True
                if dist01 < 4:
                    pt1 = ((road_dic[each_id]['xyz'][0, :2]+road_dic[each_id]['xyz'][1, :2])/2).tolist()
                    pt2 = ((road_dic[each_id]['xyz'][2, :2] + road_dic[each_id]['xyz'][3, :2]) / 2).tolist()
                elif dist12 < 4:
                    pt1 = ((road_dic[each_id]['xyz'][1, :2]+road_dic[each_id]['xyz'][2, :2])/2).tolist()
                    pt2 = ((road_dic[each_id]['xyz'][0, :2] + road_dic[each_id]['xyz'][3, :2]) / 2).tolist()
                else:
                    narrow = False
                if narrow:
                    dist = util.euclidean_distance(pt1, pt2)
                    parking_lot_shape = [2.44, 4.88]
                    num = int(dist / parking_lot_shape[1])
                    line_yaw = util.get_angle_of_a_line(pt1, pt2)
                    for i in range(num):
                        x, y = [pt1[0], pt1[1] + (i+0.5) * parking_lot_shape[1]]
                        x, y = util.rotate(origin=pt1, point=[x, y], angle=-math.pi/2+line_yaw)
                        new_parking_lot_id = f'{each_id}-PLot{i}'
                        new_dic[new_parking_lot_id] = {
                            'xyz': [x, y],
                            'shape': parking_lot_shape, 'type': 99,
                            'dir': line_yaw,
                            'next_lanes': [], 'previous_lanes': [],
                            'outbound': 0, 'marking': 0,
                            'speed_limit': 5,  # in mph,
                            'upper_level': [each_id], 'lower_level': [],
                        }
                        road_dic[each_id]['lower_level'].append(new_parking_lot_id)
                    else:
                        # TODO
                        pass
        road_dic.update(new_dic)
        return road_dic

    def load_new_file(self, first_file=False):
        if self.max_file_number is not None and self.current_file_index >= (self.start_file_number + self.max_file_number):
            print("Reached max number:", self.current_file_index, self.max_file_number, self.start_file_number, self.total_file_num)
            self.end = True
            return
        if self.current_file_index < self.total_file_num:
            # if "." not in self.file_names[self.current_file_index] or "tf" not in self.file_names[self.current_file_index]:
            #     print("skipping invalid file: ", self.file_names[self.current_file_index])
            #     self.current_file_index += 1
            #     self.load_new_file(first_file=first_file)
            #     return
            print("Loading file from: ", self.current_dataset.log_dbs[self.current_file_index]._load_path, " with index of ", self.current_file_index)
            # self.current_file_index += 1
            self.current_file_total_scenario = len(self.current_dataset.log_dbs[self.current_file_index].scenario_tag)
            if not first_file:
                self.current_scenario_index = 0
            print(" with ", self.current_file_total_scenario, " scenarios and current is ", self.current_scenario_index)
        else:
            self.end = True

    def get_next(self, process_intersection=True, relation=False, agent_only=False, only_predict_interest_agents=False,
                 filter_config={}, calculate_gt_relation=False, load_prediction=True, detect_gt_relation=False, seconds_in_future=TOTAL_FRAMES_IN_FUTURE):
        new_files_loaded = False

        self.current_scenario_index += 1

        if not self.current_scenario_index < self.current_file_total_scenario:
            self.current_file_index += 1
            self.load_new_file()
            new_files_loaded = True

        if self.end:
            return None, new_files_loaded

        log_db = self.current_dataset.log_dbs[self.current_file_index]
        lidar_token = None
        while self.current_scenario_index < self.current_file_total_scenario:
            try:
                lidar_token = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[self.current_scenario_index].lidar_pc_token
                break
            except:
                print(f"Failed to fetch lidar pc token with exceptions!!!!!!!!!!! {self.current_file_index}/{self.total_file_num} {self.current_scenario_index}/{self.current_file_total_scenario}")
                self.current_scenario_index += 1
                if not self.current_scenario_index < self.current_file_total_scenario:
                    self.current_file_index += 1
                    self.load_new_file()
                    new_files_loaded = True
                if self.end:
                    return None, new_files_loaded

        if lidar_token is None:
            print(f"scenario loaded failed: {self.current_file_index}/{self.total_file_num} {self.current_scenario_index}/{self.current_file_total_scenario}")
            self.end = True
            return None, True

        lidar_token_timestamp = nuplan_scenario_queries.get_lidarpc_token_timestamp_from_db(log_db.load_path, lidar_token)

        scenario = get_default_scenario_from_token(log_db, lidar_token, lidar_token_timestamp)

        scenario_id = scenario.token
        self.timestamp = lidar_token_timestamp
        data_to_return = self.get_datadic(scenario=scenario,
                                          scenario_id=scenario_id, agent_only=agent_only, detect_gt_relation=detect_gt_relation,
                                          seconds_in_future=seconds_in_future)
        if data_to_return is None:
            data_to_return = {'skip': True}
            return data_to_return, new_files_loaded

        scenario_type = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[self.current_scenario_index].type
        data_to_return['type'] = scenario_type
        # goal_state = scenario.get_mission_goal()
        goal_state = scenario.get_expert_goal_state()
        if goal_state is None:
            data_to_return['ego_goal'] = None
        else:
            data_to_return['ego_goal'] = [goal_state.point.x, goal_state.point.y, 0, goal_state.heading]

        data_to_return['dataset'] = 'NuPlan'

        return data_to_return, new_files_loaded

    def get_next_file(self, specify_file_index=None):
        if specify_file_index is None:
            self.current_file_index += 1
            file_index = self.current_file_index
        else:
            # for non-sequential parallel loading
            file_index = specify_file_index
        if not file_index < self.total_file_num:
            print('index exceed total file number', file_index, self.total_file_num)
            self.end = True
            return None
        log_db = self.current_dataset.log_dbs[file_index]
        total_frames = len(log_db.lidar_pc)
        first_lidar_pc = log_db.lidar_pc[0]
        last_lidar_pc = log_db.lidar_pc[-1]
        # print("test: ", [each_pc.token for each_pc in log_db.lidar_pc])
        # print("test2: ", first_lidar_pc.prev_token,
        #       last_lidar_pc.next_token, log_db.lidar_pc[-2].next_token == last_lidar_pc,
        #       log_db.lidar_pc[-1].prev_token == log_db.lidar_pc[-2])
        starting_timestamp = first_lidar_pc.timestamp
        last_timestamp = last_lidar_pc.timestamp
        starting_token = first_lidar_pc.token
        last_token = last_lidar_pc.token
        total_time = (last_timestamp - starting_timestamp) / 1000000.0

        starting_scenario = get_default_scenario_from_token(log_db, starting_token, starting_timestamp)
        last_scenario = get_default_scenario_from_token(log_db, last_token, last_timestamp)
        future_ego_states = starting_scenario.get_ego_future_trajectory(0, 1)
        future_ego_states = [each_obj for each_obj in future_ego_states]
        any_ego_state = future_ego_states[0]

        scenario_list = []
        for i in range(0, total_frames, 400):
            if i == 0:
                scenario_list.append(starting_scenario)
            lidar_pc = log_db.lidar_pc[i]
            scenario = get_default_scenario_from_token(log_db, lidar_pc.token, lidar_pc.timestamp)
            scenario_list.append(scenario)
        scenario_list.append(last_scenario)

        self.total_frames = total_frames

        # init agent dic
        agent_dic = {}
        new_dic = {'pose': np.ones([total_frames, 4]) * -1,
                   'shape': np.ones([total_frames, 3]) * -1,
                   'speed': np.ones([total_frames, 2]) * -1,
                   'type': 0,
                   'is_sdc': 0, 'to_predict': 0}
        # pack ego
        agent_dic['ego'] = copy.deepcopy(new_dic)
        poses_np = agent_dic['ego']['pose']
        shapes_np = agent_dic['ego']['shape']
        speeds_np = agent_dic['ego']['speed']
        agent_dic['ego']['type'] = 7
        # get ego
        current_pc = first_lidar_pc
        current_ego_pose = current_pc.ego_pose
        for i in range(total_frames):
            poses_np[i, :] = [current_ego_pose.x, current_ego_pose.y, 0,
                              math.atan2(2.0 * (current_ego_pose.qw * current_ego_pose.qz + current_ego_pose.qx * current_ego_pose.qy), current_ego_pose.qw * current_ego_pose.qw + current_ego_pose.qx * current_ego_pose.qx - current_ego_pose.qy * current_ego_pose.qy - current_ego_pose.qz * current_ego_pose.qz)]
            shapes_np[i, :] = [any_ego_state.car_footprint.width,
                               any_ego_state.car_footprint.length, 2]
            speeds_np[i, :] = [current_ego_pose.vx, current_ego_pose.vy]
            if current_pc != last_lidar_pc:
                current_pc = current_pc.next
                current_ego_pose = current_pc.ego_pose
        # get other agents
        current_pc = first_lidar_pc
        # selected_agent_types = [0, 7]
        selected_agent_types = None
        # VEHICLE = 0, 'vehicle'
        # PEDESTRIAN = 1, 'pedestrian'
        # BICYCLE = 2, 'bicycle'
        # TRAFFIC_CONE = 3, 'traffic_cone'
        # BARRIER = 4, 'barrier'
        # CZONE_SIGN = 5, 'czone_sign'
        # GENERIC_OBJECT = 6, 'generic_object'
        # EGO = 7, 'ego'

        for i in range(total_frames):
            tracks = DetectionsTracks(
                extract_tracked_objects(current_pc.token, self.global_file_names[file_index])
            )
            all_agents = tracks.tracked_objects.get_agents()
            for each_agent in all_agents:
                token = each_agent.track_token
                agent_type = each_agent.tracked_object_type.value
                if selected_agent_types is not None and agent_type not in selected_agent_types:
                    print('skip type: ', agent_type, selected_agent_types)
                    continue
                if token not in agent_dic:
                    # init
                    new_dic = {'pose': np.ones([total_frames, 4]) * -1,
                               'shape': np.ones([total_frames, 3]) * -1,
                               'speed': np.ones([total_frames, 2]) * -1,
                               'type': int(agent_type),
                               'is_sdc': 0, 'to_predict': 0}
                    agent_dic[token] = new_dic
                poses_np = agent_dic[token]['pose']
                shapes_np = agent_dic[token]['shape']
                speeds_np = agent_dic[token]['speed']
                poses_np[i, :] = [each_agent.center.x, each_agent.center.y, 0, each_agent.center.heading]
                shapes_np[i, :] = [each_agent.box.width, each_agent.box.length, 2]
                speeds_np[i, :] = [each_agent.velocity.x, each_agent.velocity.y]
            if current_pc != last_lidar_pc:
                current_pc = current_pc.next
                current_ego_pose = current_pc.ego_pose

        road_dic, traffic_dic = self.pack_scenario_to_roaddic(starting_scenario, map_radius=100, scenario_list=scenario_list)
        # road_dic = {}
        # traffic_dic = {}
        data_to_return = {
            "road": road_dic,
            "agent": agent_dic,
            "traffic_light": traffic_dic,
        }
        skip = False
        # sanity check
        if agent_dic is None or road_dic is None or traffic_dic is None:
            print("Invalid Scenario Loaded: ", agent_dic is None, road_dic is None, traffic_dic is None)
            skip = True

        route_road_ids = starting_scenario.get_route_roadblock_ids()
        # handle '' empty string in route_road_ids
        road_ids_processed = []
        for each_id in route_road_ids:
            if each_id != '':
                try:
                    road_ids_processed.append(int(each_id))
                except:
                    print(f"Invalid road id in route {each_id}")
        route_road_ids = road_ids_processed

        # category = classify_scenario(data_to_return)
        data_to_return["category"] = 1
        data_to_return['scenario'] = f'{self.file_names[file_index]}'
        data_to_return['edges'] = []
        data_to_return['skip'] = skip
        data_to_return['edge_type'] = []
        data_to_return['route'] = [] if route_road_ids is None else route_road_ids

        data_to_return['starting_timestamp'] = starting_timestamp
        data_to_return['lidar_pc_tokens'] = log_db.lidar_pc
        data_to_return['ego_goal'] = None

        return data_to_return

    def pack_scenario_to_agentdic(self, scenario, total_frames_future=TOTAL_FRAMES_IN_FUTURE, total_frames_past=2):
        total_frames = total_frames_past * 20 + 1 + total_frames_future * 20
        new_dic = {'pose': np.ones([total_frames, 4]) * -1,
                   'shape': np.ones([total_frames, 3]) * -1,
                   'speed': np.ones([total_frames, 2]) * -1,
                   'type': 0,
                   'is_sdc': 0, 'to_predict': 0}
        is_init = True

        selected_agent_types = [0, 7]
        selected_agent_types = None

        # VEHICLE = 0, 'vehicle'
        # PEDESTRIAN = 1, 'pedestrian'
        # BICYCLE = 2, 'bicycle'
        # TRAFFIC_CONE = 3, 'traffic_cone'
        # BARRIER = 4, 'barrier'
        # CZONE_SIGN = 5, 'czone_sign'
        # GENERIC_OBJECT = 6, 'generic_object'
        # EGO = 7, 'ego'

        agent_dic = {}

        # pack ego
        agent_dic['ego'] = new_dic
        poses_np = agent_dic['ego']['pose']
        shapes_np = agent_dic['ego']['shape']
        speeds_np = agent_dic['ego']['speed']
        # past
        try:
            past_ego_states = scenario.get_ego_past_trajectory(0, total_frames_past, num_samples=total_frames_past*20)
            past_ego_states = [each_obj for each_obj in past_ego_states]
        except:
            print("Skipping invalid past trajectory with ", total_frames_past)
            return None

        short = max(0, total_frames_past*20 - len(past_ego_states))
        for current_t in range(total_frames_past*20):
            if current_t < short:
                continue
            ego_agent = past_ego_states[current_t-short]
            poses_np[current_t, :] = [ego_agent.car_footprint.center.x, ego_agent.car_footprint.center.y,
                                      0, ego_agent.car_footprint.center.heading]
            shapes_np[current_t, :] = [ego_agent.car_footprint.width, ego_agent.car_footprint.length, 2]
            speeds_np[current_t, :] = [ego_agent.dynamic_car_state.center_velocity_2d.x,
                                       ego_agent.dynamic_car_state.center_velocity_2d.y]

        current_ego_state = scenario.get_ego_state_at_iteration(0)
        poses_np[total_frames_past * 20, :] = [current_ego_state.car_footprint.center.x,
                                               current_ego_state.car_footprint.center.y, 0,
                                               current_ego_state.car_footprint.center.heading]
        shapes_np[total_frames_past * 20, :] = [current_ego_state.car_footprint.width,
                                                current_ego_state.car_footprint.length, 2]
        speeds_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.center_velocity_2d.x,
                                                current_ego_state.dynamic_car_state.center_velocity_2d.y]

        try:
            future_ego_states = scenario.get_ego_future_trajectory(0, total_frames_future, num_samples=total_frames_future*20)
            future_ego_states = [each_obj for each_obj in future_ego_states]
        except:
            print("Skipping invalid future trajectory with ", total_frames_future)
            return None

        for current_t in range(total_frames_future*20):
            if current_t >= len(future_ego_states):
                break
            ego_agent = future_ego_states[current_t]
            poses_np[current_t + total_frames_past * 20 + 1, :] = [ego_agent.car_footprint.center.x,
                                                                   ego_agent.car_footprint.center.y,
                                                                   0,
                                                                   ego_agent.car_footprint.center.heading]
            shapes_np[current_t + total_frames_past * 20 + 1, :] = [ego_agent.car_footprint.width,
                                                                    ego_agent.car_footprint.length, 2]
            speeds_np[current_t + total_frames_past * 20 + 1, :] = [ego_agent.dynamic_car_state.center_velocity_2d.x,
                                                                    ego_agent.dynamic_car_state.center_velocity_2d.y]

        # for other agents
        try:
            past_tracked_obj = scenario.get_past_tracked_objects(0, total_frames_past, num_samples=total_frames_past*20)
            # past_tracked_obj is a generator
            past_tracked_obj = [each_obj for each_obj in past_tracked_obj]
        except:
            print("Skipping invalid past trajectory with ", total_frames_past)
            return None


        short = max(0, total_frames_past*20 - len(past_tracked_obj))
        for current_t in range(total_frames_past*20):
            if current_t < short:
                continue
            all_agents = past_tracked_obj[current_t-short].tracked_objects.get_agents()
            for each_agent in all_agents:
                token = each_agent.track_token
                agent_type = each_agent.tracked_object_type.value
                if selected_agent_types is not None and agent_type not in selected_agent_types:
                    continue
                if token not in agent_dic:
                    # init
                    new_dic = {'pose': np.ones([total_frames, 4]) * -1,
                               'shape': np.ones([total_frames, 3]) * -1,
                               'speed': np.ones([total_frames, 2]) * -1,
                               'type': int(agent_type),
                               'is_sdc': 0, 'to_predict': 0}
                    agent_dic[token] = new_dic
                poses_np = agent_dic[token]['pose']
                shapes_np = agent_dic[token]['shape']
                speeds_np = agent_dic[token]['speed']
                poses_np[current_t, :] = [each_agent.center.x, each_agent.center.y, 0, each_agent.center.heading]
                shapes_np[current_t, :] = [each_agent.box.width, each_agent.box.length, 2]
                speeds_np[current_t, :] = [each_agent.velocity.x, each_agent.velocity.y]

        current_tracked_obj = scenario.get_tracked_objects_at_iteration(0)
        all_agents = current_tracked_obj.tracked_objects.get_agents()
        for each_agent in all_agents:
            token = each_agent.track_token
            agent_type = each_agent.tracked_object_type.value
            if selected_agent_types is not None and agent_type not in selected_agent_types:
                continue

            if token not in agent_dic:
                # init
                new_dic = {'pose': np.ones([total_frames, 4]) * -1,
                           'shape': np.ones([total_frames, 3]) * -1,
                           'speed': np.ones([total_frames, 2]) * -1,
                           'type': int(agent_type),
                           'is_sdc': 0, 'to_predict': 0}
                agent_dic[token] = new_dic
            poses_np = agent_dic[token]['pose']
            shapes_np = agent_dic[token]['shape']
            speeds_np = agent_dic[token]['speed']
            poses_np[total_frames_past * 20, :] = [each_agent.center.x, each_agent.center.y, 0,
                                                                   each_agent.center.heading]
            shapes_np[total_frames_past * 20, :] = [each_agent.box.width, each_agent.box.length, 2]
            speeds_np[total_frames_past * 20, :] = [each_agent.velocity.x, each_agent.velocity.y]

        try:
            future_tracked_obj = scenario.get_future_tracked_objects(0, total_frames_future, num_samples=total_frames_future*20)
            # future_tracked_obj is a generator (unstable now)
            # looping generator raise assertion error:
            # next_token = row["next_token"].hex() if "next_token" in keys else None,
            # AttributeError: 'NoneType' object has no attribute 'hex'
            future_tracked_obj = [each_obj for each_obj in future_tracked_obj]
        except:
            print("Skipping invalid future trajectory with ", total_frames_future)
            return None

        # future_tracked_obj = [each_obj for t, each_obj in enumerate(future_tracked_obj)]

        for current_t in range(total_frames_future*20):
            if current_t >= len(future_tracked_obj):
                break
            all_agents = future_tracked_obj[current_t].tracked_objects.get_agents()
            for each_agent in all_agents:
                token = each_agent.track_token
                agent_type = each_agent.tracked_object_type.value
                if selected_agent_types is not None and agent_type not in selected_agent_types:
                    continue

                if token not in agent_dic:
                    # init
                    new_dic = {'pose': np.ones([total_frames, 4]) * -1,
                               'shape': np.ones([total_frames, 3]) * -1,
                               'speed': np.ones([total_frames, 2]) * -1,
                               'type': int(agent_type),
                               'is_sdc': 0, 'to_predict': 0}
                    agent_dic[token] = new_dic
                poses_np = agent_dic[token]['pose']
                shapes_np = agent_dic[token]['shape']
                speeds_np = agent_dic[token]['speed']
                poses_np[current_t + total_frames_past * 20 + 1, :] = [each_agent.center.x, each_agent.center.y, 0, each_agent.center.heading]
                shapes_np[current_t + total_frames_past * 20 + 1, :] = [each_agent.box.width, each_agent.box.length, 2]
                speeds_np[current_t + total_frames_past * 20 + 1, :] = [each_agent.velocity.x, each_agent.velocity.y]

        # total shape of agent pose is 181
        return agent_dic

    def pack_scenario_to_trafficdic(self, scenario, map_radius=MAP_RADIUS, scenario_list=None):
        traffic_dic = {}
        map_api = scenario.map_api
        self.map_api = map_api
        map_api = scenario.map_api
        # currently NuPlan only supports these map obj classes
        selected_objs = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        selected_objs += [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        selected_objs += [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK]
        selected_objs += [SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]

        traffic_light_data = scenario.get_traffic_light_status_at_iteration(0)

        green_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.GREEN
        ]
        red_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.RED
        ]

        ego_state = scenario.get_ego_state_at_iteration(0)
        all_selected_map_instances = map_api.get_proximal_map_objects(ego_state.car_footprint.center, 999999,
                                                                      selected_objs)
                                                                      
        all_selected_objs_to_render = []
        if scenario_list is None:
            all_selected_map_instances_to_render = map_api.get_proximal_map_objects(ego_state.car_footprint.center,
                                                                                    map_radius, selected_objs)
            
            for layer_name in all_selected_map_instances_to_render:
                objs_to_render = all_selected_map_instances_to_render[layer_name]
                for each_obj in objs_to_render:
                    all_selected_objs_to_render.append(each_obj.id)
            
        else:
            for each_scenario in scenario_list:
                ego_state = each_scenario.get_ego_state_at_iteration(0)
                all_selected_map_instances_to_render = map_api.get_proximal_map_objects(ego_state.car_footprint.center,
                                                                                        map_radius, selected_objs)
                for layer_name in all_selected_map_instances_to_render:
                    objs_to_render = all_selected_map_instances_to_render[layer_name]
                    for each_obj in objs_to_render:
                        all_selected_objs_to_render.append(each_obj.id)
        
        for layer_name in list(all_selected_map_instances.keys()):
        
            all_selected_obj = all_selected_map_instances[layer_name]
            map_layer_type = layer_name.value
            for selected_obj in all_selected_obj:
                map_obj_id = selected_obj.id

                # Add traffic light data.
                traffic_light_status = 0
                # status follow waymo's data coding
                if map_obj_id in green_lane_connectors:
                    traffic_light_status = 6
                elif map_obj_id in red_lane_connectors:
                    traffic_light_status = 4

                if traffic_light_status != 0:
                    total_frames = self.total_frames
                    valid_np = np.ones((total_frames, 1))
                    state_np = np.ones((total_frames, 1)) * traffic_light_status
                    traffic_dic[int(map_obj_id)] = {
                        'valid': valid_np,
                        'state': state_np
                    }
            
            #print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
        print("Traffic loaded with ", len(list(traffic_dic.keys())), " traffic elements.")
        return traffic_dic

    def pack_scenario_to_trafficdic(self, scenario, map_radius=MAP_RADIUS, scenario_list=None):
        traffic_dic = {}
        map_api = scenario.map_api
        self.map_api = map_api
        map_api = scenario.map_api
        # currently NuPlan only supports these map obj classes
        selected_objs = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        selected_objs += [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        selected_objs += [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK]
        selected_objs += [SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]

        traffic_light_data = scenario.get_traffic_light_status_at_iteration(0)

        green_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.GREEN
        ]
        red_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.RED
        ]

        ego_state = scenario.get_ego_state_at_iteration(0)
        all_selected_map_instances = map_api.get_proximal_map_objects(ego_state.car_footprint.center, 999999,
                                                                      selected_objs)
                                                                      
        all_selected_objs_to_render = []
        if scenario_list is None:
            all_selected_map_instances_to_render = map_api.get_proximal_map_objects(ego_state.car_footprint.center,
                                                                                    map_radius, selected_objs)
            
            for layer_name in all_selected_map_instances_to_render:
                objs_to_render = all_selected_map_instances_to_render[layer_name]
                for each_obj in objs_to_render:
                    all_selected_objs_to_render.append(each_obj.id)
            
        else:
            for each_scenario in scenario_list:
                ego_state = each_scenario.get_ego_state_at_iteration(0)
                all_selected_map_instances_to_render = map_api.get_proximal_map_objects(ego_state.car_footprint.center,
                                                                                        map_radius, selected_objs)
                for layer_name in all_selected_map_instances_to_render:
                    objs_to_render = all_selected_map_instances_to_render[layer_name]
                    for each_obj in objs_to_render:
                        all_selected_objs_to_render.append(each_obj.id)
        
        for layer_name in list(all_selected_map_instances.keys()):
        
            all_selected_obj = all_selected_map_instances[layer_name]
            map_layer_type = layer_name.value
            for selected_obj in all_selected_obj:
                map_obj_id = selected_obj.id

                # Add traffic light data.
                traffic_light_status = 0
                # status follow waymo's data coding
                if map_obj_id in green_lane_connectors:
                    traffic_light_status = 6
                elif map_obj_id in red_lane_connectors:
                    traffic_light_status = 4

                if traffic_light_status != 0:
                    total_frames = self.total_frames
                    valid_np = np.ones((total_frames, 1))
                    state_np = np.ones((total_frames, 1)) * traffic_light_status
                    traffic_dic[int(map_obj_id)] = {
                        'valid': valid_np,
                        'state': state_np
                    }
            
            #print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
        print("Traffic loaded with ", len(list(traffic_dic.keys())), " traffic elements.")
        return traffic_dic

    def pack_scenario_to_roaddic(self, scenario, map_radius=MAP_RADIUS, scenario_list=None):
        """
        Road types:
        LANE = 0
        INTERSECTION = 1
        STOP_LINE = 2
        TURN_STOP = 3
        CROSSWALK = 4
        DRIVABLE_AREA = 5
        YIELD = 6
        TRAFFIC_LIGHT = 7
        STOP_SIGN = 8
        EXTENDED_PUDO = 9
        SPEED_BUMP = 10
        LANE_CONNECTOR = 11
        BASELINE_PATHS = 12
        BOUNDARIES = 13
        WALKWAYS = 14
        CARPARK_AREA = 15
        PUDO = 16
        ROADBLOCK = 17
        ROADBLOCK_CONNECTOR = 18
        PRECEDENCE_AREA = 19
        """
        
        
        road_dic = {}
        traffic_dic = {}
        map_api = scenario.map_api
        self.map_api = map_api
        all_map_obj = map_api.get_available_map_objects()

        # Collect lane information, following nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils.get_neighbor_vector_map
        map_api = scenario.map_api
        # currently NuPlan only supports these map obj classes
        selected_objs = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        selected_objs += [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        selected_objs += [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK]
        selected_objs += [SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]

        # selected_objs = []
        # for each_obj in all_map_obj:
        #     if each_obj.value in [0, 11, 16, 17]:
        #         # lanes
        #         selected_objs.append(each_obj)

        traffic_light_data = scenario.get_traffic_light_status_at_iteration(0)

        green_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.GREEN
        ]
        red_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.RED
        ]

        ego_state = scenario.get_ego_state_at_iteration(0)
        all_selected_map_instances = map_api.get_proximal_map_objects(ego_state.car_footprint.center, 999999,
                                                                      selected_objs)
                                                                      
                                                                      
        all_selected_objs_to_render = []
        if scenario_list is None:
            all_selected_map_instances_to_render = map_api.get_proximal_map_objects(ego_state.car_footprint.center,
                                                                                    map_radius, selected_objs)
            
            
            for layer_name in all_selected_map_instances_to_render:
                objs_to_render = all_selected_map_instances_to_render[layer_name]
                for each_obj in objs_to_render:
                    all_selected_objs_to_render.append(each_obj.id)
            
            
        else:
            for each_scenario in scenario_list:
                ego_state = each_scenario.get_ego_state_at_iteration(0)
                all_selected_map_instances_to_render = map_api.get_proximal_map_objects(ego_state.car_footprint.center,
                                                                                        map_radius, selected_objs)
                for layer_name in all_selected_map_instances_to_render:
                    objs_to_render = all_selected_map_instances_to_render[layer_name]
                    for each_obj in objs_to_render:
                        all_selected_objs_to_render.append(each_obj.id)
        
        
        for layer_name in list(all_selected_map_instances.keys()):
        
        
            all_selected_obj = all_selected_map_instances[layer_name]
            map_layer_type = layer_name.value
            for selected_obj in all_selected_obj:
                map_obj_id = selected_obj.id
                if map_obj_id in road_dic:
                    continue
                speed_limit = 80
                has_traffic_light = -1
                incoming = []
                outgoing = []
                upper_level = []
                lower_level = []
                connector = 0
                if layer_name in [SemanticMapLayer.STOP_LINE]:
                    # PED_CROSSING = 0
                    # STOP_SIGN = 1
                    # TRAFFIC_LIGHT = 2
                    # TURN_STOP = 3
                    # YIELD = 4
                    line_x, line_y = selected_obj.polygon.exterior.coords.xy
                    # if selected_obj.stop_line_type not in [0, 1]:
                    #     continue
                elif layer_name in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
                    line_x, line_y = selected_obj.baseline_path.linestring.coords.xy
                    if selected_obj.speed_limit_mps is not None:
                        speed_limit = selected_obj.speed_limit_mps * 3600 / 1609.34  # mps(meters per second) to mph(miles per hour)
                    if selected_obj.has_traffic_lights() is not None:
                        has_traffic_light = 1 if selected_obj.has_traffic_lights() else 0
                    incoming = [int(obj.id) for obj in selected_obj.incoming_edges]
                    outgoing = [int(obj.id) for obj in selected_obj.outgoing_edges]
                    upper_level = [int(selected_obj.get_roadblock_id())]
                    connector = 1 if layer_name == SemanticMapLayer.LANE_CONNECTOR else 0
                elif layer_name in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
                    line_x, line_y = selected_obj.polygon.exterior.coords.xy
                    incoming = [int(obj.id) for obj in selected_obj.incoming_edges]
                    outgoing = [int(obj.id) for obj in selected_obj.outgoing_edges]
                    lower_level = [int(obj.id) for obj in selected_obj.interior_edges]
                    connector = 1 if layer_name == SemanticMapLayer.ROADBLOCK_CONNECTOR else 0
                else:
                    line_x, line_y = selected_obj.polygon.exterior.coords.xy

                # Add traffic light data.
                traffic_light_status = 0
                # status follow waymo's data coding
                if map_obj_id in green_lane_connectors:
                    traffic_light_status = 6
                elif map_obj_id in red_lane_connectors:
                    traffic_light_status = 4

                num_of_pts = len(line_x)
                
                
                road_xy_np = np.ones([num_of_pts, 3]) * -1
                road_dir_np = np.ones([num_of_pts, 1]) * -1
                
                
                for i in range(num_of_pts):
                    road_xy_np[i, 0] = line_x[i]
                    road_xy_np[i, 1] = line_y[i]
                    if i != 0:
                        road_dir_np[i, 0] = util.get_angle_of_a_line(pt1=[road_xy_np[i-1, 0], road_xy_np[i-1, 1]],
                                                                     pt2=[road_xy_np[i, 0], road_xy_np[i, 1]])
                
                
                new_dic = {
                    'dir': road_dir_np, 'type': int(map_layer_type), 'turning': connector,
                    'next_lanes': outgoing, 'previous_lanes': incoming,
                    'outbound': 0, 'marking': 0,
                    'vector_dir': road_dir_np, 'xyz': road_xy_np[:, :3],
                    'speed_limit': speed_limit,  # in mph,
                    'upper_level': upper_level, 'lower_level': lower_level,
                    'render': map_obj_id in all_selected_objs_to_render,
                }
                road_dic[int(map_obj_id)] = new_dic

                if traffic_light_status != 0:
                    total_frames = self.total_frames
                    valid_np = np.ones((total_frames, 1))
                    state_np = np.ones((total_frames, 1)) * traffic_light_status
                    traffic_dic[int(map_obj_id)] = {
                        'valid': valid_np,
                        'state': state_np
                    }

        # print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
        return road_dic, traffic_dic


    def get_datadic(self, scenario: AbstractScenario,
                    scenario_id,
                    process_intersection=True,
                    include_relation=True,
                    loading_prediction_relation=False,  # NuPlan 1.1 reassign scenario_ids causing problems matching 1.0 relations to 1.1
                    detect_gt_relation=False,
                    agent_to_interact_np=None, agent_only=False,
                    only_predict_interest_agents=False, filter_config={},
                    seconds_in_future=TOTAL_FRAMES_IN_FUTURE):

        skip = False
        agent_dic = self.pack_scenario_to_agentdic(scenario=scenario, total_frames_future=seconds_in_future)


        if agent_dic is None:
            return None
        # get relation as edges [[A->B], ..]
        edges = []
        edge_type = []

        if include_relation and not skip:
            if loading_prediction_relation:
                # currently only work for one pair relation visualization
                if self.gt_relation_path is not None:
                    loading_file_name = self.gt_relation_path
                    with open(loading_file_name, 'rb') as f:
                        loaded_dictionary = pickle.load(f)

                # file_to_read = open(loading_file_name, "rb")
                # loaded_dictionary = pickle.load(file_to_read)
                # file_to_read.close()
                # old version
                # old_version = False
                old_version = True
                one_pair = False
                multi_time_edges = True

                if old_version:
                    if scenario_id in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id]
                        edges = []
                        for each_info in relation:
                            if len(each_info) == 3:
                                agent_inf, agent_reactor, relation_label = each_info
                            elif len(each_info) == 4:
                                agent_inf, agent_reactor, inf_passing_frame, reactor_passing_frame = each_info
                            else:
                                assert False, f'Unknown relation format loaded {each_info}'
                            # for agent_inf, agent_reactor, relation_label in relation:
                            edges.append([agent_inf, agent_reactor, 0, 1])
                    else:
                        print("scenario_id not found in loaded dic 1:", scenario_id, list(loaded_dictionary.keys())[0])
                        # skip unrelated scenarios
                        skip = True
                elif one_pair:
                    threshold = 0.8
                    if scenario_id.encode() in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id.encode()]
                        edges = []
                        for reactor_id in relation:
                            # print("debug: ", reactor_id, relation[reactor_id])
                            labels = relation[reactor_id]['pred_inf_label']
                            agent_ids = relation[reactor_id]['pred_inf_id']
                            scores = relation[reactor_id]['pred_inf_scores']
                            for i, label in enumerate(labels):
                                if label == 1 and scores[i][1] > threshold:
                                    edges.append([agent_ids[i], reactor_id, 0, 1])
                    else:
                        print("scenario_id not found in loaded dic 2:", scenario_id.encode(),
                              list(loaded_dictionary.keys())[0])
                        skip = True
                elif multi_time_edges:
                    threshold = 0.8
                    if scenario_id.encode() in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id.encode()]
                        edges = {}
                        for reactor_id in relation:
                            for time_offset in relation[reactor_id]:
                                labels = relation[reactor_id][time_offset]['pred_inf_label']
                                agent_ids = relation[reactor_id][time_offset]['pred_inf_id']
                                scores = relation[reactor_id][time_offset]['pred_inf_scores']
                                time_offset += 11
                                for i, label in enumerate(labels):
                                    if label == 1 and scores[i][1] > threshold:
                                        if time_offset not in edges:
                                            edges[time_offset] = []
                                        # rescale the score
                                        bottom = 0.6
                                        score = (scores[i][1] - threshold) / (1 - threshold) * (1 - bottom) + bottom
                                        edges[time_offset].append([agent_ids[i], reactor_id, 0, score])
                    else:
                        print("scenario_id not found in loaded dic 3:", scenario_id.encode(),
                              list(loaded_dictionary.keys())[0])
                        skip = True
                else:
                    if scenario_id.encode() in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id.encode()]
                        agent_ids = []
                        edges = []
                        for reactor_id in relation:
                            print("relation from loaded dic: ", reactor_id, relation[reactor_id])
                            inf_ids = relation[reactor_id]['pred_inf_ids']
                            inf_scores = relation[reactor_id]['pred_inf_scores']
                            inf_ids_list = inf_ids.tolist()
                            for k, inf_id in enumerate(inf_ids_list):
                                if int(inf_id) != 0 and inf_scores[k] > 0.5:
                                    edges.append([int(inf_id), int(reactor_id), 0, 1])
                                if inf_scores[k] > 0.8:
                                    print('skipping over 0.8 scenes')
                                    skip = True

                    else:
                        print(
                            f"scenario_id not found in loaded dic: {scenario_id}. Loaded sample: {list(loaded_dictionary.keys())[0]}")
                        # skip unrelated scenarios
                        skip = True

                # inspect only on inconsistant cases
                if False:
                    edges_detected = get_relation_on_crossing(agent_dic=agent_dic,
                                                              only_prediction_agents=only_predict_interest_agents)
                    if len(edges) > 0:
                        if len(edges_detected) > 0:
                            if edges_detected[0][0] == edges[0][0] and edges_detected[0][1] == edges[0][1]:
                                print("skip duplicate edges")
                                skip = True
                    # end of inspect codes
                    else:
                        if len(edges_detected) < 1:
                            print("skip no edge detected scenario")
                            skip = True
                    if not skip:
                        print("detected edge:", edges_detected, "\nprediction edge", edges)
            elif detect_gt_relation:
                edges = get_relation_on_crossing(agent_dic=agent_dic,
                                                 only_prediction_agents=only_predict_interest_agents,
                                                 total_frame_number=self.total_frames,
                                                 to_predict=False,
                                                 agent_types=[0, 1, 2, 7], fast=True)

                form_a_tree = False
                if not only_predict_interest_agents and form_a_tree:
                    edges = form_tree_from_edges(edges)

                temp_loading_flag = True
                if temp_loading_flag and not skip:
                    for edge in edges:
                        if len(edge) != 4:
                            # [agent_id_influencer, agent_id_reactor, frame_idx_reactor_passing_cross_point, abs(frame_diff)]
                            print("invalid edge: ", edge)
                            skip = True
                            break
                        # one type per agent
                        # relation type: 1-vv, 2-vp, 3-vc, 4-others
                        agent_types = []
                        agent_id1, agent_id2, _, _ = edge
                        for agent_id in agent_dic:
                            if agent_id in [agent_id1, agent_id2]:
                                agent_types.append(agent_dic[agent_id]['type'])
                        if len(agent_types) != 2:
                            print("WARNING: Skipping an solo interactive agent scene - ", str(agent_types),
                                  str(scenario_id))
                            skip = True
                        else:
                            if agent_types[0] == 0 and agent_types[1] == 0:
                                edge_type.append(1)
                            elif 0 in agent_types and 1 in agent_types:
                                edge_type.append(2)
                            elif 0 in agent_types and 2 in agent_types:
                                edge_type.append(3)
                            else:
                                print("other type:", agent_types)
                                edge_type.append(4)

        if not agent_only:
            if self.running_mode == 1:
                if self.road_dic_path is not None:
                    loading_file_name = self.road_dic_path
                    #print(loading_file_name)
                    with open(loading_file_name, 'rb') as f:
                        road_dic = pickle.load(f)
                        #print(road_dic)
                traffic_dic = self.pack_scenario_to_trafficdic(scenario)
                route_road_ids = scenario.get_route_roadblock_ids()
                # handle '' empty string in route_road_ids
                road_ids_processed = []
                for each_id in route_road_ids:
                    if each_id != '':
                        try:
                            road_ids_processed.append(int(each_id))
                        except:
                            print(f"Invalid road id in route {each_id}")
                route_road_ids = road_ids_processed
                
            else:             
                road_dic, traffic_dic = self.pack_scenario_to_roaddic(scenario)
                route_road_ids = scenario.get_route_roadblock_ids()
                
                # handle '' empty string in route_road_ids
                road_ids_processed = []
                for each_id in route_road_ids:
                    if each_id != '':
                        try:
                            road_ids_processed.append(int(each_id))
                        except:
                            print(f"Invalid road id in route {each_id}")
                route_road_ids = road_ids_processed
                # route_road_ids = [int(each_id) for each_id in route_road_ids]
            if self.running_mode == 1:
                if self.road_dic_path is not None:
                    loading_file_name = self.road_dic_path
                    #print(loading_file_name)
                    with open(loading_file_name, 'rb') as f:
                        road_dic = pickle.load(f)
                        #print(road_dic)
                traffic_dic = self.pack_scenario_to_trafficdic(scenario)
                route_road_ids = scenario.get_route_roadblock_ids()
                # handle '' empty string in route_road_ids
                road_ids_processed = []
                for each_id in route_road_ids:
                    if each_id != '':
                        try:
                            road_ids_processed.append(int(each_id))
                        except:
                            print(f"Invalid road id in route {each_id}")
                route_road_ids = road_ids_processed
                
            else:             
                road_dic, traffic_dic = self.pack_scenario_to_roaddic(scenario)
                route_road_ids = scenario.get_route_roadblock_ids()
                
                # handle '' empty string in route_road_ids
                road_ids_processed = []
                for each_id in route_road_ids:
                    if each_id != '':
                        try:
                            road_ids_processed.append(int(each_id))
                        except:
                            print(f"Invalid road id in route {each_id}")
                route_road_ids = road_ids_processed
                # route_road_ids = [int(each_id) for each_id in route_road_ids]
        else:
            road_dic = {}
            traffic_dic = {}

        if road_dic is None or traffic_dic is None:
            return None

        if len(route_road_ids) == 0:
            print("Invalid route given, Skipping")
            return None
        
        
        # mark still agents is the past
        for agent_id in agent_dic:
            is_still = False
            for i in range(10):
                if agent_dic[agent_id]['pose'][i, 0] == -1:
                    continue
                if util.euclidean_distance(agent_dic[agent_id]['pose'][i, :2],
                                      agent_dic[agent_id]['pose'][10, :2]) < 1:
                    is_still = True
            agent_dic[agent_id]['still_in_past'] = is_still
        
        
        data_to_return = {
            "road": road_dic,
            "agent": agent_dic,
            "traffic_light": traffic_dic,
        }

        if 'ego' not in agent_dic:
            print("no ego and skip")
            skip = True
        elif agent_dic['ego']['pose'][20][0] == -1:
            print("invalid ego pose")
            skip = True

        # sanity check
        if agent_dic is None or road_dic is None or traffic_dic is None:
            print("Invalid Scenario Loaded: ", agent_dic is None, road_dic is None, traffic_dic is None)
            skip = True

        # category = classify_scenario(data_to_return)
        data_to_return["category"] = 1
        data_to_return['scenario'] = scenario_id

        data_to_return['edges'] = edges
        data_to_return['skip'] = skip
        data_to_return['edge_type'] = edge_type

        data_to_return['route'] = route_road_ids

        # if process_intersection:
        #     intersection_dic = get_intersection(data_to_return)
        #     data_to_return["intersection"] = intersection_dic

        return data_to_return
