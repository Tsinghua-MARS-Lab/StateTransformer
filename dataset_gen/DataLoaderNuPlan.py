import copy
import os
from pathlib import Path
import tempfile
import hydra
import time

"""
This code is currently tested on nuPlan devkit v1.0.0
"""

FILE_TO_START = 0
SCENE_TO_START = 0  # nuplan 1-17 unreasonable stuck by ped nearby  2-62 wrong route in dataset
# 107 for nudging  # 38 for turning large intersection failure
SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD = 20
SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD = 0.1

TOTAL_FRAMES_IN_FUTURE = 8
FREQUENCY = 0.05

MAP_RADIUS = 100

import math
import os
import numpy as np
import pickle
from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import extract_tracked_objects
from nuplan.database.nuplan_db.nuplan_db_utils import (
    SensorDataSource,
)

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)

from nuplan.database.nuplan_db import nuplan_scenario_queries
import dataset_gen.utils as util

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
                 gt_relation_path=None, cpus=10, db=None, data_path=None, road_dic_path=None, running_mode=None,
                 filter_scenario=None, keep_future_steps=False):

        NUPLAN_MAP_VERSION = 'nuplan-maps-v1.0'
        if data_path is None:
            NUPLAN_DATA_ROOT = "/Users/qiaosun/nuplan/dataset"
            NUPLAN_MAPS_ROOT = "/Users/qiaosun/nuplan/dataset/maps"
            NUPLAN_DB_FILES = "/Users/qiaosun/nuplan/dataset/nuplan-v1.0/public_set_boston_train/"
        else:
            NUPLAN_DATA_ROOT = data_path['NUPLAN_DATA_ROOT']
            NUPLAN_MAPS_ROOT = data_path['NUPLAN_MAPS_ROOT']
            NUPLAN_DB_FILES = data_path['NUPLAN_DB_FILES']

        files_names = [os.path.join(NUPLAN_DB_FILES, each_path) for each_path in os.listdir(NUPLAN_DB_FILES) if
                       each_path[0] != '.']
        for each_path in files_names:
            if '.db' not in each_path:
                print('ERROR', each_path)
        self.data_path = data_path
        files_names = sorted(files_names)

        if file_to_start is not None and max_file_number is None:
            files_names = files_names[file_to_start:]
        if file_to_start is not None and max_file_number is not None:
            files_names = files_names[file_to_start:file_to_start + max_file_number]

        self.global_file_names = files_names
        self.filter_scenario = filter_scenario

        self.map_api = None
        self.total_frames = None
        # self.loaded_playback = None
        self.running_mode = running_mode
        self.road_dic_path = road_dic_path
        # self.traffic_dic_path = traffic_dic_path
        self.gt_relation_path = gt_relation_path
        self.timestamp = None
        self.road_dic_mem = None
        self.route_idx_mem = None
        if db is None:
            db = NuPlanDBWrapper(
                data_root=NUPLAN_DATA_ROOT,
                map_root=NUPLAN_MAPS_ROOT,
                db_files=files_names,
                map_version=NUPLAN_MAP_VERSION,
                max_workers=cpus
            )

        self.current_dataset = db

        self.total_file_num = len(self.current_dataset.log_dbs)
        self.current_file_index = FILE_TO_START
        if file_to_start is not None and file_to_start >= 0:
            # self.current_file_index = file_to_start
            self.current_file_index = 0

        self.file_names = [nuplanDB.name for nuplanDB in self.current_dataset.log_dbs]
        if self.current_file_index >= self.total_file_num:
            self.current_file_total_scenario = 0
            self.end = True
            print("Init with index out of max file number ", self.current_file_index, self.total_file_num)
        else:
            self.current_file_total_scenario = len(db.log_dbs[self.current_file_index].scenario_tag)
            self.end = False

        self.max_file_number = max_file_number
        self.start_file_number = self.current_file_index
        if scenario_to_start is not None and scenario_to_start >= 0:
            self.current_scenario_index = scenario_to_start - 1
        else:
            self.current_scenario_index = SCENE_TO_START - 1
        self.cache_previous_token_time_step = None
        self.keep_future_steps = keep_future_steps

    def load_new_file(self, first_file=False):
        if self.max_file_number is not None and self.current_file_index >= (
                self.start_file_number + self.max_file_number):
            self.end = True
            return
        if self.current_file_index < self.total_file_num:
            print("Loading file from: ", self.current_dataset.log_dbs[self.current_file_index]._load_path,
                  " with index of ", self.current_file_index)
            # self.current_file_index += 1
            self.current_file_total_scenario = len(self.current_dataset.log_dbs[self.current_file_index].scenario_tag)
            if not first_file:
                self.current_scenario_index = 0
            print(" with ", self.current_file_total_scenario, " scenarios and current is ", self.current_scenario_index)
        else:
            self.end = True

    def get_next(self, 
                 sample_interval,  
                 agent_only=False, 
                 seconds_in_future=TOTAL_FRAMES_IN_FUTURE,
                 map_name=None,
                 scenarios_to_keep=None,
                 filter_still=False,
                 sensor_blob_path=None,
                 sensor_meta_path=None):
        """
        :param sample_interval:
        :param agent_only:
        :param seconds_in_future:
        :param map_name:
        :param scenarios_to_keep: past scenario tokens for val14 to generate val14 indices
        :return:
        """
        new_files_loaded = False

        self.current_scenario_index += sample_interval

        if not self.current_scenario_index < self.current_file_total_scenario:
            self.current_file_index += 1
            self.load_new_file()
            new_files_loaded = True
            self.road_dic_mem = None
            self.route_idx_mem = None

        if self.end:
            return None, new_files_loaded

        log_db = self.current_dataset.log_dbs[self.current_file_index]
        lidar_token = None
        while self.current_scenario_index < self.current_file_total_scenario:
            try:
                lidar_token = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[
                    self.current_scenario_index].lidar_pc_token
                lidar_pc = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[
                    self.current_scenario_index].lidar_pc
                break
            except:
                print(
                    f"Failed to fetch lidar pc token with exceptions!!!!!!!!!!! {self.current_file_index}/{self.total_file_num} {self.current_scenario_index}/{self.current_file_total_scenario}")
                self.current_scenario_index += 1
                if not self.current_scenario_index < self.current_file_total_scenario:
                    self.current_file_index += 1
                    self.load_new_file()
                    new_files_loaded = True
                if self.end:
                    return None, new_files_loaded

        if lidar_token is None:
            print(
                f"scenario loaded failed: {self.current_file_index}/{self.total_file_num} {self.current_scenario_index}/{self.current_file_total_scenario}")
            self.end = True
            return None, True

        sensor_data_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', '')
        # lidar_token_timestamp = nuplan_scenario_queries.get_sensor_data_token_timestamp_from_db(log_db.load_path,
        #                                                                                         sensor_data_source,
        #                                                                                         lidar_token)
        lidar_token_timestamp = lidar_pc.timestamp

        scenario_type = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[self.current_scenario_index].type
        if self.filter_scenario is not None and scenario_type not in self.filter_scenario:
            return None, True
        if map_name is not None and log_db.map_name != map_name:
            return None, True
        scenario = get_default_scenario_from_token(log_db, lidar_token, lidar_token_timestamp)
        # filter scenarios by token id
        scenario_id = scenario.token
        self.timestamp = lidar_token_timestamp
        # total_frames are 150
        total_frames = scenario.get_number_of_iterations()
        current_frame_idx = round(20 * (lidar_token_timestamp - log_db.lidar_pc[0].timestamp) / 1e6)
        if scenarios_to_keep is not None and scenario_id not in scenarios_to_keep:
            return None, True
        if self.keep_future_steps:
            # return a list of dictionary instead of one
            dic_list_to_returen = []
            if total_frames < 150:
                print('test total iterations: ', total_frames)
            lidar_pc_in_loop = lidar_pc
            for i in range(0, total_frames, 10):
                # i is in 10hz from iterations
                current_frame_idx_in_loop = round(20 * (lidar_pc_in_loop.timestamp - log_db.lidar_pc[0].timestamp) / 1e6)
                if current_frame_idx_in_loop >= self.current_file_total_scenario:
                    # will skip if the future is not enough for 15s
                    self.current_file_index += 1
                    self.load_new_file()
                    new_files_loaded = True
                if self.end:
                    return None, new_files_loaded
                # lidar_token_in_loop = self.current_dataset.log_dbs[self.current_file_index].scenario_tag[self.current_scenario_index + i * 2].lidar_pc_token
                # time_stamp_in_loop_from = nuplan_scenario_queries.get_sensor_data_token_timestamp_from_db(log_db.load_path,
                #                                                                                            sensor_data_source,
                #                                                                                            lidar_token_in_loop)
                scenario_in_loop = get_default_scenario_from_token(log_db, lidar_pc_in_loop.token, lidar_pc_in_loop.timestamp)
                data_to_return = self.get_datadic(scenario=scenario_in_loop,
                                                  scenario_id=scenario_id,
                                                  agent_only=agent_only,
                                                  seconds_in_future=seconds_in_future,
                                                  filter_still=filter_still)
                if data_to_return is None:
                    continue
                data_to_return['scenario_type'] = scenario_type
                mission_goal_state = scenario.get_mission_goal()
                expert_goal_state = scenario.get_expert_goal_state()
                data_to_return['mission_goal'] = []
                data_to_return['expert_goal'] = []
                if mission_goal_state is None:
                    print('###### ERROR: mission goal missing for val set, skipping ######')
                    return None, new_files_loaded
                else:
                    data_to_return['mission_goal'] = [mission_goal_state.point.x, mission_goal_state.point.y, 0,
                                                      mission_goal_state.heading]
                if expert_goal_state is not None:
                    data_to_return['expert_goal'] = [expert_goal_state.point.x, expert_goal_state.point.y, 0,
                                                     expert_goal_state.heading]

                if sensor_meta_path is not None:
                    # using meta should be faster than looping the whole sensor path
                    from nuplan.planning.simulation.observation.observation_type import CameraChannel, LidarChannel
                    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
                    from typing import cast
                    scenario._sensor_root = sensor_blob_path
                    # local_store, remote_store = scenario._create_blob_store_if_needed()
                    retrieved_images = get_images_from_lidar_tokens(
                        scenario._log_file, [lidar_pc_in_loop.token], [cast(str, channel.value) for channel in
                                                                       CameraChannel]
                    )
                    data_to_return['images_path'] = [each.filename_jpg for each in retrieved_images]

                    # check if image exist at all
                    # read plain text utf8 file
                    file_path = sensor_meta_path
                    with open(file_path, 'r') as f:
                        meta_folders = f.read()
                        for each in data_to_return['images_path']:
                            if each.split('/')[0] not in meta_folders:
                                print('###### ERROR: image missing for val set, skipping ###### ', each)
                                return None, new_files_loaded

                elif sensor_blob_path is not None:
                    from nuplan.planning.simulation.observation.observation_type import CameraChannel, LidarChannel
                    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
                    from typing import cast
                    scenario._sensor_root = sensor_blob_path
                    # local_store, remote_store = scenario._create_blob_store_if_needed()
                    retrieved_images = get_images_from_lidar_tokens(
                        scenario._log_file, [lidar_pc_in_loop.token], [cast(str, channel.value) for channel in CameraChannel]
                    )
                    data_to_return['images_path'] = [each.filename_jpg for each in retrieved_images]

                    # check if image exist at all
                    for each in data_to_return['images_path']:
                        image_path = os.path.join(sensor_blob_path, each)
                        if not os.path.exists(image_path):
                            print('###### ERROR: image missing for val set, skipping ###### ', image_path)
                            return None, new_files_loaded

                data_to_return['dataset'] = 'NuPlan'
                data_to_return['lidar_pc_tokens'] = lidar_pc_in_loop.lidar_token
                data_to_return['frame_id'] = current_frame_idx_in_loop
                data_to_return['timestamp'] = lidar_pc_in_loop.timestamp
                data_to_return['file_name'] = log_db.log_name
                data_to_return['map'] = log_db.map_name
                data_to_return['scenario_id'] = scenario_in_loop.token
                data_to_return['t0_frame_id'] = current_frame_idx  # this should never be None
                dic_list_to_returen.append(data_to_return)
                for _ in range(20):
                    # step 1s
                    lidar_pc_in_loop = lidar_pc_in_loop.next
            return dic_list_to_returen, new_files_loaded

        data_to_return = self.get_datadic(scenario=scenario,
                                          scenario_id=scenario_id, 
                                          agent_only=agent_only,
                                          seconds_in_future=seconds_in_future,
                                          filter_still=filter_still)
        if data_to_return is None:
            data_to_return = {'skip': True}
            return data_to_return, new_files_loaded

        data_to_return['scenario_type'] = scenario_type
        mission_goal_state = scenario.get_mission_goal()
        expert_goal_state = scenario.get_expert_goal_state()
        data_to_return['mission_goal'] = []
        data_to_return['expert_goal'] = []
        if expert_goal_state is not None:
            data_to_return['expert_goal'] = [expert_goal_state.point.x, expert_goal_state.point.y, 0, expert_goal_state.heading]
        if mission_goal_state is not None:
            data_to_return['mission_goal'] = [mission_goal_state.point.x, mission_goal_state.point.y, 0, mission_goal_state.heading]

        if sensor_meta_path is not None:
            # using meta should be faster than looping the whole sensor path
            from nuplan.planning.simulation.observation.observation_type import CameraChannel, LidarChannel
            from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
            from typing import cast
            scenario._sensor_root = sensor_blob_path
            # local_store, remote_store = scenario._create_blob_store_if_needed()
            retrieved_images = get_images_from_lidar_tokens(
                scenario._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in
                                                               CameraChannel]
            )
            data_to_return['images_path'] = [each.filename_jpg for each in retrieved_images]

            # check if image exist at all
            # read plain text utf8 file
            file_path = sensor_meta_path
            with open(file_path, 'r') as f:
                meta_folders = f.read()
                for each in data_to_return['images_path']:
                    if each.split('/')[0] not in meta_folders:
                        return None, new_files_loaded
        elif sensor_blob_path is not None:
            from nuplan.planning.simulation.observation.observation_type import CameraChannel, LidarChannel
            from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
            from typing import cast
            scenario._sensor_root = sensor_blob_path
            # local_store, remote_store = scenario._create_blob_store_if_needed()
            retrieved_images = get_images_from_lidar_tokens(
                scenario._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in CameraChannel]
            )
            data_to_return['images_path'] = [each.filename_jpg for each in retrieved_images]

            # check if image exist at all
            for each in data_to_return['images_path']:
                image_path = os.path.join(sensor_blob_path, each)
                if not os.path.exists(image_path):
                    # print('###### ERROR: image missing for training set, skipping ###### ', image_path)
                    return None, new_files_loaded

        data_to_return['dataset'] = 'NuPlan'
        data_to_return['lidar_pc_tokens'] = log_db.lidar_pc
        data_to_return['frame_id'] = current_frame_idx
        data_to_return['timestamp'] = lidar_token_timestamp
        data_to_return['file_name'] = log_db.log_name
        data_to_return['map'] = log_db.map_name
        data_to_return['scenario_id'] = scenario_id
        data_to_return['t0_frame_id'] = -1
        return data_to_return, new_files_loaded

    def get_next_file(self, specify_file_index=None, map_name=None, agent_only=False, sample_interval=2):
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
        if map_name is not None and log_db.map_name != map_name:
            print('unmatched map name(loaded, asked): ', log_db.map_name, map_name)
            return None
        total_frames = len(log_db.lidar_pc)
        first_lidar_pc = log_db.lidar_pc[0]
        last_lidar_pc = log_db.lidar_pc[-1]
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
                   'speed': np.ones([total_frames, 3]) * -1,  # [v, a, angular_v]
                   'type': 0,
                   'is_sdc': 0, 'to_predict': 0,
                   'starting_frame': 0,
                   'ending_frame': -1,}
        # pack ego
        agent_dic['ego'] = copy.deepcopy(new_dic)
        poses_np = agent_dic['ego']['pose']
        shapes_np = agent_dic['ego']['shape']
        speed_np = agent_dic['ego']['speed']  # [v, a, angular_v]
        agent_dic['ego']['type'] = 7
        # get ego
        current_pc = first_lidar_pc
        current_ego_pose = current_pc.ego_pose
        for i in range(total_frames):
            poses_np[i, :] = [current_ego_pose.x, current_ego_pose.y, 0,
                              math.atan2(2.0 * (
                                          current_ego_pose.qw * current_ego_pose.qz + current_ego_pose.qx * current_ego_pose.qy),
                                         current_ego_pose.qw * current_ego_pose.qw + current_ego_pose.qx * current_ego_pose.qx - current_ego_pose.qy * current_ego_pose.qy - current_ego_pose.qz * current_ego_pose.qz)]
            shapes_np[i, :] = [any_ego_state.car_footprint.width,
                               any_ego_state.car_footprint.length, 2]
            speed_np[i, :] = [math.sqrt(current_ego_pose.vx ** 2 + current_ego_pose.vy ** 2 + current_ego_pose.vz ** 2),
                              math.sqrt(current_ego_pose.acceleration_x ** 2 + current_ego_pose.acceleration_y ** 2 + current_ego_pose.acceleration_z ** 2),
                              math.sqrt(current_ego_pose.angular_rate_x ** 2 + current_ego_pose.angular_rate_y ** 2 + current_ego_pose.angular_rate_z ** 2)]
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
                    new_dic = {'pose': np.ones([total_frames, 4], dtype=np.float32) * -1,
                               'shape': np.ones([total_frames, 3], dtype=np.float32) * -1,
                               'type': int(agent_type),
                               'is_sdc': 0, 'to_predict': 0,
                               'starting_frame': i,
                               'ending_frame': -1}
                    agent_dic[token] = new_dic
                poses_np = agent_dic[token]['pose']
                shapes_np = agent_dic[token]['shape']
                poses_np[i, :] = [each_agent.center.x, each_agent.center.y, 0, each_agent.center.heading]
                shapes_np[i, :] = [each_agent.box.width, each_agent.box.length, 2]
                agent_dic[token]['ending_frame'] = i
            if current_pc != last_lidar_pc:
                current_pc = current_pc.next
                current_ego_pose = current_pc.ego_pose

        # trim agent_dic
        for key in agent_dic.keys():
            """
            trim unused frames save disk space for almost 10x (used 0.1 percent), also loading/saving pickles 10x faster
            """
            if key == 'ego':
                # nothing to trim for ego
                continue
            starting_frame = agent_dic[key]['starting_frame']
            ending_frame = agent_dic[key]['ending_frame']
            if ending_frame == -1:
                agent_dic[key]['pose'] = agent_dic[key]['pose'][starting_frame:, :]
                agent_dic[key]['shape'] = agent_dic[key]['shape'][starting_frame:, :]
            elif ending_frame <= 0 or ending_frame <= starting_frame:
                if ending_frame == starting_frame:
                    # skip agent with only one valid frame
                    pass
                else:
                    print('warning: illegal ending frame: ', agent_dic[key], ending_frame, starting_frame)
            else:
                agent_dic[key]['pose'] = agent_dic[key]['pose'][starting_frame:ending_frame, :]
                agent_dic[key]['shape'] = agent_dic[key]['shape'][starting_frame:ending_frame, :]

        # convert to float16 to save disk space: ERROR: float16 does not have enough space for pose
        for key in agent_dic.keys():
            # change 20hz into 10hz to save disk space
            agent_dic[key]['pose'] = agent_dic[key]['pose'][::sample_interval, :].astype(np.float32)
            agent_dic[key]['shape'] = agent_dic[key]['shape'][::sample_interval, :].astype(np.float16)
            if key == 'ego':
                agent_dic[key]['speed'] = agent_dic[key]['speed'][::sample_interval, :].astype(np.float32)
        skip = False
        if not agent_only:
            road_dic = self.pack_scenario_to_roaddic(starting_scenario, map_radius=100,
                                                     scenario_list=scenario_list)
            traffic_dic = self.pack_scenario_to_trafficdic(starting_scenario, map_radius=100,
                                                           scenario_list=scenario_list)
            data_to_return = {
                "road": road_dic,
                "agent": agent_dic,
                "traffic_light": traffic_dic,
            }
            # sanity check
            if agent_dic is None or road_dic is None or traffic_dic is None:
                print("Invalid Scenario Loaded: ", agent_dic is None, road_dic is None, traffic_dic is None)
                skip = True
            # loop route road ids from all scenarios in this file
            route_road_ids = []
            log_db = self.current_dataset.log_dbs[file_index]
            sensor_data_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', '')
            for each_scenario_tag in log_db.scenario_tag:
                # fetch lidar token (as time stamp) from scenario tag
                each_lidar_token = each_scenario_tag.lidar_pc_token
                # get scenario from lidar_token
                lidar_token_timestamp = nuplan_scenario_queries.get_sensor_data_token_timestamp_from_db(log_db.load_path,
                                                                                                        sensor_data_source,
                                                                                                        each_lidar_token)
                scenario = get_default_scenario_from_token(log_db, each_lidar_token, lidar_token_timestamp)
                route_road_ids += scenario.get_route_roadblock_ids()

            route_road_ids = list(set(route_road_ids))
            # route_road_ids = starting_scenario.get_route_roadblock_ids()

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
        else:
            # sanity check
            if agent_dic is None:
                print("Invalid Scenario Loaded: ", agent_dic is None)
                skip = True
            data_to_return = {
                "agent": agent_dic,
                'scenario': f'{self.file_names[file_index]}',
                'skip': skip,
            }
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

        agent_dic = {'ego': new_dic}

        # pack ego
        poses_np = agent_dic['ego']['pose']
        shapes_np = agent_dic['ego']['shape']
        speeds_np = agent_dic['ego']['speed']
        
        # ego additional features
        speed1d_np = np.ones([total_frames_past * 20 + 1, 1]) * -1
        acceleration_np = np.ones([total_frames_past * 20 + 1, 2]) * -1
        acceleration1d_np = np.ones([total_frames_past * 20 + 1, 1]) * -1
        angular_vel_np = np.ones([total_frames_past * 20 + 1, 1]) * -1  # rad/s
        angular_acc_np = np.ones([total_frames_past * 20 + 1, 1]) * -1  # rad/s^2
        tire_steering_rate_np = np.ones([total_frames_past * 20 + 1, 1]) * -1  # rad/s
        
        # past
        try:
            past_ego_states = scenario.get_ego_past_trajectory(0, total_frames_past, num_samples=total_frames_past * 20)
            past_ego_states = [each_obj for each_obj in past_ego_states]
        except:
            # print("Skipping invalid past trajectory with ", total_frames_past)
            return None

        short = max(0, total_frames_past * 20 - len(past_ego_states))
        for current_t in range(total_frames_past * 20):
            if current_t < short:
                continue
            ego_agent = past_ego_states[current_t - short]
            poses_np[current_t, :] = [ego_agent.car_footprint.center.x, ego_agent.car_footprint.center.y,
                                      0, ego_agent.car_footprint.center.heading]
            shapes_np[current_t, :] = [ego_agent.car_footprint.width, ego_agent.car_footprint.length, 2]
            speeds_np[current_t, :] = [ego_agent.dynamic_car_state.center_velocity_2d.x,
                                       ego_agent.dynamic_car_state.center_velocity_2d.y]

            # additional features
            speed1d_np[current_t, :] = [ego_agent.dynamic_car_state.speed]
            acceleration_np[current_t, :] = [ego_agent.dynamic_car_state.center_acceleration_2d.x,
                                             ego_agent.dynamic_car_state.center_acceleration_2d.y]
            acceleration1d_np[current_t, :] = [ego_agent.dynamic_car_state.acceleration]
            angular_vel_np[current_t, :] = [ego_agent.dynamic_car_state.angular_velocity]
            angular_acc_np[current_t, :] = [ego_agent.dynamic_car_state.angular_acceleration]
            tire_steering_rate_np[current_t, :] = [ego_agent.dynamic_car_state.tire_steering_rate]

        current_ego_state = scenario.get_ego_state_at_iteration(0)
        poses_np[total_frames_past * 20, :] = [current_ego_state.car_footprint.center.x,
                                               current_ego_state.car_footprint.center.y, 0,
                                               current_ego_state.car_footprint.center.heading]
        shapes_np[total_frames_past * 20, :] = [current_ego_state.car_footprint.width,
                                                current_ego_state.car_footprint.length, 2]
        speeds_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.center_velocity_2d.x,
                                                current_ego_state.dynamic_car_state.center_velocity_2d.y]

        # additional features
        speed1d_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.speed]
        acceleration_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.center_acceleration_2d.x,
                                                      current_ego_state.dynamic_car_state.center_acceleration_2d.y]
        acceleration1d_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.acceleration]
        angular_vel_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.angular_velocity]
        angular_acc_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.angular_acceleration]
        tire_steering_rate_np[total_frames_past * 20, :] = [current_ego_state.dynamic_car_state.tire_steering_rate]

        try:
            future_ego_states = scenario.get_ego_future_trajectory(0, total_frames_future,
                                                                   num_samples=total_frames_future * 20)
            future_ego_states = [each_obj for each_obj in future_ego_states]
        except:
            # print("Skipping invalid future trajectory with ", total_frames_future)
            return None

        for current_t in range(total_frames_future * 20):
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
            past_tracked_obj = scenario.get_past_tracked_objects(0, total_frames_past,
                                                                 num_samples=total_frames_past * 20)
            # past_tracked_obj is a generator
            past_tracked_obj = [each_obj for each_obj in past_tracked_obj]
        except:
            # print("Skipping invalid past trajectory with ", total_frames_past)
            return None

        short = max(0, total_frames_past * 20 - len(past_tracked_obj))
        for current_t in range(total_frames_past * 20):
            if current_t < short:
                continue
            all_agents = past_tracked_obj[current_t - short].tracked_objects.get_agents()
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
            future_tracked_obj = scenario.get_future_tracked_objects(0, total_frames_future,
                                                                     num_samples=total_frames_future * 20)
            # future_tracked_obj is a generator (unstable now)
            # looping generator raise assertion error:
            # next_token = row["next_token"].hex() if "next_token" in keys else None,
            # AttributeError: 'NoneType' object has no attribute 'hex'
            future_tracked_obj = [each_obj for each_obj in future_tracked_obj]
        except:
            # print("Skipping invalid future trajectory with ", total_frames_future)
            return None

        # future_tracked_obj = [each_obj for t, each_obj in enumerate(future_tracked_obj)]

        for current_t in range(total_frames_future * 20):
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
                poses_np[current_t + total_frames_past * 20 + 1, :] = [each_agent.center.x, each_agent.center.y, 0,
                                                                       each_agent.center.heading]
                shapes_np[current_t + total_frames_past * 20 + 1, :] = [each_agent.box.width, each_agent.box.length, 2]
                speeds_np[current_t + total_frames_past * 20 + 1, :] = [each_agent.velocity.x, each_agent.velocity.y]

        # clean agents invalid at current frame
        agent_to_delete = []
        for each_agent in agent_dic:
            if agent_dic[each_agent]['pose'][40, 0] == -1 and each_agent != 'ego':
                agent_to_delete.append(each_agent)
        for each_agent in agent_to_delete:
            del agent_dic[each_agent]

        # check agent number
        if len(list(agent_dic.keys())) > 200:
            # can compress about 40-60 peds
            print("Too many agents in this scenario!!!!!!!!!!!!!!!!")
            print("Before: ", len(list(agent_dic.keys())))
            agent_num_counter = [0] * 8
            from scipy.cluster.hierarchy import ward, fcluster
            from scipy.spatial.distance import pdist
            X = []
            ids = []
            for each_agent in agent_dic:
                agent_num_counter[agent_dic[each_agent]['type']] += 1
                if agent_dic[each_agent]['type'] == 1:
                    X.append(agent_dic[each_agent]['pose'][40, :2].tolist())
                    ids.append(each_agent)
            Z = ward(pdist(X))
            # cluster and keep one agent in each cluster
            cluster_ids = fcluster(Z, 1, criterion='distance')
            prev_ids = []
            for i in range(len(cluster_ids)):
                if cluster_ids[i] not in prev_ids:
                    prev_ids.append(cluster_ids[i])
                else:
                    agent_dic.pop(ids[i])
            print('#########################')
            print('inspect number: ', agent_num_counter, 'total after: ', len(list(agent_dic.keys())))
            print('#########################')

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
        yellow_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.YELLOW
        ]
        unknown_lane_connectors = [
            str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.UNKNOWN
        ]

        ego_state = scenario.get_ego_state_at_iteration(0)
        all_selected_map_instances = map_api.get_proximal_map_objects(ego_state.car_footprint.center, 1e8,
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
                traffic_light_status = -1
                # status follow waymo's data coding
                if map_obj_id in green_lane_connectors:
                    traffic_light_status = 0
                elif map_obj_id in red_lane_connectors:
                    traffic_light_status = 1
                elif map_obj_id in yellow_lane_connectors:
                    traffic_light_status = 2
                elif map_obj_id in unknown_lane_connectors:
                    traffic_light_status = 3

                if traffic_light_status != -1:
                    traffic_dic[int(map_obj_id)] = {
                        'state': traffic_light_status
                    }

            # print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
        # print("Traffic loaded with ", len(list(traffic_dic.keys())), " traffic elements.")
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


        ego_state = scenario.get_ego_state_at_iteration(0)
        all_selected_map_instances = map_api.get_proximal_map_objects(ego_state.car_footprint.center, 1e8,
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
                if int(map_obj_id) in road_dic:
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

                num_of_pts = len(line_x)

                road_xy_np = np.ones([num_of_pts, 3]) * -1
                road_dir_np = np.ones([num_of_pts, 1]) * -1

                for i in range(num_of_pts):
                    road_xy_np[i, 0] = line_x[i]
                    road_xy_np[i, 1] = line_y[i]
                    if i != 0:
                        road_dir_np[i, 0] = util.get_angle_of_a_line(pt1=[road_xy_np[i - 1, 0], road_xy_np[i - 1, 1]],
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


        # print("Road loaded with ", len(list(road_dic.keys())), " road elements.")
        return road_dic

    def get_datadic(self, scenario: AbstractScenario,
                    scenario_id,
                    include_relation=False,
                    loading_prediction_relation=False,
                    agent_only=False,
                    seconds_in_future=TOTAL_FRAMES_IN_FUTURE,
                    routes_per_file=False,
                    include_intentions=True,
                    include_navigation=True,
                    filter_still=False,) -> dict:

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

        if not agent_only:
            if self.running_mode == 1:
                if self.road_dic_mem is None:
                    self.road_dic_mem = self.pack_scenario_to_roaddic(scenario, map_radius=1e2)
                traffic_dic = self.pack_scenario_to_trafficdic(scenario)
            else:
                if self.road_dic_mem is None:
                    self.road_dic_mem = self.pack_scenario_to_roaddic(scenario, map_radius=1e2)
                if routes_per_file and self.route_idx_mem is None and self.current_dataset is not None:
                     # loop route road ids from all scenarios in this file
                    route_road_ids = []
                    log_db = self.current_dataset.log_dbs[self.current_file_index]
                    sensor_data_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', '')
                    for each_scenario_tag in log_db.scenario_tag:
                        # fetch lidar token (as time stamp) from scenario tag
                        each_lidar_token = each_scenario_tag.lidar_pc_token
                        # get scenario from lidar_token
                        lidar_token_timestamp = nuplan_scenario_queries.get_sensor_data_token_timestamp_from_db(
                            log_db.load_path,
                            sensor_data_source,
                            each_lidar_token)
                        default_scenario = get_default_scenario_from_token(log_db, each_lidar_token, lidar_token_timestamp)
                        route_road_ids += default_scenario.get_route_roadblock_ids()
                    route_road_ids = list(set(route_road_ids))
                    road_ids_processed = list()
                    for each_id in route_road_ids:
                        if each_id != '':
                            try:
                                road_ids_processed.append(int(each_id))
                            except:
                                print(f"Invalid road id in route {each_id}")
                    self.route_idx_mem = road_ids_processed
                traffic_dic = self.pack_scenario_to_trafficdic(scenario, map_radius=200)
           
        else:
            road_dic = {}
            traffic_dic = {}

        # mark still agents is the past
        # for agent_id in agent_dic:
        #     is_still = False
        #     for i in range(10):
        #         if agent_dic[agent_id]['pose'][i, 0] == -1:
        #             continue
        #         if util.euclidean_distance(agent_dic[agent_id]['pose'][i, :2],
        #                                    agent_dic[agent_id]['pose'][10, :2]) < 1:
        #             is_still = True
        #     agent_dic[agent_id]['still_in_past'] = is_still

        # check if ego is still
        if filter_still and 'ego' in agent_dic and abs(agent_dic['ego']['pose'][0, 0] - agent_dic['ego']['pose'][-1, 0]) < 0.5 and abs(agent_dic['ego']['pose'][0, 1] - agent_dic['ego']['pose'][-1, 1]) < 0.5:
            # print('skipping still ego scenario')
            return None

        if self.road_dic_mem is None or traffic_dic is None:
            return None

        # check traffic dic
        filtered_traffic_dic = {}
        for each_id in traffic_dic:
            if each_id not in self.road_dic_mem:
                print('invalid traffic light id: ', each_id)
            else:
                filtered_traffic_dic[each_id] = traffic_dic[each_id]

        data_to_return = {
            "road": self.road_dic_mem,
            "agent": agent_dic,
            "traffic_light": traffic_dic,
        }

        if not routes_per_file or self.current_dataset is None:
            # no db mode, fetch current scenario route ids
            try:
                route_road_ids = scenario.get_route_roadblock_ids()
            except:
                print("Invalid scenario, cannot get route!, skipping")
                return None
            route_road_ids = list(set(route_road_ids))
            route_ids_processed = []
            for each_id in route_road_ids:
                if each_id != '':
                    try:
                        route_ids_processed.append(int(each_id))
                    except:
                        print(f"Invalid road id in route {each_id}")
            data_to_return['route'] = route_ids_processed
        else:
            if len(self.route_idx_mem) == 0:
                print("Invalid route given, Skipping")
                return None
            else:
                data_to_return['route'] = self.route_idx_mem

        if include_intentions:
            # WARNING: this only works for NuPlan (since using index of 40 as current frame)
            # get constant velocity position
            data_to_return['intentions'] = []
            ego_poses = copy.deepcopy(agent_dic['ego']['pose'])  # in 20Hz, past 2s, future 15s, 341 frames in total
            # i = 40 + 10  # current pose in 2s + 0.5s
            # for i in range(2*20, min(8*20 + 5, ego_poses.shape[0] + 5), 10):  # looping over 2s to 8s with interval of 0.5s
            for i in range(2 * 20, min(10 * 20 + 5, ego_poses.shape[0] + 5), 10):  # looping over 2s to 8s with interval of 0.5s
                current_pose = ego_poses[i]
                # normalize at current pose
                cos_, sin_ = math.cos(-current_pose[3]), math.sin(-current_pose[3])
                ego_poses -= current_pose
                rotated_poses = [ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                                 ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_]
                rotated_poses = np.stack(rotated_poses, axis=1)
                assert rotated_poses[i, 0] == 0 and rotated_poses[i, 1] == 0, f'rotated pose not zero at 40: {rotated_poses[i]}'

                # yaw in 1 s
                future_yaw = np.mean(rotated_poses[i + 10: i + 30, -1])
                # normalize yaw angle to [-pi, pi]
                if future_yaw > math.pi:
                    future_yaw -= 2 * math.pi
                elif future_yaw < -math.pi:
                    future_yaw += 2 * math.pi
                yaw_threshold = 0.5

                velocity = rotated_poses[i - 20:i + 20, :2] - rotated_poses[i - 40:i, :2]  # 0-20
                estimated_pose = rotated_poses[i - 20:i + 20, :2] + velocity * 3
                delta_pose = np.mean(estimated_pose - rotated_poses[i + 40:i + 80, :2], axis=0)  # 0-60
                # y_threshold = 4
                x_threshold = 5

                # if future_yaw > yaw_threshold:
                #     intention = 0  # left
                # elif future_yaw < -yaw_threshold:
                #     intention = 1  # right
                # # if delta_pose[1] > y_threshold:
                # #     intentions[i] = 0  # left
                # # elif delta_pose[1] < -y_threshold:
                # #     intentions[i] = 1  # right
                # elif delta_pose[0] > x_threshold:
                #     intention = 3  # accelerate
                # elif delta_pose[0] < -x_threshold:
                #     intention = 2  # decelerate
                # else:
                #     intention = 4  # keep

                if delta_pose[0] > x_threshold:
                    intention = 0  # accelerate
                elif delta_pose[0] < -x_threshold:
                    intention = 1  # decelerate
                else:
                    intention = 2  # keep

                # validity check
                if abs(delta_pose[0]) > 100:
                    # invalid due to -1 values
                    # print('invalid intention, skipping ', current_pose, rotated_poses[35:51, :], agent_dic['ego']['pose'][35:51, :])
                    skip = True
                    break
                data_to_return['intentions'].append(intention)
                # only classify the current intention
                break

        if include_navigation and 'route' in data_to_return and len(data_to_return['route']) > 0:
            # WARNING: Currently only works for nuplan with route infos
            from transformer4planning.utils import nuplan_utils
            dist_threshold = 10
            nav_lanes = []
            for i in range(40, agent_dic['ego']['pose'].shape[0], 10):
                # loop over from 2s to 10s with 0.5s interval
                # 1. get current lane
                current_pose = agent_dic['ego']['pose'][i]
                current_lane_id, dist = nuplan_utils.get_closest_lane_on_route(current_pose, data_to_return['route'], self.road_dic_mem)
                if dist < dist_threshold and current_lane_id not in nav_lanes:
                    nav_lanes.append(int(current_lane_id))
            data_to_return['navigation'] = nav_lanes
            if len(nav_lanes) == 0 and filter_still:
                # skip off route
                skip = True

        if 'ego' not in agent_dic:
            print("no ego and skip")
            skip = True
        elif agent_dic['ego']['pose'][40][0] == -1:
            print("invalid ego pose")
            skip = True

        # sanity check
        for each_key in data_to_return:
            if data_to_return[each_key] is None:
                print("Invalid Scenario Loaded for key", each_key)
                skip = True
        # category = classify_scenario(data_to_return)
        data_to_return["category"] = 1
        data_to_return['scenario'] = scenario_id
        data_to_return['edges'] = edges
        data_to_return['edge_type'] = edge_type
        data_to_return['skip'] = skip
        return data_to_return

    def get_scenario_num(self):
        from tqdm import tqdm
        zero_scenario_file_number = 0
        total_scenario = 0
        for i in tqdm(range(self.total_file_num)):
            route_road_ids = []
            log_db = self.current_dataset.log_dbs[i]
            scenarios = len(log_db.scenario_tag)
            if scenarios == 0:
                zero_scenario_file_number += 1
                continue
            sensor_data_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', '')
            for each_scenario_tag in log_db.scenario_tag:
                # fetch lidar token (as time stamp) from scenario tag
                each_lidar_token = each_scenario_tag.lidar_pc_token
                # get scenario from lidar_token
                lidar_token_timestamp = nuplan_scenario_queries.get_sensor_data_token_timestamp_from_db(
                    log_db.load_path,
                    sensor_data_source,
                    each_lidar_token)
                scenario = get_default_scenario_from_token(log_db, each_lidar_token, lidar_token_timestamp)
                route_road_ids += scenario.get_route_roadblock_ids()
            route_road_ids = list(set(route_road_ids))
            if len(route_road_ids) == 0:
                continue
            total_scenario += scenarios
        return total_scenario, zero_scenario_file_number