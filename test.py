from datasets import Dataset
import os
from datasets.arrow_dataset import _concatenate_map_style_datasets
import math
import torch
from pathlib import Path
import numpy as np
import pickle
from dataset_gen.DataLoaderNuPlan import NuPlanDL
from dataset_gen.nuplan_obs import get_observation_for_nsm
from transformers import HfArgumentParser
from transformer4planning.models.model import build_models
from transformer4planning.utils import ModelArguments
from transformer4planning.checkratser import visulize_raster, visulize_trajectory, visulize_raster_without_route, visulize_raster_perchannel
import datasets

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def visulize_backbone():
    import matplotlib.pyplot as plt
    # 12k dataset, 30million gpt
    epochs = np.arange(start=0 ,stop=251, step=50)
    gpt_loss = np.array([108.44, 0.579, 0.132, 0.087, 0.063, 0.048])
    xlnet_loss = np.array([126.27, 0.927, 0.170, 0.114, 0.082, 0.062])
    transxl_loss = np.array([127.28, 1.189, 0.175, 0.117, 0.089, 0.068])
    t5_loss = np.array([131.84, 0.619, 0.175, 0.123, 0.095, 0.077])
    
    plt.plot(epochs, gpt_loss, marker='o', label='GPT Loss')
    plt.plot(epochs, xlnet_loss, marker='o', label='XLNet Loss')
    plt.plot(epochs, transxl_loss, marker='o', label='TransXL Loss')
    plt.plot(epochs, t5_loss, marker='o', label='T5 Loss')
    plt.title("Loss over Epochs of Diff Backbones")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def visulize_gpt_scale():
    import matplotlib.pyplot as plt
    epochs = np.arange(start=0, stop=46, step=5)
    gpt_30m_small_data_loss = np.array([137.60, 57.00, 3.03, 1.43, 0.98, 0.65, 0.53, 0.44, 0.39, 0.32])
    gpt_50m_samlldata_loss = np.array([123.48, 45.47, 2.19, 1.26, 0.75, 0.57, 0.48, 0.39, 0.34, 0.28])
    gpt_1_5b_largedata_loss = np.array([27.84, 3.11, 1.91, 1.01, 0.76, 0.58, 0.47, 0.39, 0.34, 0.30])
    plt.plot(epochs, gpt_30m_small_data_loss, marker='o', label='30million param&12k data')
    plt.plot(epochs, gpt_50m_samlldata_loss, marker='o', label='50million param&12k data')
    # plt.plot(epochs, gpt_1_5b_largedata_loss, marker='o', label='1.5B param&260k data')
    plt.title("Loss over Epochs of Diff Model&Dataset Size")
    plt.yscale("log")
    # plt.ylim(top=1, bottom=0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def visulize_xl_scale():
    import matplotlib.pyplot as plt
    epochs = np.arange(start=10, stop=46, step=5)
    xl_30m_large_data_loss = np.array([2.35,1.49,1.30,1.24,0.7,0.58,0.56,0.56])
    xl_200m_large_data_loss = np.array([2.07,1.20,0.81,0.60,0.48,0.41,0.35,0.3])
    plt.plot(epochs, xl_30m_large_data_loss, marker='o', label='30million param&240k data')
    plt.plot(epochs, xl_200m_large_data_loss, marker='o', label='200million param&240k data')
    plt.title("Loss over Epochs of Diff Model Size")
    plt.yscale("log")
    # plt.ylim(top=1, bottom=0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle

def dataset_unitest():
    from transformer4planning.checkratser import visulize_raster
    dataset = datasets.load_from_disk("/media/shiduozhang/My Passport/nuplan/pittsburgh_byscenario")
    example = dataset[20]
    high_res_raster = example["high_res_raster"].detach().cpu().numpy()
    low_res_raster = example["low_res_raster"].detach().cpu().numpy()
    context_action = example["context_actions"].detach().cpu().numpy()
    trajectory = example["trajectory_label"].detach().cpu().numpy()
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    model_args.model_pretrain_name_or_path = "/home/shiduozhang/nuplan/test-checkpoint"
    # model = build_models(model_args)
    # pred_trajectory = model(intended_maneuver_vector=torch.zeros((1), dtype=torch.int32), \
    #                         current_maneuver_vector=torch.zeros((1, 12), dtype=torch.float32), \
    #                         context_actions=torch.tensor(context_action).unsqueeze(0), \
    #                         high_res_raster=torch.tensor(high_res_raster).unsqueeze(0), \
    #                         low_res_raster=torch.tensor(low_res_raster).unsqueeze(0),
    #                         trajectory_label=torch.zeros((1, 160, 4))).logits.squeeze(0).detach().cpu().numpy()
    # diff = pred_trajectory - trajectory[1::2]

    visulize_raster_perchannel("visulization/rasters/pittsburgh_scenario_test/high", high_res_raster)
    visulize_raster_perchannel("visulization/rasters/pittsburgh_scenario_test/low", low_res_raster)
    visulize_trajectory("visulization/rasters/pittsburgh_scenario_test/low",context_action)
    visulize_trajectory("visulization/rasters/pittsburgh_scenario_test/high",context_action, scale=4)
    print("done")

def planner_post_process_unitest():
    data_path={'NUPLAN_DATA_ROOT': str(Path.home()) + "/nuplan/dataset",
                    'NUPLAN_MAPS_ROOT': str(Path.home()) + "/nuplan/dataset/maps",
                    'NUPLAN_DB_FILES': str(Path.home()) + "/nuplan/dataset/testset/",}
    road_path=None
    data_loader = NuPlanDL(scenario_to_start=0,
                        file_to_start=0,
                        max_file_number=20,
                        data_path=data_path, db=None, gt_relation_path=None,
                        road_dic_path=road_path,
                        running_mode=1)
    observation_kwargs = dict(
            max_dis=500,
            high_res_raster_shape=[224, 224],  # for high resolution image, we cover 50 meters for delicated short-term actions
            high_res_raster_scale=4.0,
            low_res_raster_shape=[224, 224],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
            low_res_raster_scale=0.77,
            past_frame_num=40,
            future_frame_num=160,
            frame_sample_interval=4,
            action_label_scale=100,)

        
    loaded_dic=data_loader.get_next_file(specify_file_index=0)
    start_id = 820
    # #start_id = 1489
    # time_us = 1629742338999991
    # actions = list()
    # for frame_id in range(start_id, 1640):
    #     total_frames = len(loaded_dic['lidar_pc_tokens'])
    #     observation_dic = get_observation_for_nsm(
    #                         observation_kwargs, loaded_dic, frame_id, total_frames, nsm_result=None)
    #     context_action = observation_dic["context_actions"]
    #     new_dict = dict(
    #         time = time_us,
    #         action = context_action[:, :2]
    #     )
    #     time_us += 5e4
    #     actions.append(new_dict)
    # with open("actions.pkl", "wb") as f:
    #     pickle.dump(actions, f)
    # print("done")
    # with open("road_dic.pkl", "rb") as f:
    #     loaded_dic = pickle.load(f)
    for frame_id in range(start_id, start_id+1, 10): 
        ego_trajectory = loaded_dic["agent"]["ego"]["pose"][frame_id-40:frame_id]
        total_frames = len(loaded_dic['lidar_pc_tokens'])
        observation_dic = get_observation_for_nsm(
                            observation_kwargs, loaded_dic, frame_id, total_frames, nsm_result=None)
        high_res_raster = observation_dic["high_res_raster"]
        low_res_raster = observation_dic["low_res_raster"]
        context_action = observation_dic["context_actions"]
        print("x:", context_action[:, 0])
        print("y:", context_action[:, 1])
        relative_gt = observation_dic["trajectory_label"][1::2]
        absolute_gt = loaded_dic["agent"]["ego"]["pose"][frame_id+1:frame_id+161][::2]
        parser = HfArgumentParser((ModelArguments))
        model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
        model_args.model_pretrain_name_or_path = "/home/shiduozhang/nuplan/test-checkpoint"
        # model = build_models(model_args)
        # output = model(intended_maneuver_vector=torch.zeros((1), dtype=torch.int32), \
        #                     current_maneuver_vector=torch.zeros((1, 12), dtype=torch.float32), \
        #                     context_actions=torch.tensor(context_action).unsqueeze(0), \
        #                     high_res_raster=torch.tensor(high_res_raster).unsqueeze(0), \
        #                     low_res_raster=torch.tensor(low_res_raster).unsqueeze(0),
        #                     trajectory_label=torch.zeros((1, 160, 4)))
        # pred_traj = output.logits.squeeze(0).detach().cpu().numpy()
        
        # relative_traj = pred_traj.copy()
        # diff_relative = relative_traj - relative_gt
        # # pred_traj = relative_gt.copy()
        # cos_, sin_ = math.cos(-ego_trajectory[-1][-1]), math.sin(-ego_trajectory[-1][-1])
        # step = 5
        # for i in range(pred_traj.shape[0]):
        #     if i == 0:
        #         delta_heading = get_angle_of_a_line(np.zeros(2), relative_traj[i + step, :2])
                
        #     else:
        #         delta_heading = get_angle_of_a_line(relative_traj[i, :2], relative_traj[i, :2]) 
        #     heading = ego_trajectory[-1, -1] + delta_heading
        #     new_x = pred_traj[i, 0].copy() * cos_ + pred_traj[i, 1].copy() * sin_ + ego_trajectory[-1][0]
        #     new_y = pred_traj[i, 1].copy() * cos_ - pred_traj[i, 0].copy() * sin_ + ego_trajectory[-1][1]
        #     pred_traj[i, 0] = new_x
        #     pred_traj[i, 1] = new_y
        #     pred_traj[i, 2] = 0
        #     pred_traj[i, -1] = heading
        # for i in range(pred_traj.shape[0]):
        #     pred_traj[i, -1] = normalize_angle(pred_traj[i, -1])
        # abs_diff = absolute_gt - pred_traj
        # next_world_coor_points = pred_traj.copy()
        # relative_error = relative_traj - relative_gt
        # absolute_error = next_world_coor_points - absolute_gt

        # fig = plt.figure(figsize=(200,100))
        # ax1 = fig.add_subplot(1,2,1)
        # ax1.set_title("relative points")
        # ax1.set_xlim([-100, 100])
        # ax1.set_ylim([-100, 100])
        # ax1.scatter(relative_gt[::4, 0], relative_gt[::4, 0], color='green')
        # ax1.scatter(relative_traj[::4, 0], relative_traj[::4, 0], color='red')
        # ax2 = fig.add_subplot(1,2,2)
        # ax2.set_title("gt_points")
        # ax2.set_xlim([-100, 100])
        # ax2.set_ylim([-100, 100])
        # absolute_x, absolute_y = absolute_gt[::4, 0]-ego_trajectory[-1][0], absolute_gt[::4, 1]-ego_trajectory[-1][1]
        # absolute_x_gt, absolute_y_gt = next_world_coor_points[::4, 0]-ego_trajectory[-1][0], next_world_coor_points[::4, 1]-ego_trajectory[-1][1]
        # ax2.scatter(absolute_x_gt, absolute_y_gt, color='green')
        # ax2.scatter(absolute_x, absolute_y, color='red')
        # plt.show()
    if not os.path.exists("visulization/rasters/rasters/compare-nuplan"):
        os.mkdir("visulization/rasters/rasters/compare-nuplan")
    visulize_raster("visulization/rasters/rasters/compare-nuplan", "high_res", high_res_raster, context_length=11)
    visulize_raster("visulization/rasters/rasters/compare-nuplan", "low_res", low_res_raster, context_length=11)
    # with open("/home/shiduozhang/Pictures/planner_rasters/pred_traj100.pkl", "rb") as f:
    #     nuplan_traj = pickle.load(f)
    #     error = nuplan_traj - next_world_coor_points
    #     fig = plt.figure(figsize=(200,100))
    #     ax = fig.add_subplot(1,2,1)
    #     ax.set_title("nuplan_compare")
    #     ax.set_xlim([-100, 100])
    #     ax.set_ylim([-100, 100])
    #     ax.scatter(absolute_x_gt, absolute_y_gt, color='green')
    #     nuplan_x, nuplan_y = nuplan_traj[::4, 0]-ego_trajectory[-1][0], nuplan_traj[::4, 1]-ego_trajectory[-1][1]
    #     ax.scatter(nuplan_x, nuplan_y, color='red')
    #     plt.show()
            
    print("done")
    # return relative_error, absolute_error

def history_compare_test():
    with open('actions.pkl', "rb") as f:
        gt_actions = pickle.load(f)
    with open("/home/shiduozhang/Pictures/planner_context/history130.pkl", "rb") as f:
        history = pickle.load(f)
    ego_states = history.ego_state_buffer
    ego_trajectory = list()
    for i in range(len(ego_states)):
        waypoint = ego_states[i].waypoint
        pos = np.array([waypoint.center.x, waypoint.center.y, waypoint.heading])
        ego_trajectory.append(pos)
    
    # process to relative actions
    ego_pose = ego_trajectory[-1]
    cos_, sin_ = math.cos(-ego_pose[2]), math.sin(-ego_pose[2])
    # new_ego_trajectory = list()
    # # upsample and interpolate
    # for idx in range(0, len(ego_states) - 1):
    #     new_ego_trajectory.append(ego_trajectory[idx])
    #     new_ego_trajectory.append((ego_trajectory[idx] + ego_trajectory[idx + 1]) / 2)
    # new_ego_trajectory.append(ego_pose)
    # ego_trajectory = np.array(new_ego_trajectory)[::5].copy()
    ego_trajectory = ego_trajectory[::2]
    context_actions = list()
    ego_poses = ego_trajectory - ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                                ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                                np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
    for i in range(len(rotated_poses)- 1):
        action = rotated_poses[i+1]
        context_actions.append(action)

    nuplan_action = dict(
        time=ego_states[-1].time_us,
        actions=np.array(context_actions)
    )
    
    for id, item in enumerate(gt_actions):
        if abs(item["time"] - nuplan_action["time"]) < 1e4:
            print("index in gt_actions:", id)
            print("time error:", (item["time"] - nuplan_action["time"])/1e6)
            nuplan_a = nuplan_action["actions"][:, :2]
            gt_a = gt_actions[id+4]["action"][:, :2]
            error = nuplan_a[:10] - gt_a
            print(np.average(error[:, 0]), np.average(error[:, 1])) 
            break
    print("done")

def planner_test():
    from transformer4planning.submission.planner import ControlTFPlanner
    data_path={'NUPLAN_DATA_ROOT': str(Path.home()) + "/nuplan/dataset",
                    'NUPLAN_MAPS_ROOT': str(Path.home()) + "/nuplan/dataset/maps",
                    'NUPLAN_DB_FILES': str(Path.home()) + "/nuplan/dataset/nuplan-v1.0/public_set_boston_train/",}
    road_path=str(Path.home()) + "/nuplan/dataset/pickles/road_dic.pkl"
    data_loader = NuPlanDL(scenario_to_start=0,
                        file_to_start=0,
                        max_file_number=2,
                        data_path=data_path, db=None, gt_relation_path=None,
                        road_dic_path=road_path,
                        running_mode=1)
    observation_kwargs = dict(
            max_dis=500,
            high_res_raster_shape=[224, 224],  # for high resolution image, we cover 50 meters for delicated short-term actions
            high_res_raster_scale=4.0,
            low_res_raster_shape=[224, 224],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
            low_res_raster_scale=0.77,
            past_frame_num=40,
            future_frame_num=160,
            frame_sample_interval=4,
            action_label_scale=100,)

    # loaded_dic=data_loader.get_next_file(specify_file_index=1)
    # map_api = data_loader.map_api
    with open("map_api.pkl","rb") as f:
        map_api = pickle.load(f)
   
    with open("road_dic.pkl", "rb") as f:
        loaded_dic = pickle.load(f)
    route_ids = loaded_dic["route"]
    time_stamp = loaded_dic['starting_timestamp']
    planner = ControlTFPlanner(
        horizon_seconds=10.0,
        sampling_time=0.1,
        acceleration=[5.0, 5.0],
    )

    planner.map_api = map_api
    planner.route_roadblock_ids = route_ids
    planner.goal = None
    with open("/home/shiduozhang/Pictures/history/history1.pkl", "rb") as f:
        history = pickle.load(f)
    ego_states = history.ego_states
    time = ego_states[-1].time_us
    diff_time = time-time_stamp
    frame = int(diff_time//5e4)
    ego_pose = loaded_dic["agent"]["ego"]["pose"][frame]
    relative_traj, global_traj, high_raster, low_raster, context_action = planner.test_trajectory(history)
    total_frames = len(loaded_dic['lidar_pc_tokens'])
    gt = get_observation_for_nsm(observation_kwargs, loaded_dic, frame, total_frames, None)
    diff_high = gt["high_res_raster"].astype(np.int64) - high_raster.astype(np.int64)
    diff_low = gt["low_res_raster"].astype(np.int64) - low_raster.astype(np.int64)
    diff_action = gt["context_actions"] - context_action
    if not os.path.exists("visulization/rasters/rasters/compare-gt"):
        os.mkdir("visulization/rasters/rasters/compare-gt")
    if not os.path.exists("visulization/rasters/rasters/compare-nuplan"):
        os.mkdir("visulization/rasters/rasters/compare-nuplan")
    visulize_raster("visulization/rasters/rasters/compare-gt", "high", gt["high_res_raster"], context_length=11)
    visulize_raster("visulization/rasters/rasters/compare-nuplan", "high", high_raster, context_length=11)
    print(np.max(diff_high))
    print(np.max(diff_low))
    print("done")
    pass

def generate_by_scenario_test():
    data_path = {
                'NUPLAN_DATA_ROOT': str(Path.home()) + "/nuplan/dataset",
                'NUPLAN_MAPS_ROOT': str(Path.home()) + "/nuplan/dataset/maps",
                'NUPLAN_DB_FILES': str(Path.home()) + "/nuplan/dataset/nuplan-v1.0/public_set_boston_train/",
            }
    road_dic_path = "road_dic.pkl"
    dl = NuPlanDL(scenario_to_start=0,
                    file_to_start=0,
                    max_file_number=1,
                    data_path=data_path, db=None, gt_relation_path=None,
                    road_dic_path=road_dic_path,
                    running_mode=1)
    observation_kwargs = dict(
        max_dis=500,
        high_res_raster_shape=[224, 224],  # for high resolution image, we cover 50 meters for delicated short-term actions
        high_res_raster_scale=4.0,
        low_res_raster_shape=[224, 224],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
        low_res_raster_scale=0.77,
        past_frame_num=40,
        future_frame_num=160,
        frame_sample_interval=4,
        action_label_scale=100,
    )
    
    filter_scenario = ["starting_straight_traffic_light_intersection_traversal","high_lateral_acceleration",
        "changing_lane", "high_magnitude_speed", "low_magnitude_speed", "starting_left_turn",
        "starting_right_turn", "stopping_with_lead", "following_lane_with_lead","near_multiple_vehicles",
        "traversing_pickup_dropoff", "behind_long_vehicle", "waiting_for_pedestrian_to_cross", "stationary_in_traffic"]  
    while not dl.end:
        loaded_dic, _ = dl.get_next(seconds_in_future=8)
        if loaded_dic["skip"]:
            continue
        if loaded_dic["agent"]["ego"]["pose"][0][0] == -1:
            continue
        if loaded_dic["type"] not in filter_scenario:
            continue
        observation_dic = get_observation_for_nsm(
            observation_kwargs, loaded_dic, 40, 201, nsm_result=None)
        # yield observation_dic

def waymo_test():
    waymo_dataset = datasets.load_from_disk("/home/shiduozhang/waymo/t4p_waymo")
    for i in range(len(waymo_dataset)):
        example = waymo_dataset[i]
        high_res_raster = example["high_res_raster"]
        low_res_raster = example["low_res_raster"]
        trajectory = example["trajectory_label"].detach().cpu().numpy()
        context_actions = example["context_actions"]
        scenario_id = example["scenario_id"]
        # visulize_raster_without_route(f"visulization/rasters/waymo/waymo_{i}", "high", high_res_raster, context_length=11)
        # visulize_raster_without_route(f"visulization/rasters/waymo/waymo_{i}", "low", low_res_ratser, context_length=11)
        visulize_raster_perchannel(f"visulization/rasters/waymo/{scenario_id}/high", high_res_raster)
        visulize_raster_perchannel(f"visulization/rasters/waymo/{scenario_id}/low", low_res_raster)
        visulize_trajectory(f"visulization/rasters/waymo/{scenario_id}/high", trajectory.copy(), 4)
        visulize_trajectory(f"visulization/rasters/waymo/{scenario_id}/low", trajectory.copy(), 0.77)
    print("done")

def waymo_raster_test():
    with open("ratser.pkl","rb") as f:
        raster = pickle.load(f)
    raster = raster[:, :, :, :86]
    for i in range(raster.shape[0]):
        raster_i = raster[i]
        visulize_raster_perchannel(f"visulization/rasters/waymo/batch{i}", raster_i)

def model_param_test():
    from transformers.models.gpt2 import GPT2Config, GPT2Model
    from torchvision.models import resnet18
    from transformer4planning.models.decoders import DecoderResCat
    resnet = resnet18(False)
    print("encoder params", sum(p.numel() for p in resnet.parameters()))
    config = GPT2Config()
    ## samllest one
    config.n_embd = 64
    config.n_inner = 256
    config.n_layer = 1
    config.n_head = 1
    small_one = GPT2Model(config)
    print("smallest params", sum(p.numel() for p in small_one.parameters()))
    decoder = DecoderResCat(config.n_inner, config.n_embd ,4)
    print("decoder params", sum(p.numel() for p in decoder.parameters()))
    ## 30m one
    config.n_embd = 256
    config.n_inner = 1024
    config.n_layer = 4
    config.n_head = 8
    _30m = GPT2Model(config)
    print("30m params", sum(p.numel() for p in _30m.parameters()))
    decoder = DecoderResCat(config.n_inner, config.n_embd ,4)
    print("decoder params", sum(p.numel() for p in decoder.parameters()))
    ## 117m one
    config.n_embd = 768
    config.n_inner = 768 * 4
    config.n_layer = 12
    config.n_head = 12
    _117one = GPT2Model(config)
    print("117m params", sum(p.numel() for p in _117one.parameters()))
    decoder = DecoderResCat(config.n_inner, config.n_embd ,4)
    print("decoder params", sum(p.numel() for p in decoder.parameters()))
    ## 762m one
    config.n_embd = 1280
    config.n_inner = 1280 * 4
    config.n_layer = 36
    config.n_head = 20
    _762m = GPT2Model(config)
    print("762m params", sum(p.numel() for p in _762m.parameters()))
    decoder = DecoderResCat(config.n_inner, config.n_embd ,4)
    print("decoder params", sum(p.numel() for p in decoder.parameters()))
    ## 1.5b one
    config.n_embd = 1600
    config.n_inner = 6400
    config.n_layer = 48
    config.n_head = 25
    _1500m = GPT2Model(config)
    print("1500m params", sum(p.numel() for p in _1500m.parameters()))
    decoder = DecoderResCat(config.n_inner, config.n_embd ,4)
    print("decoder params", sum(p.numel() for p in decoder.parameters()))

def eval_xl():
    from runner import ModelArguments, HfArgumentParser
    import collections
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    from tqdm import tqdm
    
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args()
    model_args.model_name = "pretrain-transxl"
    model_args.task = "nuplan"
    checkpoint_path = "/localdata_ssd/nuplan/transxl-checkpoint-addition"
    checkpoints = os.listdir(checkpoint_path)

    dataset = datasets.load_from_disk("/public/MARS/datasets/nuPlanCache/5hz_test_boston_fixroute")
    dataset.set_format(type='torch')
    dataset.shuffle()
    dataset = dataset.select(range(4880))
    def preprocess_data(examples, device):
        # take a batch of texts
        for each_key in examples:
            if isinstance(examples[each_key], type(torch.tensor(0))):
                examples[each_key] = examples[each_key].to(device)
        return examples
    
    def nuplan_collate_fn(batch):
        import collections
        expect_keys = ["file_name", "frame_index", "high_res_raster", "low_res_raster", "context_actions", "trajectory_label"]
        
        elem = batch[0]
        if isinstance(elem, collections.abc.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in expect_keys}
    test_dataloader = DataLoader(
                dataset=dataset,
                batch_size=4,
                num_workers=40,
                collate_fn=nuplan_collate_fn,
                pin_memory=True,
                drop_last=True
            )
    for checkpoint in checkpoints:
        checkpoint = os.path.join(checkpoint_path, checkpoint)
        model_args.model_pretrain_name_or_path = checkpoint
        model = build_models(model_args)
        model.to("cuda:0")
        loss = 0
        for i, input in enumerate(tqdm(test_dataloader)):
            input = preprocess_data(input, "cuda:0")
            output = model(**input)
            loss += output.loss.item()
        print(f"{checkpoint} eval loss is", loss/(i+1))
        model.to("cpu")
        del model

def read_nuplan_metric():
    import pandas as pd
    from pandas import read_parquet
    open_loop = dict(
        boston_1500 = "/home/shiduozhang/nuplan/exp/exp/1.5B-1data-boston/open_loop_boxes/2023.06.03.10.11.24/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.03.10.11.24.parquet",
        boston_762 = "/home/shiduozhang/nuplan/exp/exp/762M-1data-boston/open_loop_boxes/2023.06.07.01.23.46/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.07.01.23.46.parquet",
        boston_117 = "/home/shiduozhang/nuplan/exp/exp/117M-1data-boston/open_loop_boxes/2023.06.03.10.50.34/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.03.10.50.34.parquet",
        boston_30 = "/home/shiduozhang/nuplan/exp/exp/30M-1data-boston/open_loop_boxes/2023.06.03.11.29.12/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.03.11.29.12.parquet",
        pitts_1500 = "/home/shiduozhang/nuplan/exp/exp/1.5B-1data-pittsburgh/open_loop_boxes/2023.06.07.02.34.50/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.07.02.34.50.parquet",
        pitts_762 = "/home/shiduozhang/nuplan/exp/exp/762M-1data-pittsburgh/open_loop_boxes/2023.06.07.02.19.00/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.07.02.19.00.parquet",
        pitts_117 = "/home/shiduozhang/nuplan/exp/exp/117M-1data-pittsburgh/open_loop_boxes/2023.06.07.03.11.53/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.07.03.11.53.parquet",
        pitts_30 = "/home/shiduozhang/nuplan/exp/exp/30M-1data-pittsburgh/open_loop_boxes/2023.06.07.02.42.34/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.07.02.42.34.parquet"
    )
    close_loop = dict(
        boston_1500 = "/home/shiduozhang/nuplan/exp/exp/1.5B-1data-boston/closed_loop_reactive_agents/2023.06.03.11.39.00/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.03.11.39.00.parquet",
        boston_762 = "/home/shiduozhang/nuplan/exp/exp/762M-1data-boston/closed_loop_reactive_agents/2023.06.07.01.23.44/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.07.01.23.44.parquet",
        boston_117 = "/home/shiduozhang/nuplan/exp/exp/117M-1data-boston/closed_loop_reactive_agents/2023.06.03.11.50.30/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.03.11.50.30.parquet",
        boston_30 = "/home/shiduozhang/nuplan/exp/exp/30M-1data-boston/closed_loop_reactive_agents/2023.06.03.18.02.00/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.03.18.02.00.parquet",
        pitts_1500 = "/home/shiduozhang/nuplan/exp/exp/1.5B-1data-pittsburgh/closed_loop_reactive_agents/2023.06.07.02.34.51/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.07.02.34.51.parquet",
        pitts_762 = "/home/shiduozhang/nuplan/exp/exp/762M-1data-pittsburgh/open_loop_boxes/2023.06.07.02.19.00/aggregator_metric/open_loop_boxes_weighted_average_metrics_2023.06.07.02.19.00.parquet",
        pitts_117 = "/home/shiduozhang/nuplan/exp/exp/117M-1data-pittsburgh/closed_loop_reactive_agents/2023.06.07.03.11.26/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.07.03.11.26.parquet",
        pitts_30 = "/home/shiduozhang/nuplan/exp/exp/30M-1data-pittsburgh/closed_loop_reactive_agents/2023.06.07.02.42.43/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.06.07.02.42.43.parquet"
    )
    # for name, open_path in open_loop.items():
    #     data = read_parquet(open_path)
    #     data.to_csv(f"openloop_{name}.csv", index=False)
    for name, close_path in close_loop.items():
        data = read_parquet(close_path)
        data.to_csv(f"closeloop_{name}.csv", index=False)
    print(data.head())

def autoregressive_unitest():
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate

    dataset = datasets.load_from_disk("/public/MARS/datasets/nuPlanCache/nuplan_autoregressive_full/nsm_autoregressive_1400-1647")
    min_x, max_x, min_y, max_y, min_yaw, max_yaw = 0, 0, 0, 0, 0, 0
    def nuplan_collate_fn(batch):
        import collections
        expect_keys = ["high_res_raster", "low_res_raster", "trajectory"]
        elem = batch[0]
        if isinstance(elem, collections.abc.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in expect_keys}
    
    dataloader =  DataLoader(
        dataset=dataset,
        batch_size=20,
        num_workers=20,
        collate_fn=nuplan_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    for i, data in enumerate(tqdm(dataloader)):
        traj = data["trajectory"].detach().cpu().numpy()
        max_x = max(np.max(traj[..., 0]), max_x)
        min_x = min(np.min(traj[..., 0]), min_x)
        max_y = max(np.max(traj[..., 1]), max_y)
        min_y = min(np.min(traj[..., 1]), min_y)
        max_yaw = max(np.max(traj[..., -1]), max_yaw)
        min_yaw = min(np.min(traj[..., -1]), min_yaw)
    print("MAX X:", max_x, " MIN X", min_x)
    print("MAX Y:", max_y, " MIN Y:", min_y)
    print("MAX YAW: ", max_yaw, "MIN YAW: ", min_yaw)
        
def dataset_statitcs():
    trainval_path = "/public/MARS/datasets/nuPlan/nuplan-v1.1/trainval"
    city_path = "/public/MARS/datasets/nuPlan/nuplan-v1.1/data/cache"
    cities = ["train_boston", "train_pittsburgh", "train_singapore", "val",
              "train_vegas_1", "train_vegas_2", "train_vegas_3", "train_vegas_4", "train_vegas_5", "train_vegas_6"]
    trainval_files = set(os.listdir(trainval_path))
    city_files = list()
    for city in cities:
        city_files.extend(os.listdir(os.path.join(city_path, city)))
    if set(city_files) == trainval_files:
        print("trainval and city files are the same")
    else:
        print("trainval and city files are not the same")
    print("start to check scenario tag")
    nuplan_dataloader = NuPlanDL(

    )

def planner_test():
    from transformer4planning.submission.planner import ControlTFPlanner
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    model_args.model_name = "scratch-gpt"
    model_args.model_pretrain_name_or_path = None
    model = build_models(model_args)
    model.eval()
    model.to("cuda")
    planner = ControlTFPlanner(horizon_seconds=5,
                               sampling_time=0.1,
                               acceleration=np.zeros(2),
                               model=model, use_backup_planner=False)
    with open("pickles/history.pkl", "rb") as f:
        history = pickle.load(f)
    with open("pickles/init.pkl", "rb") as f:
        initial = pickle.load(f)
    planner.initialize(initial)
    model_args.history = history
    trajectory = planner.compute_planner_trajectory(model_args)
    print("done")

if __name__ == "__main__":
    # print((338999991-277599788)/1e6)
    # history_compare_test()
    # planner_post_process_unitest()
    # visulize_backbone()
    # visulize_xl_scale()
    # planner_test()
    # generate_by_scenario_test()
    # dataset_unitest()
    # with open("vehicle.pkl", "rb") as f:
    #     vehicles = pickle.load(f)
    #     print(vehicles)
    # print("done")
    # waymo_test()
    # waymo_raster_test()
    # model_param_test()
    # eval_xl()
    # read_nuplan_metric()
    # autoregressive_unitest()
    # print(datasetsize)
    # dataset_statitcs()
    planner_test()
# 2021.08.23.18.02.44_veh-40_01747_01868 
# 2021.08.23.18.07.38_veh-28_00015_00137
# local:  1629742277599788  
# nuplan: 1629742338999991 + 61.4s => 1227 /nuplan'50th frame correspond to local 1327/ 100->1427 / 130->1488