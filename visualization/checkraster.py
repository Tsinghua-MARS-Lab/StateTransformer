import datasets
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_gen.DataLoaderNuPlan import NuPlanDL

def visulize_raster(savepath, name, raster, context_length=9):
    """
    raster shape: (224, 224, 93)
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if isinstance(raster, torch.Tensor):
        raster = raster.numpy()
    # agents_rasters = raster[:, 21:, :, :,]
    agents_rasters = raster[:, :, 21:]
    road_rasters = raster[:, :, :21]
    goalchannel = 255 - 255*road_rasters[:, :, 0].astype(np.uint8)
    cv2.imwrite(os.path.join(savepath, f"{name}_goal.png"), goalchannel)
    for i in range(1, 21):
        channel = 255 - 255*road_rasters[:, :, i].astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_roadtype_{i-1}.png"), channel)
    
    for i in range(8):
        historic_channel = np.zeros_like(channel)
        for j in range(context_length):
            historic_channel += agents_rasters[:, :, i*context_length + j]
        for w in range(historic_channel.shape[0]):
            for h in range(historic_channel.shape[1]):
                if historic_channel[w, h] > 0:
                    historic_channel[w, h] = 1        
        historic_channel = 255 - 255*historic_channel.astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_agenttype{i}.png"), historic_channel)        
        
def visulize_trajectory(savepath, trajectory, scale=0.77):
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()
    raster = 255 * np.ones((224, 224)).astype(np.uint8)
    trajectory *= scale
    trajectory = trajectory.astype(np.float32) + 112.0
    future_keypoints = list()
    for traj in trajectory:
        future_keypoints.append(cv2.KeyPoint(x=traj[1], y=traj[0], _size=2))
    future_raster = cv2.drawKeypoints(raster, future_keypoints, 0, [0, 0, 0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if scale < 1:
        name = "low"
    else:
        name = "high"
    cv2.imwrite(os.path.join(savepath, name + "_trajectory.png"), future_raster)

def visulize_raster_perchannel(savepath, raster):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if isinstance(raster, torch.Tensor):
        raster = raster.numpy()
    for i in range(raster.shape[-1]):
        channel = 255 - 255*raster[:, :, i].astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"channel_{i}.png"), channel)

def main(args):
    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)
    data_path={'NUPLAN_DATA_ROOT': str(Path.home()) + "/nuplan/dataset",
                'NUPLAN_MAPS_ROOT': str(Path.home()) + "/nuplan/dataset/maps",
                'NUPLAN_DB_FILES': str(Path.home()) + "/nuplan/dataset/nuplan-v1.0/public_set_boston_train/",}
    road_path=str(Path.home()) + "/nuplan/dataset/pickles/road_dic.pkl"
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
            frame_sample_interval=5,
            action_label_scale=100,)
    
    # loaded_dic=data_loader.get_next_file(specify_file_index=0)
    # total_frames = len(loaded_dic['lidar_pc_tokens'])
    # observation_dic = get_observation_for_autoregression_nsm(
    #                     observation_kwargs, loaded_dic, args.frame_id, total_frames, nsm_result=None)

    dataset = datasets.load_from_disk(args.dataset_disk_path)
    for i in range(len(dataset)):
        example = dataset[i]
        high_rasters = example["high_res_raster"]
        low_rasters = example["low_res_raster"]
        trajectory = np.concatenate([example["context_actions"], example["trajectory_label"]], axis=0)
    
        visulize_raster_perchannel(os.path.join(args.path_to_save + f"_{i}","high_res_channel"), high_rasters)
        visulize_raster_perchannel(os.path.join(args.path_to_save + f"_{i}", "low_res_channel"), low_rasters) 
        visulize_trajectory(args.path_to_save, trajectory, 0.77)
        visulize_trajectory(args.path_to_save, trajectory, 4)
        print("done!")
