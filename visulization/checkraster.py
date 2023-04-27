import datasets
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_gen.DataLoaderNuPlan import NuPlanDL
from dataset_gen.nuplan_obs import get_observation_for_autoregression_nsm

def visulize_raster(savepath, name, raster, vis_id):
    if isinstance(raster, torch.Tensor):
        raster = raster.numpy()
    agents_rasters = raster[:, 21:, :, :,]
    road_rasters = raster[vis_id, :21, :, :]
    goalchannel = 255 - 255*road_rasters[0].astype(np.uint8)
    cv2.imwrite(os.path.join(savepath, f"{name}_goal.png"), goalchannel)
    for i in range(1, 21):
        channel = 255 - 255*road_rasters[i].astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_roadtype_{i-1}.png"), channel)
    
    history = agents_rasters.shape[0]
    for i in range(8):
        historic_channel = np.zeros_like(channel)
        historic_channel += agents_rasters[vis_id, i]
        for w in range(historic_channel.shape[0]):
            for h in range(historic_channel.shape[1]):
                if historic_channel[w, h] > 0:
                    historic_channel[w, h] = 1        
        historic_channel = 255 - 255*historic_channel.astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_agenttype{i}.png"), historic_channel)          
        
def visulize_trajectory(savepath, future_trajectory, context_actions, scale=0.77):
    if isinstance(future_trajectory, torch.Tensor):
        future_trajectory = future_trajectory.numpy()
    if isinstance(context_actions, torch.Tensor):
        context_actions = context_actions.numpy()
    raster = 255 * np.ones((224, 224)).astype(np.uint8)
    future_trajectory *= scale
    future_trajectory = future_trajectory.astype(np.float32) + 112.0
    future_keypoints = list()
    for traj in future_trajectory:
        future_keypoints.append(cv2.KeyPoint(x=traj[1], y=traj[0], _size=2))
    future_raster = cv2.drawKeypoints(raster, future_keypoints, 0, [0, 0, 0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(savepath, "future_trajectory.png"), future_raster)

    startpose = np.array([112, 112], dtype=np.float32) - context_actions[0] * scale
    context_pose = list()
    for i in range(1, context_actions.shape[0]):
        startpose += context_actions[i]
        context_pose.append(startpose.copy() * scale)
    past_keypoints = list()
    for pose in context_pose:
        past_keypoints.append(cv2.KeyPoint(x=pose[1], y=pose[0], _size=2))
    past_raster = cv2.drawKeypoints(raster, past_keypoints, 0, [0, 0, 0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(savepath, "past_trajectory.png"), past_raster)

    keypoints = list()
    keypoints.extend(future_keypoints)
    keypoints.extend(past_keypoints)
    total_raster = cv2.drawKeypoints(raster, keypoints, 0, [0, 0, 0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(savepath, "trajectory.png"), total_raster)

# def visulize_context_trajectory(savepath, context_actions, scale=0.77):
#     if isinstance(context_actions, torch.Tensor):
#         context_actions = context_actions.numpy()
#     raster = np.zeros((224, 224)).astype(np.uint8)
#     startpose = np.array([112, 112], dtype=np.float32) + context_actions[0] * scale
#     context_pose = list()
#     for i in range(1, context_actions.shape[0]):
#         startpose += context_actions[i]
#         context_pose.append(startpose.copy() * scale)
#     keypoints = list()
#     for pose in context_pose:s
#         keypoints.append(cv2.KeyPoint(x=pose[0], y=pose[1], _size=2))
#     raster = cv2.drawKeypoints(raster, keypoints, 0, [255, 255, 255], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv2.imwrite(os.path.join(savepath, "context_trajectory.png"), raster)  
    
def main(args):
    import pickle
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

    # dataset = datasets.load_from_disk(args.dataset_disk_path)
    # example = dataset[0]
    # trajectory = example["trajectory_label"][:, :2]
    # high_res_raster = example["high_res_raster"].permute(2, 0, 1)
    # low_res_raster = example["low_res_raster"].permute(2, 0, 1)
    # context_actions = example["context_actions"][:, :2]
    # visulize_raster(args.path_to_save, "high_res_channel", high_res_raster)
    # visulize_raster(args.path_to_save, "low_res_channel", low_res_raster)
    # # visulize_trajectory(args.path_to_save, trajectory, context_actions)
    # # visulize_context_trajectory(args.path_to_save, context_actions)
    with open("autoregressive_data_3d.pkl","rb") as f:
        example = pickle.load(f)
        high_rasters = example["high_res_raster"].reshape(224, 224, -1, 29).permute(2, 3, 0, 1)
        low_rasters = example["low_res_raster"].reshape(224, 224, -1, 29).permute(2, 3, 0, 1)
    # trajectory = observation_dic["trajectory"]
    # high_rasters = observation_dic["high_res_raster"].transpose(0, 3, 1, 2)
    # low_rasters = observation_dic["low_res_raster"].transpose(0, 3, 1, 2)
    visulize_raster(args.path_to_save, "high_res_channel", high_rasters, args.vis_id)
    visulize_raster(args.path_to_save, "low_res_channel", low_rasters, args.vis_id) 
    print("done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_disk_path", default="/home/shiduozhang/nuplan/dataset/nsm_sparse_balance")  
    parser.add_argument("--path_to_save", default="visualization/rasters/example-array3d")
    parser.add_argument("--vis_id",type=int, default=0)
    parser.add_argument("--frame_id", type=int, default=80)
    args = parser.parse_args()

    main(args)
