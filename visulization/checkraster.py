import datasets
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def visulize_raster(savepath, name, raster):
    if isinstance(raster, torch.Tensor):
        raster = raster.numpy()
    goalchannel = 255 - 255*raster[0].astype(np.uint8)
    cv2.imwrite(os.path.join(savepath, f"{name}_goal.png"), goalchannel)
    for i in range(1, 21):
        channel = 255 - 255*raster[i].astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_roadtype_{i-1}.png"), channel)
    agent_history_start = 21
    for i in range(8):
        historic_channel = np.zeros_like(channel)
        for j in range(9):
            historic_channel += raster[agent_history_start+j*8+i]
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
        future_keypoints.append(cv2.KeyPoint(x=traj[0], y=traj[1], _size=2))
    future_raster = cv2.drawKeypoints(raster, future_keypoints, 0, [0, 0, 0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(savepath, "future_trajectory.png"), future_raster)

    startpose = np.array([112, 112], dtype=np.float32) - context_actions[0] * scale
    context_pose = list()
    for i in range(1, context_actions.shape[0]):
        startpose += context_actions[i]
        context_pose.append(startpose.copy() * scale)
    past_keypoints = list()
    for pose in context_pose:
        past_keypoints.append(cv2.KeyPoint(x=pose[0], y=pose[1], _size=2))
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
#     for pose in context_pose:
#         keypoints.append(cv2.KeyPoint(x=pose[0], y=pose[1], _size=2))
#     raster = cv2.drawKeypoints(raster, keypoints, 0, [255, 255, 255], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv2.imwrite(os.path.join(savepath, "context_trajectory.png"), raster)  
    
def main(args):
    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)
    dataset = datasets.load_from_disk(args.dataset_disk_path)
    example = dataset[0]
    trajectory = example["trajectory_label"][:, :2]
    high_res_raster = example["high_res_raster"].permute(2, 0, 1)
    low_res_raster = example["low_res_raster"].permute(2, 0, 1)
    context_actions = example["context_actions"][:, :2]
    visulize_raster(args.path_to_save, "high_res_channel", high_res_raster)
    visulize_raster(args.path_to_save, "low_res_channel", low_res_raster)
    visulize_trajectory(args.path_to_save, trajectory)
    # visulize_context_trajectory(args.path_to_save, context_actions)
    print("done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_disk_path", default="/home/shiduozhang/nuplan/dataset/nsm_sparse_balance")  
    parser.add_argument("--path_to_save", default="visualizition/rasters/example3")
    args = parser.parse_args()

    main(args)
