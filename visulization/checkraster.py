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
                    historic_channel[w, h] = 255        
        historic_channel = 255 - 255*historic_channel.astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_agenttype{i+1}.png"), channel)          
        
def visulize_trajectory(savepath, trajectory):
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()
    x = trajectory[::2, 0]
    y = trajectory[::2, 1]
    plt.scatter(x, y)
    plt.savefig(os.path.join(savepath, f"trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visulize_context_trajectory(savepath, context_actions):
    if isinstance(context_actions, torch.Tensor):
        context_actions = context_actions.numpy()
    startpose = np.zeros(2)
    context_pose = list()
    for i in range(context_actions.shape[0]):
        startpose += context_actions[i]
        context_pose.append(startpose.copy())
    context_pose = np.array(context_pose)
    x = context_pose[:,0]
    y = context_pose[:, 1]
    plt.scatter(x, y)
    plt.savefig(os.path.join(savepath, f"context_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
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
    visulize_context_trajectory(args.path_to_save, context_actions)
    print("done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_disk_path", default="/home/shiduozhang/nuplan/dataset/nsm_sparse_balance")  
    parser.add_argument("--path_to_save", default="visualizition/rasters/example3")
    args = parser.parse_args()

    main(args)
