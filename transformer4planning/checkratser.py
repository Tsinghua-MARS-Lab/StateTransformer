import datasets
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    
    for i in range(9):
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

def visulize_raster_without_route(savepath, name, raster, context_length=9):
    """
    raster shape: (224, 224, 86)
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if isinstance(raster, torch.Tensor):
        raster = raster.numpy()
    # agents_rasters = raster[:, 21:, :, :,]
    agents_rasters = raster[:, :, 20:]
    road_rasters = raster[:, :, :20]
    for i in range(0, 20):
        channel = 255 - 255*road_rasters[:, :, i].astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_roadtype_{i}.png"), channel)
    
    for i in range(6):
        historic_channel = np.zeros_like(channel)
        for j in range(context_length):
            historic_channel += agents_rasters[:, :, i*context_length + j]
        for w in range(historic_channel.shape[0]):
            for h in range(historic_channel.shape[1]):
                if historic_channel[w, h] > 0:
                    historic_channel[w, h] = 1        
        historic_channel = 255 - 255*historic_channel.astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"{name}_agenttype{i}.png"), historic_channel)

def visulize_raster_perchannel(savepath, raster):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if isinstance(raster, torch.Tensor):
        raster = raster.numpy()
    for i in range(raster.shape[-1]):
        channel = 255 - 255*raster[:, :, i].astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"channel_{i}.png"), channel)