import torch
import torch.nn as nn
import random

class DataAugmentation(nn.Module):
    """
    Augmentation module for large trajectory model encoders.
    Any augmentation method should be implemented in this class.
    """
    def __init__(self):
        super().__init__()

    def trajectory_augmentation(self, target_traj, x_noise_scale, y_noise_scale, expanded_indices=1):
        raise NotImplementedError
        if self.training and x_noise_scale > 0:
            x_noise = (torch.rand(target_traj.shape, device=target_traj.device) * x_noise_scale * 2 - x_noise_scale) * expanded_indices
            target_traj[..., 0] += x_noise[..., 0] 
        if self.training and y_noise_scale > 0:
            y_noise = (torch.rand(target_traj.shape, device=target_traj.device) * y_noise_scale * 2 - y_noise_scale) * expanded_indices
            target_traj[..., 1] += y_noise[..., 1]
        return target_traj


    def trajectory_linear_augmentation(self, target_traj, x_noise_scale, y_noise_scale, reverse_scale=True):
        """
        Args:
            target_traj: trajectory to be augmented
            x_noise_scale: max value of x noise
            y_noise_scale: max value of y noise
            reverse_scale: if Ture, first point has the largest noise, otherwise the last point has the largest noise

        Returns: Augmented trajectory
        """
        if x_noise_scale == 0 and y_noise_scale == 0:
            return target_traj
        # augment with linear scale
        device = target_traj.device
        context_length = target_traj.shape[1]
        scale = (torch.arange(context_length, device=device, dtype=torch.float32) + 1) / context_length
        if reverse_scale:
            scale = scale.flip(0)
        scale = scale.unsqueeze(0).unsqueeze(-1).repeat(target_traj.shape[0], 1, target_traj.shape[-1])
        if self.training and x_noise_scale > 0:
            x_noise = 1 + scale * random.random() * x_noise_scale * 2 - x_noise_scale
            target_traj[..., 0] *= x_noise[..., 0]
        if self.training and y_noise_scale > 0:
            y_noise = 1 + scale * random.random() * y_noise_scale * 2 - y_noise_scale
            target_traj[..., 1] *= y_noise[..., 1]
        return target_traj

    def raster_augmentation(self, raster):
        raise NotImplementedError

    def kp_denoising(self, future_key_points):
        if not self.training:
            return future_key_points
        device = future_key_points.device
        noise_scale = []

        # linear
        # batch_size = future_key_points.shape[0]
        # for i in range(10):
        #     noise = torch.FloatTensor(batch_size, 2).uniform_(0.1*i, 2-0.1*i).to(device)
        #     # element wise multiply
        #     future_key_points[:, i, :] *= noise

        # exponential
        for i in range(10):
            noise_scale.append(0.8 ** i)
        noise_scale = torch.tensor(noise_scale).unsqueeze(-1).repeat(1, 2).to(device)
        kp_to_add_noise = future_key_points[:, :10, :]
        future_key_points[:, :10, :] += (torch.randn_like(kp_to_add_noise) * noise_scale * 2 - noise_scale) * kp_to_add_noise
        return future_key_points