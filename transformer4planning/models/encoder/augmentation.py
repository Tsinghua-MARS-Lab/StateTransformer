import torch
import torch.nn as nn

class DataAugmentation(nn.Module):
    """
    Augmentation module for large trajectory model encoders.
    Any augmentation method should be implemented in this class.
    """
    def __init__(self):
        super().__init__()
    
    def trajectory_augmentation(self, target_traj, x_noise_scale, y_noise_scale, expanded_indices=1):
        if self.training and x_noise_scale > 0:
            x_noise = (torch.rand(target_traj.shape, device=target_traj.device) * x_noise_scale * 2 - x_noise_scale) * expanded_indices
            target_traj[..., 0] += x_noise[..., 0] 
        if self.training and y_noise_scale > 0:
            y_noise = (torch.rand(target_traj.shape, device=target_traj.device) * y_noise_scale * 2 - y_noise_scale) * expanded_indices
            target_traj[..., 1] += y_noise[..., 1]
        return target_traj

    def raster_augmentation(self, raster):
        raise NotImplementedError