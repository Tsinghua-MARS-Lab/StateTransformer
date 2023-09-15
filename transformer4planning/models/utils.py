import torch
import numpy as np
import math
from typing import Optional

DEFAULT_TOKEN_CONFIG = dict(
    x_range=[0, 4],
    y_range=[-0.4, 0.4],
    x_class=80,
    y_class=40,
    sample_frequency=4
)

def cat_raster_seq(raster:Optional[torch.LongTensor], framenum=9, traffic=True):
    """
    input raster can be either high resolution raster or low resolution raster
    expected input size: [bacthsize, channel, h, w], and channel is consisted of goal(1d)+roadtype(20d)+agenttype*time(8*9d)
    """
    b, c, h, w = raster.shape
    agent_type = 8
    road_type = 20
    traffic_light_type = 4

    goal_raster = raster[:, :2, :, :].reshape(b, 2, h, w)  # updated as route raster
    road_ratser = raster[:, 2:2+road_type, :, :]
    traffic_raster = raster[:, 2+road_type:2+road_type+traffic_light_type, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type + traffic_light_type + 2, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, 2 + road_type + traffic_light_type + i::framenum, :, :]
        if traffic:
            raster_i = torch.cat([goal_raster, road_ratser, traffic_raster, agent_raster], dim = 1)  # expected format (b, 1+20+8, h, w)
        else:
            assert False, 'not supported anymore'
            raster_i = torch.cat([goal_raster, road_ratser, agent_raster], dim = 1)
        result[:, i, :, :, :] = raster_i
    # return format (batchsize, history_frame_number, channels_per_frame, h, w)
    return result

def cat_raster_seq_for_waymo(raster, framenum=11):
    b, c, h, w = raster.shape
    agent_type = 3
    road_type = 16
    road_raster = raster[:, :road_type, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, road_type + i::framenum, :, :]
        raster_i = torch.cat([road_raster, agent_raster], dim=1)
        assert raster_i.shape[1] == agent_type + road_type
        result[:, i, :, :, :] = raster_i
    return result

def cat_raster_seq_for_waymo_intention(raster, framenum=11):
    b, c, h, w = raster.shape
    agent_type = 3
    road_type = 16
    road_raster = raster[:, :road_type, :, :]
    intention_raster = raster[:, -1:, :, :]
    agent_raster_ori = raster[:, road_type:-1, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type + 1, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = agent_raster_ori[:, i::framenum, :, :]
        raster_i = torch.cat([road_raster, agent_raster, intention_raster], dim=1)
        result[:, i, :, :, :] = raster_i
    return result

def normalize(x):
    y = torch.zeros_like(x)
    # mean(x[...,0]) = 9.517, mean(sqrt(x[...,0]**2))=9.517
    y[..., 0] += (x[..., 0] / 10)
    y[..., 0] -= 0
    # mean(x[..., 1]) = -0.737, mean(sqrt(x[..., 1]**2))=0.783
    y[..., 1] += (x[..., 1] / 10)
    y[..., 1] += 0
    if x.shape[-1]==2:
        return y
    # mean(x[..., 2]) = 0, mean(sqrt(x[..., 2]**2)) = 0
    y[..., 2] = x[..., 2] * 10
    # mean(x[..., 3]) = 0.086, mean(sqrt(x[..., 3]**2))=0.090
    y[..., 3] += x[..., 3] / 2
    y[..., 3] += 0
    return y

def denormalize(y):
    x = torch.zeros_like(y)
    x[..., 0] = (y[..., 0]) * 10
    x[..., 1] = (y[..., 1]) * 10
    if y.shape[-1]==2:
        return x
    x[..., 2] = y[..., 2] / 10
    x[..., 3] = y[..., 3] * 2
    return x


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exp_positional_embedding(key_point_num, feat_dim):
    point_num = key_point_num
    # Creating the position tensor where the first position is 2^point_num, the second is 2^{point_num-1}, and so on.
    position = torch.tensor([2 ** (point_num - i - 1) for i in range(point_num)]).unsqueeze(1).float()

    # Create a table of divisors to divide each position. This will create a sequence of values for the divisor.
    div_term = torch.exp(torch.arange(0, feat_dim, 2).float() * (-math.log(100.0) / feat_dim))

    # Generate the positional encodings
    # For even positions use sin, for odd use cos
    pos_embedding = torch.zeros((point_num, feat_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    return pos_embedding

def uniform_positional_embedding(key_point_num, feat_dim):
    point_num = key_point_num
    position = torch.tensor([6 * (point_num - i)] for i in range(point_num))
    # Create a table of divisors to divide each position. This will create a sequence of values for the divisor.
    div_term = torch.exp(torch.arange(0, feat_dim, 2).float() * (-math.log(100.0) / feat_dim))

    # Generate the positional encodings
    # For even positions use sin, for odd use cos
    pos_embedding = torch.zeros((point_num, feat_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    return pos_embedding

def nll_loss_gmm_direct(pred_trajs, gt_trajs, gt_valid_mask,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    nearest_trajs = pred_trajs  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_trajs)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss