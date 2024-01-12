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
    # updated to dynamic route types
    route_type = c - agent_type * framenum - road_type - traffic_light_type

    goal_raster = raster[:, :route_type, :, :].reshape(b, route_type, h, w)  # updated as route raster
    road_ratser = raster[:, route_type : route_type+road_type, :, :]
    traffic_raster = raster[:, route_type + road_type : route_type + road_type + traffic_light_type, :, :]
    if traffic:
        result = torch.zeros((b, framenum, agent_type + road_type + traffic_light_type + route_type, h, w), device=raster.device)
    else:
        result = torch.zeros((b, framenum, agent_type + road_type + route_type, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, route_type + road_type + traffic_light_type + i::framenum, :, :]
        if traffic:
            raster_i = torch.cat([goal_raster, road_ratser, traffic_raster, agent_raster], dim = 1)  # expected format (b, 1+20+8, h, w)
        else:
            raster_i = torch.cat([goal_raster, road_ratser, agent_raster], dim = 1)
        result[:, i, :, :, :] = raster_i

    return result

def cat_raster_seq_for_waymo(raster, framenum=11):
    b, c, h, w = raster.shape
    agent_type = 3
    road_type = 20
    road_raster = raster[:, :road_type, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, road_type + i::framenum, :, :]
        raster_i = torch.cat([road_raster, agent_raster], dim=1)
        assert raster_i.shape[1] == agent_type + road_type
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