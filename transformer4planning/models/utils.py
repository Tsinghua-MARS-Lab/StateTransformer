import torch
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

    goal_raster = raster[:, 0, :, :].reshape(b, 1, h, w)  # updated as route raster
    road_ratser = raster[:, 1:1+road_type, :, :]
    traffic_raster = raster[:, 1+road_type:1+road_type+traffic_light_type, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type + traffic_light_type + 1, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, 1 + road_type + traffic_light_type + i::framenum, :, :]
        if traffic:
            raster_i = torch.cat([goal_raster, road_ratser, traffic_raster, agent_raster], dim = 1)  # expected format (b, 1+20+8, h, w)
        else:
            raster_i = torch.cat([goal_raster, road_ratser, agent_raster], dim = 1)
        result[:, i, :, :, :] = raster_i
    # return format (batchsize, history_frame_number, channels_per_frame, h, w)
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