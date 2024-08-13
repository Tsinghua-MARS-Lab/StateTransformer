import numpy as np
import torch
import torch.nn as nn

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa,  sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, :cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor

def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets

def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1) 
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) 

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
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

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

def _num_to_str(nums):
    string_list = []
    
    for str in nums:
        s = ""
        for char in str:
            if char == -100: continue
            s += chr(char)
        string_list.append(s)

    return string_list

def _str_to_num(string):
    "Encodes `string` to a decodeable number and breaks it up by `batch_size`"
    nums = [[ord(char) for char in str] for str in string]
    length = [len(n) for n in nums]
    max_length = max(length)
    for i in range(len(nums)):
        nums[i] += [-100] * (max_length - length[i])

    return nums

def str_to_tensor(string) -> torch.tensor:
    """
    Encodes `string` to a tensor of shape [1,N,batch_size] where 
    `batch_size` is the number of characters and `n` is
    (len(string)//batch_size) + 1
    """
    return torch.tensor(_str_to_num(string), dtype=torch.long)

def tensor_to_str(x:torch.Tensor) -> str:
    """
    Decodes `x` to a string. `x` must have been encoded from
    `str_to_tensor`
    """
    return _num_to_str(x.tolist())

import yaml

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    return config

import tensorflow as tf

from google.protobuf import text_format

all_gpus = tf.config.experimental.list_physical_devices('GPU')
if all_gpus:
    try:
        for cur_gpu in all_gpus:
            tf.config.experimental.set_memory_growth(cur_gpu, True)
    except RuntimeError as e:
        print(e)

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2


object_type_to_id = {
    'TYPE_UNSET': 0,
    'TYPE_VEHICLE': 1,
    'TYPE_PEDESTRIAN': 2,
    'TYPE_CYCLIST': 3,
    'TYPE_OTHER': 4
}


def _default_metrics_config(eval_second, num_modes_for_eval=6):
    assert eval_second in [3, 5, 8]
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
    }
    """
    config_text += f"""
    max_predictions: {num_modes_for_eval}
    """
    if eval_second == 3:
        config_text += """
        track_future_samples: 30
        """
    elif eval_second == 5:
        config_text += """
        track_future_samples: 50
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
        }
        """

    text_format.Parse(config_text, config)
    return config

def transform_preds_to_waymo_format(pred_dicts, top_k_for_eval=-1, eval_second=8):
    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    if num_future_frames in [30, 50, 80]:
        sampled_interval = 5
    assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
    num_frame_to_eval = num_future_frames // sampled_interval

    if eval_second == 3:
        num_frames_in_total = 41
        num_frame_to_eval = 6
    elif eval_second == 5:
        num_frames_in_total = 61
        num_frame_to_eval = 10
    else:
        num_frames_in_total = 91
        num_frame_to_eval = 16

    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=np.int)
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=np.int)
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.int)
    scenario_id = np.zeros((num_scenario), dtype=np.object)

    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
            cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
            cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]
            cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()

            batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, [0, 1, 3, 4, 6, 7, 8]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, -1]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
            object_id[scene_idx, obj_idx] = cur_pred['object_id']

            object_type_cnt_dict[cur_pred['object_type']] += 1

    gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
        'pred_gt_indices': pred_gt_idxs,
        'pred_gt_indices_mask': pred_gt_idx_valid_mask
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict

def waymo_evaluation(pred_dicts, top_k=-1, eval_second=8, num_modes_for_eval=6):

    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = transform_preds_to_waymo_format(
        pred_dicts, top_k_for_eval=top_k, eval_second=eval_second,
    )
    eval_config = _default_metrics_config(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval)

    pred_score = tf.convert_to_tensor(pred_score, np.float32)
    pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
    gt_trajs = tf.convert_to_tensor(gt_infos['gt_trajectory'], np.float32)
    gt_is_valid = tf.convert_to_tensor(gt_infos['gt_is_valid'], np.bool)
    pred_gt_indices = tf.convert_to_tensor(gt_infos['pred_gt_indices'], tf.int64)
    pred_gt_indices_mask = tf.convert_to_tensor(gt_infos['pred_gt_indices_mask'], np.bool)
    object_type = tf.convert_to_tensor(gt_infos['object_type'], tf.int64)

    metric_results = py_metrics_ops.motion_metrics(
        config=eval_config.SerializeToString(),
        prediction_trajectory=pred_trajs,  # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, )
        prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
        ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
        ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
        prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
        prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (batch_size, num_pred_groups, num_agents_per_group)
        object_type=object_type  # (batch_size, num_total_agents)
    )

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)

    result_dict = {}
    avg_results = {}
    for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']):
        avg_results.update({
            f'{m} - VEHICLE': [0.0, 0], f'{m} - PEDESTRIAN': [0.0, 0], f'{m} - CYCLIST': [0.0, 0]
        })
        for j, n in enumerate(metric_names):
            cur_name = n.split('_')[1]
            avg_results[f'{m} - {cur_name}'][0] += float(metric_results[i][j])
            avg_results[f'{m} - {cur_name}'][1] += 1
            result_dict[f'{m} - {n}\t'] = float(metric_results[i][j])

    for key in avg_results:
        avg_results[key] = avg_results[key][0] / avg_results[key][1]

    result_dict.update(avg_results)

    final_avg_results = {}
    result_format_list = [
        ['Waymo', 'mAP', 'minADE', 'minFDE', 'MissRate', '\n'],
        ['VEHICLE', None, None, None, None, '\n'],
        ['PEDESTRIAN', None, None, None, None, '\n'],
        ['CYCLIST', None, None, None, None, '\n'],
        ['Avg', None, None, None, None, '\n'],
    ]
    name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
    name_to_col = {'mAP': 1, 'minADE': 2, 'minFDE': 3, 'MissRate': 4}

    for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP']:
        final_avg_results[cur_metric_name] = 0
        for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
            final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name} - {cur_name}']

            result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = '%.4f,' % avg_results[f'{cur_metric_name} - {cur_name}']

        final_avg_results[cur_metric_name] /= 3
        result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]

    result_format_str = ' '.join([x.rjust(12) for items in result_format_list for x in items])

    result_dict.update(final_avg_results)
    result_dict.update(object_type_cnt_dict)

    return result_dict, result_format_str