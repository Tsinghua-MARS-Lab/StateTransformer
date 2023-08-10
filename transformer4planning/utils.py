from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: str = field(
        default="scratch-mini-gpt-raster",
        metadata={"help": "Name of a planning model backbone"}
    )
    model_pretrain_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    predict_result_saving_dir: Optional[str] = field(
        default=False,
        metadata={"help": "The target folder to save prediction results."},
    )
    predict_trajectory: Optional[bool] = field(
        default=True,
    )
    d_embed: Optional[int] = field(
        default=256,
    )
    d_model: Optional[int] = field(
        default=256,
    )
    d_inner: Optional[int] = field(
        default=1024,
    )
    n_layers: Optional[int] = field(
        default=4,
    )
    n_heads: Optional[int] = field(
        default=8,
    )
    # Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    activation_function: Optional[str] = field(
        default="silu"
    )
    loss_fn: Optional[str] = field(
        default="mse",
    )
    task: Optional[str] = field(
        default="nuplan" # only for mmtransformer
    )
    with_traffic_light: Optional[bool] = field(
        default=True
    )
    autoregressive: Optional[bool] = field(
        default=False
    )
    k: Optional[int] = field(
        default=1,
        metadata={"help": "Set k for top-k predictions, set to -1 to not use top-k predictions."},
    )
    next_token_scorer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use next token scorer for prediction."},
    )
    past_seq: Optional[int] = field(
        default=10,
        metadata={"help": "past frames to include for prediction/planning."},
    )
    x_random_walk: Optional[float] = field(
        default=0.0
    )
    y_random_walk: Optional[float] = field(
        default=0.0
    )
    tokenize_label: Optional[bool] = field(
        default=True
    )
    raster_channels: Optional[int] = field(
        default=33,
        metadata={"help": "default is 0, automatically compute. [WARNING] only supports nonauto-gpt now."},
    )
    predict_yaw: Optional[bool] = field(
        default=False
    )
    ar_future_interval: Optional[int] = field(
        default=20,
        metadata={"help": "default is 0, don't use auturegression. [WARNING] only supports nonauto-gpt now."},
    )
    arf_x_random_walk: Optional[float] = field(
        default=0.0
    )
    arf_y_random_walk: Optional[float] = field(
        default=0.0
    )
    trajectory_loss_rescale: Optional[float] = field(
        default=1.0
    )
    visualize_prediction_to_path: Optional[str] = field(
        default=None
    )
    pred_key_points_only: Optional[bool] = field(
        default=False
    )
    specified_key_points: Optional[bool] = field(
        default=True
    )
    forward_specified_key_points: Optional[bool] = field(
        default=True
    )
    token_scenario_tag: Optional[bool] = field(
        default=False
    )
    token_scenario_tag: Optional[bool] = field(
        default=False
    )
    max_token_len: Optional[int] = field(
        default=20
    )
    resnet_type: Optional[str] = field(
        default="resnet18"
    )
    pretrain_encoder: Optional[bool] = field(
        default=False
    )
    encoder_type: Optional[str] = field(
        default='raster'
    )
    past_sample_interval: Optional[int] = field(
        default=5
    )
    future_sample_interval: Optional[int] = field(
        default=2
    )
    debug_raster_path: Optional[str] = field(
        default=None
    )


def rotate_array(origin, points, angle, tuple=False):
    """
    Rotate a numpy array of points counter-clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    assert isinstance(points, type(np.array([]))), type(points)
    ox, oy = origin
    px = points[:, 0]
    py = points[:, 1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if tuple:
        return (qx, qy)
    else:
        rst_array = np.zeros_like(points)
        rst_array[:, 0] = qx
        rst_array[:, 1] = qy
        return rst_array


def change_coordination(target_point, ego_center, ego_to_global=False):
    target_point_new = target_point.copy()
    if ego_to_global:
        cos_, sin_ = math.cos(ego_center[-1]), math.sin(ego_center[-1])
        # ego to global
        new_x, new_y = target_point_new[0] * cos_ - target_point_new[1] * sin_, \
                       target_point_new[0] * sin_ + target_point_new[1] * cos_
        target_point_new[0], target_point_new[1] = new_x, new_y
        target_point_new[:2] += ego_center[:2]
    else:
        cos_, sin_ = math.cos(-ego_center[-1]), math.sin(-ego_center[-1])
        target_point_new[:2] -= ego_center[:2]
        # global to ego
        new_x, new_y = target_point_new[0] * cos_ - target_point_new[1] * sin_, \
                       target_point_new[0] * sin_ + target_point_new[1] * cos_
        target_point_new[0], target_point_new[1] = new_x, new_y
    return target_point_new


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

def check_collision(checking_agent, target_agent):
    # return check_collision_for_two_agents_dense_scipy(checking_agent, target_agent)  # slower
    # return check_collision_for_two_agents_dense(checking_agent, target_agent)
    return check_collision_for_two_agents_rotate_and_dist_check(checking_agent=checking_agent,
                                                                target_agent=target_agent)

def check_collision_for_two_agents_rotate_and_dist_check(checking_agent, target_agent, vertical_margin=0.7, vertical_margin2=0.7, horizon_margin=0.7):
    # center_c = [checking_agent.x, checking_agent.y]
    # center_t = [target_agent.x, target_agent.y]

    length_sum_top_threshold = checking_agent.length + target_agent.length
    if checking_agent.x == -1 or target_agent.x == -1:
        return False
    if abs(checking_agent.x - target_agent.x) > length_sum_top_threshold:
        return False
    if abs(checking_agent.y - target_agent.y) > length_sum_top_threshold:
        return False

    if euclidean_distance([checking_agent.x, checking_agent.y], [target_agent.x, target_agent.y]) <= (checking_agent.width + target_agent.width)/2:
        return True
    collision_box_t = [(target_agent.x - target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y - target_agent.length/2 * vertical_margin2 - checking_agent.y),
                       (target_agent.x - target_agent.width / 2 * horizon_margin - checking_agent.x,
                        target_agent.y - checking_agent.y),
                       (target_agent.x - target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y + target_agent.length/2 * vertical_margin2 - checking_agent.y),
                       (target_agent.x + target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y + target_agent.length/2 * vertical_margin2 - checking_agent.y),
                       (target_agent.x + target_agent.width / 2 * horizon_margin - checking_agent.x,
                        target_agent.y - checking_agent.y),
                       (target_agent.x + target_agent.width/2 * horizon_margin - checking_agent.x,
                        target_agent.y - target_agent.length/2 * vertical_margin2 - checking_agent.y)]
    rotated_checking_box_t = rotate_array(origin=(target_agent.x - checking_agent.x, target_agent.y - checking_agent.y),
                                          points=np.array(collision_box_t),
                                          angle=normalize_angle( - target_agent.yaw))
    rotated_checking_box_t = np.insert(rotated_checking_box_t, 0, [target_agent.x - checking_agent.x, target_agent.y - checking_agent.y], 0)

    rotated_checking_box_t = rotate_array(origin=(0, 0),
                                          points=np.array(rotated_checking_box_t),
                                          angle=normalize_angle( - checking_agent.yaw))

    rst = False
    for idx, pt in enumerate(rotated_checking_box_t):
        x, y = pt
        if abs(x) < checking_agent.width/2 * horizon_margin and abs(y) < checking_agent.length/2 * vertical_margin:
            rst = True
            # print('test: ', idx)
            break
    return rst


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle

def generate_contour_pts(center_pt, w, l, direction):
    pt1 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]-l/2), direction, tuple=True)
    pt2 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]-l/2), direction, tuple=True)
    pt3 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]+l/2), direction, tuple=True)
    pt4 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]+l/2), direction, tuple=True)
    return pt1, pt2, pt3, pt4

def rotate(origin, point, angle, tuple=False):
    """
    Rotate a point counter-clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if tuple:
        return (qx, qy)
    else:
        return qx, qy