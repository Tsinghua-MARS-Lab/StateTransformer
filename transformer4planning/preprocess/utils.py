from typing import Dict, List, Tuple

import numpy as np
import os
import cv2
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMapFactory,
)

from nuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)
def compute_derivative(x, interval=0.05, drop_z_axis=True):
    """
    compute derivative of array x, such as velocity&acceleration
    computation from positions&velocities

    input:
        x: origin array, shape (bsz, 4) or (bsz, 3) for positions&velocities
        interval: time interval between two frames, default to 0.05 in 20hz
        drop_z_axis: whether to drop z axis, default to True
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    derivative = list()
    for i in range(x.shape[0] - 1):
        derivative.append((x[i + 1, ...] - x[i, ...]) / interval)
    padded_item = derivative[-1]
    derivative.append(padded_item)
    derivative = np.array(derivative)
    if drop_z_axis:
        derivative = np.concatenate([derivative[:, :2], derivative[:, -1].reshape(-1, 1)], axis=1)
    return derivative

def route_roadblock_correction(
    ego_position,
    map_api: AbstractMap,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    """
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_position, map_api, route_roadblock_dict
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route roadblock
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            if path_found:
                end_roadblock_idx = np.argmax(
                    np.array(route_roadblock_ids) == path_id[-1]
                )

                route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
                route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]

                route_roadblocks[:0] = path
                route_roadblock_ids[:0] = path_id

    # Fix 2: check if roadblocks are linked, search for links if not
    roadblocks_to_append = {}
    for i in range(len(route_roadblocks) - 1):
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    return route_roadblock_ids

def remove_route_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[str], List[RoadBlockGraphEdgeMapObject]]:
    """
    Remove ending of route, if the roadblock are intersecting the route (forming a loop).
    :param route_roadblocks: input route roadblocks
    :param route_roadblock_ids: input route roadblocks ids
    :return: tuple of ids and roadblocks of route without loops
    """

    roadblock_occupancy_map = None
    loop_idx = None

    for idx, roadblock in enumerate(route_roadblocks):
        # loops only occur at intersection, thus searching for roadblock-connectors.
        if str(roadblock.__class__.__name__) == "NuPlanRoadBlockConnector":
            if not roadblock_occupancy_map:
                roadblock_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry(
                    [roadblock.polygon], [roadblock.id]
                )
                continue

            strtree, index_by_id = roadblock_occupancy_map._build_strtree()
            indices = strtree.query(roadblock.polygon)
            if len(indices) > 0:
                for geom in strtree.geometries.take(indices):
                    area = geom.intersection(roadblock.polygon).area
                    if area > 1:
                        loop_idx = idx
                        break
                if loop_idx:
                    break

            roadblock_occupancy_map.insert(roadblock.id, roadblock.polygon)

    if loop_idx:
        route_roadblocks = route_roadblocks[:loop_idx]
        route_roadblock_ids = route_roadblock_ids[:loop_idx]

    return route_roadblocks, route_roadblock_ids

def get_current_roadblock_candidates(
    ego_position,
    map_api: AbstractMap,
    route_roadblocks_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
    """
    Determines a set of roadblock candidate where ego is located
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param heading_error_thresh: maximum heading error, defaults to np.pi/4
    :param displacement_error_thresh: maximum displacement, defaults to 3
    :return: tuple of most promising roadblock and other candidates
    """
    ego_pose = StateSE2(x=ego_position[0], y=ego_position[1], heading=ego_position[-1])
    roadblock_candidates = []

    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_pose.point, radius=1.0, layers=layers
    )
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )

    if not roadblock_candidates:
        for layer in layers:
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer
            )
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            if roadblock:
                roadblock_candidates.append(roadblock)

    on_route_candidates, on_route_candidate_displacement_errors = [], []
    candidates, candidate_displacement_errors = [], []

    roadblock_displacement_errors = []
    roadblock_heading_errors = []

    for idx, roadblock in enumerate(roadblock_candidates):
        lane_displacement_error, lane_heading_error = np.inf, np.inf

        for lane in roadblock.interior_edges:
            lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path], dtype=np.float64
            )
            lane_state_distances = (
                (lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0
            ).sum(axis=-1) ** 0.5
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(lane_discrete_path[argmin].heading - ego_pose.heading)
            )
            displacement_error = lane_state_distances[argmin]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if (
                heading_error < heading_error_thresh
                and displacement_error < displacement_error_thresh
            ):
                if roadblock.id in route_roadblocks_dict.keys():
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(displacement_error)
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        roadblock_displacement_errors.append(lane_displacement_error)
        roadblock_heading_errors.append(lane_heading_error)

    if on_route_candidates:  # prefer on-route roadblocks
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)],
            on_route_candidates,
        )
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close roadblock
    return (
        roadblock_candidates[np.argmin(roadblock_displacement_errors)],
        roadblock_candidates,
    )

def save_raster(result_dic, debug_raster_path, agent_type_num, past_frames_num, image_file_name, split,
                high_scale, low_scale):
    # save rasters
    path_to_save = debug_raster_path
    # check if path not exist, create
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        file_number = 0
    else:
        file_number = len(os.listdir(path_to_save))
        if file_number > 200:
            return
    image_shape = None
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = result_dic[each_key]
        goal = each_img[:, :, 0]
        road = each_img[:, :, :21]
        traffic_lights = each_img[:, :, 21:25]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = np.random.randint(0, 255, size=(21, 3)) * 0.5
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float32)
        image_shape = target_image.shape
        for i in range(21):
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()

        # print('test: ', past_frames_num, agent_type_num, agent.shape)
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                # if j == 7:
                #     print('debug', np.sum(agent[:, :, j * 9 + i]), agent[:, :, j * 9 + i])
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '.png'), target_image)
    for each_key in ['context_actions', 'trajectory_label']:
        pts = result_dic[each_key]
        for scale in [high_scale, low_scale]:
            target_image = np.zeros(image_shape, dtype=np.float32)
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, :] = [255, 255, 255]
            cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '_' + str(scale) +'.png'), target_image)
    print('length of action and labels: ', result_dic['context_actions'].shape, result_dic['trajectory_label'].shape)
    print('debug images saved to: ', path_to_save, file_number)

def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, no_time_dim=False):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    if no_time_dim:
        num_objects, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_attrs).repeat(num_center_objects, 1,  1)
        obj_trajs[:, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, :]
        obj_trajs[:, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, 2)

        obj_trajs[:, :, heading_index] -= center_heading[:, None]
    else:
        num_objects, num_frame, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_frame, num_attrs).repeat(num_center_objects, 1,  1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_frame, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    return obj_trajs

def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

    # # CHECK the results
    # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
    # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
    # assert center_dist.max() < 10
    return ret_polylines, ret_polylines_mask

def create_map_data_for_center_objects(center_objects, heading, map_infos, center_offset):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """
    num_center_objects = center_objects.shape[0]

    # transform object coordinates by center objects
    def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-heading
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
            angle=-heading
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
        xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    polylines = torch.from_numpy(map_infos['all_polylines'].copy())

    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines.numpy(), point_sampled_interval=1,
        vector_break_dist_thresh=1.0,
        num_points_each_polyline=20,
    )  # (num_polylines, num_points_each_polyline, 9), (num_polylines, num_points_each_polyline)

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = 768
    map_polylines = torch.zeros((num_center_objects, num_of_src_polylines, batch_polylines.shape[-2], batch_polylines.shape[-1]))
    map_polylines_mask = torch.zeros((num_center_objects, num_of_src_polylines, batch_polylines.shape[-2]))
    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
        center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
        center_offset_rot = rotate_points_along_z(
            points=center_offset_rot.view(num_center_objects, 1, 2),
            angle=heading
        ).view(num_center_objects, 2)

        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

        dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
        topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
        map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines[:, :len(batch_polylines), :, :] = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
        map_polylines_mask[:, :len(batch_polylines), :] = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )

    map_polylines = map_polylines.numpy()
    map_polylines_mask = map_polylines_mask.numpy()

    return map_polylines, map_polylines_mask

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

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False