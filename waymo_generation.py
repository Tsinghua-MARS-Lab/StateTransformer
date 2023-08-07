import pickle

from datasets import Dataset, Features, Sequence, Value
from dataset_gen.DataLoaderWaymo import WaymoDL

import os
import logging
import argparse

import pickle

import math
import numpy as np
import cv2
import shapely

import torch
from waymo_open_dataset.protos import scenario_pb2
from waymo_process_to_pickles.datapreprocess import decode_tracks_from_proto, decode_map_features_from_proto

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

def generate_contour_pts(center_pt, w, l, direction):
    pt1 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]-l/2), direction, tuple=True)
    pt2 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]-l/2), direction, tuple=True)
    pt3 = rotate(center_pt, (center_pt[0]+w/2, center_pt[1]+l/2), direction, tuple=True)
    pt4 = rotate(center_pt, (center_pt[0]-w/2, center_pt[1]+l/2), direction, tuple=True)
    return pt1, pt2, pt3, pt4
    
def get_observation_for_waymo(observation_kwargs, data_dic, scenario_frame_number, total_frames, nsm_result=None):
    # hyper parameters setting
    max_dis = observation_kwargs["max_dis"]
    high_res_raster_shape = observation_kwargs["high_res_raster_shape"]
    low_res_raster_shape = observation_kwargs["low_res_raster_shape"]
    assert len(high_res_raster_shape) == len(
        low_res_raster_shape) == 2, f'{high_res_raster_shape}, {low_res_raster_shape}'
    high_res_raster_scale = observation_kwargs["high_res_raster_scale"]
    low_res_raster_scale = observation_kwargs["low_res_raster_scale"]

    result_to_return = {}
    past_frames_number = observation_kwargs["past_frame_num"]
    future_frames_number = observation_kwargs["future_frame_num"]
    frame_sample_interval = observation_kwargs["frame_sample_interval"]

    total_road_types = 20
    total_agent_types = 6
    sample_frames = list(range(scenario_frame_number - past_frames_number, scenario_frame_number, frame_sample_interval))
    sample_frames.append(scenario_frame_number)
    total_raster_channels = total_road_types + total_agent_types * len(sample_frames)
    rasters_high_res = np.zeros([high_res_raster_shape[0],
                                 high_res_raster_shape[1],
                                 total_raster_channels], dtype=np.uint8)
    rasters_low_res = np.zeros([low_res_raster_shape[0],
                                low_res_raster_shape[1],
                                total_raster_channels], dtype=np.uint8)
    rasters_high_res_channels = cv2.split(rasters_high_res)
    rasters_low_res_channels = cv2.split(rasters_low_res)

    # trajectory label
    trajectory_label = data_dic['agent']['ego']['pose'][
                       scenario_frame_number + 1:scenario_frame_number + future_frames_number + 1, :].copy()
    trajectory_label[:, 2] = 0
    result_to_return['trajectory_label'] = np.array(trajectory_label)

    # 'map_raster': (n, w, h),  # n is the number of road types and traffic lights types
    xyz = data_dic["road"]["xyz"].copy()
    road_type = data_dic['road']['type'].astype('int32')
    high_res_road = (xyz * high_res_raster_scale).astype('int32')
    low_res_road = (xyz * low_res_raster_scale).astype('int32')
    high_res_road += observation_kwargs["high_res_raster_shape"][0] // 2
    low_res_road += observation_kwargs["low_res_raster_shape"][0] // 2
    high_res_road[:, 0] = 224 - high_res_road[:, 0]
    low_res_road[:, 0] = 224 - low_res_road[:, 0]
    for j, pt in enumerate(xyz):
        if abs(pt[0]) > max_dis or abs(pt[1]) > max_dis:
            continue
        cv2.circle(rasters_high_res_channels[road_type[j]], tuple(high_res_road[j, :2]), 1, (255, 255, 255), -1)
        cv2.circle(rasters_low_res_channels[road_type[j]], tuple(low_res_road[j, :2]), 1, (255, 255, 255), -1)

    for i, key in enumerate(data_dic['agent']):
        for j, sample_frame in enumerate(sample_frames):
            pose = data_dic['agent'][key]['pose'][sample_frame, :].copy()
            if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
                continue
            agent_type = int(data_dic['agent'][key]['type'])
            shape = data_dic['agent'][key]['shape'][scenario_frame_number, :]
            # rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
            #   (shape[1], shape[0]), np.rad2deg(pose[3])))
            rect_pts = generate_contour_pts((pose[1], pose[0]), w=3, l=3,
                                            direction=pose[3])
            rect_pts = np.array(rect_pts, dtype=np.int32)
            # draw on high resolution
            rect_pts_high_res = int(high_res_raster_scale) * rect_pts
            rect_pts_high_res += observation_kwargs["high_res_raster_shape"][0] // 2
            rect_pts_high_res[:, 0] = 224 - rect_pts_high_res[:, 0]
            cv2.drawContours(rasters_high_res_channels[total_road_types + agent_type * len(sample_frames) + j],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_raster_scale * rect_pts).astype(np.int64)
            rect_pts_low_res += observation_kwargs["low_res_raster_shape"][0] // 2
            rect_pts_low_res[:, 0] = 224 - rect_pts_low_res[:, 0] 
            cv2.drawContours(rasters_low_res_channels[total_road_types + agent_type * len(sample_frames) + j],
                             [rect_pts_low_res], -1, (255, 255, 255), -1)

    rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
    rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)
    
    result_to_return['high_res_raster'] = np.array(rasters_high_res, dtype=bool)
    result_to_return['low_res_raster'] = np.array(rasters_low_res, dtype=bool)
    # context action computation
    context_action = data_dic['agent']['ego']['pose'][:10]
    context_action[:, 2] = 0 
    result_to_return["context_actions"] = np.array(context_action)

    # inspect actions
    max_action = np.max(result_to_return['context_actions'][:, :2])
    min_action = np.min(result_to_return['context_actions'][:, :2])
    if abs(max_action) > 1000 or abs(min_action) > 1000:
        print(result_to_return['context_actions'].shape)
        print(result_to_return['context_actions'][:10, :])
        assert False, f'Invalid actions to filter: {max_action}, {min_action}'

    # inspect labels
    max_label = np.max(result_to_return['trajectory_label'][:, :2])
    min_label = np.min(result_to_return['trajectory_label'][:, :2])
    if abs(max_label) > 1000 or abs(min_label) > 1000:
        print(result_to_return['trajectory_label'].shape)
        print(result_to_return['trajectory_label'][:80, :])
        assert False, f'Invalid labels to filter: {max_label}, {min_label}'

    return result_to_return

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

def create_map_data_for_center_objects(center_objects, heading, map_infos):
    map_polylines = []
    last_polyline_index = 0
    polyline_index = []
    for i, info in enumerate(map_infos['all_polylines']):
        if len(info) == 0: continue
        elif len(info) == 1:
            map_polylines.append(info[:, :2])
            polyline_index.append(last_polyline_index)
            last_polyline_index += 1
            continue 

        pts = list(zip(info[:, 0], info[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy

        assert len(simplified_x) == len(simplified_y) and len(simplified_x) > 0, len(simplified_x)

        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y

        map_polylines.append(simplified_xyz)
        polyline_index.append(last_polyline_index)
        last_polyline_index += len(simplified_xyz)
    
    map_polylines = np.concatenate(map_polylines, axis=0)

    num_objects, num_attrs = map_polylines.shape
    num_center_objects = center_objects.shape[0]
    assert center_objects.shape[0] == heading.shape[0]
    assert center_objects.shape[1] in [3, 2]

    map_polylines = torch.from_numpy(map_polylines).to(torch.float)
    polylines = map_polylines.clone().view(1, num_objects, num_attrs).repeat(num_center_objects, 1,  1)
    polylines[:, :, 0:center_objects.shape[1]] -= center_objects[:, None, :2]
    polylines[:, :, 0:2] = rotate_points_along_z(
        points=polylines[:, :, 0:2].view(num_center_objects, -1, 2),
        angle=-heading
    ).view(num_center_objects, num_objects, 2)

    mask = np.ones_like(polylines)

    return polylines, mask, polyline_index


def create_map_data_for_center_objects_MTR(center_objects, heading, map_infos, center_offset):
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

def process_agents(agent_trajs):
    agent_dict = {}
    for i, traj in enumerate(agent_trajs): 
        first_valid_frame, last_valid_frame = 999, -1
        for t, frame in enumerate(traj):
            if frame[-1] == 1:
                if t < first_valid_frame: first_valid_frame = t
                if t > last_valid_frame: last_valid_frame = t
        
        if last_valid_frame <= first_valid_frame: continue

        if abs(traj[last_valid_frame, 0] - traj[first_valid_frame, 0]) + abs(traj[last_valid_frame, 1] - traj[first_valid_frame, 1]) < 0.1:
            agent_dict[i] = traj[first_valid_frame].to(torch.float32)
        else:
            agent_dict[i] = traj

    return agent_dict

def create_map_data(map_infos):
    map_polylines = []
    last_polyline_index = 0
    polyline_index = []
    for i, info in enumerate(map_infos['all_polylines']):
        if len(info) == 0: continue
        elif len(info) == 1:
            map_polylines.append(info[:, :2])
            polyline_index.append(last_polyline_index)
            last_polyline_index += 1
            continue 

        pts = list(zip(info[:, 0], info[:, 1]))
        line = shapely.geometry.LineString(pts)
        simplified_xyz_line = line.simplify(1)
        simplified_x, simplified_y = simplified_xyz_line.xy

        assert len(simplified_x) == len(simplified_y) and len(simplified_x) > 0, len(simplified_x)

        simplified_xyz = np.ones((len(simplified_x), 2)) * -1
        simplified_xyz[:, 0], simplified_xyz[:, 1] = simplified_x, simplified_y

        map_polylines.append(simplified_xyz)
        polyline_index.append(last_polyline_index)
        last_polyline_index += len(simplified_xyz)
    
    map_polylines = np.concatenate(map_polylines, axis=0, dtype=np.float32)
    mask = np.ones_like(map_polylines, dtype=bool)

    return map_polylines, mask, polyline_index

def main(args):
    data_path = args.data_path

    def yield_data(shards, dl, output_path):
        for shard in shards:
            tf_dataset, file_name = dl.get_next_file(specify_file_index=shard)
            if tf_dataset is None:
                continue
            
            file_name = file_name + ".pkl"
            dicts_to_save = {}
            for data in tf_dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytearray(data.numpy()))

                track_infos = decode_tracks_from_proto(scenario.tracks)
                track_index_to_predict = torch.tensor([cur_pred.track_index for cur_pred in scenario.tracks_to_predict])
                map_infos = decode_map_features_from_proto(scenario.map_features)
                
                agent_trajs = torch.from_numpy(track_infos['trajs'])  # (num_objects, num_timestamp, 10)
                ego_index = torch.tensor([idx for idx in track_index_to_predict if agent_trajs[idx, 10, -1] == 1], dtype=torch.int32)
                # agent_dict = process_agents(agent_trajs=agent_trajs)
                map_polylines_data, map_polylines_mask, polyline_index = create_map_data(map_infos=map_infos,
                        )
                    
                dicts_to_save[scenario.scenario_id] = {
                    "agent_trajs": agent_trajs,
                    "track_index_to_predict": ego_index,
                    "map_polyline": map_polylines_data, 
                    "map_polylines_mask": map_polylines_mask,
                    "polyline_index": polyline_index,
                    }
                
                for ego_id in ego_index:
                    yield {
                        "file_name": file_name,
                        "scenario_id": scenario.scenario_id,
                        "ego_id": ego_id,
                    }

            with open(os.path.join(output_path, file_name), "wb") as f:
                pickle.dump(dicts_to_save, f)
                f.close()
    
    data_loader = WaymoDL(data_path=data_path, mode="train" if args.train else "test")
    file_indices = []
    for i in range(args.num_proc):
        file_indices += range(data_loader.total_file_num)[i::args.num_proc]

    total_file_number = len(file_indices)
    print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}')

    os.makedirs(args.output_path, exist_ok=True)
    waymo_dataset = Dataset.from_generator(yield_data,
                                            gen_kwargs={'shards': file_indices, 'dl': data_loader, 'output_path': args.output_path},
                                            writer_batch_size=10, cache_dir=args.cache_folder,
                                            num_proc=args.num_proc)
    print('Saving dataset')
    waymo_dataset.set_format(type="torch")
    waymo_dataset.save_to_disk(os.path.join(args.cache_folder, args.dataset_name), num_proc=args.num_proc)
    print('Dataset saved')

if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--data_path", type=dict, default={
            "WAYMO_DATA_ROOT": "/public/MARS/datasets/waymo_prediction_v1.2.0/scenario",
            "SPLIT_DIR": {
                    'train': "training", 
                    'test': "validation"
                },
        })
    
    parser.add_argument('--train', default=False, action='store_true')   
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--output_path', type=str, default='/public/MARS/datasets/waymo_prediction_v1.2.0/t4p_training/t4p_waymo')
    parser.add_argument('--cache_folder', type=str, default='/public/MARS/datasets/waymo_prediction_v1.2.0/t4p_training/waymo_cache')
    parser.add_argument('--dataset_name', type=str, default='t4p_waymo')

    args_p = parser.parse_args()
    main(args_p)