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
    total_raster_channels = 1 + total_road_types + total_agent_types * len(sample_frames)
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

    # sample and draw the goal
    goal_sample_frame = total_frames - 1
    goal_point = data_dic['agent']['ego']['pose'][goal_sample_frame, :].copy()
    shape = data_dic['agent']['ego']['shape'][scenario_frame_number, :]

    goal_contour = generate_contour_pts((goal_point[1], goal_point[0]), w=shape[1], l=shape[2],
                                    direction=goal_point[3])
    goal_contour = np.array(goal_contour, dtype=np.int32)
    goal_contour_high_res = int(high_res_raster_scale) * goal_contour
    goal_contour_high_res += observation_kwargs["high_res_raster_shape"][0] // 2
    cv2.drawContours(rasters_high_res_channels[0], [goal_contour_high_res], -1, (255, 255, 255), -1)
    goal_contour_low_res = (low_res_raster_scale * goal_contour).astype(np.int64)
    goal_contour_low_res += observation_kwargs["low_res_raster_shape"][0] // 2
    cv2.drawContours(rasters_low_res_channels[0], [goal_contour_low_res], -1, (255, 255, 255), -1)

    # 'map_raster': (n, w, h),  # n is the number of road types and traffic lights types
    xyz = data_dic["road"]["xyz"].copy()
    road_type = data_dic['road']['type'].astype('int32')
    high_res_road = (xyz * high_res_raster_scale).astype('int32')
    low_res_road = (xyz * low_res_raster_scale).astype('int32')
    high_res_road += observation_kwargs["high_res_raster_shape"][0] // 2
    low_res_road += observation_kwargs["low_res_raster_shape"][0] // 2
    for j, pt in enumerate(xyz):
        if abs(pt[0]) > max_dis or abs(pt[1]) > max_dis:
            continue
        cv2.circle(rasters_high_res_channels[road_type[j] + 1], tuple(high_res_road[j, :2]), 1, (255, 255, 255), -1)
        cv2.circle(rasters_low_res_channels[road_type[j] + 1], tuple(low_res_road[j, :2]), 1, (255, 255, 255), -1)

    for i, key in enumerate(data_dic['agent']):
        for j, sample_frame in enumerate(sample_frames):
            pose = data_dic['agent'][key]['pose'][sample_frame, :].copy()
            if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
                continue
            agent_type = int(data_dic['agent'][key]['type'])
            shape = data_dic['agent'][key]['shape'][scenario_frame_number, :]
            # rect_pts = cv2.boxPoints(((rotated_pose[0], rotated_pose[1]),
            #   (shape[1], shape[0]), np.rad2deg(pose[3])))
            rect_pts = generate_contour_pts((pose[1], pose[0]), w=shape[1], l=shape[0],
                                            direction=pose[3])
            rect_pts = np.array(rect_pts, dtype=np.int32)
            # draw on high resolution
            rect_pts_high_res = int(high_res_raster_scale) * rect_pts
            rect_pts_high_res += observation_kwargs["high_res_raster_shape"][0] // 2
            cv2.drawContours(rasters_high_res_channels[1 + total_road_types + agent_type * len(sample_frames) + j],
                             [rect_pts_high_res], -1, (255, 255, 255), -1)
            # draw on low resolution
            rect_pts_low_res = (low_res_raster_scale * rect_pts).astype(np.int64)
            rect_pts_low_res += observation_kwargs["low_res_raster_shape"][0] // 2
            cv2.drawContours(rasters_low_res_channels[1 + total_road_types + agent_type * len(sample_frames) + j],
                             [rect_pts_low_res], -1, (255, 255, 255), -1)

    rasters_high_res = cv2.merge(rasters_high_res_channels).astype(bool)
    rasters_low_res = cv2.merge(rasters_low_res_channels).astype(bool)
    
    result_to_return['high_res_raster'] = np.array(rasters_high_res, dtype=bool)
    result_to_return['low_res_raster'] = np.array(rasters_low_res, dtype=bool)
    # context action computation
    context_action = data_dic['agent']['ego']['pose'][:10]
    context_action[:, 2] = 0 
    result_to_return["context_actions"] = np.array(context_action)

    return result_to_return


def main(args):
    data_path = args.data_path

    observation_kwargs = dict(
        max_dis=500,
        high_res_raster_shape=[224, 224],  # for high resolution image, we cover 50 meters for delicated short-term actions
        high_res_raster_scale=4.0,
        low_res_raster_shape=[224, 224],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
        low_res_raster_scale=0.77,
        past_frame_num=10,
        future_frame_num=80,
        frame_sample_interval=1,
    )

    def yield_data(shards, dl):
        for shard in shards:
            loaded_dic_list = dl.get_next_file(specify_file_index=shard)
            file_name = dl.global_file_names[shard]
            for loaded_dic in loaded_dic_list:                
                total_frames = loaded_dic['total_frames']
                
                observation_dic = get_observation_for_waymo(
                    observation_kwargs, loaded_dic, 10, total_frames, nsm_result=None)
                other_info = {
                    'file_name': file_name,
                    'scenario_id': loaded_dic["scenario_id"],  # empty for NuPlan
                }
                if observation_dic is not None:
                    observation_dic.update(other_info)
                    yield observation_dic
                else:
                    continue
    
    data_loader = WaymoDL(data_path=data_path)
    data_dic = data_loader.get_next_file(0)
    example_dic = get_observation_for_waymo(observation_kwargs, data_dic[0], 10, 91, None)
    print("here")
    file_indices = list(range(data_loader.total_file_num))
    total_file_number = len(file_indices)
    print(f'Loading Dataset,\n  File Directory: {data_path}\n  Total File Number: {total_file_number}')
    
    waymo_dataset = Dataset.from_generator(yield_data,
                                            gen_kwargs={'shards': file_indices, 'dl': data_loader},
                                            writer_batch_size=2, cache_dir=args.cache_folder,
                                            num_proc=args.num_proc)
    print('Saving dataset')
    waymo_dataset.set_format(type="torch")
    waymo_dataset.save_to_disk(os.path.join(args.cache_folder, args.dataset_name))
    print('Dataset saved')

if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--running_mode", type=int, default=1)
    parser.add_argument("--data_path", type=dict, default={
            "WAYMO_DATA_ROOT": "/home/shiduozhang/waymo",
            "SPLIT_DIR": {
                    'train': "processed_scenarios_training", 
                    'test': "processed_scenarios_validation"
                },
            "INFO_FILE": {
                    'train': "processed_scenarios_training_infos.pkl", 
                    'test': "processed_scenarios_val_infos.pkl"
                }
        })
    
    parser.add_argument('--starting_file_num', type=int, default=0)
    parser.add_argument('--ending_file_num', type=int, default=1000)
    parser.add_argument('--starting_scenario', type=int, default=-1)
    parser.add_argument('--cache_folder', type=str, default='/home/shiduozhang/waymo')

    parser.add_argument('--train', default=False, action='store_true')   
    parser.add_argument('--num_proc', type=int, default=1)

    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default='t4p_waymo')

    args_p = parser.parse_args()
    main(args_p)