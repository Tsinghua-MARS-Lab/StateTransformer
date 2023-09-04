import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from tqdm import tqdm
import torch

import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

class Scenario:
    def __init__(self, scenario_bytes):
        self.scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
        self.id = self.scenario.scenario_id
        self.map_features = self.scenario.map_features

    def visualize_map(self, ax, zorder=1):
        feature_attrs = [
            ('lane', '#d6d6d6'),
            ('road_line', 'gray'),
            ('road_edge', 'black'),
            ('crosswalk', '#f1f289'),
            ('speed_bump', '#fab6e6'),
            ('driveway', '#b3b3b3')
        ]

        # draw the road graph
        all_polylines_x = []
        all_polylines_y = []
        for feature in self.map_features:
            for attr_name, color in feature_attrs:
                if not hasattr(feature, attr_name):
                    continue
                
                attr = getattr(feature, attr_name)
                if hasattr(attr, 'polyline') and attr.polyline:
                    x, y, z = self.poly2list(attr.polyline)
                    all_polylines_x.append(x)
                    all_polylines_y.append(y)
                    ax.plot(x, y, '-', c=color, lw=2, zorder=1)
                elif hasattr(attr, 'polygon') and attr.polygon:
                    x, y, z = self.poly2list(attr.polygon)
                    if len(x) > 0:
                        x += (x[0],)
                        y += (y[0],)
                        all_polylines_x.append(x)
                        all_polylines_y.append(y)
                        ax.plot(x, y, '-', c=color, lw=4, zorder=1)
        all_polylines = np.column_stack((np.concatenate(all_polylines_x), np.concatenate(all_polylines_y)))
        return all_polylines

    def visualize_result(self, predictions, eval_res=None, current_time_idx=10, visualize_past=True, output_file=None):    
        f, ax = plt.subplots(1, 1, figsize=(50, 50))
        all_polylines = self.visualize_map(ax)
        ax.set_xlim([all_polylines[:, 0].min(), all_polylines[:, 0].max()])
        ax.set_ylim([all_polylines[:, 1].min(), all_polylines[:, 1].max()])
        ax.set_aspect('equal')

        # visualize the ground truth trajectory
        for prediction in predictions:
            gt_trajs = prediction['gt_trajs']
            object_id = prediction['object_id']
            pred_scores = torch.from_numpy(prediction['pred_scores']).softmax(0)

            valid_mask = (gt_trajs[:,-1] == 1)
            valid_gt = gt_trajs[valid_mask]
            
            color = np.random.rand(3,)
            # visualize the gt trajectory
            ax.scatter(
                valid_gt[:, 0],
                valid_gt[:, 1],
                linewidths=3,
                color=color,
                alpha=0.5,
                facecolors='none',
                zorder=5,
            )
            # visualize the starting point
            ax.scatter(
                gt_trajs[current_time_idx, 0],
                gt_trajs[current_time_idx, 1],
                marker='o',
                s=200,
                linewidths=3,
                color=color,
                alpha=0.5,
                facecolors=color,
                zorder=5,
            )

            num_mode = prediction['pred_trajs'].shape[0]
            for t_i in range(num_mode):
                pred_trajs = prediction['pred_trajs'][t_i]
                pred_len = pred_trajs.shape[0]
                valid_pred = pred_trajs[valid_mask[-pred_len:]]
                # visualize the trajectory prediction
                ax.plot(
                    valid_pred[:, 0],
                    valid_pred[:, 1],
                    linewidth=2,
                    color=color,
                    alpha=1.0,
                    zorder=5,
                )
                
                ax.text(valid_pred[-1, 0]+1, valid_pred[-1, 1]+1, 's:%.3f'% (pred_scores[t_i]), color=color)
                
                if "pred_kps" in prediction:
                    pred_kps = prediction['pred_kps'][t_i] # (num_kps, 2)
                    ax.scatter(
                        pred_kps[:, 0],
                        pred_kps[:, 1],
                        marker='*',
                        linewidth=5,
                        color=color,
                        alpha=1.0,
                        zorder=5,
                    )

        if eval_res is not None:
            x_shift = all_polylines[:, 0].min()
            y_shift = all_polylines[:, 1].min()
            
            ax.text(x_shift+20, y_shift+25, 'minADE     minFDE     MissRate     OverlapRate    mAP', color=(1,0,0), fontsize=16)
            ax.text(x_shift+1,  y_shift+20, 'VEHICLE', color=(1,0,0), fontsize=16)
            ax.text(x_shift+1,  y_shift+15, 'PEDESTRIAN', color=(1,0,0), fontsize=16)
            ax.text(x_shift+1,  y_shift+10, 'CYCLIST', color=(1,0,0), fontsize=16)
            
            c_i=0
            for m in ['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']:
                ax.text(x_shift+20+c_i, y_shift+20, '%.3f  '% (eval_res[m + ' - VEHICLE']), color=(1,0,0), fontsize=16)
                ax.text(x_shift+20+c_i, y_shift+15, '%.3f  '% (eval_res[m + ' - PEDESTRIAN']), color=(1,0,0), fontsize=16)
                ax.text(x_shift+20+c_i, y_shift+10, '%.3f  '% (eval_res[m + ' - CYCLIST']), color=(1,0,0), fontsize=16)
                c_i += 12
            
        f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # save the image to output_file
        if output_file is not None:
            plt.savefig(output_file)
        plt.close()
        plt.clf()

    @staticmethod
    def poly2list(poly):
        return [list(coord) for coord in zip(*[(point.x, point.y, point.z) for point in poly])]

def parse_config():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize the trajectory prediction result")
    parser.add_argument('--pkl_file', type=str, default='/home/wangyu/projects/MTR-org/waymo_vector_gpt_result.pkl')
    parser.add_argument('--scenario_folder', type=str, default='/mnt/emc/opensource/waymo_open_dataset_motion_v_1_2_0/scenario')
    parser.add_argument('--out_folder', type=str, default='/home/wangyu/projects/MTR-org/demo')
    parser.add_argument('--dataset', type=str, default='validation')
    args = parser.parse_args()
    return args

# python visulization/vis_prediction_waymo.py --pkl_file /data_3/madanjiao/model_res/z_new_gpt_small_k1_KP0_anchored_ep100_gpu7_allType_anchorLogits/training_results/checkpoint-400000/eval_output/result_z_new_gpt_small_k1_KP0_anchored_ep100_gpu7_allType_anchorLogits___checkpoint-400000_20230827-150052.pkl  
#   --out_folder /data_3/madanjiao/model_res/z_new_gpt_small_k1_KP0_anchored_ep100_gpu7_allType_anchorLogits/training_results/checkpoint-400000/figs

def main():
    args = parse_config()
    args.pkl_file = "/data_3/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits/training_results/checkpoint-330000/eval_output/result_vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits___checkpoint-330000_20230904-225356.pkl"
    args.out_folder = "/data_3/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits/training_results/checkpoint-330000/figs_2"

    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    eval_data = data.pop(-1)

    scenario_folder = args.scenario_folder
    test_files = os.path.join(args.scenario_folder, f'{args.dataset}/*')

    filenames = tf.io.matching_files(test_files)
    print(' tot len ', len(filenames))
    file_idx = 0
    for filename in tqdm(filenames):
        shard_dataset = tf.data.TFRecordDataset(filename)
        shard_iterator = shard_dataset.as_numpy_iterator()
        scenario_idx = 0
        for scenario_bytes in shard_iterator:
            scenario = Scenario(scenario_bytes)

            scenario_id = scenario.id
            # print(f'process {scenario_id}...')

            # filter the data to get the data with the same scenario_id
            predictions = [data[i] for i in range(len(data)) if data[i]['scenario_id'] == scenario_id]
            
            if len(predictions) == 0:
                continue

            # visualize the result
            output_file = os.path.join(args.out_folder, f'{file_idx}_{scenario_idx}_{scenario_id}')
            os.makedirs(args.out_folder, exist_ok=True)
            
            scenario.visualize_result(predictions, eval_res=eval_data[scenario_id], output_file=output_file)
            
            scenario_idx += 1
        # print(f'saved to {output_file}...')
        
        file_idx += 1
        
        # break
        
if __name__ == '__main__':
    main()