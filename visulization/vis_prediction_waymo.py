import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from tqdm import tqdm

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

    def visualize_result(self, predictions, current_time_idx=10, visualize_past=True, output_file=None):    
        f, ax = plt.subplots(1, 1, figsize=(50, 50))
        all_polylines = self.visualize_map(ax)
        ax.set_xlim([all_polylines[:, 0].min(), all_polylines[:, 0].max()])
        ax.set_ylim([all_polylines[:, 1].min(), all_polylines[:, 1].max()])
        ax.set_aspect('equal')

        # visualize the ground truth trajectory
        for prediction in predictions:
            gt_trajs = prediction['gt_trajs']
            object_id = prediction['object_id']
            pred_trajs = prediction['pred_trajs'][0]
            pred_scores = prediction['pred_scores']

            color = np.random.rand(3,)
            # visualize the gt trajectory
            ax.scatter(
                gt_trajs[:, 0],
                gt_trajs[:, 1],
                linewidths=3,
                color=color,
                alpha=1.0,
                facecolors='none',
                zorder=5,
            )

            # visualize the trajectory prediction
            ax.plot(
                pred_trajs[:, 0],
                pred_trajs[:, 1],
                linewidth=3,
                color=color,
                alpha=1.0,
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
                alpha=1.0,
                facecolors=color,
                zorder=5,
            )

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

# python visualization/vis_prediction_waymo.py --pkl_file result.pkl  
#   --out_folder /data_3/madanjiao/model_res/vector_gpt_1.5B_mse_FI1_PI1_k1/training_results/checkpoint-145000/figs

def main():
    args = parse_config()
    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)

    scenario_folder = args.scenario_folder
    test_files = os.path.join(args.scenario_folder, f'{args.dataset}/*')

    filenames = tf.io.matching_files(test_files)
    print('---- ', len(filenames))
    for filename in tqdm(filenames):
        shard_dataset = tf.data.TFRecordDataset(filename)
        shard_iterator = shard_dataset.as_numpy_iterator()
        for scenario_bytes in shard_iterator:
            scenario = Scenario(scenario_bytes)

        scenario_id = scenario.id
        print(f'process {scenario_id}...')

        # filter the data to get the data with the same scenario_id
        predictions = [data[i] for i in range(len(data)) if data[i]['scenario_id'] == scenario_id]

        # visualize the result
        output_file = os.path.join(args.out_folder, f'{scenario_id}.png')
        os.makedirs(args.out_folder, exist_ok=True)
        
        scenario.visualize_result(predictions, output_file=output_file)
        print(f'saved to {output_file}...')
        
        # break
        
if __name__ == '__main__':
    main()