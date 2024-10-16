"""
This script is used to convert parquet file to csv file, to statics the metric scores.
"""
import pyarrow.parquet as pq 
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_path",
    type=str,
    default="{PATH_TO_SIMULATION}"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="{PATH_TO_SAVE_METRICS}"
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="gpt30m_kp"
)
parser.add_argument(
    "--simulation_type",
    type=str,
    default="open_loop_l2"
)

def convert_parquet_to_csv(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # assume metrics follow nuplan simulation's default setting
    aggregator_metric_folder = os.path.join(args.file_path, "aggregator_metric")
    if os.path.isdir(aggregator_metric_folder):
        for file in os.listdir(aggregator_metric_folder):
            if file.endswith(".parquet"):
                table = pq.read_table(os.path.join(aggregator_metric_folder, file))
                df = table.to_pandas()
                save_folder = os.path.join(args.save_dir, f"{args.exp_name}", f"{args.simulation_type}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = os.path.join(args.save_dir, f"{args.exp_name}", f"{args.simulation_type}", f"{file.split('.')[0]}.csv")
                df.to_csv(save_path, index=False)
                print(f"save {save_path}")
    metrics_folder = os.path.join(args.file_path, "metrics")
    if os.path.isdir(metrics_folder):
        for file in os.listdir(metrics_folder):
            if file.endswith(".parquet"):
                table = pq.read_table(os.path.join(metrics_folder, file))
                df = table.to_pandas()
                save_folder = os.path.join(args.save_dir, f"{args.exp_name}", f"{args.simulation_type}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = os.path.join(args.save_dir, f"{args.exp_name}", f"{args.simulation_type}", f"{file.split('.')[0]}.csv")
                df.to_csv(save_path, index=False)
                print(f"save {save_path}")
    # save_path = os.path.join(args.save_dir, f"{args.exp_name}", f"{args.simulation_type}.csv")
    # df.to_csv(save_path, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    convert_parquet_to_csv(args)
