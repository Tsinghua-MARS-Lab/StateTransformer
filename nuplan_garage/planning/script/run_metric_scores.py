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
    default = "/home/sunq/nuplan/exp/exp/simulation/open_loop_boxes/2023.08.02.22.05.25/metrics/planner_expert_final_l2_error_within_bound.parquet"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="/home/sunq/nuplan/metrics"
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
    table = pq.read_table(args.file_path)
    df = table.to_pandas()
    save_path = os.path.join(args.save_dir, f"{args.exp_name}", f"{args.simulation_type}.csv")
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    convert_parquet_to_csv(args)
