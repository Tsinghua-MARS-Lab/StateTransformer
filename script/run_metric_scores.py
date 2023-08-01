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
    default = "/home/zhangsd/nuplan/exp/exp/kp_30m/closed_loop_reactive_agents/2023.07.31.23.20.05/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.07.31.23.20.05.parquet"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="/home/zhangsd/nuplan/metrics"
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="gpt30m_kp"
)
parser.add_argument(
    "--simulation_type",
    type=str,
    default="open_loop"
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