import os
import pickle
import multiprocessing
import argparse
from functools import partial

def split_file(file_name, data_path=None, output_path=None, delete_origin=False):
    with open(os.path.join(data_path, file_name), "rb") as f:
        data_dict = pickle.load(f)

    for key in data_dict.keys():
        output_file = os.path.join(output_path, key + ".pkl")
        with open(output_file, "wb") as file:
            pickle.dump(data_dict[key], file)
    
    if delete_origin: os.remove(os.path.join(data_path, file_name))

def split_scenarios(args):
    dirs = os.listdir(args.data_path)

    if os.path.isdir(os.path.join(args.data_path, dirs[0])):
        for d in dirs:
            data_path = os.path.join(args.data_path, d)
            output_path = os.path.join(args.output_path, d)
            os.makedirs(output_path)

            files = os.listdir(data_path)
            assert os.path.isfile(os.path.join(data_path, files[0]))

            func = partial(split_file, data_path=data_path, output_path=output_path, delete_origin=args.delete_origin)
            with multiprocessing.Pool(args.num_proc) as p:
                p.map(func, files)
        
            print("Split files in", data_path, "to", output_path)

    elif os.path.isfile(os.path.join(args.data_path, dirs[0])):
        os.makedirs(args.output_path)

        func = partial(split_file, data_path=args.data_path, output_path=args.output_path, delete_origin=args.delete_origin)
        with multiprocessing.Pool(args.num_proc) as p:
            p.imap(func, dirs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--data_path", type=str, default="/localdata_ssd/liderun/t4p_waymo_full/origin")
    parser.add_argument('--output_path', type=str, default="/localdata_ssd/liderun/t4p_waymo_full/")
    parser.add_argument('--delete_origin', type=bool, default=False)
    parser.add_argument('--num_proc', type=int, default=50)

    args = parser.parse_args()
    split_scenarios(args)