from datasets import Dataset
import os
import logging
import argparse

import numpy as np
import math

def main(args):
    total_file_num = 10000
    def yield_data(shards):
        for _ in shards:
            velocity = np.random.uniform(0.5, 1.5)
            theta_past = np.random.uniform(-math.pi, math.pi)
            agent = np.zeros((5, 2))

            agent[0][0] = velocity * 2 * math.cos(theta_past)
            agent[0][1] = velocity * 2 * math.sin(theta_past)
            agent[1][0] = velocity * 1 * math.cos(theta_past)
            agent[1][1] = velocity * 1 * math.sin(theta_past)

            theta_future = theta_past + np.random.randint(-1, 2) * math.pi / 4
            agent[3][0] = velocity * (-1) * math.cos(theta_future)
            agent[3][1] = velocity * (-1) * math.sin(theta_future)
            agent[4][0] = velocity * (-2) * math.cos(theta_future)
            agent[4][1] = velocity * (-2) * math.sin(theta_future)

            
                
            yield {"agent_trajs": agent}

    file_indices = []
    for i in range(args.num_proc):
        file_indices += range(total_file_num)[i::args.num_proc]

    total_file_number = len(file_indices)
    print(f'Total File Number: {total_file_number}')
    
    demo_dataset = Dataset.from_generator(yield_data,
                                            gen_kwargs={'shards': file_indices},
                                            writer_batch_size=10, cache_dir=args.cache_folder,
                                            num_proc=args.num_proc)
    print('Saving dataset')
    demo_dataset.set_format(type="torch")
    demo_dataset.save_to_disk(os.path.join(args.cache_folder, args.dataset_name), num_proc=args.num_proc)
    print('Dataset saved')

if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())


    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--cache_folder', type=str, default='/localdata_ssd/liderun/demo_cache')

    parser.add_argument('--train', default=False, action='store_true')   
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default='t4p_demo')

    args_p = parser.parse_args()
    main(args_p)