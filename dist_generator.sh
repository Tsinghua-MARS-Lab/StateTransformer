#!/usr/bin/env bash

# mini
# python generation.py  --num_proc 40 --sample_interval 100  --dataset_name mini_dic  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_demo/  \
#         --data_path mini  --only_data_dic

python generation.py  --num_proc 40 --sample_interval 1  --dataset_name mini_index  --starting_file_num 0  \
        --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_demo/  \
        --data_path mini  --only_index  

# python generation.py  --num_proc 40 --sample_interval 1 --dataset_name mini_index_full  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_mini_jul  --data_path mini --save_map


# boston
# python generation.py  --num_proc 40 --sample_interval 100  --dataset_name boston_index_demo  --starting_file_num 0  \
#     --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_demo/  
#     --data_path train_boston  --only_data_dic

# python generation.py  --num_proc 40 --sample_interval 100  --dataset_name boston_index_interval100  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_demo/  \
#         --data_path train_boston  --only_index  

# python generation.py  --num_proc 40 --sample_interval 1 --dataset_name boston_index_full  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_boston_jul  --data_path train_boston --save_map


# pittsburgh                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
# python generation.py  --num_proc 40 --sample_interval 1 --dataset_name pittsburgh_index_full  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_pittsburgh_jul  --data_path train_pittsburgh --save_map


# vegas
# python generation.py  --num_proc 40 --sample_interval 1  --dataset_name vegas2_datadic_float32  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/vegas2_datadic_float32  --data_path train_vegas_2 --save_map


# test
# python generation.py  --num_proc 40 --sample_interval 100  --dataset_name test_dic  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_demo/  \
#         --data_path test  --only_data_dic

# python generation.py  --num_proc 40 --sample_interval 1  --dataset_name test_index  --starting_file_num 0  \
#         --ending_file_num 10000  --cache_folder /data_3/madanjiao/nuplan/online_demo/  \
#         --data_path test  --only_index  
