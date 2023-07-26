#!/usr/bin/env bash

# python -m torch.distributed.run --nproc_per_node=8 runner.py \
#         --model_name scratch-gpt --model_pretrain_name_or_path None \
#         --saved_dataset_folder  /data_3/madanjiao/nuplan/online_demo/mini_index_interval100 \
#         --output_dir /data_3/madanjiao/model_res/gpt_1.5B_mse_FI1_PI1_k1/training_results  \
#         --logging_dir /data_3/madanjiao/model_res/gpt_1.5B_mse_FI1_PI1_k1/training_logs \
#         --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
#         --per_device_train_batch_size 1 --warmup_steps 50 \
#         --weight_decay 0.01 --logging_steps 2 --save_strategy steps \
#         --save_steps 1000 --dataloader_num_workers 10 \
#         --save_total_limit 2  --predict_trajectory True \
#         --dataloader_drop_last True --do_train \
#         --d_embed 1600 --d_model 1600 --d_inner 6400 --n_layers 48 --n_heads 25 \
#         --activation_function silu --dataset_scale 1 \
#         --task nuplan --with_traffic_light True --k 1 \
#         --online_preprocess True \
#         --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_index_demo \
#         --remove_unused_columns False --future_sample_interval 1 \
#         --past_sample_interval 1 --do_eval \
#         --evaluation_strategy steps --eval_steps 100 \
#         --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_index_interval100 \
#         --overwrite_output_dir --loss_fn mse

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run --nproc_per_node=8 runner_waymo.py \
        --model_name scratch-vector-gpt --model_pretrain_name_or_path None \
        --saved_dataset_folder  /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
        --output_dir /data_3/madanjiao/model_res/vector_gpt_1.5B_mse_FI1_PI1_k1/training_results  \
        --logging_dir /data_3/madanjiao/model_res/vector_gpt_1.5B_mse_FI1_PI1_k1/training_logs \
        --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
        --per_device_train_batch_size 16 --warmup_steps 50 \
        --weight_decay 0.01 --logging_steps 2 --save_strategy steps \
        --save_steps 1000 --dataloader_num_workers 10 \
        --save_total_limit 2  --predict_trajectory True \
        --dataloader_drop_last True --do_train \
        --d_embed 256 --d_model 256 --d_inner 1024 --n_layers 4 --n_heads 4 \
        --activation_function silu --dataset_scale 1 \
        --task nuplan --with_traffic_light True --k 1 \
        --online_preprocess True \
        --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_demo \
        --remove_unused_columns False --future_sample_interval 1 \
        --past_sample_interval 1 \
        --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
        --overwrite_output_dir --loss_fn mse

# python -m torch.distributed.run --nproc_per_node=1 runner.py \
#         --model_name scratch-gpt --model_pretrain_name_or_path None \
#         --saved_dataset_folder  /data_3/madanjiao/nuplan/online_demo/boston_demo/index/train/train-index_boston \
#         --output_dir /data_3/madanjiao/model_res/gpt_1.5B_mse_FI1_PI1_k1/training_results  \
#         --logging_dir /data_3/madanjiao/model_res/gpt_1.5B_mse_FI1_PI1_k1/training_logs \
#         --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
#         --per_device_train_batch_size 1 --warmup_steps 50 \
#         --weight_decay 0.01 --logging_steps 2 --save_strategy steps \
#         --save_steps 1000 --dataloader_num_workers 10 \
#         --save_total_limit 2  --predict_trajectory True \
#         --dataloader_drop_last True --do_train \
#         --d_embed 1600 --d_model 1600 --d_inner 6400 --n_layers 48 --n_heads 25 \
#         --activation_function silu --dataset_scale 1 \
#         --task nuplan --with_traffic_light True --k 1 \
#         --online_preprocess True \
#         --datadic_path /data_3/madanjiao/nuplan/online_demo/boston_demo \
#         --remove_unused_columns False --future_sample_interval 1 \
#         --past_sample_interval 1 \
#         --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/boston_demo/index/train/train-index_boston \
#         --overwrite_output_dir --loss_fn mse

# root-
#  |--train
#       |--us-ma-boston
#          --*.pkl
#       |--us-pa-pittsburgh-hazelwood
#     ...
#  |--test
#      |--us-ma-pittsburgh
#         --*.pkl
#      ...
#  |--map
#     --us-ma-boston.pkl
#     --us-pa-pittsburgh-hazelwood.pkl
#     --us-nv-las-vegas-strip.pkl
#     --sg-one-north
#     ...
#  |--index (can be organized dynamically)
#     |--train
#         |--train-index_boston
#             --*.arrow
#     |--test
#         |--test-index_pittsburgh
#             --*.arrow  


    