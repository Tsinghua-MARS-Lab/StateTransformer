#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; \
python -m torch.distributed.run --nproc_per_node=7 --master_port=29502 runner_waymo.py \
        --model_name scratch-vector-gpt --model_pretrain_name_or_path None \
        --saved_dataset_folder  /data3/madanjiao/nuplan/online_demo/mini_demo/index \
        --output_dir /data/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits/training_results  \
        --logging_dir /data/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits/training_logs \
        --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
        --per_device_train_batch_size 16 --warmup_steps 50 \
        --weight_decay 0.01 --logging_steps 100 --save_strategy steps \
        --save_steps 5000 --dataloader_num_workers 10 \
        --save_total_limit 50  --use_key_points None\
        --dataloader_drop_last True --do_train \
        --d_model 256 --dataset_scale 1 \
        --task waymo --with_traffic_light True --k 1 \
        --online_preprocess True \
        --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_demo \
        --remove_unused_columns False --future_sample_interval 1 \
        --past_sample_interval 1 \
        --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
        --overwrite_output_dir --loss_fn mse

# export CUDA_VISIBLE_DEVICES=0; \
# python -m torch.distributed.run --nproc_per_node=1 --master_port=29510 runner_waymo.py \
#         --model_name scratch-vector-gpt --model_pretrain_name_or_path None  \
#         --saved_dataset_folder  /data/madanjiao/nuplan/online_demo/mini_demo/index \
#         --output_dir /data/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_vehicle/training_results  \
#         --logging_dir /data/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_vehicle/training_logs \
#         --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
#         --per_device_train_batch_size 16 --warmup_steps 50 \
#         --weight_decay 0.01 --logging_steps 100 --save_strategy steps \
#         --save_steps 5000 --dataloader_num_workers 10 \
#         --save_total_limit 50  --use_key_points None\
#         --dataloader_drop_last True --do_train \
#         --d_model 256 --dataset_scale 1 \
#         --task waymo --with_traffic_light True --k 1 \
#         --online_preprocess True \
#         --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_demo \
#         --remove_unused_columns False --future_sample_interval 1 \
#         --past_sample_interval 1 \
#         --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
#         --overwrite_output_dir --loss_fn mse

    