#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0;
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 runner_waymo.py \
        --model_name pretrain-vector-gpt \
        --model_pretrain_name_or_path /data/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits/training_results/checkpoint-350000 \
        --output_dir /data/madanjiao/model_res/vector_gpt_small_k1_KP0_anchored_ep100_gpu7_vheicle_masked_anchorLogits/training_results  \
        --run_name gpt_mse_FI1_PI1_k1_test \
        --per_device_eval_batch_size 16 --dataloader_num_workers 10 --predict_trajectory True --ar_future_interval 0 --specified_key_points True --forward_specified_key_points False \
        --dataloader_drop_last True \
        --d_model 256 --dataset_scale 1 \
        --dataset_name waymo --task waymo \
        --with_traffic_light True --k 1 \
        --online_preprocess True \
        --remove_unused_columns False --future_sample_interval 1 \
        --past_sample_interval 1 --do_eval\
        --overwrite_output_dir --loss_fn mse

# python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 runner_waymo.py \
#         --model_name pretrain-vector-gpt \
#         --model_pretrain_name_or_path /data_3/madanjiao/model_res/vector_gpt_small_k6_forwardKP5_ep100/training_results/checkpoint-30000 \
#         --saved_dataset_folder  /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
#         --output_dir /data_3/madanjiao/model_res/vector_gpt_small_k6_forwardKP5_ep100/training_results  \
#         --run_name gpt_mse_FI1_PI1_k1_test \
#         --per_device_eval_batch_size 16 --dataloader_num_workers 10 --predict_trajectory True --ar_future_interval 5 --specified_key_points True --forward_specified_key_points True \
#         --dataloader_drop_last True \
#         --d_embed 256 --d_model 256 --d_inner 1024 --n_layers 4 --n_heads 4 \
#         --activation_function silu --dataset_scale 1 \
#         --dataset_name waymo --task waymo \
#         --with_traffic_light True --k 6 \
#         --online_preprocess True \
#         --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_demo \
#         --remove_unused_columns False --future_sample_interval 1 \
#         --past_sample_interval 1 --do_eval\
#         --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
#         --overwrite_output_dir --loss_fn mse
