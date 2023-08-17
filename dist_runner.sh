#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; \
python -m torch.distributed.run --nproc_per_node=7 --master_port=29502 runner_waymo.py \
        --model_name scratch-vector-gpt --model_pretrain_name_or_path None \
        --saved_dataset_folder  /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
        --output_dir /data_3/madanjiao/model_res/vector_gpt_small_k6_reverseKP5_correctA_kptsLogitsCls_ep100/training_results  \
        --logging_dir /data_3/madanjiao/model_res/vector_gpt_small_k6_reverseKP5_correctA_kptsLogitsCls_ep100/training_logs \
        --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
        --per_device_train_batch_size 16 --warmup_steps 50 \
        --weight_decay 0.01 --logging_steps 100 --save_strategy steps \
        --save_steps 5000 --dataloader_num_workers 10 \
        --save_total_limit 50  --predict_trajectory True --ar_future_interval 5 --specified_key_points True --forward_specified_key_points False \
        --dataloader_drop_last True --do_train \
        --d_model 256 --dataset_scale 1 \
        --task waymo --with_traffic_light True --k 6 \
        --online_preprocess True \
        --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_demo \
        --remove_unused_columns False --future_sample_interval 1 \
        --past_sample_interval 1 \
        --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
        --overwrite_output_dir --loss_fn mse

# python -m torch.distributed.run --nproc_per_node=8 runner_waymo.py \
#         --model_name scratch-gpt --model_pretrain_name_or_path None \
#         --saved_dataset_folder  /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
#         --output_dir /data_3/madanjiao/model_res/waymo_raster_gpt_dembed256/training_results  \
#         --logging_dir /data_3/madanjiao/model_res/waymo_raster_gpt_dembed256/training_logs \
#         --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 30 \
#         --per_device_train_batch_size 8 --warmup_steps 50 \
#         --weight_decay 0.01 --logging_steps 2 --save_strategy steps \
#         --save_steps 1000 --dataloader_num_workers 10 \
#         --save_total_limit 2  --predict_trajectory True \
#         --dataloader_drop_last True --do_train \
#         --d_embed 256 --d_model 256 --d_inner 1024 --n_layers 4 --n_heads 4 \
#         --activation_function silu --dataset_scale 1 \
#         --task waymo --with_traffic_light True --k 1 \
#         --online_preprocess True \
#         --datadic_path /data_3/madanjiao/nuplan/online_demo/mini_demo \
#         --remove_unused_columns False --future_sample_interval 1 \
#         --past_sample_interval 1 \
#         --saved_valid_dataset_folder /data_3/madanjiao/nuplan/online_demo/mini_demo/index \
#         --overwrite_output_dir --loss_fn mse

    