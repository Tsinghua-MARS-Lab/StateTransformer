#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7;
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 runner_waymo.py \
        --model_name pretrain-vector-gpt \
        --model_pretrain_name_or_path /localdata_ssd/liderun/tmp/vector_ipclr/training_results/checkpoint-30000 \
        --output_dir /localdata_ssd/liderun/tmp/debug/training_results \
        --run_name debug \
        --per_device_eval_batch_size 8 --dataloader_num_workers 10 --ar_future_interval 0 --specified_key_points True --forward_specified_key_points False \
        --dataloader_drop_last True \
        --d_model 256 --dataset_scale 1 \
        --task waymo \
        --with_traffic_light True --k 1 \
        --remove_unused_columns False --future_sample_interval 1 \
        --past_sample_interval 1 --do_eval\
        --overwrite_output_dir --loss_fn mse --raster_channels 20

# python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 runner_waymo.py \
#         --model_name pretrain-vector-gpt \
#         --model_pretrain_name_or_path /data_3/madanjiao/model_res/z_gptS_vehicle_k1_KP0_anchored1_e100_finetuneWithAnchorClsOnly/training_results/checkpoint-430000 \
#         --output_dir /data_3/madanjiao/model_res/z_gptS_vehicle_k1_KP0_anchored1_e100_finetuneWithAnchorClsOnly/training_results  \
#         --run_name gpt_mse_FI1_PI1_k1_test \
#         --per_device_eval_batch_size 1 --dataloader_num_workers 10 --ar_future_interval 0 --specified_key_points True --forward_specified_key_points False \
#         --dataloader_drop_last True \
#         --d_model 256 --dataset_scale 1 \
#         --task waymo_save_feature \
#         --with_traffic_light True --k 1 \
#         --remove_unused_columns False --future_sample_interval 1 \
#         --past_sample_interval 1 --do_eval\
#         --overwrite_output_dir --loss_fn mse
