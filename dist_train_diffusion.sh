#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; \
python -m torch.distributed.run --nproc_per_node=7 --master_port=29502 runner_waymo.py \
        --model_name scratch-gpt-small --model_pretrain_name_or_path None \
        --saved_dataset_folder  /data_3/madanjiao/model_res/z_gptS_vehicle_k1_KP0_anchored1_e100_finetuneWithAnchorClsOnly/training_results/checkpoint-430000/eval_output/feature_out \
        --output_dir /data_3/madanjiao/model_res/diffusion_vehicle_k1/training_results  \
        --logging_dir /data_3/madanjiao/model_res/diffusion_vehicle_k1/training_logs \
        --run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
        --per_device_train_batch_size 16 --warmup_steps 50 \
        --learning_rate 1e-4 --weight_decay 1e-5 --logging_steps 100 --save_strategy steps \
        --save_steps 5000 --dataloader_num_workers 10 \
        --save_total_limit 100  --ar_future_interval 0 --specified_key_points True --forward_specified_key_points False \
        --dataloader_drop_last True --do_train \
        --d_model 256 --dataset_scale 1 \
        --task train_diffusion_decoder --with_traffic_light True --k 1 \
        --remove_unused_columns False --future_sample_interval 1 \
        --past_sample_interval 1 \
        --overwrite_output_dir --loss_fn mse \
        --key_points_num 1 \
        --diffusion_condition_sequence_lenth 22

    