export CUDA_VISIBLE_DEVICES=0; python -m torch.distributed.run --nproc_per_node=1 --master_port 12346 runner.py --model_name scratch-mixtral-small-diffusion-dit --model_pretrain_name_or_path /cephfs/zhanjh/exp/MOE_aux/checkpoint-150000 --saved_dataset_folder /localssd/zhanjh/online_s6 --output_dir /cephfs/zhanjh/StrDiff_result/DiT --logging_dir /cephfs/zhanjh/StrDiff_result_2 --run_name Small_Str_DiT_2 --num_train_epochs 50 --per_device_train_batch_size 32 --warmup_steps 50 --weight_decay 0.01 --logging_steps 100 --save_strategy steps --save_steps 3000 --dataloader_num_workers 24 --dataloader_drop_last True --save_total_limit 10 --do_train --task nuplan --remove_unused_columns False --do_eval --evaluation_strategy steps --eval_steps 300 --per_device_eval_batch_size 1 --predict_yaw True --use_proposal 0 --selected_exponential_past True --mean_circular_loss True --raster_channels 34 --use_mission_goal False --raster_encoder_type vit --vit_intermediate_size 768 --lr_scheduler_type cosine --use_speed --use_key_points specified_backward --augment_index 5 --attn_implementation flash_attention_2 --sync_norm True --bf16 True --nuplan_sim_exp_root /localssd/zhanjh/online_s6 --nuplan_sim_data_path /localssd/zhanjh/online_s6/val --nuplan_sim_map_folder /localssd/zhanjh/online_s6/map --nuplan_sim_split_filter_yaml nuplan_simulation/val14_split.yaml --max_sim_samples 64 --inspect_kp_loss --num_local_experts 24 --num_experts_per_token 2 --router_aux_loss_coef 0.1 --overwrite_output_dir  --ddp_find_unused_parameters False --debug_train True --eval_on_start True  --max_eval_samples 10  --resume_from_checkpoint /cephfs/zhanjh/StrDiff_result/DiT_Debug/checkpoint-1800