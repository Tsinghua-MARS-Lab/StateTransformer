# transformer4planning

To train:

model_name consist of ['scratch','pretrain']-['xl','gpt']

`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run \
--nproc_per_node=8 \
runner.py --model_name scratch-gpt \
--model_pretrain_name_or_path transfo-xl-wt103 \
--saved_dataset_folder /localdata_ssd/nuplan_nsm/nsm_sparse_balance \
--output_dir nsm_perchannel_rebalance_2m_perInsFalse_layer4_lossx10000/training_results \
--logging_dir nsm_perchannel_rebalance_2m_perInsFalse_layer4_lossx10000/training_logs \
--run_name nsm_perchannel_rebalance_2m_perInsFalse_layer4_lossx10000 \
--num_train_epochs 50 \
--per_device_train_batch_size 2 \
--warmup_steps 500 \
--weight_decay 0.01 \
--logging_steps 200 \
--save_strategy steps \
--save_steps 1000 \
--past_index 2 \
--dataloader_num_workers 5 \
--save_total_limit 2 \
--use_nsm True \
--predict_intended_maneuver True \
--predict_current_maneuver True \
--predict_pose False \
--predict_trajectory False \
--dataloader_drop_last True \
--per_instance_encoding False \
--do_train \
--maneuver_repeat False \
--d_embed 256 \
--d_model 256 \
--d_inner 1024 \
--n_layers 4 \
`



To predict and evaluate:

`
export CUDA_VISIBLE_DEVICES=3; \
python -m torch.distributed.run \
--nproc_per_node=1 \
--master_port 12345 \
runner.py --model_name pretrain \
--model_pretrain_name_or_path nsm_perchannel_rebalance_current_only_layer4_lossx100/training_results/checkpoint-19320 \
--saved_dataset_folder /public/MARS/datasets/nuPlan/nuplan-v1.1/nsm_cache_boston_0412/nsm_sparse_balance \
--output_dir nsm_perchannel_rebalance_current_only_layer4_lossx100/prediction_results/checkpoint-19320 \
--per_device_eval_batch_size 20 \
--past_index 2 \
--dataloader_num_workers 5 \
--use_nsm True \
--predict_intended_maneuver True \
--predict_current_maneuver True \
--predict_pose False \
--predict_trajectory False \
--do_predict \
--max_predict_samples 500 \
--dataloader_drop_last True \
--per_instance_encoding False \
--maneuver_repeat False \
`
 To generate dataset:
 '
 python generation.py --config configs/nuplan_training_config_server.py --num_proc 40  --sample_interval 10 --dataset_name nonsm_boston_full --ending_file_num 1647

 if you need nsm label:
  add '--use_nsm' at the end of command
 '
