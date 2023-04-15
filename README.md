# transformer4planning

To train:

`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run \
--nproc_per_node=8 \
runner.py --model_name TransfoXLModelNuPlan_Config \
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
--save_strategy epoch \
--past_index 2 \
--dataloader_num_workers 5 \
--save_total_limit 2 \
--use_nsm True \
--predict_intended_maneuver True \
--predict_pose False \
--predict_trajectory False \
--dataloader_drop_last True \
--per_instance_encoding False \
--do_train
`



To predict and evaluate:

`
export CUDA_VISIBLE_DEVICES=3; \
python -m torch.distributed.run \
--nproc_per_node=1 \
--master_port 12345 \
runner.py --model_name TransfoXLModelNuPlan_Config \
--model_pretrain_name_or_path nsm_perchannel_rebalance_current_only_layer4_lossx100/training_results/checkpoint-19320 \
--saved_dataset_folder /public/MARS/datasets/nuPlan/nuplan-v1.1/nsm_cache_boston_0412/nsm_sparse_balance \
--output_dir nsm_perchannel_rebalance_current_only_layer4_lossx100/prediction_results/checkpoint-19320 \
--per_device_eval_batch_size 20 \
--past_index 2 \
--dataloader_num_workers 5 \
--use_nsm True \
--predict_intended_maneuver False \
--predict_pose False \
--predict_trajectory False \
--do_predict \
--max_predict_samples 500 \
--dataloader_drop_last True \
--per_instance_encoding False
`

