# transformer4planning

To train:
`
python -m torch.distributed.run \
--nproc_per_node=4 \
runner.py --model_name TransfoXLModelNuPlan \
--model_pretrain_name_or_path transfo-xl-wt103 \
--saved_dataset_folder /localdata_ssd/nuplan_nsm/0404 \
--output_dir transformerXL_nopose_0405/training_results \
--logging_dir transformerXL_nopose_0405/training_logs \
--num_train_epochs 50 \
--per_device_train_batch_size 3 \
--warmup_steps 500 \
--weight_decay 0.01 \
--logging_steps 50 \
--save_strategy epoch \
--past_index 2 \
--dataloader_num_workers 5 \
--save_total_limit 2 \
--predict_pose False \
--predict_trajectory False \
--do_train
`