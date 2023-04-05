# transformer4planning

To train:
`
python -m torch.distributed.run; \
--nproc_per_node=4; \
runner.py --model_name TransfoXLModelNuPlan; \
--model_pretrain_name_or_path {path_for_pretrained_model}; \
--saved_dataset_folder {path_for_dataset}; \
--output_dir {path_for_training_results}; \
--logging_dir {path_for_training_logs}; \
--num_train_epochs 50; \
--per_device_train_batch_size 4; \
--warmup_steps 500; \
--weight_decay 0.01; \
--logging_steps 50; \
--save_strategy 'epoch'; \
--past_index 2; \
--do_train;
`