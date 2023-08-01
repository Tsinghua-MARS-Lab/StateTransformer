python -m torch.distributed.run --nproc_per_node=1 waymo_runner.py --model_name scratch-demo \
    --model_pretrain_name_or_path None --saved_dataset_folder /localdata_ssd/liderun/demo_cache/t4p_demo \
    --output_dir /localdata_ssd/liderun/tmp/demo/training_results  --logging_dir /localdata_ssd/liderun/tmp/demo/training_logs \
    --run_name waymo-debug --num_train_epochs 300 --per_device_train_batch_size 10 --warmup_steps 500 \
    --weight_decay 0.01 --logging_steps 100 --save_strategy steps --save_steps 1000 --dataloader_num_workers 10 \
    --save_total_limit 2  --predict_trajectory True  --dataloader_drop_last True --do_train --d_embed 256 \
    --d_model 256 --d_inner 1024 --n_layers 4 --n_heads 8 --activation_function silu --dataset_scale 1 --autoregressive True --k 1 --online_preprocess False \
    --remove_unused_columns False --next_token_scorer False --tokenize_label False --dataloader_pin_memory False