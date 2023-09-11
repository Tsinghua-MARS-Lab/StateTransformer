export CUDA_VISIBLE_DEVICES=7;
nohup python3 -m torch.distributed.run --nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_K6_WOMD_VECTOR/checkpoint-100000 \
--do_eval True --do_train False --label_names center_gt_trajs_src \
--saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
--run_name GPTS_SKP_K6_WOMD_VECTOR_400-22_Waymo_test \
--num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--max_train_samples 1140000 \
--max_eval_samples 1140000 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00  \
--logging_steps 2 --save_strategy epoch --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 12 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--output_dir /localdata_ssd/waymo_1/result_and_output/400-22_Waymo/otpt_dir/ \
--logging_dir /localdata_ssd/waymo_1/result_and_output/400-22_Waymo/log_dir/ >> 400-22_Waymo_test.log &

export CUDA_VISIBLE_DEVICES=7;
nohup python3 -m torch.distributed.run --nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_K6_WOMD_VECTOR/checkpoint-100000 \
--do_eval True --do_train False --label_names center_gt_trajs_src \
--saved_dataset_folder /localdata_ssd/waymo_1/dataset/t4p_waymo/motion \
--run_name GPTS_SKP_K6_WOMD_VECTOR_412_Waymo_Newckpt_test \
--num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 256 \
--max_train_samples 1140000 \
--max_eval_samples 1140000 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00  \
--logging_steps 2 --save_strategy epoch --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 12 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--output_dir /localdata_ssd/waymo_1/result_and_output/412_Waymo_Newckpt/otpt_dir/ \
--generation_method diffusion \
--key_points_diffusion_decoder_load_from /localdata_ssd/waymo_1/waymo_Newckpt100000_diff_new_keypoints_decoderTFBased_saving_dir/Waymo70002-allvalid-256FeatDim-run-LargeTFBased-keypoints_AllTrainAllTest-ade6valloss/2023_09_01___15_30_52/epoch=146-step=307817.ckpt \
--logging_dir /localdata_ssd/waymo_1/result_and_output/412_Waymo_Newckpt/log_dir/ >> 412_Waymo_Newckpt_test.log &


export CUDA_VISIBLE_DEVICES=7;
nohup python3 -m torch.distributed.run --nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--do_eval True \
--do_train True --label_names center_gt_trajs_src \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_K6_WOMD_VECTOR/checkpoint-100000 \
--saved_dataset_folder /localdata_ssd/waymo_1/dataset/t4p_waymo/motion \
--run_name GPTS_SKP_K6_WOMD_VECTOR_410_NewCkpt100000Gen_Waymo \
--num_train_epochs 1 --per_device_train_batch_size 32 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00 \
--logging_steps 2 --save_strategy steps --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 20 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--generate_diffusion_dataset_for_key_points_decoder True \
--diffusion_dataset_save_dir /localdata_ssd/waymo_1/410_NewCkpt100000_diffusion_dataset/ \
--output_dir /localdata_ssd/waymo_1/result_and_output/410_NewCkpt100000Gen_Waymo/otpt_dir/ \
--logging_dir /localdata_ssd/waymo_1/result_and_output/410_NewCkpt100000Gen_Waymo/log_dir/ >> 410_NewCkpt100000Gen_Waymo.log &


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
python -m torch.distributed.run --nproc_per_node=8 runner.py \
        --model_name scratch-gpt-small --task waymo --encoder_type vector \
        --do_train \
        --model_pretrain_name_or_path None \
        --saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
        --run_name GPTS_SKP_K6_WOMD_VECTOR\
        --num_train_epochs 100 --per_device_train_batch_size 4 \
        --warmup_steps 50 --weight_decay 0.01 \
        --logging_steps 2 --save_strategy steps --save_steps 5000 --save_total_limit 2 \
        --dataloader_num_workers 8 --dataloader_drop_last True --dataset_scale 1 \
        --ar_future_interval 5 --specified_key_points True \
        --activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
        --remove_unused_columns False \
        --overwrite_output_dir \
        --output_dir /localdata_ssd/liderun/tmp/GPTS_SKP_K6_WOMD_VECTOR/training_results \
        --logging_dir /localdata_ssd/liderun/tmp/GPTS_SKP_K6_WOMD_VECTOR/training_logs \

export CUDA_VISIBLE_DEVICES=0;
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 runner.py \
        --model_name scratch-gpt-medium --task waymo --encoder_type vector \
        --do_eval --label_names center_gt_trajs_src \
        --model_pretrain_name_or_path None \
        --saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
        --run_name eval_debug \
        --per_device_eval_batch_size 16 \
        --warmup_steps 50 --weight_decay 0.01 \
        --dataloader_num_workers 8 --dataloader_drop_last True --dataset_scale 1 \
        --ar_future_interval 5 --specified_key_points True \
        --activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
        --remove_unused_columns False \
        --overwrite_output_dir \
        --output_dir /localdata_ssd/liderun/tmp/eval_debug/training_results \
        --logging_dir /localdata_ssd/liderun/tmp/eval_debug/training_logs \
        --model_pretrain_name_or_path /localdata_ssd/liderun/ckpts/checkpoint-60000








export CUDA_VISIBLE_DEVICES=7;
nohup python3 -m torch.distributed.run --nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_MTR_WOMD_VECTOR/checkpoint-330000 \
--do_eval True --do_train False --label_names center_gt_trajs_src \
--saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
--run_name ckpt330k_GPTS_SKP_MTR_WOMD_VECTOR_400-24_Waymo_test \
--num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--max_train_samples 1140000 \
--max_eval_samples 1140000 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00  \
--logging_steps 2 --save_strategy epoch --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 12 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--mtr_config_path /home/shijz/projects/LM_diffusion_decoder/working_dir_waymo_1/transformer4planning/config/gpt_small.yaml \
--output_dir /localdata_ssd/shijz/waymo_ckpt330k/result_and_output/400-24_Waymo/otpt_dir/ \
--logging_dir /localdata_ssd/shijz/waymo_ckpt330k/result_and_output/400-24_Waymo/log_dir/ >> 400-24_Waymo_test.log &

export CUDA_VISIBLE_DEVICES=6;
nohup python3 -m torch.distributed.run \
--master_port 12316 \
--nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_MTR_WOMD_VECTOR/checkpoint-380000 \
--do_eval True --do_train False --label_names center_gt_trajs_src \
--saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
--run_name ckpt330k_GPTS_SKP_MTR_WOMD_VECTOR_400-24-2_Waymo_test \
--num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--max_train_samples 1140000 \
--max_eval_samples 1140000 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00  \
--logging_steps 2 --save_strategy epoch --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 12 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--mtr_config_path /home/shijz/projects/LM_diffusion_decoder/working_dir_waymo_1/transformer4planning/config/gpt_small.yaml \
--output_dir /localdata_ssd/shijz/waymo_ckpt380k/result_and_output/400-24-2_Waymo/otpt_dir/ \
--logging_dir /localdata_ssd/shijz/waymo_ckpt380k/result_and_output/400-24-2_Waymo/log_dir/ >> 400-24-2_Waymo_test.log &

export CUDA_VISIBLE_DEVICES=5;
nohup python3 -m torch.distributed.run \
--master_port 12315 \
--nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_MTR_WOMD_VECTOR/checkpoint-275000 \
--do_eval True --do_train False --label_names center_gt_trajs_src \
--saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
--run_name ckpt330k_GPTS_SKP_MTR_WOMD_VECTOR_400-24-3_Waymo_test \
--num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--max_train_samples 1140000 \
--max_eval_samples 1140000 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00  \
--logging_steps 2 --save_strategy epoch --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 12 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--mtr_config_path /home/shijz/projects/LM_diffusion_decoder/working_dir_waymo_1/transformer4planning/config/gpt_small.yaml \
--output_dir /localdata_ssd/shijz/waymo_ckpt275k/result_and_output/400-24-2_Waymo/otpt_dir/ \
--logging_dir /localdata_ssd/shijz/waymo_ckpt275k/result_and_output/400-24-2_Waymo/log_dir/ >> 400-24-3_Waymo_test.log &

export CUDA_VISIBLE_DEVICES=3;
nohup python3 -m torch.distributed.run \
--master_port 12313 \
--nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small --task waymo --encoder_type vector \
--model_pretrain_name_or_path /public/MARS/liderun/ckpts/GPTS_SKP_MTR_WOMD_VECTOR/checkpoint-225000 \
--do_eval True --do_train False --label_names center_gt_trajs_src \
--saved_dataset_folder /localdata_ssd/liderun/t4p_waymo/motion \
--run_name ckpt330k_GPTS_SKP_MTR_WOMD_VECTOR_400-24-5_Waymo_test \
--num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--max_train_samples 1140000 \
--max_eval_samples 1140000 \
--warmup_steps 50 --weight_decay 0.00 --learning_rate 0.00  \
--logging_steps 2 --save_strategy epoch --save_steps 5000 --save_total_limit 2 \
--dataloader_num_workers 12 --dataloader_drop_last True --dataset_scale 1 \
--ar_future_interval 5 --specified_key_points True \
--activation_function silu --with_traffic_light True --k 6 --loss_fn mse \
--remove_unused_columns False \
--overwrite_output_dir True \
--mtr_config_path /home/shijz/projects/LM_diffusion_decoder/working_dir_waymo_1/transformer4planning/config/gpt_small.yaml \
--output_dir /localdata_ssd/shijz/waymo_ckpt225k/result_and_output/400-24-5_Waymo/otpt_dir/ \
--logging_dir /localdata_ssd/shijz/waymo_ckpt225k/result_and_output/400-24-5_Waymo/log_dir/ >> 400-24-5_Waymo_test.log &