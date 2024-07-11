
python -m torch.distributed.run \
--nproc_per_node=1 \
runner.py \
--do_predict \
--do_test \
--do_dagger False\
--output_dir "/cephfs/zhanjh/experiment/output" \
--model_name "scratch-mixtral-medium" \
--model_pretrain_name_or_path "/cephfs/zhanjh/checkpoint/checkpoint-150000" \
--save_analyze_result_to_path "/cephfs/zhanjh/experiment/moe_prone" \
--run_name "moe_prone" \
--dataloader_num_workers 10 \
--save_total_limit 2 \
--dataloader_drop_last \
--task "nuplan" \
--saved_dataset_folder "/cephfs/shared/nuplan/online_s6" \
--analyze_dataset_target "test" \
--remove_unused_columns False \
--per_device_eval_batch_size 1 \
--max_test_samples 20 


python -m torch.distributed.run \
--nproc_per_node=1 \
runner.py \
--do_eval False \
--do_predict False \
--do_test \
--do_dagger \
--predict_trajectory False \
--output_dir "/cephfs/zhanjh/experiment/moe_prone" \
--model_name "scratch-mixtral-medium" \
--model_pretrain_name_or_path "/cephfs/zhanjh/checkpoint/checkpoint-150000" \
--run_name "moe_prone" \
--dataloader_num_workers 10 \
--save_total_limit 2 \
--dataloader_drop_last \
--task "nuplan" \
--saved_dataset_folder "/cephfs/shared/nuplan/online_s6" \
--analyze_dataset_target "test" \
--remove_unused_columns False \
--per_device_eval_batch_size 1 \
--max_test_samples 4