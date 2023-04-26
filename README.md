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
--output_dir data/example/training_results \
--logging_dir data/example/training_logs \
--run_name example \
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
--predict_intended_maneuver False \
--predict_current_maneuver False \
--predict_pose False \
--predict_trajectory False \
--recover_obs False \
--dataloader_drop_last True \
--per_instance_encoding False \
--do_train \
--maneuver_repeat False \
--d_embed 288 \
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
--model_pretrain_name_or_path data/example/training_results/checkpoint-xxxxx \
--saved_dataset_folder /public/MARS/datasets/nuPlan/nuplan-v1.1/nsm_cache_boston_0412/nsm_sparse_balance \
--output_dir data/example/prediction_results/checkpoint-xxxxx \
--per_device_eval_batch_size 20 \
--past_index 2 \
--dataloader_num_workers 5 \
--use_nsm True \
--predict_intended_maneuver False \
--predict_current_maneuver False \
--predict_pose False \
--predict_trajectory False \
--recover_obs False \
--do_predict \
--max_predict_samples 500 \
--dataloader_drop_last True \
--per_instance_encoding False \
--maneuver_repeat False \
`

 To generate dataset:
`
 python generation.py \
 --config configs/nuplan_training_config_server.py \
 --num_proc 40  \
 --sample_interval 10 \
 --dataset_name nonsm_boston_full \
 --ending_file_num 1647
`

 if you need nsm label:

  add `--use_nsm` at the end of command
  
 '
 
 To evaluate in nuboard:

### Modify the following variables
nuplan/planning/script/config/common/default_experiment.yaml

`job_name: open_loop_boxes`

nuplan/planning/script/config/common/worker/ray_distributed.yaml

`threads_per_node: 6`

nuplan/planning/script/config/common/worker/single_machine_thread_pool.yaml

`max_workers: 6`

nuplan/planning/script/config/simulation/default_simulation.yaml

`observation: box_observation`

`ego_controller: perfect_tracking_controller`

`planner: control_tf_planner`

nuplan/planning/script/config/common/default_common.yaml

debug mode `- worker: sequential`
    
multi-threading `- worker: ray_distributed`

nuplan/planning/script/config/common/scenario_filter/all_scenarios.yaml

`num_scenarios_per_type: 1`

`limit_total_scenarios: 5`

### Add the following at the beginning of run_xxx.py

`os.environ['USE_PYGEOS'] = '0'`

`import geopandas`

`os.environ['HYDRA_FULL_ERROR'] = '1'`

`os.environ['NUPLAN_DATA_ROOT'] = ''`

`os.environ['NUPLAN_MAPS_ROOT'] = ''`

`os.environ['NUPLAN_DB_FILES'] = ''`

`os.environ['NUPLAN_MAP_VERSION'] = ''`

### Add this file nuplan/planning/script/config/simulation/planner/control_tf_planner.yaml
Add the following content to the control_tf_planner.yaml file

Change the target path name to the actual planner path name

`control_tf_planner:
  _target_: nuplan.planning.simulation.planner.transformer_planner.ControlTFPlanner
  horizon_seconds: 10.0
  sampling_time: 0.1
  acceleration: [5.0, 5.0] # x (longitudinal), y (lateral)
  thread_safe: true`


### Run the following command
`python nuplan/planning/script/run_simulation.py`

`python nuplan/planning/script/run_nuboard.py simulation_path='[/home/xiongx/nuplan/exp/exp/simulation/open_loop_boxes/2023.04.21.21.47.58]'`

