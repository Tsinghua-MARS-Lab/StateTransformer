# transformer4planning

## To train:

model_name consist of ['scratch','pretrain']-['xl','gpt']

`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run \
--nproc_per_node=8 \
runner.py --model_name scratch-xl \
--model_pretrain_name_or_path transfo-xl-wt103 \
--saved_dataset_folder /localdata_ssd/nuplan_nsm/nsm_sparse_balance_new_4seq \
--output_dir data/example/training_results \
--logging_dir data/example/training_logs \
--run_name example \
--num_train_epochs 500 \
--per_device_train_batch_size 2 \
--warmup_steps 500 \
--weight_decay 0.01 \
--logging_steps 200 \
--save_strategy steps \
--save_steps 2000 \
--past_index 2 \
--dataloader_num_workers 40 \
--save_total_limit 5 \
--use_nsm False \
--predict_trajectory True \
--predict_trajectory_with_stopflag False \
--dataloader_drop_last True \
--do_train \
--maneuver_repeat True \
--d_embed 256 \
--d_model 256 \
--d_inner 1024 \
--n_layers 4 \
--activation_function silu \
`



## To predict and evaluate:

`
export CUDA_VISIBLE_DEVICES=3; \
python -m torch.distributed.run \
--nproc_per_node=1 \
--master_port 12345 \
runner.py --model_name pretrain-gpt \
--model_pretrain_name_or_path data/example/training_results/checkpoint-xxxxx \
--saved_dataset_folder /localdata_ssd/nuplan/nsm_autoregressive_rapid \
--output_dir data/example/prediction_results/checkpoint-xxxxx \
--per_device_eval_batch_size 20 \
--past_index 2 \
--dataloader_num_workers 40 \
--use_nsm False \
--predict_intended_maneuver False \
--predict_current_maneuver False \
--predict_trajectory True \
--recover_obs False \
--do_predict \
--max_predict_samples 500 \
--dataloader_drop_last True \
--per_instance_encoding False \
--maneuver_repeat False \
`

## To generate dataset:
`
python generation.py \ 
--num_proc 1 \
--sample_interval 20 \
--dataset_name single_test \
--starting_file_num 0 \
--ending_file_num 1 \
--cache_folder /localdata_hdd/nuplan_nsm \
--auto_regressive False
`

 if you need nsm label:

  add `--use_nsm` at the end of command
  
 '

## To evaluate on NuBoard:

### Install Transformer4Planning

run `pip install -e .` from the root directory of Transformer4Planning.

### Install NuPlan-Devkit
(tested with v1.2)
run `pip install -e .` from the root directory of NuPlan-Devkit.
Then install these packages:

    pip install aioboto3
    pip install retry
    pip install aiofiles
    pip install bokeh==2.4.1


### Register the planner
Create a new yaml file for Hydra at: `script/config/simulation/planner/control_tf_planner.yaml` with:


    control_tf_planner:
        _target_: transformer4planning.submission.planner.ControlTFPlanner
        horizon_seconds: 10.0
        sampling_time: 0.1
        acceleration: [5.0, 5.0]  # x (longitudinal), y (lateral)
        thread_safe: true

### Run simulation without yaml changing


1. Install Transformer4Planning and NuPlan-Devkit
2. (Optional) Copy the script folder from NuPlan's Official Repo to update
3. Modify dataset path in the `run_simulation.py` and run it to evaluate the model with the tran-xl planner


    python script/run_simulation.py 'planner=control_tf_planner' 
    'scenario_filter.limit_total_scenarios=2' 'scenario_filter.num_scenarios_per_type=1' 
    'job_name=test' 'scenario_builder=nuplan' 
    'ego_controller=perfect_tracking_controller' 'observation=box_observation'
    '+model_pretrain_name_or_path=/public/MARS/datasets/nuPlanCache/checkpoint/nonauto-regressive/xl-silu-fde1.1' 


### Or Modify yaml files and py scripts 
#### Modify the following variables in yaml files
nuplan/planning/script/config/common/default_experiment.yaml

`job_name: open_loop_boxes`

nuplan/planning/script/config/common/worker/ray_distributed.yaml

`threads_per_node: 6`

nuplan/planning/script/config/common/worker/single_machine_thread_pool.yaml

`max_workers: 6`

nuplan/planning/script/config/simulation/default_simulation.yaml

    observation: box_observation
    ego_controller: perfect_tracking_controller
    planner: control_tf_planner

nuplan/planning/script/config/common/default_common.yaml

debug mode `- worker: sequential`
    
multi-threading `- worker: ray_distributed`

nuplan/planning/script/config/common/scenario_filter/all_scenarios.yaml

`num_scenarios_per_type: 1`

`limit_total_scenarios: 5`

#### Add the following at the beginning of run_xxx.py

    os.environ['USE_PYGEOS'] = '0'
    import geopandas
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['NUPLAN_DATA_ROOT'] = ''
    os.environ['NUPLAN_MAPS_ROOT'] = ''
    os.environ['NUPLAN_DB_FILES'] = ''
    os.environ['NUPLAN_MAP_VERSION'] = ''


#### Run the following command from NuPlan-Devkit

`python nuplan/planning/script/run_simulation.py`

### Launch nuboard for visualization

``python script/run_nuboard.py simulation_path='[/home/xiongx/nuplan/exp/exp/simulation/open_loop_boxes/2023.04.21.21.47.58]'``

or

`python nuplan/planning/script/run_nuboard.py simulation_path='[/home/xiongx/nuplan/exp/exp/simulation/open_loop_boxes/2023.04.21.21.47.58]'`

