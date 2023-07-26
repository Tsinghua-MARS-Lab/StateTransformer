# transformer4planning


## NuPlan Dataset Pipeline

Usage:

1. Generate dataDic
2. Generate index as a cached dataset
3. Train the model and evaluate during training

### Prepare Environment

(Instructions for different OS TBD)

### Process the dataset (NuPlan Only)

1. process NuPlan .db files to .pkl files (to agent dictionaries)
2. generate filtered scenario index and cache in .arrow files
3. generate map dictionary to pickles

Step 1: Process .db to .pkl by running:
```
    python generation.py  --num_proc 40 --sample_interval 100  
    --dataset_name boston_index_demo  --starting_file_num 0  
    --ending_file_num 10000  --cache_folder /localdata_hdd/nuplan/online_demo/  
    --data_path train_boston  --only_data_dic
```

Step 2: Generate scenarios to .arrow datasets
```
    python generation.py  --num_proc 40 --sample_interval 100  
    --dataset_name boston_index_interval100  --starting_file_num 0  
    --ending_file_num 10000  --cache_folder /localdata_hdd/nuplan/online_demo/  
    --data_path train_boston  --only_index  
```

Step 3: Generate Map files to .pickle files

```
    python generation.py  --num_proc 40 --sample_interval 1 --dataset_name pittsburgh_index_full  
    --starting_file_num 0  --ending_file_num 10000  
    --cache_folder /localdata_hdd/nuplan/online_pittsburgh_jul  --data_path train_pittsburgh --save_map
```

```
    python generation.py  --num_proc 40 --sample_interval 1  
    --dataset_name vegas2_datadic_float32  --starting_file_num 0  --ending_file_num 10000  
    --cache_folder /localdata_hdd/nuplan/vegas2_datadic_float32  --data_path train_vegas_2 --save_map
```

You only need to process Vegas's map once for all Vegas subsets.


Why process .db files to .pkl files? Lower disk usage (lower precision) and faster loading (without initiate NuPlan DataWrapper)


```
root-
 |--train
      |--us-ma-boston
         --*.pkl
      |--us-pa-pittsburgh-hazelwood
    ...
 |--test
     |--us-ma-pittsburgh
        --*.pkl
     ...
 |--map
    --us-ma-boston.pkl
    --us-pa-pittsburgh-hazelwood.pkl
    --us-nv-las-vegas-strip.pkl
    --sg-one-north
    ...
 |--index (can be organized dynamically)
    |--train
        |--train-index_boston
            --*.arrow
    |--test
        |--test-index_pittsburgh
            --*.arrow    
```


## To train and evaluate during training:

model_name consist of ['scratch','pretrain']-['xl','gpt']

`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run \
--nproc_per_node=8 runner.py \
--model_name scratch-gpt --model_pretrain_name_or_path None \
--saved_dataset_folder  /localdata_ssd/nuplan/train-index_boston/boston_index_full \
--output_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/training_results  \
--logging_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/training_logs \
--run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
--per_device_train_batch_size 4 --warmup_steps 50 \
--weight_decay 0.01 --logging_steps 2 --save_strategy steps \
--save_steps 1000 --dataloader_num_workers 10 \
--save_total_limit 2  --predict_trajectory True \
--dataloader_drop_last True --do_train \
--d_embed 1600 --d_model 1600 --d_inner 6400 --n_layers 48 --n_heads 25 \
--activation_function silu --dataset_scale 1 \
--task nuplan --with_traffic_light True --k 1 \
--online_preprocess True \
--datadic_path /localdata_ssd/nuplan/online \
--remove_unused_columns False --future_sample_interval 1 \
--past_sample_interval 1 --do_eval \
--evaluation_strategy steps --eval_steps 100 \
--saved_valid_dataset_folder /localdata_ssd/nuplan/test-index_boston_full \
--overwrite_output_dir --loss_fn mse --max_eval_samples 1e5\
--next_token_scorer True \
--x_random_walk 3.0 --y_random_walk 3.0 \
--arf_x_random_walk 3.0 --arf_y_random_walk 3.0 \
--trajectory_loss_rescale 1e-3 \ 
--pred_key_points_only False \
--specified_key_points True \
--forward_specified_key_points False \
`



## To predict:

to be update for online process

`
export CUDA_VISIBLE_DEVICES=3; \
python -m torch.distributed.run \
--nproc_per_node=1 \
--master_port 12345 \
runner.py --model_name pretrain-gpt \
--model_pretrain_name_or_path data/example/training_results/checkpoint-xxxxx \
--saved_dataset_folder /localdata_ssd/nuplan/nsm_autoregressive_rapid \
--output_dir data/example/prediction_results/checkpoint-xxxxx \
--per_device_eval_batch_size 32 \
--dataloader_num_workers 16 \
--predict_trajectory True \
--do_predict \
--saved_valid_dataset_folder /localdata_ssd/nuplan/boston_test_byscenario/
--max_predict_samples 1e3 \
--dataloader_drop_last True \
--with_traffic_light True \
--remove_unused_columns True \
--next_token_scorer True \
--trajectory_loss_rescale 1e-3 \
--pred_key_points_only False \
--specified_key_points True \
--forward_specified_key_points False \
`

## To generate dataset:

### Generate data dictionary

`
python generation.py  --num_proc 40 \
--sample_interval 1  --dataset_name boston_datadic_float32 \
--starting_file_num 0  --ending_file_num 10000  \
--cache_folder /localdata_hdd/nuplan/boston_datadic_float32 \
--data_path train_boston  --only_data_dic
`

### Generate index

`
python generation.py  --num_proc 96 --sample_interval 1 \
--dataset_name boston_index_full \
--starting_file_num 0  --ending_file_num 10000 \
--cache_folder /localdata_hdd/nuplan/online_boston \
--data_path train_boston  --only_index
`



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

Create a new yaml file for Hydra at: `script/config/simulation/planner/rule_based_planner.yaml` with:
    rule_based_planner:
        _target_: transformer4planning.submission.rule_based_planner.RuleBasedPlanner
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
    'job_name=open_loop_boxes' 'scenario_builder=nuplan' 
    'ego_controller=perfect_tracking_controller' 'observation=box_observation'
    '+model_pretrain_name_or_path=/public/MARS/datasets/nuPlanCache/checkpoint/nonauto-regressive/xl-silu-fde1.1' 


### Or Modify yaml files and py scripts 
#### Modify the following variables in yaml files
nuplan/planning/script/config/common/default_experiment.yaml

`job_name: open_loop_boxes` or
`job_name: closed_loop_nonreactive_agents` or 
`job_name: closed_loop_reactive_agents`

nuplan/planning/script/config/common/default_common.yaml

`scenario_builder: nuplan` correspond to the yaml filename in scripts/config/common/scenario_builder

`scenario_filter: all_scenarios` correspond to the yaml filenmae in scripts/config/common/scnario_filter 

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
   
`log_names: []` to filter the db files to run simulation

#### Add the following at the beginning of run_xxx.py

    os.environ['USE_PYGEOS'] = '0'
    import geopandas
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['NUPLAN_DATA_ROOT'] = ''
    os.environ['NUPLAN_MAPS_ROOT'] = ''
    os.environ['NUPLAN_DB_FILES'] = ''
    os.environ['NUPLAN_MAP_VERSION'] = ''

#### Filter log files or maps
set `scenario_builder` `scenario_filter` to correspond scenario_builder yaml file and scenario_filter yaml file in `script/config/common/default_common.yaml`. scenario_builder path is `script/config/common/scenario_builder` and scenario_filter path is `script/config/common/scenario_filter`. For example, choose builder yaml as nuplan, and filter yaml as all_scenarios, if you want to filter specify logs or maps, please add `[log_name1, ..., log_namen]` to log_names.

#### Run the following command from NuPlan-Devkit

`python nuplan/planning/script/run_simulation.py`

to set configs: 
planner choice: `planner=control_tf_planner` Optional `[control_tf_planner, rule_based_planner]`
chanllenge choice: `+simulation=closed_loop_reactive_agents` Optional `[closed_loop_reactive_agents, open_loop_boxes, closed_loop_nonreactive_agents]`

### Launch nuboard for visualization

``python script/run_nuboard.py simulation_path='[/home/sunq/nuplan/exp/exp/simulation/test/2023.05.08.19.17.16]' 'scenario_builder=nuplan' 'port_number=5005'``

or

`python nuplan/planning/script/run_nuboard.py simulation_path='[/home/sunq/nuplan/exp/exp/simulation/open_loop_boxes/2023.04.21.21.47.58]'`

