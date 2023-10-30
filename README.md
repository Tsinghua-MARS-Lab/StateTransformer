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

`--model_name` can choose from ["gpt-large","gpt-medium","gpt-small","gpt-mini"] and with prefix of `scratch-` or `pretrain-` to determine wether load pretrained weights from existed checkpoints, whose attributes is `--model_pretrain_name_or_path`

The common training settings are shown below.

`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run \
--nproc_per_node=8 runner.py \
--do_train --do_eval\
--model_name scratch-gpt-mini --model_pretrain_name_or_path None \
--saved_dataset_folder  /localdata_ssd/nuplan/online_dataset \
--output_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/training_results  \
--logging_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/training_logs \
--run_name gpt_1.5B_mse_FI1_PI1_k1 --num_train_epochs 100 \
--per_device_train_batch_size 4 --warmup_steps 50 \
--weight_decay 0.01 --logging_steps 2 --save_strategy steps \
--dataloader_num_workers 10 \
--save_total_limit 2 \
--dataloader_drop_last True \
--dataset_scale 1 \
--task nuplan \
--k 1 \
--future_sample_interval 2 \
--past_sample_interval 5 \
--evaluation_strategy steps --eval_steps 100 \
--overwrite_output_dir --loss_fn mse --max_eval_samples 10000\
--next_token_scorer True \
--ar_future_interval 20 \
--x_random_walk 3.0 --y_random_walk 3.0 \
--arf_x_random_walk 3.0 --arf_y_random_walk 3.0 \
--trajectory_loss_rescale 1e-3 \
--pred_key_points_only False \
--specified_key_points True \
--forward_specified_key_points False \
`

### Train with different encoder(same backbone)
To choose different encoder, please set the attribute `--encoder_type`. The choices are [`raster`, `vector`]. With different `--task` setting, the encoder can be initilized with different classes.  

### Train with differnet decoder(same backbone)
To choose different decoder, please set the attribute `--decoder_type`. The choices are [`mlp`, `diffusion`]. 
Note that only mlp decoders can be trained together with the backbone.

### Train only diffusion decoder, without backbone

To train the diffusion decoder, you need first to train using an mlp decoder to obtain a pretrained backbone. After that, you need to generate the dataset for training the diffusion decoder using `generate_diffusion_feature.py`: this is done using the same command for eval except that you need to set `--generate_diffusion_dataset_for_key_points_decoder` to True and `--diffusion_feature_save_dir` to the dir to save the pth files for training diffusion decoders.
An example:
`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
python -m torch.distributed.run \
--nproc_per_node=8 generate_diffusion_feature.py \
--model_name pretrain-gpt-mini --model_pretrain_name_or_path /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/training_results/checkpoint-20000/ \
--saved_dataset_folder  /localdata_ssd/nuplan/online_dataset \
--output_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/dummy_generating_results  \
--logging_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/dummy_generating_logs \
--run_name gpt_1.5B_mse_FI1_PI1_k1_genDiffFeat
--dataloader_num_workers 10 \
--dataloader_drop_last True \
--dataset_scale 1 \
--task nuplan \
--k 1 \
--future_sample_interval 2 \
--past_sample_interval 5 \
--evaluation_strategy steps \
--overwrite_output_dir --loss_fn mse --max_eval_samples 10000\
--next_token_scorer True \
--ar_future_interval 20 \
--pred_key_points_only False \
--specified_key_points True \
--forward_specified_key_points False \
--diffusion_feature_save_dir /localdata_hdd1/sunq/gpt_1.5B_mse_FI1_PI1_k1/diffusion_feature_pth_files/ \
`
After running these, you are supposed to get three folders under `diffusion_feature_save_dir` for `train`, `val` and `test` diffusion features respectively.

```
diffusion_feature_save_dir
 |--train
    --future_key_points_[0-9]*.pth
    --future_key_points_hidden_state_[0-9]*.pth
 |--val
    --future_key_points_[0-9]*.pth
    --future_key_points_hidden_state_[0-9]*.pth
 |--test
    --future_key_points_[0-9]*.pth
    --future_key_points_hidden_state_[0-9]*.pth
...

After saving the pth files, you need to run `convert_diffusion_dataset.py` to convert them into arrow dataset which is consistent with the training format we are using here.
`
python3 convert_diffusion_dataset.py \
    --save_dir /localdata_ssd/nuplan_diff_dataset/arrow_dataset \
    --data_dir diffusion_feature_save_dir/train/ \
    --num_proc 10 \
    --dataset_name nuplan_diffusion_train \
    --map_dir /localdata_hdd/nuplan/map/ \
    --saved_datase_folder /localdata_ssd/nuplan/online_dataset/ \
    --use_centerline False \
    --split train \
`
After which you are expected to see such structures in save_dir:
```
save_dir
    |--generator
        |--*
    |--train
        *.arrow
        dataset_info.json
        state.json
    *.lock
```

Fianlly, please set `--task` to `train_diffusion_decoder`. In this case, the model is initilized without a transformer backbone(Reduce the infence time to build backbone feature). 
In the meanwhile, please change the `--saved_dataset_folder` to the folder which stores 'backbone features dataset', obtained previously by `convert_diffusion_dataset.py`. In this case it should be consistent with `save_dir` in the previous step.
`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
nohup python3 -m torch.distributed.run \
--master_port 12363 \
--nproc_per_node=8 runner.py \
--model_name pretrain-gpt-small --model_pretrain_name_or_path /public/MARS/t4p/checkpoints/Small_Oct9/Small_SKPY_PI2_x1_auXYd1_auCurrentd1_VAL_Oct9/training_results/checkpoint-324000/ \
--saved_dataset_folder  /localdata_ssd/nuplan_diff_save_dir/diff_dataset/arrow_dataset/ \
--output_dir /localdata_hdd2/nuplan_diff_save_dir/otpt_dir/small_skpy_pi2_x10_auxyd1_val_ckptoct9_324k_trainDiff__/output_dir/ \
--logging_dir /localdata_hdd2/nuplan_diff_save_dir/otpt_dir/small_skpy_pi2_x10_auxyd1_val_ckptoct9_324k_trainDiff__/result_dir/ \
--run_name small_skpy_pi2_x10_auxyd1_val_ckptoct9_324k_trainDiff__ \
--num_train_epochs 10 \
--weight_decay 0.00001 --learning_rate 0.0001 --logging_steps 2 --save_strategy steps \
--dataloader_num_workers 10 \
--per_device_train_batch_size 2 \
--save_total_limit 2  --predict_yaw True \
--dataloader_drop_last True \
--task train_diffusion_decoder \
--remove_unused_columns False \
--do_train \
--loss_fn mse \
--per_device_eval_batch_size 2 --max_eval_samples 99999999 \
--max_train_samples 99999999 \
--max_predict_samples 99999999 \
--past_sample_interval 2 \
--trajectory_loss_rescale 0.00001 \
`

After training the diffusion keypoint decoder separately, you may use such command to eval its performance:
`
export CUDA_VISIBLE_DEVICES=0; \
nohup python3 -m torch.distributed.run \
--master_port 12363 \
--nproc_per_node=1 runner.py \
--model_name pretrain-gpt-small-gen1by1 --model_pretrain_name_or_path /public/MARS/t4p/checkpoints/Small_Oct9/Small_SKPY_PI2_x1_auXYd1_auCurrentd1_VAL_Oct9/training_results/checkpoint-324000/ \
--saved_dataset_folder  /localdata_ssd/nuplan/online_float32_opt \
--output_dir /localdata_hdd2/nuplan_diff_save_dir/otpt_dir/Small_SKPY_PI2_x10_auXYd1_VAL_ckpt407k_sjztest_0100_wiDiffusion_______gen1by1_filtered_Oct9ckpt324k_val14_1k_/output_dir/ \
--logging_dir /localdata_hdd2/nuplan_diff_save_dir/otpt_dir/Small_SKPY_PI2_x10_auXYd1_VAL_ckpt407k_sjztest_0100_wiDiffusion_______gen1by1_filtered_Oct9ckpt324k_val14_1k_/result_dir/ \
--run_name Small_SKPY_PI2_x10_auXYd1_VAL_ckpt407k_sjztest_0100_wiDiffusion_______gen1by1_filtered_Oct9ckpt324k_val14_1k_ \
--num_train_epochs 1 \
--weight_decay 0.00 --learning_rate 0.00 --logging_steps 2 --save_strategy steps \
--dataloader_num_workers 10 \
--per_device_train_batch_size 2 \
--save_total_limit 2  --predict_yaw True \
--dataloader_drop_last True \
--task nuplan \
--remove_unused_columns False \
--do_eval \
--evaluation_strategy epoch \
--loss_fn mse \
--per_device_eval_batch_size 2 --max_eval_samples 99999999 \
--max_train_samples 99999999 \
--max_predict_samples 99999999 \
--past_sample_interval 2 \
--kp_decoder_type diffusion --key_points_diffusion_decoder_feat_dim 256 --diffusion_condition_sequence_lenth 1 --key_points_diffusion_decoder_load_from /localdata_hdd2/nuplan_diff_save_dir/otpt_dir/small_skpy_pi2_x10_auxyd1_val_ckptoct9_324k_trainDiff__/output_dir/checkpoint-500/pytorch_model.bin \
`
## To eval only:


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
--max_predict_samples 1000 \
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

### Generate WOMD training data dictionary and index

`
python waymo_generation.py --train --save_dict
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

### Statics Simulation Scores
`python script/run_metric_scores.py --file_path ... --save_dir ... --exp_name ... --simulation_type ...`
