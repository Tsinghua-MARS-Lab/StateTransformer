from dataclasses import dataclass, field
from typing import Optional, List
from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: str = field(
        default="scratch-mini-gpt",
        metadata={"help": "Name of a planning model backbone"}
    )
    model_pretrain_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    d_embed: Optional[int] = field(
        default=256,
    )
    d_model: Optional[int] = field(
        default=256,
    )
    d_inner: Optional[int] = field(
        default=1024,
    )
    n_layers: Optional[int] = field(
        default=4,
    )
    n_heads: Optional[int] = field(
        default=8,
    )
    activation_function: Optional[str] = field(
        default="gelu_new",
        metadata={"help": "Activation function, to be selected in the list `[relu, silu, gelu, tanh, gelu_new]"},
    )
    task: Optional[str] = field(
        default="nuplan"
    )
    encoder_type: Optional[str] = field(
        default='raster',
        metadata={"help": "choose from [raster, vector]"}
    )
    raster_channels: Optional[int] = field(
        default=34,  # updated channels (added both block and lanes for route), change to 33 for older version
        metadata={"help": "default is 0, automatically compute. [WARNING] only supports nonauto-gpt now."},
    )
    raster_encoder_type: Optional[str] = field(
        default='resnet18',
        metadata={"help": "choose from [vit, resnet18, resnet34, resnet50, resnet101, resnet152]"}
    )
    vit_intermediate_size: Optional[int] = field(
        default=3072,
    )
    pretrain_encoder: Optional[bool] = field(
        default=False,
    )
    k: Optional[int] = field(
        default=1,
        metadata={"help": "Set k for top-k predictions, set to -1 to not use top-k predictions."},
    )
    x_random_walk: Optional[float] = field(
        default=0.0
    )
    y_random_walk: Optional[float] = field(
        default=0.0
    )
    predict_yaw: Optional[bool] = field(
        default=True
    )
    loss_fn: Optional[str] = field(
        default="mse",
    )
    trajectory_loss_rescale: Optional[float] = field(
        default=1.0
    )
    mean_circular_loss: Optional[bool] = field(
        default=True
    )

    ######## begin of proposal args ########
    use_proposal: Optional[int] = field(
        default=0,
        metadata={"help": "number of proposal candidates. 0: not using proposal"}
    )
    ######## end of speed args ########
    use_speed: Optional[bool] = field(
        default=False
    )
    ######## begin of key points args ########
    use_key_points: Optional[str] = field(
        default='specified_backward',
        metadata={"help": "no: not using key points,"
                          "universal: using universal key points, with interval of 20 frames."
                          "specified_forward: using specified key points, with exponentially growing frame indices."
                          "specified_backward: using specified key points, with exponentially growing frame indices."
                          "specified_two_backward: 8s, and 0.5s only"
                          "specified_first_second: 1s only"}
    )
    separate_kp_encoder: Optional[bool] = field(
        default=True
    )
    pred_key_points_only: Optional[bool] = field(
        default=False
    )
    pred_traj_only: Optional[bool] = field(
        default=False
    )
    arf_x_random_walk: Optional[float] = field(
        default=0.0
    )
    arf_y_random_walk: Optional[float] = field(
        default=0.0
    )
    kp_decoder_type: Optional[str] = field(
        default='mlp',
        metadata={"help": "choose from [linear, mlp, diffusion]"}
    )
    kp_dropout: Optional[float] = field(
        default=0.0
    )
    trajectory_decoder_type: Optional[str] = field(
        default='mlp',
        metadata={"help": "choose from [linear, mlp]"}
    )
    traj_dropout: Optional[float] = field(
        default=0.0
    )
    ######## end of key points args ########

    ######## begin of diffusion decoder args ########
    mc_num: Optional[int] = field(
        default=200, metadata={"help": "The number of sampled KP trajs the diffusionKPdecoder is going to generate. After generating this many KP trajs, they go through the EM algorithm and give a group of final KP trajs of number k. This arg only works when we use diffusionKPdecoder and set k > 1."}
    )
    key_points_diffusion_decoder_feat_dim: Optional[int] = field(
        default=256, metadata={"help": "The feature dimension for key_poins_diffusion_decoder. 256 for a diffusion KP decoder of #parameter~10M and 1024 for #parameter~100M."}
    )
    key_points_num: Optional[int] = field(
        default=5, metadata={"help": "Number of key points. Only used to initialize diffusion KP decoder."}
    )
    diffusion_condition_sequence_lenth: Optional[int] = field(
        default=1, metadata={"help": "Lenth of condition input into diffusion KP decoder. It should be equal to: 1."}

    )
    key_points_diffusion_decoder_load_from: Optional[str] = field(
        default=None, metadata={"help": "From which file to load the pretrained key_points_diffusion_decoder."}
    )
    ######## end of diffusion decoder args ########

    ######## begin of camera images args ########
    camera_image_encoder: Optional[str] = field(
        default=None, metadata={"help": "choose from [dinov2], set None to not use camera images"}
    )
    train_camera_image_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder of camera images for training. Set None to not use camera images."}
    )
    val_camera_image_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder of camera images for validation. Set None to not use camera images."}
    )
    ####### end of camera images args ########

    ######## begin of nuplan args ########
    with_traffic_light: Optional[bool] = field(
        default=True
    )
    past_sample_interval: Optional[int] = field(
        default=2
    )
    selected_exponential_past: Optional[bool] = field(
        default=True
    )
    current_frame_only: Optional[bool] = field(
        default=False
    )
    future_sample_interval: Optional[int] = field(
        default=2
    )
    augment_current_pose_rate: Optional[float] = field(
        # currently this only works for raster preprocess, and aug_x, aug_y are default to 1.0
        default=0.0, metadata={"help": "The rate of augmenting current pose in the preprocess"}
    )
    augment_current_ratio: Optional[float] = field(
        default=0.3, metadata={"help": "The ratio of augmenting current pose in the preprocess"}
    )
    augment_current_with_past_linear_changes: Optional[bool] = field(
        default=False, metadata={"help": "Whether to augment past poses with linear changes"}
    )
    augment_current_with_future_linear_changes: Optional[bool] = field(
        default=False, metadata={"help": "Whether to augment future poses with linear changes"}
    )
    generate_diffusion_dataset_for_key_points_decoder: Optional[bool] = field(
        default = False, metadata={"help": "Whether to generate and save the diffusion_dataset_for_keypoint_decoder. This is meant to train the diffusion decoder for class TrajectoryGPTDiffusionKPDecoder, in which ar_future_interval > 0 and the key_poins_decoder is a diffusion decoder while the traj_decoder is a plain decoder. Need to be used with a pretrained model of name pretrain-gpt and ar_future_interval > 0."}
    )
    diffusion_feature_save_dir: Optional[str] = field(
        default = None, metadata = {"help":"where to save diffusion dataset."}
    )
    ######## end of nuplan args ########

    ######## begin of WOMD args ########
    proposal_path: Optional[str] = field(
        default="data/waymo/cluster_64_center_dict.pkl"
    )
    dense_pred: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use dense prediction in MTR model"}
    )
    ######## end of WOMD args ########

    # WIP args
    autoregressive_proposals: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use autoregressive proposals in MTR model"}
    )
    proposal_num: Optional[int] = field(
        default=13
    )

    ######## begin of Mamba args ########
    rms_norm: Optional[bool] = field(
        default=False
    )
    residual_in_fp32: Optional[bool] = field(
        default=False
    )
    fused_add_norm: Optional[bool] = field(
        default=False
    )
    ######## end of Mamba args ########

    use_mission_goal: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use mission goal in the model"}
    )
    mission_goal_dropout: Optional[float] = field(
        default=1.0, metadata={"help": "Rate to dropout the mission goal, 1 is no dropout"}
    )
    attention_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "The dropout ratio for attention layers."}
    )
    num_local_experts: Optional[int] = field(
        default=8, metadata={"help": "The number of local experts."}
    )
    debug_raster_path: Optional[str] = field(
        default=None, metadata={"help": "The path of raster image for debugging."}
    )
    augment_index: Optional[int] = field(
        default=0, metadata={"help": "The index of augmenting current pose in the preprocess"}
    )

    use_cache: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use cache in the model"}
    )
    attn_implementation: Optional[str] = field(
        default=None, metadata={"help": "The implementation of attention layers."}
    )
    sync_norm: Optional[bool] = field(
        default=False, metadata={"help": "use SyncBatchNorm over all GPUs."}
    )
    ######## temporal model args, check your model config before using it! ########
    trajectory_prediction_mode: Optional[str] = field(
        default='all_frames', metadata={"help": "choose from [all_frames, start_frame, both, rebalance]"}
    )
    finetuning_with_stepping: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use finetuning with stepping"}
    )
    finetuning_gamma: Optional[float] = field(
        default=1, metadata={"help": "Discount of loss from the next step"}
    )
    finetuning_with_stepping_random_steps: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use finetuning with stepping with random steps"}
    )
    finetuning_with_stepping_minimal_step: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use finetuning with stepping 0.1s"}
    )
    finetuning_with_stepping_without_first_step_loss: Optional[bool] = field(  # about to delete
        default=False, metadata={"help": "Whether to use finetuning with stepping without first step loss"}
    )
    no_yaw_with_stepping: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use finetuning with stepping without yaw"}
    )

    first_frame_scele: Optional[float] = field(
        default=1000.0, metadata={"help": "The scale of the first frame"}
    )
    kp_loss_rescale: Optional[float] = field(
        default=1.0, metadata={"help": "Whether to rescale the key points loss"}
    )
    # end of auturegressive args
    reverse_traj_index_order: Optional[bool] = field(
        default=False, metadata={"help": "Whether to reverse the trajectory index order"}
    )
    inspect_kp_loss: Optional[bool] = field(
        default=False, metadata={"help": "Whether to inspect the key points loss one by one"}
    )

    # begin of MoE configs
    num_local_experts: Optional[int] = field(
        default=8, metadata={"help": "The number of local experts."}
    )
    num_experts_per_token: Optional[int] = field(
        default=2, metadata={"help": "The number of experts to route per-token, can be also interpreted as the top-k routing parameter."}
    )
    router_jitter_noise: Optional[float] = field(
        default=0.0, metadata={"help": "The noise added to the router logits. From 0 to 1."}
    )
    router_aux_loss_coef: Optional[float] = field(
        default=0.001, metadata={"help": "The coefficient of the router auxiliary loss."}
    )
    # end of MoE configs

    # WIP: key point tokenize
    kp_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "choose from [none, uniform, cluster, gaussian]"}
    )
    kp_cluster_files: Optional[str] = field(
        default=None, metadata={"help": "csv files which record all cluster center info for 8s 4s 2s 1s 0.5s"}
    )

    regression_long_class_short: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use regression long class short"}
    )

    add_regression_loss: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add regression loss"}
    )

    # WIP: training with the simulation scores as loss
    finetuning_with_simulation_on_val: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use finetuning with simulation on val"}
    )
    simulate_one_step_on_training: Optional[bool] = field(
        default=False, metadata={"help": "Whether to simulate one step on training"}
    )
    simulate_with_5f_smoothing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to simulate with 5 frame smoothing"}
    )
    sim_eval_with_gt: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gt for sim"}
    )
    skip_yaw_norm: Optional[bool] = field(
        default=False, metadata={"help": "Skip rotation normalization during preprocess"}
    )

    autoregressive: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use autoregressive prediction"}
    )
    future_seconds: Optional[int] = field(
        default=8, metadata={"help": "The number of seconds to predict in the future"}
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    saved_dataset_folder: Optional[str] = field(
        default=None, metadata={"help": "The path of a pre-saved dataset folder. The dataset should be saved by Dataset.save_to_disk())."}
    )
    saved_valid_dataset_folder: Optional[str] = field(
        default=None, metadata={"help": "The path of a pre-saved validation dataset folder. The dataset should be saved by Dataset.save_to_disk())."}
    )
    camera_images_path: Optional[str] = field(
        default=None, metadata={"help": "path to the folder containing camera images"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
            )
        },
    )
    max_sim_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of simulation examples to this "
                "value if set."
            )
        },
    )
    dataset_scale: Optional[float] = field(
        default=1, metadata={"help": "The dataset size, choose from any float <=1, such as 1, 0.1, 0.01"}
    )
    dagger: Optional[bool] = field(
        default=False, metadata={"help": "(WIP) Whether to save dagger results"}
    )
    nuplan_map_path: Optional[str] = field(
        default=None, metadata={"help": "The root path of map file, to init map api used in nuplan package"}
    )
    agent_type: Optional[str] = field(
        default="all", metadata={"help": "all: no filter on WOMD"
                                        "1: vehicle on WOMD"
                                        "2: pedestrian on WOMD"
                                        "3: cyclist on WOMD"
                                        "any combination of numbers will be decoded into list of int (1 2;2 3;1 3)"}
    )


@dataclass
class ConfigArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    analyze_dataset_target: Optional[str] = field(
        default=None, metadata={"help": "choose from train/val/test to choose which dataset to analyze"}
    )
    save_analyze_result_to_path: Optional[str] = field(
        default=None, metadata={"help": "save analyze result to path if not None"}
    )


@dataclass
class PlanningTrainingArguments(TrainingArguments):
    """
    Warnings: This overrides the TrainingArguments in transformers. DOES NOT WORK FOR UNKNOWN REASONs.
    """
    eval_interval: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "how many epoch the model perform an evaluation."
            )
        },
    )
    do_test: Optional[bool] = field(
        default=False,
    )
    images_cleaning_to_folder: Optional[str] = field(
        default=None, metadata={"help": "Pass a target folder to clean the raw image folder to the target folder."}
    )
    do_sim_val: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do simulation validation"}
    )
    do_sim_test: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do simulation test"}
    )
    sim_test_type: Optional[str] = field(
        default='closed_loop_nonreactive_agents', metadata={"help": "choose from test/val to choose which simulation dataset to use"}
    )
    nuplan_sim_data_path: Optional[str] = field(
        default=None, metadata={"help": "The path of nuplan simulation raw db data"}
    )
    nuplan_sim_map_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder of nuplan simulation raw map"}
    )
    nuplan_sim_exp_root: Optional[str] = field(
        default=None, metadata={"help": "The root path of nuplan simulation experiment data"}
    )
    nuplan_sim_split_filter_yaml: Optional[str] = field(
        default=None, metadata={"help": "The path of nuplan simulation split filter yaml"}
    )
    nuplan_train_data_path: Optional[str] = field(
        default=None, metadata={"help": "The path of nuplan training raw db data"}
    )
    nuplan_train_split_filter_yaml: Optional[str] = field(
        default=None, metadata={"help": "The path of nuplan train split filter yaml"}
    )
    sim_steps: Optional[int] = field(
        default=None, metadata={"help": "The number of simulation steps"}
    )
    sim_controller: Optional[str] = field(
        default='two_stage_controller', metadata={"help": "The controller for simulation, choose from [perfect_controller, two_stage_controller]"}
    )

    # WIP
    # long_model_ckpt_path: Optional[str] = field(
    #     default=None, metadata={"help": "The path of the long model checkpoint"}
    # )


    # label_names: Optional[List[str]] = field(
    #     default=lambda: ['trajectory_label']
    # )
    # prediction_loss_only: Optional[bool] = field(
    #     default=False,
    # )
