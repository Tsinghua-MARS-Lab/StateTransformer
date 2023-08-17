from dataclasses import dataclass, field
from typing import Optional
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
        default="silu",
        metadata={"help": "Activation function, to be selected in the list `[relu, silu, gelu, tanh, gelu_new]"},
    )
    loss_fn: Optional[str] = field(
        default="mse",
    )
    task: Optional[str] = field(
        default="nuplan" # only for mmtransformer
    )
    with_traffic_light: Optional[bool] = field(
        default=True
    )
    autoregressive: Optional[bool] = field(
        default=False
    )
    k: Optional[int] = field(
        default=1,
        metadata={"help": "Set k for top-k predictions, set to -1 to not use top-k predictions."},
    )
    next_token_scorer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use next token scorer for prediction."},
    )
    x_random_walk: Optional[float] = field(
        default=0.0
    )
    y_random_walk: Optional[float] = field(
        default=0.0
    )
    tokenize_label: Optional[bool] = field(
        default=True
    )
    raster_channels: Optional[int] = field(
        default=33,
        metadata={"help": "default is 0, automatically compute. [WARNING] only supports nonauto-gpt now."},
    )
    predict_yaw: Optional[bool] = field(
        default=False
    )
    ar_future_interval: Optional[int] = field(
        default=20,
        metadata={"help": "default is 0, don't use auturegression. [WARNING] only supports nonauto-gpt now."},
    )
    arf_x_random_walk: Optional[float] = field(
        default=0.0
    )
    arf_y_random_walk: Optional[float] = field(
        default=0.0
    )
    trajectory_loss_rescale: Optional[float] = field(
        default=1.0
    )
    visualize_prediction_to_path: Optional[str] = field(
        default=None
    )
    pred_key_points_only: Optional[bool] = field(
        default=False
    )
    specified_key_points: Optional[bool] = field(
        default=True
    )
    forward_specified_key_points: Optional[bool] = field(
        default=False
    )
    token_scenario_tag: Optional[bool] = field(
        default=False
    )
    max_token_len: Optional[int] = field(
        default=20
    )
    resnet_type: Optional[str] = field(
        default="resnet18",
        metadata={"help": "choose from [resnet18, resnet34, resnet50, resnet101, resnet152]"}
    )
    pretrain_encoder: Optional[bool] = field(
        default=False,
    )
    encoder_type: Optional[str] = field(
        default='raster',
        metadata={"help": "choose from [raster, vector]"}
    )
    past_sample_interval: Optional[int] = field(
        default=5
    )
    future_sample_interval: Optional[int] = field(
        default=2
    )
    debug_raster_path: Optional[str] = field(
        default=None
    )
    generate_diffusion_dataset_for_key_points_decoder: Optional[bool] = field(
        default = False, metadata={"help": "Whether to generate and save the diffusion_dataset_for_keypoint_decoder. This is meant to train the diffusion decoder for class TrajectoryGPTDiffusionKPDecoder, in which ar_future_interval > 0 and the key_poins_decoder is a diffusion decoder while the traj_decoder is a plain decoder. Need to be used with a pretrained model of name pretrain-gpt and ar_future_interval > 0."}
    )
    mc_num: Optional[int] = field(
        default = 200, metadata = {"help": "The number of sampled KP trajs the diffusionKPdecoder is going to generate. After generating this many KP trajs, they go through the EM algorithm and give a group of final KP trajs of number k. This arg only works when we use diffusionKPdecoder and set k > 1."}
    )
    key_points_diffusion_decoder_feat_dim: Optional[int] = field(
        default = 256, metadata = {"help": "The feature dimension for key_poins_diffusion_decoder. 256 for a diffusion KP decoder of #parameter~10M and 1024 for #parameter~100M."}
    )
    key_points_num: Optional[int] = field(
        default = 5, metadata = {"help": "Number of key points. Only used to initialize diffusion KP decoder."}
    )
    diffusion_condition_sequence_lenth: Optional[int] = field(
        default = 16, metadata = {"help": "Lenth of condition input into diffusion KP decoder. It should be equal to: scenario_type_len + context_length * 2."}
    )
    key_points_diffusion_decoder_load_from: Optional[str] = field(
        default = None, metadata = {"help": "From which file to load the pretrained key_points_diffusion_decoder."}
    )
    interaction: Optional[bool] = field(
        default=False
    )
    mtr_config_path: Optional[str] = field(
        default="/home/ldr/workspace/transformer4planning/config/gpt.yaml"
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )    

    dataset_scale: Optional[float] = field(
        default=1, metadata={"help":"The dataset size, choose from any float <=1, such as 1, 0.1, 0.01"}
    )
    dagger: Optional[bool] = field(
        default=False, metadata={"help":"Whether to save dagger results"}
    )
    nuplan_map_path: Optional[str] = field(
        default=None, metadata={"help":"The root path of map file, to init map api used in nuplan package"}
    )

@dataclass
class ConfigArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    save_model_config_to_path: Optional[str] = field(
        default=None, metadata={"help": "save current model config to a json file if not None"}
    )
    save_data_config_to_path: Optional[str] = field(
        default=None, metadata={"help": "save current data config to a json file if not None"}
    )
    load_model_config_from_path: Optional[str] = field(
        default=None, metadata={"help": "load model config from a json file if not None"}
    )
    load_data_config_from_path: Optional[str] = field(
        default=None, metadata={"help": "load data config to a json file if not None"}
    )

@dataclass
class PlanningTrainingArguments(TrainingArguments):
    eval_interval: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "how many epoch the model perform an evaluation."
            )
        },
    )