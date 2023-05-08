from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: str = field(
        default="TransfoXLModelNuPlan_Config",
        metadata={"help": "Name of a planning model backbone"}
    )
    model_pretrain_name_or_path: str = field(
        default="transfo-xl-wt103",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    predict_result_saving_dir: Optional[str] = field(
        default=False,
        metadata={"help": "The target folder to save prediction results."},
    )
    use_nsm: Optional[bool] = field(
        default=False,
    )
    predict_intended_maneuver: Optional[bool] = field(
        default=False,
    )
    predict_current_maneuver: Optional[bool] = field(
        default=False,
    )
    predict_trajectory: Optional[bool] = field(
        default=True,
    )
    recover_obs: Optional[bool] = field(
        default=False,
    )
    maneuver_repeat: Optional[bool] = field(
        default=False,
    )
    predict_trajectory_with_nsm: Optional[bool] = field(
        default=False,
    )
    predict_trajectory_with_stopflag: Optional[bool] = field(
        default=False,
    )
    mask_history_intended_maneuver: Optional[bool] = field(
        default=False,
    )
    mask_history_current_maneuver: Optional[bool] = field(
        default=False,
    )
    predict_intended_maneuver_change: Optional[bool] = field(
        default=False,
    )
    predict_intended_maneuver_change_non_persuasive: Optional[bool] = field(
        default=False,
    )
    predict_current_maneuver_change: Optional[bool] = field(
        default=False,
    )
    d_embed: Optional[int] = field(
        default=384,
    )
    d_model: Optional[int] = field(
        default=384,
    )
    d_inner: Optional[int] = field(
        default=1024,
    )
    n_layers: Optional[int] = field(
        default=4,
    )
    # Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    activation_function: Optional[str] = field(
        default = "silu"
    )