from transformer4planning.models.backbone.str_base import STR, STRConfig
from transformers import (MixtralModel, MixtralPreTrainedModel, MixtralConfig)

# import load_balancing_loss_func from transformers
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func


class STRMixtralConfig(MixtralConfig, STRConfig):
    pass


class STR_Mixtral(STR, MixtralPreTrainedModel):
    """
    STR with GPT2 as backbone
    MRO (Method Resolution Order) is important here, will call STR's forward and generate method
    """
    def __init__(self, config):
        super().__init__(config)
        self.transformer = MixtralModel(config)
        if self.config.output_router_logits:
            self.load_balancing_loss_func = load_balancing_loss_func
        # assert self.config.output_router_logits is False, "Router z-loss is not supported yet"
