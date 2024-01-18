from transformer4planning.models.backbone.str_base import STR, STRConfig
from transformers import (GPT2Model, GPT2PreTrainedModel, GPT2Config)


class STRGPT2Config(GPT2Config, STRConfig):
    pass


class STR_GPT2(STR, GPT2PreTrainedModel):
    """
    STR with GPT2 as backbone
    MRO (Method Resolution Order) is important here, will call STR's forward and generate method
    """
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
