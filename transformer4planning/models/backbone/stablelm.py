from transformer4planning.models.backbone.str_base import STR, STRConfig
from transformers import (StableLmModel, StableLmPreTrainedModel, StableLmConfig)


class STRStableLMConfig(StableLmConfig, STRConfig):
    pass


class STR_StableLM(STR, StableLmPreTrainedModel):
    """
    STR with GPT2 as backbone
    MRO (Method Resolution Order) is important here, will call STR's forward and generate method
    """
    def __init__(self, config):
        super().__init__(config)
        self.transformer = StableLmModel(config)
