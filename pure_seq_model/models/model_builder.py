from models.pure_seq_model_v1 import PureSeqModelV1
from models.pure_seq_model_v1_aug import PureSeqModelV1Aug

__all__ = {
    'PureSeqModelV1': PureSeqModelV1,
    'PureSeqModelV1Aug': PureSeqModelV1Aug,
}

def build_model(config):
    model = __all__[config.MODEL.model_name](
        config=config
    )
    return model