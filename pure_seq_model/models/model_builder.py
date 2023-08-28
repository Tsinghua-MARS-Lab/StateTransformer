from models.pure_seq_model_v1_aug import PureSeqModelV1Aug
from models.constant_vel_model import ConstantVelModel
from models.pure_seq_model_v1_no_vel import PureSeqModelV1NoVel

__all__ = {
    'PureSeqModelV1Aug': PureSeqModelV1Aug,
    'PureSeqModelV1NoVel': PureSeqModelV1NoVel,
    'ConstantVelModel': ConstantVelModel,
}

def build_model(config):
    model = __all__[config.MODEL.model_name](
        config=config
    )
    return model