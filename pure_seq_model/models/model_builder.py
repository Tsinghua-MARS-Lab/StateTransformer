from models.pure_seq_model_v1_aug import PureSeqModelV1Aug
from models.constant_vel_model import ConstantVelModel
from models.pure_seq_model_v1_no_vel import PureSeqModelV1NoVel
from models.pure_seq_model_v2 import PureSeqModelV2
from models.base_encoder_decoder_model import BaseEncoderDecoderModel
from models.learned_anchor_model import LearnedAnchorModel
from models.wayformer_model import WayformerModel
from models.wayformer_gmm_model import WayformerGMMModel

__all__ = {
    'PureSeqModelV1Aug': PureSeqModelV1Aug,
    'PureSeqModelV1NoVel': PureSeqModelV1NoVel,
    'ConstantVelModel': ConstantVelModel,
    'PureSeqModelV2': PureSeqModelV2,
    'BaseEncoderDecoderModel': BaseEncoderDecoderModel,
    'LearnedAnchorModel': LearnedAnchorModel,
    'WayformerModel': WayformerModel,
    'WayformerGMMModel': WayformerGMMModel,
}

def build_model(config):
    model = __all__[config.MODEL.model_name](
        config=config
    )
    return model