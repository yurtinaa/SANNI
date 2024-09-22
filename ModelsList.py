from enum import Enum

from .AbstractModel.AbstractImpute import AbstractImpute
from .PyPOTSAdapter.BRITS.BRITS import BRITSImpute
from .PyPOTSAdapter.SAITS.SAITS import SAITSImpute
from .SAETI import SAETI
from .SANNI import SANNI


class ModelType(str, Enum):
    SANNI = "SANNI"
    SAETI = "SAETI"
    BRITS = "BRITS"
    SAITS = "SAITS"



_model_dict = {
    ModelType.SANNI: SANNI,
    ModelType.SAETI: SAETI,
    ModelType.BRITS: BRITSImpute,
    ModelType.SAITS: SAITSImpute
}


def get_model(score_type: ModelType) -> AbstractImpute:
    return _model_dict[score_type]
