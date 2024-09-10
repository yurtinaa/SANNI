from enum import Enum

from AbstractModel.AbstractImpute import AbstractImpute
from SAETI import SAETI
from SANNI import SANNI


class ModelType(str, Enum):
    SANNI = "SANNI"
    SAETI = "SAETI"


_model_dict = {
    ModelType.SANNI: SANNI,
    ModelType.SAETI: SAETI
}


def get_model(score_type: ModelType) -> AbstractImpute:
    return _model_dict[score_type]
