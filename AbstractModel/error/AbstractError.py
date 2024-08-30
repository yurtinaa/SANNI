from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from AbstractModel.FrameParam import FrameworkType
class ErrorType(str, Enum):
    MSE = "mean_squared_error"
    MAE = "mean_absolute_error"
    CE = "cross_entropy"
    MPDE = "mean_profile_distante_error"


class AbstractError(ABC):

    @abstractmethod
    def __call__(self, X, Y, Y_pred):
        pass
