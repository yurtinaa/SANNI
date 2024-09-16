from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from AbstractModel.FrameParam import FrameworkType


class ErrorType(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    CE = "cross_entropy"
    MPDE = "MPDE"
    LogCosh = "LogCosh"
    QuantileLoss = "quantile"


class AbstractError(ABC):
    __name: str

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @abstractmethod
    def __call__(self, X, Y, Y_pred):
        pass


class AbstractErrorFactory(ABC):
    name: str

    @abstractmethod
    def __call__(self, frame_type: FrameworkType):
        pass

    def __repr__(self):
        return f"{self.name}"
