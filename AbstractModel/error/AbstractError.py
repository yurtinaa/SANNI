from abc import ABC, abstractmethod
from enum import Enum


from ..FrameParam import FrameworkType


class ErrorType(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    CE = "cross_entropy"
    MPDE = "MPDE"
    LogCosh = "LogCosh"
    QuantileLoss = "QuantileLoss"
    DTW = "DTW"


base_error_list = [ErrorType.MSE, ErrorType.MAE, ErrorType.LogCosh, ErrorType.QuantileLoss]


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
