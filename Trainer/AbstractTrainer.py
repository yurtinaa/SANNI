from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from importlib.abc import Loader

import numpy as np

from AbstractModel.Parametrs import NeuralNetworkConfig
from EnumConfig import EpochType
# from AbstractModel.AbstractLogger import AbstractLogger
from Trainer.Loader import AbstractLoader
from Logger.AbstractLogger import AbstractLogger


@dataclass
class AbstractModel(ABC):

    @property
    @abstractmethod
    def model(self):
        pass

    @model.setter
    @abstractmethod
    def model(self, value):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def __call__(self, X):
        pass

@dataclass
class TrainParams:
    epoch: int = 0
    error: float = np.inf

    def copy(self):
        return TrainParams(epoch=self.epoch,
                           error=self.error)


@dataclass
class AbstractTrainer(ABC):
    loader: AbstractLoader
    current_model: AbstractModel
    config: NeuralNetworkConfig
    logger: AbstractLogger
    device: str = None
    best_model: AbstractModel = None
    train_params: TrainParams = field(default_factory=TrainParams)
    _best_params: TrainParams = field(default_factory=TrainParams)

    @abstractmethod
    def __post_init__(self):
        pass
    #
    # @abstractmethod
    # def __get_batch(self):
    #     pass

    @abstractmethod
    def _one_epoch(self, type_: EpochType):
        pass

    @abstractmethod
    def _log_data(self,log_dict):
        pass

    @abstractmethod
    def _update_model(self):
        pass

    @abstractmethod
    def train(self):
        pass
