from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from importlib.abc import Loader

import numpy as np

from AbstractModel.Parametrs import NeuralNetworkConfig
from AbstractModel.AbstractLogger import AbstractLogger
from Loader import AbstractLoader


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
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass


class EpochType(Enum):
    TRAIN = 0
    EVAL = 1


@dataclass
class TrainParams:
    epoch: int = 0
    error: float = np.inf


@dataclass
class AbstractTrainer(ABC):
    loader: AbstractLoader
    current_model: AbstractModel
    config: NeuralNetworkConfig
    train_params: TrainParams
    best_model: AbstractModel
    logger: AbstractLogger
    _best_params: TrainParams

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def __get_batch(self):
        pass

    @abstractmethod
    def __one_epoch(self, type_: EpochType):
        pass

    @abstractmethod
    def __log_data(self):
        pass

    @abstractmethod
    def __update_model(self):
        pass

    @abstractmethod
    def train(self):
        pass
