from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

# from AbstractModel.FrameParam import FrameworkType
from .Parametrs import TimeSeriesConfig, NeuralNetworkConfig
from ..Logger.AbstractLogger import AbstractLogger
# from Logger.ConsoleLogger import ConsoleLogger
from ..Trainer.AbstractTrainer import AbstractTrainer, AbstractModel


@dataclass
class AbstractImpute(ABC):
    time_series: TimeSeriesConfig
    neural_network_config: NeuralNetworkConfig
    name: str = None

    device: str = 'cpu'
    logger: AbstractLogger = None
    _trainer: AbstractTrainer = None
    _model: AbstractModel = None

    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray,
              X_val: np.ndarray=None, Y_val: np.ndarray=None):
        pass

    def train_with_val(self, X: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass


class AbstractClassifier(ABC):
    neural_network_config: NeuralNetworkConfig
    logger: AbstractLogger
    device: str = 'cpu'
    logger: AbstractLogger = None
    __trainer: AbstractTrainer = None
    __model: AbstractModel = None

    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass
