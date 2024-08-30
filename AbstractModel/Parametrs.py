from dataclasses import dataclass
from enum import Enum

from AbstractModel.FrameParam import FrameworkType
from AbstractModel.optimizer.abstract_scheduler import AbstractScheduler
from optimizer.abstract_optimizer import AbstractOptimizer
from LossFunction import MPDELoss
from abc import ABC, abstractmethod

import torch




@dataclass
class NeuralNetworkConfig(ABC):
    batch_size: int
    epochs: int
    lr: float
    error_type: ErrorType
    optimizer_type: AbstractOptimizer
    scheduler_type: AbstractScheduler = None

    @property
    @abstractmethod
    def optimizer(self):
        pass

    @property
    @abstractmethod
    def error(self):
        pass




@dataclass
class TorchNNConfig(NeuralNetworkConfig):


    @property
    def optimizer(self):
        return self.optimizer_type(FrameworkType.Torch)

    @property
    def error(self):
        """Геттер для получения класса ошибки на основе error_type."""
        if self.error_type not in self._error_classes:
            raise ValueError(f"Unknown error type {self.error_type}")
        return self._error_classes[self.error_type]

    @property
    def scheduler(self):
        if self.scheduler_type is None:
            return None
        return self.scheduler(FrameworkType.Torch)


@dataclass
class TimeSeriesConfig:
    dim: int
    window_size: int
