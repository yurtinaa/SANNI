from dataclasses import dataclass
from enum import Enum

from .FrameParam import FrameworkType
from .error.AbstractError import AbstractErrorFactory, AbstractError
from .optimizer.abstract_scheduler import AbstractScheduler
from .score import Score
from .optimizer.abstract_optimizer import AbstractOptimizer
# from LossFunction import MPDELoss
from abc import ABC, abstractmethod

import torch


@dataclass
class NeuralNetworkConfig(ABC):
    batch_size: int
    epochs: int
    error_factory: AbstractErrorFactory
    optimizer_type: AbstractOptimizer
    score_factory: Score = None
    scheduler_type: AbstractScheduler = None
    early_stopping_patience: int = 0

    @property
    @abstractmethod
    def optimizer(self):
        pass

    @property
    @abstractmethod
    def error(self) -> AbstractError:
        pass

    @property
    @abstractmethod
    def score(self) -> Score:
        pass


@dataclass
class TorchNNConfig(NeuralNetworkConfig):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def optimizer(self):
        return self.optimizer_type(FrameworkType.Torch)

    @property
    def error(self):
        """Геттер для получения класса ошибки на основе error_type."""
        return self.error_factory(FrameworkType.Torch)

    @property
    def score(self) -> Score:
        """Геттер для получения класса ошибки на основе error_type."""
        if self.score_factory is None:
            return self.error_factory(FrameworkType.Torch)
        return self.score_factory

    @property
    def scheduler(self):
        if self.scheduler_type is None:
            return None
        return self.scheduler(FrameworkType.Torch)


@dataclass
class TimeSeriesConfig:
    dim: int
    window_size: int
