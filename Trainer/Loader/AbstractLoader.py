from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Any, Union

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..AbstractTrainer import EpochType


@dataclass
class DataLoader(ABC):
    X: Union[torch.Tensor, np.ndarray]
    y: Union[torch.Tensor, np.ndarray]
    X_val: Union[torch.Tensor, np.ndarray] = None
    y_val: Union[torch.Tensor, np.ndarray] = None
    batch_size: int = 64
    percent: float = 0.25
    shuffle: bool = True
    seed: int = 1233

    @abstractmethod
    def length(self, epoch_type: EpochType) -> float:
        pass

    # @abstractmethod
    # def load(self, source: str) -> None:
    #     """Загружает данные из источника"""
    #     pass

    @abstractmethod
    def __iter__(self, epoch_type: EpochType):
        """Возвращает итератор для перебора данных"""
        pass

    # @abstractmethod
    # def preprocess(self, data: Any) -> Any:
    #     """Предварительная обработка данных"""
    #     pass
