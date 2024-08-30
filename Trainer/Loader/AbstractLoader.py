import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Any

from abc import ABC, abstractmethod
from typing import Any, Dict


class DataLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> None:
        """Загружает данные из источника"""
        pass

    @abstractmethod
    def __iter__(self):
        """Возвращает итератор для перебора данных"""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Предварительная обработка данных"""
        pass

