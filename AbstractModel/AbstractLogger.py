from abc import ABC, abstractmethod
from enum import Enum


class LogKeys(Enum):
    EPOCH = "epoch"
    BATCH = "batch"
    LOSS_TRAIN = "loss_train"
    LOSS_VALID = "loss_valid"
    SCORE_TRAIN = "score_train"
    SCORE_VALID = "score_valid"


class AbstractLogger(ABC):
    @abstractmethod
    def __call__(self, log_data: dict):
        """Логгирование данных во время обучения.

        Аргументы:
        log_data: dict -- Словарь с логируемыми параметрами (например, epoch, loss, accuracy и т.д.)
        """
        pass

    def print(self, data: str):
        pass
