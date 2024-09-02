import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any


class LogKeys(Enum):
    EPOCH = "epoch"
    BATCH = "batch"
    LOSS_TRAIN = "loss_train"
    LOSS_VALID = "loss_valid"
    SCORE_TRAIN = "score_train"
    SCORE_VALID = "score_valid"

    def __repr__(self):
        # Возвращаем только значение перечисления
        return self.value


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @staticmethod
    def from_string(level: str) -> 'LogLevel':
        level = level.upper()
        try:
            return LogLevel[level]
        except KeyError:
            return LogLevel.INFO  # Уровень по умолчанию

    def to_logging_level(self) -> int:
        level_mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_mapping.get(self.value, logging.INFO)


class AbstractLogger(ABC):

    @abstractmethod
    def configure(self, level):
        """Настройка уровня логирования."""
        pass

    @abstractmethod
    def log(self, log_data: dict, level: Optional[LogLevel] = None):
        """Логгирование данных во время обучения.

        Аргументы:
        log_data: dict -- Словарь с логируемыми параметрами (например, epoch, loss, accuracy и т.д.)
        """
        pass

    @abstractmethod
    def print(self, data: Any):
        pass
