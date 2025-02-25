import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

from .AbstractLogger import AbstractLogger, LogLevel

@dataclass
class MemoryLogger(AbstractLogger):
    log_dict: Dict = field(default_factory=dict)


    def configure(self, level):
        """Настройка уровня логирования."""
        # self.log_file = level
        return self

    def log(self, log_data: dict, level: Optional[LogLevel] = None):
        """Логгирование данных во время обучения.

        Аргументы:
        log_data: dict -- Словарь с логируемыми параметрами (например, epoch, loss, accuracy и т.д.)
        """
        for key, value in log_data.items():
            if key not in self.log_dict:
                self.log_dict[key] = []
            self.log_dict[key].append(value)

    def print(self, data: Any):
        pass