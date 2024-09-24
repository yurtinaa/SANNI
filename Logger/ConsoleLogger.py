import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .AbstractLogger import AbstractLogger, LogLevel


@dataclass
class ConsoleLogger(AbstractLogger, object):
    _instance = None  # Класс-атрибут для хранения единственного экземпляра

    level: LogLevel = LogLevel.INFO  # Уровень логирования по умолчанию
    __logger: logging.Logger = field(default_factory=lambda: logging.getLogger('ConsoleLogger'))

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def configure(self, level: LogLevel = LogLevel.INFO):
        """Настройка уровня логирования."""
        self.__configure_logger(level)
        return self

    def __configure_logger(self, level: LogLevel):
        """Настраивает логгер и его обработчики."""
        # Очистка существующих обработчиков
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        self.__logger.handlers.clear()

        self.__logger.setLevel(self.level.to_logging_level())

        # Создание обработчика для вывода в консоль
        handler = logging.StreamHandler()
        handler.setLevel(level.to_logging_level())

        # Форматирование сообщений
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s\n'
                                      '----------------------------\n'
                                      '%(message)s\n'
                                      '---------------------------\n')
        handler.setFormatter(formatter)

        # Добавление обработчика к логгеру
        self.__logger.addHandler(handler)

    def log(self, log_data: dict,
            level: Optional[LogLevel] = None):
        if level is None:
            level = self.level

        log_level = level.to_logging_level()
        out_str = '\n'.join(f'[{key}]: {value}' for key, value in log_data.items())
        self.__logger.log(log_level, out_str)

    def print(self, data: Any):
        self.__logger.log(logging.INFO, str(data))
