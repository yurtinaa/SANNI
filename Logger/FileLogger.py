import logging
from dataclasses import field, dataclass
from pathlib import Path
from typing import Optional, Any

from .AbstractLogger import AbstractLogger, LogLevel


@dataclass
class FileLogger(AbstractLogger):
    _instance = None  # Класс-атрибут для хранения единственного экземпляра

    level: LogLevel = LogLevel.INFO  # Уровень логирования по умолчанию
    __logger: logging.Logger = field(default_factory=lambda: logging.getLogger('FileLogger'))
    log_file: Path = Path('application.log')  # Путь к файлу лога по умолчанию

    def configure(self, level: LogLevel = LogLevel.INFO,
                  log_file: Path = Path('application.log')):
        """Настройка уровня логирования и файла."""
        self.log_file = log_file
        self.__configure_logger(level)
        return self

    def __configure_logger(self, level: LogLevel):
        """Настраивает логгер и его обработчики."""
        # Очистка существующих обработчиков
        for handler in self.__logger.handlers[:]:
            self.__logger.removeHandler(handler)

        self.__logger.setLevel(level.to_logging_level())

        # Создание обработчика для записи в файл
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(level.to_logging_level())

        # Форматирование сообщений
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s\n'
                                      '----------------------------\n'
                                      '%(message)s\n'
                                      '---------------------------\n')
        handler.setFormatter(formatter)

        # Добавление обработчика к логгеру
        self.__logger.addHandler(handler)

    def log(self, log_data: dict, level: Optional[LogLevel] = None):
        if level is None:
            level = self.level

        log_level = level.to_logging_level()
        out_str = '\n'.join(f'[{key}]: {value}' for key, value in log_data.items())
        self.__logger.log(log_level, out_str)

    def print(self, data: Any):
        self.__logger.log(logging.INFO, str(data))