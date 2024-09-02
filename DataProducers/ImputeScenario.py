from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np

from .Convertors import Convertor


@dataclass
class BlackoutScenario(Convertor):
    size_block = 100

    def convert(self, X: np.ndarray) -> np.ndarray:
        data = X.copy()
        data[-self.size_block:] = np.nan
        return data
