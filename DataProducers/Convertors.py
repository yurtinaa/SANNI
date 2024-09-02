from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np


@dataclass
class Convertor(ABC):

    @abstractmethod
    def convert(self, X: np.ndarray) -> np.ndarray:
        pass


@dataclass
class SliceTimeSeriesConvertor(Convertor):
    window: int
    step: int = 1

    def convert(self, X: np.ndarray) -> np.ndarray:
        data = []
        step = 1
        # FIXME: Требуется оптимизация
        for idx in range(0, len(X) - self.window + 1, step):
            data.append(X[idx:idx + self.window])

        data = np.array(data).reshape(len(data), self.window, -1)
        return data


@dataclass
class FindMissingIndexConvertor(Convertor):
    dim: int = 0
    return_nan_index: bool = True

    def convert(self, X: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(X)
        dims = []
        for dim in range(len(X.shape)):
            if dim != self.dim:
                dims.append(dim)
        nan_mask = nan_mask.any(axis=tuple(dims))  # Преобразуем список dims в кортеж
        if self.return_nan_index:
            return np.where(nan_mask)[0]
        else:
            return np.where(~nan_mask)[0]


@dataclass
class DropMissingSubConvertor(Convertor):
    def convert(self, X: np.ndarray) -> np.ndarray:
        not_missing_index = FindMissingIndexConvertor(dim=0,
                                                      return_nan_index=False).convert(X)
        return X[not_missing_index]
