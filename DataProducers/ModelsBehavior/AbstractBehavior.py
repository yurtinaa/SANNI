from abc import ABC, abstractmethod
from ...AbstractModel import AbstractImpute
import numpy as np
from attr import dataclass


@dataclass
class AbstractImputeBehavior(ABC):
    model: AbstractImpute

    @abstractmethod
    def simulate(self, data: np.ndarray):
        pass


@dataclass
class SerialImputeBehavior(AbstractImputeBehavior):
    window_size: int

    def simulate(self, data: np.ndarray):
        time_series = data.copy()
        nan_index = np.isnan(time_series).any(axis=1)
        nan_index = np.where(nan_index)[0]
        for index in nan_index:
            input_data = time_series[index-self.window_size+1:index+1]
            input_data = input_data[None,:]
            # print(input_data.shape)
            result = self.model(input_data)[0]
            time_series[index-self.window_size+1:index+1] = result
        return time_series


