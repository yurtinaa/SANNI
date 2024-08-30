from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from AbstractModel.FrameParam import FrameworkType
from AbstractModel.Parametrs import TimeSeriesConfig, NeuralNetworkConfig


class AbstractImpute(ABC):
    time_series: TimeSeriesConfig
    neural_network_config: NeuralNetworkConfig


    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass

