from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from AbstractModel.FrameParam import FrameworkType


class OptimizerType(str, Enum):
    Adam = 'Adam'


class AbstractOptimizer(ABC):

    @abstractmethod
    def __call__(self, frame_type: FrameworkType):
        pass


@dataclass
class Adam(AbstractOptimizer):
    lr: float
    amsgrad: bool

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            return lambda param: torch.optim.Adam(param,
                                                  lr=self.lr,
                                                  amsgrad=self.amsgrad)
