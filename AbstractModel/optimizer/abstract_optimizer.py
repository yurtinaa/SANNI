from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from ..FrameParam import FrameworkType


class OptimizerType(str, Enum):
    Adam = 'Adam'


class AbstractOptimizer(ABC):

    @abstractmethod
    def __call__(self, frame_type: FrameworkType):
        pass


@dataclass
class Adam(AbstractOptimizer):
    lr: float = 0.001
    amsgrad: bool = True

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            return lambda param: torch.optim.Adam(param,
                                                  lr=self.lr,
                                                  amsgrad=self.amsgrad)
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")
