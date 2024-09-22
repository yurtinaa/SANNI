from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from ..FrameParam import FrameworkType


class SchedulerType(str, Enum):
    ReduceLROnPlateau = 'ReduceLROnPlateau'


class AbstractScheduler(ABC):

    @abstractmethod
    def __call__(self, frame_type: FrameworkType):
        pass


@dataclass
class ReduceLROnPlateau(AbstractScheduler):
    patience: int

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            return lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                patience=self.patience)
