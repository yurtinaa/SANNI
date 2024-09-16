from dataclasses import dataclass

import torch

from AbstractModel.FrameParam import FrameworkType
from AbstractModel.error.AbstractError import AbstractErrorFactory, AbstractError


@dataclass
class BaseErrorTorch(AbstractError):
    loss: torch.nn.Module
    index: bool = True
    __name: str = 'loss'

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name):
        self.__name = new_name

    def __call__(self, X, Y, Y_pred):
        return self.loss(Y_pred, Y)


@dataclass
class TorchImputeError(BaseErrorTorch):

    def __call__(self, X, Y, Y_pred):
        index_origin = Y != Y
        if self.index:
            index = X != X
            index[index_origin] = False
        else:
            index = index_origin
            index = ~index

        return self.loss(Y[index], Y_pred[index])

class ErrorFactoryBase(AbstractErrorFactory):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            loss = TorchImputeError(self.loss_fn)
            loss.name = self.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")

class ErrorFactoryMSE(ErrorFactoryBase):
    def __init__(self):
        super().__init__(torch.nn.MSELoss())
        self.name = 'MSE'

class ErrorFactoryMAE(ErrorFactoryBase):
    def __init__(self):
        super().__init__(torch.nn.L1Loss())
        self.name = 'MAE'