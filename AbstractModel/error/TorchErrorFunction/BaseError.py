from dataclasses import dataclass
from typing import Type

import torch

from ...FrameParam import FrameworkType
from ..AbstractError import AbstractErrorFactory, AbstractError


@dataclass
class BaseErrorTorch(AbstractError):
    loss: torch.nn.Module
    index: bool = True
    __name: str = 'loss'

    def to(self, device):
        self.loss = self.loss.to(device)
        return self
    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name):
        self.__name = new_name

    def __call__(self, X, Y, Y_pred):
        return self.loss(Y_pred, Y)


@dataclass
class ErrorFactoryWrapper(AbstractErrorFactory):
    name: str
    inside_error: AbstractErrorFactory
    wrapper_error: Type[BaseErrorTorch]

    def __call__(self, frame_type: FrameworkType) -> AbstractError:
        error = self.inside_error(frame_type)
        error.index = False
        wrapped_error = self.wrapper_error(error)
        wrapped_error.name = self.inside_error.name  # Передаем имя

        return wrapped_error


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
    def __init__(self, loss_fn,
                 impute_type=TorchImputeError):
        self.loss_fn = loss_fn
        self.impute_type = impute_type

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            loss = self.impute_type(self.loss_fn)
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


class ErrorFullTorch(BaseErrorTorch):

    def __call__(self, X, Y, Y_pred):
        return self.loss(Y_pred, Y).mean()
