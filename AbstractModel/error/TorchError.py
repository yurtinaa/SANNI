from dataclasses import dataclass

import torch

from AbstractModel.error.AbstractError import ErrorType, AbstractError


class MPDELoss:
    pass


@dataclass
class TorchRegress(AbstractError):
    loss: torch.nn.Module

    def __call__(self, X, Y, Y_pred):
        index = X != X
        index_origin = Y != Y

        index[index_origin] = False
        return self.loss(Y[index], Y_pred[index])


_error_classes = {
    ErrorType.MSE: lambda: TorchRegress(torch.nn.MSELoss()),
    ErrorType.MAE: lambda: TorchRegress(torch.nn.L1Loss()),
    ErrorType.CE: torch.nn.CrossEntropyLoss(),
    ErrorType.MPDE: lambda windows: MPDELoss(windows),
}
