import torch
import torch.nn as nn
from torchmetrics.regression import LogCoshError
from ..TorchErrorFunction.BaseError import ErrorFactoryBase


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.log(torch.cosh(diff))
        return torch.mean(loss)

class ErrorFactoryLogCosh(ErrorFactoryBase):
    def __init__(self):
        super().__init__(LogCoshError())
        self.name = 'LogCosh'