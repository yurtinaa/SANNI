import torch
import torch.nn as nn
from torchmetrics.regression import LogCoshError
from ..TorchErrorFunction.BaseError import ErrorFactoryBase


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        error_line = str((torch.any(torch.isnan(y_pred)), torch.any(torch.isnan(y_true))))
        print(error_line)
        if torch.any(torch.isnan(y_pred)) or torch.any(torch.isnan(y_true)):
            raise ValueError(f"Input contains NaN values{error_line}")
        y_pred = torch.clamp(y_pred, min=-1e10, max=1e10)
        y_true = torch.clamp(y_true, min=-1e10, max=1e10)
        diff = y_pred - y_true

        # Ограничиваем значения разницы
        diff_clamped = torch.clamp(diff, min=-1e10, max=1e10)

        # Печатаем для отладки
        # print(f"y_pred: {y_pred}, y_true: {y_true}, diff: {diff_clamped}")
        if torch.any(diff == 0):
            print(torch.mean(y_pred))
            return torch.mean(diff)
        # Вычисляем Log-Cosh
        loss = torch.log(torch.cosh(diff + 1e-6) + 1e-6)
        met_loss = torch.log(((torch.exp(diff + 1e-6) + torch.exp(-diff + 1e-6)) / 2) + 1e-6)
        # loss = torch.log(torch.cosh(diff_clamped + 1e-12) + 1e-12)
        print(torch.mean(loss), torch.mean(met_loss), torch.mean(torch.cosh(diff + 1e-6)), torch.mean(diff))
        return torch.mean(met_loss)


class ErrorFactoryLogCosh(ErrorFactoryBase):
    def __init__(self):
        super().__init__(LogCoshError())
        self.name = 'LogCosh'
