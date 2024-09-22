import torch
import torch.nn as nn

from ..TorchErrorFunction.BaseError import ErrorFactoryBase


class QuantileLoss(nn.Module):
    def __init__(self, quantile: float):
        super(QuantileLoss, self).__init__()
        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile must be between 0 and 1.")
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        # Вычисляем разницу между предсказанными и истинными значениями
        diff = y_pred - y_true
        # Вычисляем квантильную потерю
        loss = torch.mean(
            torch.max(self.quantile * diff, (self.quantile - 1) * diff)
        )
        return loss


class ErrorFactoryQuantile(ErrorFactoryBase):
    def __init__(self, quantile=0.5):
        super().__init__(QuantileLoss(quantile))
        self.name = 'QuantileLoss'
