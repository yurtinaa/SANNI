from torch import nn

from ...TorchErrorFunction.BaseError import ErrorFactoryBase, ErrorFullTorch
from ...TorchErrorFunction.DTWLoss.DTWLoss import DTWLoss
from ...TorchErrorFunction.DTWLoss.DTWLoss_CUDA import DTWLoss as DTWLossCuda


class DTWLossWrapper(nn.Module):
    def __init__(self, gamma=0.1):
        super(DTWLossWrapper, self).__init__()
        self.loss_fn = DTWLoss(gamma)  # Инициализация DTW Loss
        self.loss_fn_cuda = DTWLossCuda(gamma)

    def forward(self, y_pred, y_true):
        if y_pred.device == 'cpu':
            return self.loss_fn(y_pred, y_true)
        else:
            return self.loss_fn_cuda(y_pred, y_true)


class ErrorFactoryDTWLoss(ErrorFactoryBase):
    def __init__(self, gamma=0.1):
        super().__init__(DTWLossWrapper(gamma), ErrorFullTorch)
        self.name = 'DTW'
