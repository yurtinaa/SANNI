from dataclasses import dataclass, field
from typing import List

from AbstractModel.error.TorchErrorFunction.BaseError import ErrorFactoryMSE, ErrorFactoryMAE, BaseErrorTorch, \
    ErrorFullTorch
from AbstractModel.error.TorchErrorFunction.DTWLoss.DTWError import ErrorFactoryDTWLoss
from AbstractModel.error.TorchErrorFunction.LogCoshLoss import ErrorFactoryLogCosh
from AbstractModel.error.TorchErrorFunction.QuantileLoss import ErrorFactoryQuantile
from MPDE import MPDETorch
import torch

from AbstractModel.FrameParam import FrameworkType
from AbstractModel.error.AbstractError import ErrorType, AbstractError, AbstractErrorFactory


class ErrorFactoryCrossEntropy(AbstractErrorFactory):
    name = 'CrossEntropy'

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            loss = BaseErrorTorch(torch.nn.CrossEntropyLoss())
            loss.name = self.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")






@dataclass
class ErrorFactoryMPDE(AbstractErrorFactory):
    name = 'MPDE'

    windows: int = None
    add_mean: bool = False
    mse: bool = False
    log: bool = False
    f1_score: bool = False
    alpha_beta: List[int] = field(default_factory=lambda: [1, 1])

    def __call__(self, frame_type: FrameworkType) -> AbstractError:
        if frame_type == FrameworkType.Torch:
            loss = ErrorFullTorch(MPDETorch(windows=self.windows,
                                            add_mean=self.add_mean,
                                            log=self.log,
                                            alpha_beta=self.alpha_beta,
                                            f1_score=self.f1_score,
                                            mse=self.mse))
            loss.name = self.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")

    def __repr__(self):
        return super().__repr__() + f"_windows_{self.windows}_mse_{self.mse}_add_mean_{self.add_mean}_log_{self.log}_alhpa_{self.alpha_beta}_f1_{self.f1_score}"


# class MSEErrorFactory:
#     pass


_error_classes = {
    ErrorType.MSE: ErrorFactoryMSE,
    ErrorType.MAE: ErrorFactoryMAE,
    ErrorType.CE: ErrorFactoryCrossEntropy,
    ErrorType.MPDE: ErrorFactoryMPDE,
    ErrorType.LogCosh: ErrorFactoryLogCosh,
    ErrorType.QuantileLoss: ErrorFactoryQuantile,
    ErrorType.DTW: ErrorFactoryDTWLoss
}


def get_error(error_type: ErrorType):
    return _error_classes[error_type]
