from dataclasses import dataclass
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


class ErrorFactoryMSE(AbstractErrorFactory):
    name = 'MSE'

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            loss = TorchImputeError(torch.nn.MSELoss())
            loss.name = self.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")


class ErrorFactoryMAE(AbstractErrorFactory):
    name = 'MAE'

    def __call__(self, frame_type: FrameworkType):
        if frame_type == FrameworkType.Torch:
            loss = TorchImputeError(torch.nn.L1Loss())
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

    def __call__(self, frame_type: FrameworkType) -> AbstractError:
        if frame_type == FrameworkType.Torch:
            loss = ErrorMPDETorch(MPDETorch(windows=self.windows,
                                            add_mean=self.add_mean,
                                            mse=self.mse))
            loss.name = self.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")

    def __repr__(self):
        return super().__repr__() + f"_windows_{self.windows}_mse_{self.mse}_add_mean_{self.add_mean}"


@dataclass
class BaseErrorTorch(AbstractError):
    loss: torch.nn.Module
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
        index = X != X
        index_origin = Y != Y

        index[index_origin] = False
        return self.loss(Y[index], Y_pred[index])


class ErrorMPDETorch(BaseErrorTorch):

    def __call__(self, X, Y, Y_pred):
        return self.loss(Y_pred, Y).mean()


class MSEErrorFactory:
    pass


_error_classes = {
    ErrorType.MSE: ErrorFactoryMSE,
    ErrorType.MAE: ErrorFactoryMAE,
    ErrorType.CE: ErrorFactoryCrossEntropy,
    ErrorType.MPDE: ErrorFactoryMPDE,
}


def get_error(error_type: ErrorType):
    return _error_classes[error_type]
