from dataclasses import dataclass

import numpy as np

from .AbstractModel.FrameParam import FrameworkType
from .AbstractModel.error.AbstractError import AbstractErrorFactory, AbstractError
from .AbstractModel.error.TorchErrorFunction.BaseError import BaseErrorTorch
from .AbstractModel.score import Score
from .Models.Predictors.SAETI.model import SAETI as BaseSAETI
import torch

from .SANNI import SANNI
from .Trainer.Loader.TorchLoader import ImputeRandomDataset, ImputeLastDataset


class SAETITorchError(BaseErrorTorch):

    def __call__(self, X, Y, Y_pred):
        X[:, :, :] = np.nan
        return self.loss(X, Y, Y_pred)


@dataclass
class ErrorFactorySAETI(AbstractErrorFactory):
    name = 'SAETI error'
    inside_error: AbstractErrorFactory

    def __call__(self, frame_type: FrameworkType) -> AbstractError:
        if frame_type == FrameworkType.Torch:
            inside_error = self.inside_error(frame_type)
            inside_error.index = False
            loss = SAETITorchError(inside_error)
            loss.name = self.inside_error.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")


@dataclass
class ScoreSAETI(Score):
    score: Score = None

    def __call__(self, X, Y, Y_Pred):
        X[:, :, :] = np.nan
        return self.score(X, Y, Y_Pred)


@dataclass
class SAETI(SANNI):

    def _predictor_construct(self):
        predictor = BaseSAETI(size_seq=self.time_series.window_size,
                              n_features=self.time_series.dim,
                              hidden_size=self.time_series.window_size,
                              latent_dim=self.time_series.window_size // 2,
                              classifier=self._classifier,
                              snippet_list=torch.tensor(self._snippet_array).to(self.device))
        return predictor.to(self.device)

    @property
    def _dataset_factory(self):
        return ImputeLastDataset

    def __post_init__(self):
        if self.name is None:
            self.name = 'SAETI'
        super().__post_init__()
        # print('drop new error')
        error_factory = ErrorFactorySAETI(self.neural_network_config.error_factory)
        score_factory = ScoreSAETI(name='SAETI_score',
                                   score=self.neural_network_config.score_factory)

        self.neural_network_config.error_factory = error_factory
        self.neural_network_config.score_factory = score_factory
