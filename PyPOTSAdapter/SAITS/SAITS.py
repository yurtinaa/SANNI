from dataclasses import dataclass, field

import numpy as np
import torch

from ...AbstractModel.AbstractImpute import AbstractImpute
# from AbstractModel.FrameParam import FrameworkType
from ...AbstractModel.TorchImpute import TorchModel
# from AbstractModel.error.AbstractError import AbstractErrorFactory, AbstractError
from ...AbstractModel.error.TorchErrorFunction.BaseError import ErrorFactoryWrapper
from ...AbstractModel.score import Score
from ...Trainer.Loader.TorchLoader import ImputeRandomDataset, TorchTensorLoader
from ...Trainer.TorchTrainer import TorchTrainer
from .core import SAITSTorchError, _SAITS


# @dataclass
# class ErrorFactorySAITS(AbstractErrorFactory):
#     name = 'SAITS error'
#     inside_error: AbstractErrorFactory
#
#     def __call__(self, frame_type: FrameworkType) -> AbstractError:
#         if frame_type == FrameworkType.Torch:
#             inside_error = self.inside_error(frame_type)
#             inside_error.index = False
#             loss = SAITSTorchError(inside_error)
#             loss.name = self.inside_error.name
#             return loss
#         else:
#             raise ValueError(f"Unsupported framework type: {frame_type}")

@dataclass
class SAITSParameters:
    n_layers: int = 2
    d_model: int = 256
    n_heads: int = 4
    d_k: int = 64
    d_v: int = 64
    d_ffn: int = 128
    dropout: float = 0
    attn_dropout: float = 0
    diagonal_attention_mask: bool = True
    ORT_weight: int = 1
    MIT_weight: int = 1
    mcar_percent: float = 0.2


@dataclass
class ScoreSAITS(Score):
    score: Score = None

    def __call__(self, X, Y, Y_Pred):
        if self.score is None:
            raise ValueError("score attribute must be set before calling the object.")
        return self.score(X, Y, Y_Pred['imputed_data'])


@dataclass
class SAITSImpute(AbstractImpute):
    saits_parameters: SAITSParameters = field(default_factory=SAITSParameters)

    def __post_init__(self):
        if self.name is None:
            self.name = 'SAITS'

        saits_model = _SAITS(
            n_layers=self.saits_parameters.n_layers,
            n_steps=self.time_series.window_size,
            n_features=self.time_series.dim,
            d_model=self.saits_parameters.d_model,
            n_heads=self.saits_parameters.n_heads,
            d_k=self.saits_parameters.d_k,
            d_v=self.saits_parameters.d_v,
            d_ffn=self.saits_parameters.d_ffn,
            dropout=self.saits_parameters.dropout,
            attn_dropout=self.saits_parameters.attn_dropout,
            diagonal_attention_mask=self.saits_parameters.diagonal_attention_mask,
            ORT_weight=self.saits_parameters.ORT_weight,
            MIT_weight=self.saits_parameters.MIT_weight
        ).to(self.device)
        self._model = TorchModel(saits_model)
        base_error = self.neural_network_config.error_factory
        error_factory = ErrorFactoryWrapper(
            name=base_error.name,
            inside_error=base_error,
            wrapper_error=SAITSTorchError,
        )
        score_factory = ScoreSAITS(name='saits_score',
                                   score=self.neural_network_config.score_factory)
        self.neural_network_config.error_factory = error_factory
        self.neural_network_config.score_factory = score_factory

    @property
    def _dataset_factory(self):
        return lambda X, Y: ImputeRandomDataset(X, Y, self.saits_parameters.mcar_percent)

    def __train(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        # dataset_factory = self.dataset_factory
        loader = TorchTensorLoader(X=X,
                                   y=y,
                                   dataset_factory=self._dataset_factory,
                                   batch_size=self.neural_network_config.batch_size,
                                   shuffle=True)
        result = TorchTrainer(current_model=self._model,
                              config=self.neural_network_config,
                              loader=loader,
                              device=self.device,
                              logger=self.logger).train()
        history, self._model, _ = result
        return history

    def train(self, X: np.ndarray, Y: np.ndarray):
        history_brits = self.__train(X, Y)
        return {"brits": history_brits}

    def __call__(self, X: np.ndarray):

        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            batch_size = self.neural_network_config.batch_size
            if X.shape[0] < batch_size:
                result = self._model(X)
                result = result['imputed_data'].detach().cpu().numpy()
                return result
            else:
                result_arr = []
                for idx in range(0, X.shape[0],
                                 batch_size):
                    batch_result = self._model(X)
                    batch_result = batch_result['imputed_data'].detach().cpu().numpy()
                    result_arr.append(batch_result)
                return np.concatenate(result_arr, axis=0)
