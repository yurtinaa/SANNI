from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...AbstractModel.AbstractImpute import AbstractImpute, AbstractClassifier
from ...AbstractModel.FrameParam import FrameworkType
from ...AbstractModel.TorchImpute import TorchModel
from ...AbstractModel.error.AbstractError import AbstractErrorFactory, AbstractError
from ...AbstractModel.score import Score
# from Trainer.AbstractTrainer import AbstractModel
from ...Trainer.TorchTrainer import TorchTrainer
from .BRITSDataLoader import BRITSLoader, BRITSDataset, brits_collate_fn
from .core import _BRITS, BritsTorchError


@dataclass
class ErrorFactoryBRITS(AbstractErrorFactory):
    name = 'BRITS error'
    inside_error: AbstractErrorFactory

    def __call__(self, frame_type: FrameworkType) -> AbstractError:
        if frame_type == FrameworkType.Torch:
            inside_error = self.inside_error(frame_type)
            inside_error.index = False
            loss = BritsTorchError(inside_error)
            loss.name = self.inside_error.name
            return loss
        else:
            raise ValueError(f"Unsupported framework type: {frame_type}")


@dataclass
class ScoreBrits(Score):
    score: Score = None

    def __call__(self, X, Y, Y_Pred):
        if self.score is None:
            raise ValueError("score attribute must be set before calling the object.")
        missing_mask = X['forward']["missing_mask"]
        X = X['forward']["X"]
        X[:, :, :] = np.nan
        return self.score(X, Y, Y_Pred['reconstruction'])


@dataclass
class BRITSImpute(AbstractImpute):
    rnn_hidden_size: int = 100

    def __post_init__(self):
        if self.name is None:
            self.name = 'BRITS'
        self.__model = TorchModel(_BRITS(
            n_steps=self.time_series.window_size,
            n_features=self.time_series.dim,
            rnn_hidden_size=self.rnn_hidden_size).to(self.device))
        error_factory = ErrorFactoryBRITS(self.neural_network_config.error_factory)
        score_factory = ScoreBrits(name='brits_score',
                                   score=self.neural_network_config.score_factory)

        self.neural_network_config.error_factory = error_factory
        self.neural_network_config.score_factory = score_factory

    def __train_brits(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        loader = BRITSLoader(X=X,
                             y=y,
                             batch_size=self.neural_network_config.batch_size,
                             shuffle=True)
        result = TorchTrainer(current_model=self.__model,
                              config=self.neural_network_config,
                              loader=loader,
                              device=self.device,
                              logger=self.logger).train()
        history, self.__model, _ = result
        return history

    def train(self, X: np.ndarray, Y: np.ndarray):
        history_brits = self.__train_brits(X, Y)
        return {"brits": history_brits}

    def __call__(self, X: np.ndarray):
        X_impute = torch.tensor(X)
        test_set = BRITSDataset(
            X_impute, X_impute
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.neural_network_config.batch_size,
            shuffle=False,
            collate_fn=brits_collate_fn
        )
        imputation_collector = []
        with torch.no_grad():
            for idx, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                results = self.__model(data)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        return imputation
