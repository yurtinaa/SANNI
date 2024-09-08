from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from AbstractModel.AbstractImpute import AbstractImpute, AbstractClassifier
from AbstractModel.Parametrs import NeuralNetworkConfig, TorchNNConfig
from AbstractModel.error.AbstractError import ErrorType
from AbstractModel.error.TorchError import get_error
from AbstractModel.optimizer.abstract_optimizer import Adam
from AbstractModel.score import ScoreType, get_score
from Models import Classifier, Predictor
from Models.Predictors.NotSerial import NotSerialPredictor
from Trainer.AbstractTrainer import AbstractModel
from Trainer.Loader.TorchLoader import TorchTensorLoader, ImputeLastDataset
from Trainer.TorchTrainer import TorchTrainer, TorchModel

CLASSIFIER_SANNI_CONFIG = TorchNNConfig(
    batch_size=32,
    epochs=10,
    error_factory=get_error(ErrorType.CE)(),
    score_factory=get_score(ScoreType.F1_SCORE),
    optimizer_type=Adam(lr=0.001,
                        amsgrad=True),
    early_stopping_patience=50
)


@dataclass
class SANNI(AbstractImpute):
    _classifier: Classifier = None
    snippet_list: List[List[Tuple[int, np.ndarray]]] = None
    _snippet_array: np.ndarray = None
    snippet_dict: List[Dict[int, np.ndarray]] = None

    _snippet_count: int = None
    classifier_config: NeuralNetworkConfig = field(default_factory=lambda: CLASSIFIER_SANNI_CONFIG)

    def _predictor_construct(self):
        predictor = Predictor(classifier=self._classifier,
                              size_subsequent=self.time_series.window_size,
                              dim=self.time_series.dim,
                              snippet_list=self._snippet_array,
                              device=torch.device(self.device),
                              count_snippet=self._snippet_count).to(self.device)
        return NotSerialPredictor(predictor)

    def __post_init__(self):
        if self.name is None:
            self.name = 'SANNI'
        self.neural_network_config: TorchNNConfig
        if self._snippet_count is None:
            self._snippet_count = len(self.snippet_dict[0])
        snippet_list = []
        for dim in self.snippet_dict:
            dim_data = []
            for snippet in dim.values():
                dim_data.append(snippet)
            snippet_list.append(dim_data)
        self._snippet_array = np.array(snippet_list)

        self._classifier = Classifier(size_subsequent=self.time_series.window_size,
                                      count_snippet=self._snippet_count,
                                      dim=self.time_series.dim).to(self.device)

        self._model = TorchModel(self._predictor_construct())

        # self.__trainer = TorchTrainer(current_model=self.__model)

    def __classifier_train(self):
        X = []
        y = []
        for subseq in self.snippet_list:
            sub_label = []
            sub_data = []
            for key, data in subseq:
                sub_label.append(key)
                sub_data.append(data)
            X.append(sub_data)
            y.append(sub_label)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y).long()
        # print(X.shape)
        loader = TorchTensorLoader(X=X,
                                   y=y,
                                   batch_size=self.classifier_config.batch_size,
                                   shuffle=True)

        classifier = TorchModel(self._classifier)

        result = TorchTrainer(current_model=classifier,
                              config=self.classifier_config,
                              loader=loader,
                              device=self.device,
                              logger=self.logger).train()
        history, self._classifier, _ = result
        self._model.model.classifier = classifier.model
        return history

    def _predictor_train(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        loader = TorchTensorLoader(X=X,
                                   y=y,
                                   dataset_factory=ImputeLastDataset,
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
        history_classifier = self.__classifier_train()
        history_predictor = self._predictor_train(X, Y)
        return {'classifier': history_classifier,
                'predictor': history_predictor}

    def __call__(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            batch_size = self.classifier_config.batch_size
            if X.shape[0] < batch_size:
                return self._model(X).detach().cpu().numpy()
            else:
                result_arr = []
                for idx in range(0, X.shape[0],
                                 batch_size):
                    batch_result = self.__model(X[idx:idx + batch_size]).detach().cpu().numpy()
                    result_arr.append(batch_result)
                return np.concatenate(result_arr, axis=0)
