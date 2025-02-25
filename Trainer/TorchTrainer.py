import copy
from dataclasses import dataclass
from time import time

# from typing import Any

import torch
# from overrides import override
# from torch import nn

from ..AbstractModel.error.AbstractError import AbstractError
from ..AbstractModel.score import Score, ScoreResult
from ..Logger.AbstractLogger import LogKeys
from .AbstractTrainer import AbstractTrainer, AbstractModel
from ..EnumConfig import EpochType


@dataclass
class TorchModel(AbstractModel):
    _model: torch.nn.Module

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        print(type(new_model))
        if not isinstance(new_model, TorchModel):
            raise ValueError("new_model must be an instance of TorchModel")
        self._model = copy.deepcopy(new_model._model)

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, X):
        return self._model(X)


class TorchTrainer(AbstractTrainer):
    __optimizer: torch.optim.Optimizer = None
    __loss: AbstractError = None
    __score: Score = None

    def __post_init__(self):
        self.best_model = self.current_model
        self.__optimizer = self.config.optimizer(self.current_model.model.parameters())
        # self.__device = self.current_model.model.device
        self.__loss = self.config.error
        self.__score = self.config.score

    def _one_epoch(self, type_: EpochType):
        loss = 0
        score = self.__score.get_empty_result()
        # print(type(score))
        for x, y in self.loader(type_):
            self.__optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            # print(x.shape)
            with torch.set_grad_enabled(type_ == EpochType.TRAIN):
                y_pred = self.current_model(x)
                loss_value = self.__loss(x, y, y_pred)
                loss += loss_value.detach().cpu().item()

                new_score = self.__score(x,
                                         y,
                                         y_pred,
                                         )
                score += new_score
                # print(x[:, -1, ], y[:, -1], y_pred[:, -1])

                if type_ == EpochType.TRAIN:
                    # loss_value = self.loss(y_pred, y_true)
                    loss_value.backward()
                    self.__optimizer.step()
        return loss / self.loader.length(type_), score / self.loader.length(type_)

    def _update_model(self):
        if self.train_params.error < self._best_params.error:
            self._best_params.epoch = self.train_params.epoch
            self._best_params.error = self.train_params.error
            self.best_model = self.current_model
            self.logger.print('new best model')

    def _log_data(self, log_dict):
        self.logger.log(log_dict)

    def train(self):
        history = {"train": [],
                   "val": []}
        for self.train_params.epoch in range(self.config.epochs):
            self.current_model.train()
            start_time = time()
            train_loss, train_score = self._one_epoch(type_=EpochType.TRAIN)
            train_time = time() - start_time

            self.current_model.eval()
            start_time = time()
            val_loss, val_score = self._one_epoch(type_=EpochType.EVAL)
            eval_time = time() - start_time

            self.train_params.error = val_loss
            self._log_data({
                LogKeys.EPOCH: self.train_params.epoch,
                EpochType.TRAIN: {'loss': train_loss,
                                  'score': train_score.to_dict(),
                                  'time': train_time},
                EpochType.EVAL: {'loss': val_loss,
                                 'score': val_score.to_dict(),
                                 "time": eval_time},
            })
            self._update_model()
            history["train"].append({'loss': train_loss,
                                     'score': train_score.to_dict(),
                                     'time': train_time})
            history["val"].append({'loss': val_loss,
                                   'score': val_score.to_dict(),
                                   "time": eval_time})
        result_history = {}
        for key in history:
            result_history[key] = {}
            for epoch in history[key]:
                for epoch_data_key in epoch:
                    if epoch_data_key not in result_history[key]:
                        result_history[key][epoch_data_key] = []
                    result_history[key][epoch_data_key].append(epoch[epoch_data_key])
        return result_history, self.best_model.copy(), self.best_model.copy()
