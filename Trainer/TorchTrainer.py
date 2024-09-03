import copy
from dataclasses import dataclass
from typing import Any

import torch
from overrides import override
from torch import nn

from AbstractModel.error.AbstractError import AbstractError
from AbstractModel.score import Score, ScoreResult
from Logger.AbstractLogger import LogKeys
from Trainer.AbstractTrainer import AbstractTrainer, AbstractModel
from EnumConfig import EpochType


@dataclass
class TorchModel(AbstractModel):
    __model: torch.nn.Module

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model):
        print(type(new_model))
        if not isinstance(new_model, TorchModel):
            raise ValueError("new_model must be an instance of TorchModel")
        self.__model = copy.deepcopy(new_model.__model)

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, X):
        return self.__model(X)


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
            print(x.shape)
            with torch.set_grad_enabled(type_ == EpochType.TRAIN):
                y_pred = self.current_model(x)
                loss_value = self.__loss(x, y, y_pred)
                loss += loss_value.detach().cpu().item()

                new_score = self.__score(x.cpu().detach(),
                                         y.cpu().detach(),
                                         y_pred.cpu().detach(),
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
            train_loss, train_score = self._one_epoch(type_=EpochType.TRAIN)
            self.current_model.eval()
            val_loss, val_score = self._one_epoch(type_=EpochType.EVAL)
            self.train_params.error = val_loss
            self._log_data({
                LogKeys.EPOCH: self.train_params.epoch,
                EpochType.TRAIN: {self.__loss.name: train_loss,
                                  self.__score.name: train_score},
                EpochType.EVAL: {self.__loss.name: val_loss,
                                 self.__score.name: val_score},
            })
            self._update_model()
            history["train"].append({self.__loss.name: train_loss,
                                     self.__score.name: train_score.to_dict()}, )
            history["val"].append({self.__loss.name: val_loss,
                                   self.__score.name: val_score.to_dict()})
        return history, self.best_model.copy(), self.best_model.copy()
