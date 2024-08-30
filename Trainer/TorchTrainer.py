import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from Trainer.AbstractTrainer import AbstractTrainer, AbstractModel, EpochType


@dataclass
class TorchModel(AbstractModel):
    model: torch.nn.Module

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def model(self):
        return self.model

    @model.setter
    def model(self, new_model: AbstractModel):
        if not isinstance(new_model, TorchModel):
            raise ValueError("new_model must be an instance of TorchModel")
        self.model = copy.deepcopy(new_model.model)


class TorchTrainer(AbstractTrainer):
    __optimizer: torch.optim.Optimizer = None
    __device: torch.device = None
    __loss: Any = None

    def __post_init__(self):
        self.__optimizer = self.config.optimizer(self.current_model.model.parameters())
        self.__device = self.current_model.model.device
        self.__loss = self.config.error()


    def __one_epoch(self, type_: EpochType):
        for i, (x, y) in self.loader:
            self.__optimizer.zero_grad()
            x = x.to(self.__device)
            y = y.to(self.__device)
            with torch.set_grad_enabled(type_ == EpochType.TRAIN):
                y_pred = self.current_model.model.forward(x)
                loss_value = self.loss(x, y, y_pred)
                loss += loss_value.detach().cpu().item()
                score += self.score_func(x.cpu().detach(),
                                         y.cpu().detach(),
                                         y_pred.cpu().detach(),
                                         )
                # print(x[:, -1, ], y[:, -1], y_pred[:, -1])
                if type_ == 'train':
                    # loss_value = self.loss(y_pred, y_true)
                    loss_value.backward()
                    self.optimizer.step()


def __update_model(self):
    if self.train_params.epoch < self._best_params.error:
        self._best_params.epoch = self.train_params.epoch
        self._best_params.error = self.train_params.error
        self.best_model = self.current_model
    if self.logger is not None:
        self.logger.print('new best model')


def train(self):
    history = {"train": [],
               "val": []}
    for self.train_params.epoch in range(self.config.epochs):
        self.current_model.train()
        self.__one_epoch(type_=EpochType.TRAIN)
        self.__one_epoch(type_=EpochType.EVAL)
        self.current_model.eval()
        self.__log_data()
        self.__update_model()
