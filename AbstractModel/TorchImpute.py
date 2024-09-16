import copy
from dataclasses import dataclass, field

import torch
from torch import nn

from Trainer.AbstractTrainer import AbstractModel


@dataclass
class TorchModel(AbstractModel):
    __model: nn.Module = field(default_factory=lambda: None)  # или оставьте None и установите позже

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        if not isinstance(value, nn.Module):
            raise ValueError("New model in TorchModel"
                             " must be an instance of torch.nn.Module")
        self.__model = value

    def copy(self):
        return copy.deepcopy(self.model)

    def __call__(self, X):
        #Fixme потом вернуть
        # if not isinstance(X, torch.Tensor):
        #     raise ValueError("New model in TorchModel"
        #                      " must be an instance of torch.nn.Module")
        if X.device != self.model.device:
            X = X.to(self.model.device)
        return self.model(X)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
