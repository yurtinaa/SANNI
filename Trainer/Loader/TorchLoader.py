from dataclasses import dataclass, field
from typing import Tuple, Dict, Type

import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataset import Dataset
import torch
from sklearn.model_selection import train_test_split

from EnumConfig import EpochType
from Trainer.Loader.AbstractLoader import DataLoader


@dataclass
class BaseDataset(Dataset):
    X: torch.Tensor
    y: torch.Tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class ImputeLastDataset(BaseDataset):
    X: torch.Tensor
    y: torch.Tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X[-1, :] = np.nan
        return X, self.y[idx]


@dataclass
class TorchTensorLoader(DataLoader):
    __set_dict: Dict[EpochType, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    dataset_factory: Type[BaseDataset] = BaseDataset

    def __post_init__(self):
        # self.X = torch.tensor(self.X, dtype=torch.float32)
        # self.y = torch.tensor(self.y, dtype=torch.float32)

        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            shuffle=self.shuffle,
                                                            test_size=self.percent,
                                                            random_state=self.seed)
        print(X_train.dtype, y_test.dtype)
        self.__set_dict[EpochType.TRAIN] = X_train, y_train
        self.__set_dict[EpochType.EVAL] = X_test, y_test

    def length(self, epoch_type: EpochType) -> float:
        return self.__set_dict[epoch_type][0].shape[0] // self.batch_size

    def __iter__(self, epoch_type: EpochType):
        dataset = self.dataset_factory(*self.__set_dict[epoch_type])
        return TorchDataLoader(dataset,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle)

    def __call__(self, epoch_type: EpochType):
        dataset = self.dataset_factory(*self.__set_dict[epoch_type])
        return TorchDataLoader(dataset,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle)
