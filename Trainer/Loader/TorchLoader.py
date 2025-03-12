from dataclasses import dataclass, field
from typing import Tuple, Dict, Type

import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataset import Dataset
import torch
from sklearn.model_selection import train_test_split

from ...EnumConfig import EpochType
from .AbstractLoader import DataLoader


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
        X = self.X[idx].clone()
        X[-1, :] = np.nan
        return X, self.y[idx]


@dataclass
class ImputeRandomDataset(BaseDataset):
    X: torch.Tensor
    y: torch.Tensor
    percent: float = 0.25
    predict: bool = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].clone()
        num_elements = X.numel()  # Общее количество элементов в X
        num_nan = int(num_elements * self.percent)  # Количество элементов, которые нужно заменить на NaN

        # Выберите случайные индексы для замены на NaN
        nan_indices = np.random.choice(num_elements, num_nan, replace=False)

        # Преобразуйте 1D индексы в индексы в тензоре
        row_indices, col_indices = np.unravel_index(nan_indices, X.shape)
        col_indices = torch.tensor(col_indices.tolist()).long()
        row_indices = torch.tensor(row_indices.tolist()).long()

        # Замените выбранные элементы на NaN
        X[row_indices, col_indices] = torch.nan
        if self.predict:
            X[-1, :] = torch.nan

        return X, self.y[idx]


@dataclass
class TorchTensorLoader(DataLoader):
    _set_dict: Dict[EpochType, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    dataset_factory: Type[BaseDataset] = BaseDataset


    def __post_init__(self):
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        if self.X_val is None or self.y_val is None:
            X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            shuffle=self.shuffle,
                                                            test_size=self.percent,
                                                            random_state=self.seed)
        else:
            X_train, y_train = self.X, self.y
            X_test = torch.tensor(self.X_val, dtype=torch.float32)
            y_test = torch.tensor(self.y_val, dtype=torch.float32)
        # print(X_train.dtype, y_test.dtype)
        self._set_dict[EpochType.TRAIN] = X_train, y_train
        self._set_dict[EpochType.EVAL] = X_test, y_test

    def length(self, epoch_type: EpochType) -> float:
        # print(self._set_dict[epoch_type][0].shape[0], self.batch_size)
        length = self._set_dict[epoch_type][0].shape[0] // self.batch_size
        if length == 0:
            return 1
        else:
            return length

    def __iter__(self, epoch_type: EpochType):
        dataset = self.dataset_factory(*self._set_dict[epoch_type])
        return TorchDataLoader(dataset,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle)

    def __call__(self, epoch_type: EpochType):
        dataset = self.dataset_factory(*self._set_dict[epoch_type])
        return TorchDataLoader(dataset,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle)
