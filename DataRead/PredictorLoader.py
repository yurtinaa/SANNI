# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import json
from lib.SANNI.Preprocess.const import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class PredictorLoader(Dataset):
    def __init__(self, X, y,
                 percent=0.25,
                 type_nan='last',
                 batch_size=32):
        self.X = X
        self.type_nan = type_nan
        self.y = y
        self.percent = percent

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx].clone()
        #    print(X.shape)
        y = self.y[idx].clone()
        #  print('asd')
        #  print((X!=X).sum())
        # y[2, :] = np.nan
        # batch[:, 1:, :][index[:, 1:, :]] = np.nan
        return X, y


@dataclass
class PredictorDataset:
    arr_train: list
    arr_val: list
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 2366
    size_subsequent: int = 100
    test_size: float = 0.25

    type_nan: str = 'None'
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if len(self.arr_train.shape) <= 2:
            self.arr_train = self.arr_train[:, :, None]
            self.arr_val = self.arr_val[:, :, None]

        X_train = torch.Tensor(self.arr_train)
        #X_train = X_train
        X_val = torch.Tensor(self.arr_val)

        self.dataset = {}

        # self.dataset["test"] = [X_test, y_test]
        self.dataset["val"] = [X_val, X_val]
        self.dataset["train"] = [X_train, X_train]

    def get_loader(self, type_dataset):
        # print("батчи:",self.batch_size)

        return DataLoader(PredictorLoader(
            self.dataset[type_dataset][0],
            self.dataset[type_dataset][1], type_nan=self.type_nan),
            batch_size=self.batch_size)
