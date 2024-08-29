from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import random

from .Models.Predictors.ANNI.Predictor import ANPredictor
from .model import SANNI

mse = torch.nn.MSELoss()

models_predictor = {
    "anni_last": ANPredictor,
    # 'short_serial': ShortPredictor,
}


# fixme: добавить обработку исключени:
# 1. Остановки по кнопке
# 2. Остановки из за большой выборки.

@dataclass
class ANNI(SANNI):

    def fit(self, train, valid):
        print(f'hidden: {self.hidden}')
        print('старт обучения')
        #   self.preprocess_data_to_classifier(train)
        print('предобработка данных')
        if len(train.shape) > 2:
            self.dim = train.shape[2]
        else:
            self.dim = 1
        #  self.train_classifier()
        # print('обучил классификатора')
        self.preprocess_data_to_predictor(train, valid)
        #   print('with_h: ', self.predictor.forward_predictor.with_h)
        print('обучаю предсказателя')
        self.train_predictor()
