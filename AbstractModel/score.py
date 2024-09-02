from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Callable, List, Union

import torch
from sklearn.metrics import precision_score, mean_squared_error, accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error

import numpy as np
from sympy import true

from AbstractModel.error.AbstractError import AbstractError, ErrorType

MEAN_DICT = lambda data_dict: np.mean(list(data_dict.values()))
KEY_DICT = lambda data_dict: list(data_dict.keys())


@dataclass
class CellFuncHandler:
    cell_func: Callable[[Dict['ErrorType', List[float]]], float]


MEAN_HANDLER = CellFuncHandler(MEAN_DICT)


class KeyFuncHandler(CellFuncHandler):
    key_: str

    def __post_init__(self):
        self.cell_func = lambda data_dict: data_dict.get(self.key_, None)


@dataclass
class ScoreResult:
    handler: CellFuncHandler
    __all_result: Dict['ScoreType', float] = field(default_factory=dict)

    def add_result(self, error_type: 'ScoreType', score: float):
        self.__all_result[error_type] = score

    def get_full_result(self):
        return self.__all_result.copy()

    def __str__(self) -> str:
        """Возвращает строковое представление объекта ScoreResult."""
        results_str = '\n '.join(f'[{error_type}]: {score}' for error_type, score in self.__all_result.items())
        return results_str

    def __repr__(self) -> str:
        """Возвращает строковое представление объекта ScoreResult."""
        results_str = '\n '.join(f'[{error_type}]: {score}' for error_type, score in self.__all_result.items())
        return results_str

    @property
    def result(self) -> float:
        """Применяет функцию `func` ко всем значениям в `__all_result`."""
        if len(self.__all_result.keys()) == 0:
            raise ValueError('Empty result.')
        if len(self.__all_result.keys()) == 1:
            return list(self.__all_result.values())[0]
        return self.handler(self.__all_result)

    def __setitem__(self, error_type: 'ScoreType', score: float):
        self.__all_result[error_type] = score

    def __getitem__(self, error_type: 'ScoreType') -> float:
        return self.__all_result[error_type]

    def __eq__(self, other: 'ScoreResult') -> bool:
        return self.result == self.result

    def __lt__(self, other: 'ScoreResult') -> bool:
        return self.result < self.result

    def __le__(self, other: 'ScoreResult') -> bool:
        return self.result <= self.result

    def __gt__(self, other: 'ScoreResult') -> bool:
        return self.result > self.result

    def __ge__(self, other: 'ScoreResult') -> bool:
        return self.result >= self.result

    def __truediv__(self, other: Union[float, int]):
        for key in self.__all_result.keys():
            self.__all_result[key] /= other
        return self

    def to_dict(self):
        return self.__all_result.copy()

    def __iadd__(self, other: 'ScoreResult'):
        # fixme: не совсем ожидаемое поведение
        # Скорее всего лучше разделить
        for key in other.__all_result.keys():
            if key in self.__all_result:
                self.__all_result[key] += other.__all_result[key]
            else:
                self.__all_result[key] = other.__all_result[key]
        return self


class F1Score(AbstractError):
    __name = 'F1'
    average = 'micro'

    def __call__(self, X, Y, Y_pred):
        # print(Y.shape, Y_pred.shape)
        if len(Y_pred.shape) > 2:
            f1 = []
            Y_pred = np.argmax(Y_pred, axis=1)
            # print(Y.shape)
            for i in range(Y.shape[1]):
                f1.append(f1_score(y_pred=Y_pred[:, i],
                                   y_true=Y[:, i], average=self.average))
            return np.mean(f1)
        return f1_score(y_pred=Y_pred,
                        y_true=Y,
                        average=self.average)


class MSEScore(AbstractError):
    __name = 'MSE'

    def __call__(self, X, Y, Y_pred):
        # if len(Y.shape) > 2:
        #     mse = []
        #     for i in range(Y.shape[1]):
        #         mse.append(mean_squared_error(y_pred=Y_pred[:, i],
        #                                       y_true=Y[:, i]))
        #     return np.mean(mse)
        index = X != X
        return mean_squared_error(y_pred=Y_pred[index], y_true=Y[index])


class ScoreType(str, Enum):
    F1_SCORE = "f1_score"
    MSE = "MSE"

    def __repr__(self):
        # Возвращаем только значение перечисления
        return self.value


def to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            return tensor.cpu().numpy()
        return tensor.numpy()
    return np.array(tensor)


@dataclass
class Score:
    name: str
    handler: CellFuncHandler = field(default_factory=lambda: MEAN_HANDLER)

    __metric_cell: ErrorType = None
    __metric_dict: Dict[ScoreType, AbstractError] = field(default_factory=dict)

    def add_metric(self, key: ScoreType, value: AbstractError):
        self.__metric_dict[key] = value
        return self

    def get_empty_result(self) -> ScoreResult:
        score = ScoreResult(self.handler)
        # print(score)

        return ScoreResult(self.handler)

    def __iadd__(self, other: 'Score') -> 'Score':
        if not isinstance(other, Score):
            raise TypeError(f"Unsupported type for +=: {type(other)}")

        if self.__metric_dict.keys() != other.__metric_dict.keys():
            raise ValueError("Keys of metrics must match for += operation.")

        for key in self.__metric_dict.keys():
            self.__metric_dict[key] += other.__metric_dict[key]

        return self

    def __add__(self, other: 'Score') -> 'Score':
        if not isinstance(other, Score):
            raise TypeError(f"Unsupported type for +: {type(other)}")

        if self.__metric_dict.keys() != other.__metric_dict.keys():
            raise ValueError("Keys of metrics must match for + operation.")

        new_score = Score(handler=self.handler)
        new_score.__metric_dict = {key: self.__metric_dict[key] + other.__metric_dict[key]
                                   for key in self.__metric_dict.keys()}

        return new_score

    def __call__(self, X, Y, Y_Pred):
        X, Y, Y_Pred = map(to_numpy, [X, Y, Y_Pred])
        score = ScoreResult(handler=self.handler)
        # print(score)
        for error_type, error in self.__metric_dict.items():
            score.add_result(error_type, error(X, Y, Y_Pred))
        return score


_score_dict = {
    ScoreType.F1_SCORE: Score(ScoreType.F1_SCORE).add_metric(ScoreType.F1_SCORE,
                                                             F1Score()),
    ScoreType.MSE: Score(ScoreType.MSE).add_metric(ScoreType.MSE,
                                                   MSEScore())
}


def get_score(score_type: ScoreType) -> Score:
    return _score_dict[score_type]
