from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class Normalizer(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def re_normalize(self, X: np.ndarray) -> np.ndarray:
        pass


@dataclass
class MinMaxNormalizer(Normalizer):
    feature_range: tuple = (0, 1)
    scaler = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        self.scaler.fit(X)
        return self.scaler.transform(X)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("The normalizer has not been fitted yet. Call 'fit' first.")
        return self.scaler.transform(X)

    def re_normalize(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("The normalizer has not been fitted yet. Call 'fit' first.")
        return self.scaler.inverse_transform(X)


@dataclass
class StandardNormalizer(MinMaxNormalizer):
    def fit(self, X: np.ndarray) -> np.ndarray:
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        return self.scaler.transform(X)
    @staticmethod
    def init_from_params(scaler_params: Dict):
        normalizer = StandardNormalizer()
        # new_scaler = StandardScaler()
        normalizer.scaler.mean_ = np.array(scaler_params["mean"])
        normalizer.scaler.var_ = np.array(scaler_params["var"])
        normalizer.scaler.scale_ = np.array(scaler_params["scale"])

    def get_params(self) -> Dict:
        scaler_params = {
            "mean": self.scaler.mean_.tolist(),
            "var": self.scaler.var_.tolist(),
            "scale": self.scaler.scale_.tolist()
        }
        return scaler_params
