from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import numpy as np
from scipy.special import factorial


class Distribution(ABC):
    def __init__(self, parameter: float):
        assert parameter > 0, f"Distribution parameter {parameter} must be positive"
        self._parameter = parameter

    @property
    def parameter(self) -> float:
        return self._parameter

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sample(self, size: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _calc_empirical_moments(samples: np.ndarray, orders: np.ndarray) -> np.ndarray:
        return np.power(samples[:, :, None], orders[None, None, :]).mean(axis=1)

    def predict_parameter(self, samples: np.ndarray, orders: np.ndarray) -> np.ndarray:
        assert np.all(orders > 0), f"Orders {orders} must be positive"
        assert samples.ndim == 2, f"Samples shape {samples.shape} but it must have two dimensions"
        assert orders.ndim == 1, f"Orders shape {orders.shape} but it must have one dimension"

        res = self._predict_parameter(samples, orders)

        assert res.shape == (samples.shape[0], orders.shape[0])
        return res

    @abstractmethod
    def _predict_parameter(self, samples: np.ndarray, orders: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UniformDistribution(Distribution):
    @property
    def name(self) -> str:
        return "Uniform distribution"

    def sample(self, size: Tuple[int, int]) -> np.ndarray:
        return np.random.uniform(high=self._parameter, size=size)

    def _predict_parameter(self, samples: np.ndarray, orders: np.ndarray) -> np.ndarray:
        empirical_moments = self._calc_empirical_moments(samples, orders)
        orders = orders[None, :]
        return np.power((orders + 1) * empirical_moments, 1 / orders)


class ExponentialDistribution(Distribution):
    @property
    def name(self) -> str:
        return "Exponential distribution"

    def sample(self, size: Tuple[int, int]) -> np.ndarray:
        return np.random.exponential(scale=self._parameter, size=size)

    def _predict_parameter(self, samples: np.ndarray, orders: np.ndarray) -> np.ndarray:
        empirical_moments = self._calc_empirical_moments(samples, orders)
        orders = orders[None, :]
        return np.power(empirical_moments / factorial(orders), 1 / orders)
