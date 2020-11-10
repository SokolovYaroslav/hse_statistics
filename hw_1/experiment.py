import random

import plotly as plt
import plotly.express as px

from hw_1.distributions import Distribution
import numpy as np


class Experiment:
    def __init__(self, distribution: Distribution):
        self._distribution = distribution

    @staticmethod
    def _set_seed(seed: int):
        np.random.seed(seed)
        random.seed(seed)

    def run_method_of_moments(self, num_runs: int, num_samples: int, max_order: int, seed: int = 42) -> np.ndarray:
        Experiment._set_seed(seed)

        samples_size = (num_runs, num_samples)
        samples = self._distribution.sample(samples_size)

        orders = np.arange(1, max_order + 1)
        predicted_parameters = self._distribution.predict_parameter(samples, orders)

        return self.calc_mse(predicted_parameters)

    def calc_mse(self, pred_parameters: np.ndarray) -> np.ndarray:
        true_parameter = self._distribution.parameter
        return np.mean((pred_parameters - true_parameter) ** 2, axis=0)

    def make_fig(self, errors: np.ndarray):
        distribution_name = self._distribution.name

        fig = px.line(x=np.arange(1, len(errors) + 1), y=errors,
                      title=f"MSE for method of moments for {distribution_name}")
        fig.update_layout(xaxis={"title": "k"}, yaxis={"title": "MSE", "exponentformat": "e"})

        return fig
