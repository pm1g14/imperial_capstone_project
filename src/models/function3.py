

from typing import Tuple
import logging
from typing import Callable, List
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.base import clone

from models.acquisition.acquisition_functions import AcquisitionFunctions
from models.homoskedastic_few_d_model import HomoskedasticFewDModel

class DrugDiscoveryModel(HomoskedasticFewDModel):

    def __init__(
        self, 
        dataset: pd.DataFrame,
        nu: float,
        n_restarts_optimizer:int,
        noise_level: float = 1e-3,
        noise_level_bounds: Tuple[float, float] = (1e-6, 1e1),
        acquisition_function: Callable[[np.ndarray], float] = AcquisitionFunctions.ucb,
        n_restarts: int = 10
    ):
        super().__init__(
            dataset=dataset,
            nu=1.0,
            n_restarts_optimizer=10,
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1e1),
            acquisition_function=acquisition_function,
            n_restarts=n_restarts
        )

    def get_correlation_matrix(self) -> np.ndarray:
        X = self._dataset[['x1', 'x2', 'x3']].to_numpy()
        y = self._dataset['y'].to_numpy()
        return np.corrcoef(X, rowvar=False)


    def suggest_next_point_to_evaluate(self, bounds: List[Tuple[float, float]]) -> str: 
        self._model.fit(self._dataset[['x1', 'x2', 'x3']], self._dataset['y'])
        print("Fitted kernel:", self._model.kernel_)
        best_x, max_acq = self._maximize_acquisition_function(bounds=bounds)
        logging.info(f"Best new exploration point found with acquisition maxing at: {max_acq}")

        formatter:str = lambda arr: "-".join(f"{x:.6f}" for x in arr)
        return formatter(best_x)

    def _calculate_acquisition_function_y(self, x: np.ndarray, best_f: float | None = None) -> float:
        mu, sigma = self._model.predict(x.reshape(1, -1), return_std=True)
        return float(self._acquisition_function(mu, sigma, kappa=3.5))

