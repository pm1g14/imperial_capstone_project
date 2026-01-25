from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from models.acquisition.acquisition_functions import AcquisitionFunctions

class HomoskedasticFewDModel(ABC):

    def __init__(
        self, 
        dataset: pd.DataFrame,
        nu: float,
        n_restarts_optimizer: int, 
        noise_level: float = 1e-3,
        noise_level_bounds: Tuple[float, float] = (1e-6, 1e1),
        acquisition_function: Callable[[np.ndarray], float] = AcquisitionFunctions.ucb,
        n_restarts: int = 10,
    ):
        self._dataset = dataset
        ls_init, ls_bounds = self._ls_init_and_bounds_from_spacing(self._dataset.iloc[:, :-1].to_numpy())
        self._kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=ls_init, 
            length_scale_bounds=ls_bounds, 
            nu=nu
        ) + WhiteKernel(
            noise_level=noise_level, 
            noise_level_bounds=noise_level_bounds
        )
        self._model = GaussianProcessRegressor(
            kernel=self._kernel, 
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer
        )
        self._acquisition_function = acquisition_function
        self._num_restarts = n_restarts



    def _maximize_acquisition_function(self, bounds: List[Tuple[float, float]], best_f: float | None = None, method:str = "L-BFGS-B") -> int:
        best_x = None
        best_acq_y = -np.inf
        lows, highs = zip(*bounds)
        low = np.array(lows, dtype=float)
        high = np.array(highs, dtype=float)
        
        rng = np.random.default_rng(30)
        random_start_points = rng.uniform(low, high, size=(self._num_restarts, len(bounds)))
        
        for x0 in random_start_points:
            res = minimize(
                lambda x: - self._calculate_acquisition_function_y(x, best_f),
                x0=x0, 
                method=method, 
                jac=False, 
                bounds=bounds
            )
            if res.success == False:
                raise Exception("Something went wrong when calculating the max for the acquisition function.")

            acq_y = - res.fun
        
            if acq_y > best_acq_y:
                best_acq_y = acq_y
                best_x = np.clip(res.x, low, high)
        return best_x, best_acq_y



    @abstractmethod
    def _ls_init_and_bounds_from_spacing(self, X, *,
        lower_floor=1e-2, rho_star=0.35,   
        k_low=0.5, k_high=5.0,             
        min_log_span=np.log(1.2)):        
        pass
  
    

    @abstractmethod
    def get_correlation_matrix(self) -> np.ndarray:
        pass


    @abstractmethod
    def _calculate_acquisition_function_y(self, x: np.ndarray, best_f: float | None = None) -> float:
        pass


    @abstractmethod
    def suggest_next_point_to_evaluate(self, bounds: List[Tuple[float, float]]) -> str:
        pass
    
