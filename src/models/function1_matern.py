from decimal import Decimal, DecimalException
from typing import List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize
from domain.domain_models import AcquisitionStepDetailsModel, PredictedValuesModel
from models.acquisition.acquisition_functions import AcquisitionFunctions
from models.homoskedastic_few_d_model import HomoskedasticFewDModel
from utils.matern_bounds_calculator_utils import MaternBoundsCalculatorUtils

class FieldContaminationModel(HomoskedasticFewDModel):


    def __init__(
        self, 
        # length_scale:float, 
        dataset: pd.DataFrame,
        nu: float,
        n_restarts_optimizer:int,
        noise_level: float = 1e-3,
        noise_level_bounds: Tuple[float, float] = (1e-3, 1e1),
        # length_scale_bounds: List[Tuple[float, float]]=(1e-2, 2.0), 
        num_points_per_dimension: int = 101,
        acquisition_function: str = AcquisitionFunctions.ucb,
        n_restarts: int = 10
    ):
        super().__init__(
            dataset=dataset,
            nu=1.5,
            n_restarts_optimizer=8,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            acquisition_function=acquisition_function,
            n_restarts=n_restarts
        )
        # self._kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(2, 0.2), length_scale_bounds=ls_bounds, nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e-1))
        self._eval_grid = self._calculate_eval_grid(num_points_per_dimension)
       
        
    
    def _calculate_eval_grid(self, num_points_per_dimension: int) -> np.ndarray:
        x_lin = np.linspace(0, 1, num_points_per_dimension)
        x1_mesh, x2_mesh = np.meshgrid(x_lin, x_lin)
        x_grid = np.vstack([x1_mesh.ravel(), x2_mesh.ravel()]).T
        return x_grid


    def train_and_validate_eval_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        self._model.fit(self._dataset[['x1', 'x2']], self._dataset['y'])
        print("Fitted kernel:", self._model.kernel_)
        post_mean_predictions, post_std_predictions = self._model.predict(self._eval_grid, return_std=True)
        return post_mean_predictions, post_std_predictions




    def suggest_next_point_to_evaluate(self, bounds: List[Tuple[float, float]]) -> str: 
        self._model.fit(self._dataset[['x1', 'x2']], self._dataset['y'])
        print("Fitted kernel:", self._model.kernel_)
        best_x, max_acq = self._maximize_acquisition_function(bounds=bounds)
        logging.info(f"Best new exploration point found with acquisition maxing at: {max_acq}")

        formatter:str = lambda arr: "-".join(f"{x:.6f}" for x in arr)
        print(f"max_acq: {max_acq}")
        return formatter(best_x)



    def _calculate_acquisition_function_y(self, x: np.ndarray, best_f: float | None = None) -> float:
        mu, sigma = self._model.predict(x.reshape(1, -1), return_std=True)
        return float(self._acquisition_function(mu, sigma))


    def get_next_point_to_search(self, post_mean_predictions: np.ndarray, post_std_predictions: np.ndarray) -> Tuple[float, float]:
        peak_index = self._get_peak_acquisition_function_index(post_mean_predictions, post_std_predictions)
        return self._eval_grid[peak_index]

    
    def _get_peak_acquisition_function_index(self, post_mean_predictions: np.ndarray, post_std_predictions: np.ndarray) -> int:
        new_acquisition_values = self._acquisition_function(post_mean_predictions, post_std_predictions)
        return int(np.argmax(new_acquisition_values))


    def predict(self, X: np.ndarray) -> Tuple[np.double, np.double, np.double, np.double]:
        mu_f, std_f = self._model.predict(X.reshape(1,-1), return_std=True)
        mu_f = float(mu_f); var_f = float(std_f**2)
        noise = self._find_white_noise_level(self._model.kernel_)  # may be None
        var_y = var_f + (noise if noise is not None else 0.0)
        return PredictedValuesModel(
            mean=mu_f,
            variance=var_f,
            variance_noise=var_y,
            mean_noise=mu_f
        )

    def _ls_init_and_bounds_from_spacing(self, X, *,
        lower_floor=1e-2, rho_star=0.35,   
        k_low=0.5, k_high=5.0,             
        min_log_span=np.log(1.2)):        
        utils = MaternBoundsCalculatorUtils()
        X = np.asarray(X, float); n, d = X.shape
        global_cap = float(utils.get_matern_correlation_map.get(1.5)(rho_star=rho_star, r=1.0))

        lowers, uppers = [], []
        for j in range(d):
            xj = np.unique(np.sort(X[:, j]))
            if xj.size <= 1:
                lo, hi = lower_floor, min(global_cap, 0.1)
            else:
                diffs = np.diff(xj); diffs = diffs[diffs > 0]
                if diffs.size == 0:
                    lo, hi = lower_floor, min(global_cap, 0.1)
                else:
                    p10 = np.percentile(diffs, 10)
                    p50 = np.percentile(diffs, 50)
                    lo  = max(lower_floor, k_low  * p10)
                    hi  = min(global_cap,  k_high * p50)
                    log_lo, log_hi = np.log(lo), np.log(hi)
                    if (log_hi - log_lo) < min_log_span:
                        log_lo = log_hi - min_log_span
                        lo = np.exp(log_lo)
            lowers.append(float(lo)); uppers.append(float(hi))

        lowers = np.array(lowers); uppers = np.array(uppers)
        ls_init = np.sqrt(lowers * uppers).astype(float)
        ls_bounds = [ (float(lo), float(hi)) for lo,hi in zip(lowers, uppers) ]
        return ls_init, ls_bounds


    def _find_white_noise_level(self, kern):
        # Recursively search for a WhiteKernel and return its noise_level; else None
        if isinstance(kern, WhiteKernel):
            return float(kern.noise_level)
        if hasattr(kern, 'k1'):
            val = self._find_white_noise_level(kern.k1)
            if val is not None:
                return val
        if hasattr(kern, 'k2'):
            val = self._find_white_noise_level(kern.k2)
            if val is not None:
                return val
        return None