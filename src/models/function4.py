

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
from utils.matern_bounds_calculator_utils import MaternBoundsCalculatorUtils

class ProductsAcrossWarehouseModel(HomoskedasticFewDModel):
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
            nu=0.5,
            n_restarts_optimizer=10,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            acquisition_function=acquisition_function,
            n_restarts=n_restarts
        )

    def _suggest_length_scale_bounds(self) -> List[Tuple[float, float]]:
        X = self._dataset[['x1', 'x2', 'x3', 'x4']].to_numpy()
        y = self._dataset['y'].to_numpy()
        base_gp = GaussianProcessRegressor(
            kernel=(C(1.0, (1e-3,1e3)) *
                Matern(length_scale=np.full(4, 0.15),
                       length_scale_bounds=[(1e-2, 1.0)]*4,
                       nu=0.5) +
                WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))),
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=0,
        )
        B = 50
        Ls = []
        rng = np.random.default_rng(33)
        for _ in range(B):
            idx = rng.integers(0, len(X), size=len(X))
            Xb, yb = X[idx], y[idx]
            gp_b = clone(base_gp)                
            gp_b.fit(Xb, yb)
            Ls.append(gp_b.kernel_.k1.k2.length_scale)  # shape (3,)
        Ls = np.vstack(Ls)
        logL = np.log(Ls)
        mu = np.median(logL, axis=0)
        sd = np.std(logL, axis=0)

        lower = np.exp(mu - 2*sd)
        upper = np.exp(mu + 2*sd)
        lower = np.clip(lower, 1e-2, 0.8)
        upper = np.clip(upper, 3e-2, 0.8)    # ensure upper >= lower; adjust numbers as needed
        upper = np.maximum(upper, lower * 1.5)  # keep at least ~×1.5 span
        ls_init = np.exp(mu)
        ls_bounds = list(map(tuple, np.c_[lower, upper]))  
        return ls_init.astype(float), ls_bounds

    def get_correlation_matrix(self) -> np.ndarray:
        X = self._dataset[['x1', 'x2', 'x3', 'x4']].to_numpy()
        y = self._dataset['y'].to_numpy()
        return np.corrcoef(X, rowvar=False)

    def suggest_next_point_to_evaluate(self, bounds: List[Tuple[float, float]]) -> str: 
        self._model.fit(self._dataset[['x1', 'x2', 'x3', 'x4']], self._dataset['y'])
        print("Fitted kernel:", self._model.kernel_)
        best_x, max_acq = self._maximize_acquisition_function(bounds=bounds)
        logging.info(f"Best new exploration point found with acquisition maxing at: {max_acq}")

        formatter:str = lambda arr: "-".join(f"{x:.6f}" for x in arr)
        return formatter(best_x)

    def _calculate_acquisition_function_y(self, x: np.ndarray, best_f: float | None = None) -> float:
        mu, sigma = self._model.predict(x.reshape(1, -1), return_std=True)
        return float(self._acquisition_function(mu, sigma, kappa=4.0))

    def _ls_init_and_bounds_from_spacing(self, X, *,
        lower_floor=1e-2, rho_star=0.35,   
        k_low=0.5, k_high=5.0,             
        min_log_span=np.log(1.2)):        
        utils = MaternBoundsCalculatorUtils()
        X = np.asarray(X, float); n, d = X.shape
        global_cap = float(utils.get_matern_correlation_map().get(0.5)(rho_star=rho_star, r=1.0))

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