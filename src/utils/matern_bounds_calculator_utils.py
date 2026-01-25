from typing import Dict
from typing import Callable
import numpy as np

FloatFunc = Callable[[float], float]

class MaternBoundsCalculatorUtils:

    def __init__(self):
        self._matern_corr_map: Dict[float, FloatFunc] = {
            0.5: self._matern1_2,
            1.5: self._matern3_2,
            2.5: self._matern5_2
        }

    
    def get_matern_correlation_map(self) -> Dict[float, FloatFunc]:
        return self._matern_corr_map


    def _matern1_2(self, rho_star=0.35, r=1.0) -> float:
        lo, hi = 1e-6, 50.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            rho = (1.0 + mid) * np.exp(-mid)
            if rho > rho_star: lo = mid
            else:              hi = mid
        a = 0.5 * (lo + hi)
        return (np.sqrt(3.0) * r) / a  


    def _matern3_2(self, rho_star=0.35, r=1.0) -> float:
        lo, hi = 1e-6, 50.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            rho = (1.0 + mid) * np.exp(-mid)
            if rho > rho_star: lo = mid
            else:              hi = mid
        a = 0.5 * (lo + hi)
        return (np.sqrt(3.0) * r) / a  


    def _matern5_2(self, rho_star=0.35, r=1.0) -> float:
        lo, hi = 1e-6, 50.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            rho = ((1.0 + mid + np.pow(mid, 2)/3)) * np.exp(-mid)
            if rho > rho_star: lo = mid
            else:              hi = mid
        a = 0.5*(lo+hi)
        return (np.sqrt(5)*r) / a