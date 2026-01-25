import numpy as np
from scipy.stats import boxcox_llf
import torch
import optuna
from evaluate.evaluate import ModelEvaluator
from utils.warp_utils import WarpUtils


class TuneWarp:
    def __init__(self, R_data: torch.Tensor, e: float = 1e-6):
        self._R_data = R_data
        R_np = R_data.detach().cpu().numpy().astype(float).flatten()
        
        # Box-Cox requires positive input, so we must shift the residuals.
        # This shift ensures R_shifted is always positive.
        shift = (-R_data.min().item() + e) if R_data.min() <= 0 else e
        self._R_shifted = R_np + shift
      

    def optuna_objective(self, trial: optuna.Trial) -> float:
        """Optuna Objective: Minimizes the Negative Log-Likelihood (NLL) of the residuals."""
        lam = trial.suggest_float("lambda", low=-2.0, high=2.0)
        
        try:
            # Calculate the Log-Likelihood (LL) for the transformed residuals
            ll_bc = boxcox_llf(lam, self._R_shifted)
            
            # Return the Negative Log-Likelihood (NLL) for minimization
            return -ll_bc
        except Exception:
            # Penalty for invalid lambda values
            return 1e10

