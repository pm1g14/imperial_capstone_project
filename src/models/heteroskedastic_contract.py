
from abc import ABC
from typing import List, Tuple
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound, qLogNoisyExpectedImprovement
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.models.model import Model
import torch
import numpy as np

class HeteroskedasticContract(ABC):

    def get_model(self) -> Model:
        pass

    def get_X(self) -> torch.Tensor:
        pass

    def get_Y(self) -> torch.Tensor:
        pass

    def suggest_next_point_to_evaluate(
        self, 
        bounds: List[Tuple[float, float]], 
        num_samples: int, 
        warmup_steps: int, 
        thinning: int, 
        max_tree_depth: int, 
        raw_samples: int, 
        max_iter: int, 
        acquisition_function_str: str, 
        acquisition_function: qLogNoisyExpectedImprovement | UpperConfidenceBound | qNegIntegratedPosteriorVariance, 
        csv_output_file_name: str
    ) -> str:
        pass


    def ls_init_and_bounds_from_spacing(
        self,
        X,
        *,
        # Global bounds for lengthscales in *input units*
        global_lo: float = 1e-2,
        global_hi: float = 10.0,
        trial_no: int,
        total_budget: int,
        # How wide around the "typical" spacing to set per-dim bounds
        span_factor: float = 3.0,
        nu: float = 1.5,      # kept for signature compatibility, not used
        **kwargs,             # ignore rho_star, k_low, k_high, min_log_span, etc.
    ):
  
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n x d).")

        n, d = X.shape

        lowers = []
        uppers = []

        for j in range(d):
            xj = np.unique(np.sort(X[:, j]))
            remaining_budget = 1 - trial_no/total_budget
            if remaining_budget > 0.8:
                if xj.size <= 1:
                    # Degenerate dimension: fall back to very generic bounds
                    lo_j = global_lo
                    hi_j = global_hi
                else:
                    diffs = np.diff(xj)
                    diffs = diffs[diffs > 0]
                    if diffs.size == 0:
                        lo_j = global_lo
                        hi_j = global_hi
                    else:
                        med = float(np.median(diffs))

                        # Center a "local" window around the median spacing
                        lo_j = med / span_factor
                        hi_j = med * span_factor

                        # Intersect with global bounds
                        lo_j = max(global_lo, lo_j)
                        hi_j = min(global_hi, hi_j)

                        # Ensure non-degenerate interval
                        if hi_j <= lo_j:
                            hi_j = min(global_hi, lo_j * 1.5)

                lowers.append(lo_j)
                uppers.append(hi_j)
            else:
                lo_j = global_lo
                hi_j = global_hi
                lowers.append(lo_j)
                uppers.append(hi_j)

            lowers = np.array(lowers, dtype=float)
            uppers = np.array(uppers, dtype=float)

        # Geometric mean init inside the (lo, hi) interval
        ls_init = np.sqrt(lowers * uppers).astype(float)
        ls_bounds = [(float(lo), float(hi)) for lo, hi in zip(lowers, uppers)]
        print(f"ls bounds: {ls_bounds}")
        return ls_init, ls_bounds