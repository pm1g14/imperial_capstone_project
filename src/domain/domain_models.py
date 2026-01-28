from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from models.heteroskedastic_model import HeteroskedasticModel


class AcquisitionStepDetailsModel:
    x: np.ndarray
    acq_y: float
    gradient: List[float]

@dataclass
class PredictedValuesModel:
    mean_noise: float
    variance: float
    mean: float
    variance_noise: float


@dataclass
class EvaluationMetricsModel:
    abs_err: float
    sq_err: float
    z: float
    nlpd: float
    inside95: bool


@dataclass
class ModelOutput:
    mse: float
    best_lambda: float
    best_eval_point: str
    best_model: "HeteroskedasticModel"
    best_residuals: np.ndarray

@dataclass
class WarpConfig:
    lam_low: float
    lam_high: float
    
@dataclass
class ModelInputConfig:
    bounds: List[Tuple[float, float]]
    nu_mean: float
    nu_noise: float
    n_restarts: int
    num_samples: int
    warmup_steps: int
    thinning: int
    max_tree_depth: int
    raw_samples: int
    max_iter: int
    acquisition_function_str: str
    beta: float | None
    warp_inputs: bool
    warp_outputs: bool
    lam_config: WarpConfig | None

    
@dataclass
class TurboState:
    dim: int
    batch_size: int = 1          
    length: float = 0.05          
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 2   
    failure_tolerance: int = 2  
    best_value: float = -float("inf")
    restart_triggered: bool = False