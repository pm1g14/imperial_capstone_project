
from abc import ABC
from typing import List, Tuple
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound, qLogNoisyExpectedImprovement
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.models.model import Model
import torch


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