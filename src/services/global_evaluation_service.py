

from typing import Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
import torch
from models.heteroskedastic_model import HeteroskedasticModel
from services.evaluation_service_contract import EvaluationServiceContract
from utils.csv_utils import CsvUtils
from utils.plot_utils import PerformancePlotUtils
import numpy as np
import pandas as pd

class GlobalEvaluationService(EvaluationServiceContract):

    def __init__(self, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int):
        self._dimensions = dims
        self._dataframe = dataframe
        self._total_budget = total_budget
        self._trial_no = trial_no

    def run_suggest(self, csv_output_file_name: str) -> str:
        new_point_model, _, _, model, _ = self.init_model_and_get_new_point(
            dataset=self._dataframe,
            bounds=[(0.0, 1.0) for _ in range(self._dimensions)],
            nu_mean=1.5,
            nu_noise=0.5,
            n_restarts=40,
            num_samples=256,
            warmup_steps=512,
            thinning=16,
            max_tree_depth=6,
            raw_samples=512,
            max_iter=200,
            acquisition_function_str='nei',
            beta=5.0,
            warp_inputs=True,
            warp_outputs=True,
            csv_output_file_name=csv_output_file_name,
            display_plots=True
        )
     
        return new_point_model
     


    def init_model_and_get_new_point(
        self,
        dataset, 
        bounds, 
        nu_mean, 
        nu_noise, 
        n_restarts, 
        num_samples, 
        warmup_steps, 
        thinning, 
        max_tree_depth, 
        raw_samples, 
        max_iter, 
        csv_output_file_name,
        acquisition_function_str,
        lam: float | None = None, 
        beta: float = 1.0, 
        warp_inputs: bool = False,
        warp_outputs: bool = False,
        display_plots: bool = False,
    
    ) -> Tuple[str, float, float]:
        
            model = HeteroskedasticModel(dataset=dataset, nu_mean=nu_mean, nu_noise = nu_noise, n_restarts=n_restarts, warp_inputs=warp_inputs,warp_outputs=warp_outputs, lam=lam)
            
            new_point_model, mu, std, max_acq = model.suggest_next_point_to_evaluate(
                bounds=bounds, 
                num_samples=num_samples, 
                warmup_steps=warmup_steps, 
                thinning=thinning, 
                max_tree_depth=max_tree_depth, 
                raw_samples=raw_samples, 
                max_iter=max_iter, 
                acquisition_function_str=acquisition_function_str,
                acquisition_function=self.get_acquisition_function(model, acquisition_function_str, beta),
                csv_output_file_name=csv_output_file_name
            )

            warped_dataset = dataset.copy()
            if (warp_outputs):
                warped_dataset['y'] = model._Y_train.detach().cpu().numpy()

            if (warp_inputs):
                X_warped = model._X_train.detach().cpu().numpy()
                for i, col in enumerate(dataset.copy().drop('y', axis=1).columns):
                    warped_dataset[col] = X_warped[:, i]
        
            if (display_plots):
                self.plot_new_point(warped_dataset, new_point_model, model)

            return new_point_model, mu, std, model, max_acq
