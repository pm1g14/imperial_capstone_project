

from typing import Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
import torch
from services.turbo_service import TurboService
from domain.domain_models import TurboState
from models.heteroskedastic_turbo_model import HeteroskedasticTurboModel
import numpy as np
import pandas as pd

from services.evaluation_service_contract import EvaluationServiceContract

class TrustRegionEvaluationService(EvaluationServiceContract):

    def __init__(self, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int):
        self._dimensions = dims
        self._dataframe = dataframe
        self._total_budget = total_budget
        self._trial_no = trial_no


    def run_suggest(self, csv_output_file_name: str, function_number: int) -> str:
        
        turbo_service = TurboService()
        turbo_state = turbo_service.load_turbo_state(function_number)
        if not turbo_state:
            turbo_state = TurboState(dim=self._dimensions)


        best_new_X, max_acq_value = self._calculate_max_acquisition_value(
            nu_mean=1.5,
            nu_noise=2.5,
            n_restarts=40,
            num_samples=256,
            turbo_state=turbo_state,
            warmup_steps=512,
            thinning=16,
            max_tree_depth=6,
            raw_samples=512,
            max_iter=200,
            acquisition_function_str='nei',
            beta=2.0,
            warp_inputs=True,
            warp_outputs=True,
            output_file_name=csv_output_file_name
        )
        return best_new_X


    def _calculate_max_acquisition_value(
        self,
        nu_mean: float, 
        nu_noise: float,
        n_restarts: int,
        num_samples: int,
        warmup_steps: int,
        thinning: int,
        max_tree_depth: int,
        raw_samples: int,
        max_iter: int,
        acquisition_function_str: str,
        beta: float,
        warp_inputs: bool,
        warp_outputs: bool,
        output_file_name: str
    ) -> tuple[float, float]:
            new_point_model, mu, std, model, max_acq = self._init_model_and_get_new_point_turbo(
                bounds=[(0.0, 1.0) for _ in range(self._dimensions)],
                nu_mean=nu_mean,
                nu_noise=nu_noise,
                n_restarts=n_restarts,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                thinning=thinning,
                max_tree_depth=max_tree_depth,
                raw_samples=raw_samples,
                max_iter=max_iter,
                acquisition_function_str=acquisition_function_str,
                beta=beta,
                warp_inputs=warp_inputs,
                warp_outputs=warp_outputs,
                csv_output_file_name=output_file_name,
                display_plots=False
            )
        
        
            return new_point_model, max_acq


    def _init_model_and_get_new_point_turbo(
        self,
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
        turbo_state: TurboState,
        beta: float = 1.0, 
        warp_inputs: bool = False,
        warp_outputs: bool = False,
        display_plots: bool = False,
        
    ) -> Tuple[str, float, float]:
            
            model = HeteroskedasticTurboModel(
                dataset=self._dataset, 
                nu_mean=nu_mean, 
                nu_noise = nu_noise, 
                n_restarts=n_restarts, 
                warp_inputs=warp_inputs,
                warp_outputs=warp_outputs, 
                turbo_state=turbo_state
            )
            if (acquisition_function_str == 'nei'):
                acquisition_function=qLogNoisyExpectedImprovement(
                    model=model.get_model(), 
                    X_baseline=model.get_X(), 
                    sampler=model.get_sampler()
                )
            elif (acquisition_function_str == 'ucb'):
                acquisition_function=UpperConfidenceBound(
                    model=model.get_model(), 
                    beta=beta
                )
            else:
                d = model.get_X().shape[1]
                mc_points = torch.rand(256, d, dtype=torch.double)
                acquisition_function=qNegIntegratedPosteriorVariance(
                    model=model.get_model().mean_gp,
                    mc_points=mc_points
                )
            new_point_model, mu, std = model.suggest_next_point_to_evaluate(
                bounds=bounds, 
                num_samples=num_samples, 
                warmup_steps=warmup_steps, 
                thinning=thinning, 
                max_tree_depth=max_tree_depth, 
                raw_samples=raw_samples, 
                max_iter=max_iter, 
                acquisition_function_str=acquisition_function_str,
                acquisition_function=acquisition_function,
                csv_output_file_name=csv_output_file_name,
            )

            warped_dataset = self._dataframe.copy()
            if (warp_outputs):
                warped_dataset['y'] = model.get_Y().detach().cpu().numpy()

            if (warp_inputs):
                X_warped = model._X_train.detach().cpu().numpy()
                for i, col in enumerate(self._dataframe.copy().drop('y', axis=1).columns):
                    warped_dataset[col] = X_warped[:, i]
        
            if (display_plots):
                self.plot_new_point(warped_dataset, new_point_model, model)

            return new_point_model, mu, std, model
