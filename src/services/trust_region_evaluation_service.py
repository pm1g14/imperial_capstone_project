

from typing import Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
import torch
from services.turbo_service import TurboService
from domain.domain_models import ModelInputConfig, TurboState
from models.heteroskedastic_turbo_model import HeteroskedasticTurboModel
import numpy as np
import pandas as pd

from services.evaluation_service_contract import EvaluationServiceContract
from models.heteroskedastic_contract import HeteroskedasticContract

class TrustRegionEvaluationService(EvaluationServiceContract):

    def __init__(self, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int, function_num: int, config: ModelInputConfig):
        self._dimensions = dims
        self._dataframe = dataframe
        self._total_budget = total_budget
        self._trial_no = trial_no
        self._turbo_state_path_identifier = function_num
        self._model_input_config = config

        

    def run_suggest(self, function_identifier: int) -> Tuple[str, HeteroskedasticContract]:
        
        turbo_service = TurboService()
        turbo_state = turbo_service.load_turbo_state(self._turbo_state_path_identifier)
        if not turbo_state:
            turbo_state = TurboState(dim=self._dimensions)


        best_new_X, model = self._calculate_max_acquisition_value(
            turbo_state=turbo_state,
            output_file_name=f"function_{function_identifier}",
        )
        return best_new_X, model


    def _calculate_max_acquisition_value(
        self,
        turbo_state: TurboState,
        output_file_name: str
    ) -> Tuple[str, HeteroskedasticContract]:
            new_point_model, mu, std, model = self._init_model_and_get_new_point_turbo(
                bounds=[(0.0, 1.0) for _ in range(self._dimensions)],
                turbo_state=turbo_state,
                csv_output_file_name=output_file_name,
                display_plots=False
            )
        
        
            return new_point_model, model


    def _init_model_and_get_new_point_turbo(
        self,
        bounds, 
        turbo_state: TurboState,
        csv_output_file_name: str,
        display_plots: bool = False,
        
    ) -> Tuple[str, float, float, HeteroskedasticTurboModel]:
            
            model = HeteroskedasticTurboModel(
                dataset=self._dataframe, 
                nu_mean=self._model_input_config.nu_mean, 
                nu_noise = self._model_input_config.nu_noise, 
                n_restarts=self._model_input_config.n_restarts, 
                warp_inputs=self._model_input_config.warp_inputs,
                warp_outputs=self._model_input_config.warp_outputs, 
                turbo_state=turbo_state,
                warp_config=self._model_input_config.lam_config
            )
            if (self._model_input_config.acquisition_function_str == 'nei'):
                acquisition_function=qLogNoisyExpectedImprovement(
                    model=model.get_model(), 
                    X_baseline=model.get_X(), 
                    sampler=model.get_sampler()
                )
            elif (self._model_input_config.acquisition_function_str == 'ucb'):
                acquisition_function=UpperConfidenceBound(
                    model=model.get_model(), 
                    beta=self._model_input_config.beta
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
                num_samples=self._model_input_config.num_samples, 
                warmup_steps=self._model_input_config.warmup_steps, 
                thinning=self._model_input_config.thinning, 
                max_tree_depth=self._model_input_config.max_tree_depth, 
                raw_samples=self._model_input_config.raw_samples, 
                max_iter=self._model_input_config.max_iter, 
                acquisition_function_str=self._model_input_config.acquisition_function_str,
                acquisition_function=acquisition_function,
                csv_output_file_name=csv_output_file_name,
            )

            warped_dataset = self._dataframe.copy()
            if (self._model_input_config.warp_outputs):
                warped_dataset['y'] = model.get_Y().detach().cpu().numpy()

            if (self._model_input_config.warp_inputs):
                X_warped = model._X_train.detach().cpu().numpy()
                for i, col in enumerate(self._dataframe.copy().drop('y', axis=1).columns):
                    warped_dataset[col] = X_warped[:, i]
        
            if (display_plots):
                self.plot_new_point(warped_dataset, new_point_model, model)

            return new_point_model, mu, std, model
