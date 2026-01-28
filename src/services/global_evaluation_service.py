

from typing import Tuple
import torch
from models.heteroskedastic_model import HeteroskedasticModel
from services.evaluation_service_contract import EvaluationServiceContract
import numpy as np
import pandas as pd

from domain.domain_models import ModelInputConfig

class GlobalEvaluationService(EvaluationServiceContract):

    def __init__(self, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int, config: ModelInputConfig):
        self._dimensions = dims
        self._dataframe = dataframe
        self._total_budget = total_budget
        self._trial_no = trial_no
        self._model_input_config = config

    def run_suggest(self, function_identifier: int) -> str:

        new_point_model, _, _, model, _ = self.init_model_and_get_new_point(
            dataset=self._dataframe,
            bounds=[(0.0, 1.0) for _ in range(self._dimensions)],
            model_config=self._model_input_config,
            csv_output_file_name=f"function_{function_identifier}",
            display_plots=True
        )
     
        return new_point_model
     


    def init_model_and_get_new_point(
        self,
        dataset, 
        bounds, 
        model_config: ModelInputConfig,
        csv_output_file_name: str,
        display_plots: bool = False,
    
    ) -> Tuple[str, float, float]:
        
            model = HeteroskedasticModel(
                dataset=dataset, 
                nu_mean=model_config.nu_mean, 
                nu_noise = model_config.nu_noise, 
                n_restarts=model_config.n_restarts, 
                warp_inputs=model_config.warp_inputs,
                warp_outputs=model_config.warp_outputs, 
                lam=model_config.lam, 
                trial_no=self._trial_no, 
                total_budget=self._total_budget
            )
            
            new_point_model, mu, std, max_acq = model.suggest_next_point_to_evaluate(
                bounds=bounds, 
                num_samples=model_config.num_samples, 
                warmup_steps=model_config.warmup_steps, 
                thinning=model_config.thinning, 
                max_tree_depth=model_config.max_tree_depth, 
                raw_samples=model_config.raw_samples, 
                max_iter=model_config.max_iter, 
                acquisition_function_str=model_config.acquisition_function_str,
                acquisition_function=self.get_acquisition_function(model, model_config.acquisition_function_str, model_config.beta),
                csv_output_file_name=csv_output_file_name
            )

            warped_dataset = dataset.copy()
            if (model_config.warp_outputs):
                warped_dataset['y'] = model._Y_train.detach().cpu().numpy()

            if (model_config.warp_inputs):
                X_warped = model._X_train.detach().cpu().numpy()
                for i, col in enumerate(dataset.copy().drop('y', axis=1).columns):
                    warped_dataset[col] = X_warped[:, i]
        
            if (display_plots):
                self.plot_new_point(warped_dataset, new_point_model, model)

            return new_point_model, mu, std, model, max_acq
