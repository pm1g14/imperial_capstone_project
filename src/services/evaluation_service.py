

from typing import Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
import torch
from src.evaluate.evaluate import ModelEvaluator
from src.models.heteroskedastic_model import HeteroskedasticModel
from src.utils.csv_utils import CsvUtils
from src.utils.plot_utils import PerformancePlotUtils
import numpy as np
import pandas as pd

class EvaluationService:

    def __init__(self, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int, local_evaluation: bool):
        self._dimensions = dims
        self._dataframe = dataframe
        self._total_budget = total_budget
        self._trial_no = trial_no
        self._local_evaluation = local_evaluation

    def run_suggest(self) -> str:
      

        # dimensions = 4
        # splits = 4
        # combinations = get_new_bounds(dimensions, splits, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
        # best_new_X, max_acq_value = calculate_max_acquisition_value(
        #     combinations=combinations,
        #     dataframe=df5,
        #     nu_mean=1.5,
        #     nu_noise=2.5,
        #     n_restarts=40,
        #     num_samples=256,
        #     warmup_steps=512,
        #     thinning=16,
        #     max_tree_depth=6,
        #     raw_samples=512,
        #     max_iter=200,
        #     acquisition_function_str='nei',
        #     beta=2.0,
        #     warp_inputs=True,
        #     warp_outputs=True,
        #     output_file_name='function_5'
        # )
    
            # turbo_service = TurboService()
            # turbo_state_f2 = turbo_service.load_turbo_state(2)
            # if not turbo_state_f2:
            #     turbo_state_f2 = TurboState(dim=inputs_f2.shape[1])
    

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
            csv_output_file_name='function_2',
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
                acquisition_function=self._get_acquisition_function(model, acquisition_function_str, beta),
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
                self._plot_new_point(warped_dataset, new_point_model, model)

            return new_point_model, mu, std, model, max_acq


    def _plot_new_point(self, dataset: pd.DataFrame, new_point_model: str, model: HeteroskedasticModel):
        if (len(dataset.copy().drop('y', axis=1).columns) == 2):
            PerformancePlotUtils.plot_3d_with_candidate_botorch(dataset, new_point_model, model=model._model)
        elif (len(dataset.copy().drop('y', axis=1).columns) == 3):
            PerformancePlotUtils.plot_pairwise_slices_botorch_3d(dataset, new_point_model, model=model._model)
        elif(len(dataset.copy().drop('y', axis=1).columns) == 4):
            PerformancePlotUtils.plot_pairwise_slices_botorch_4d(dataset, new_point_model, model=model._model)


    def _get_acquisition_function(self, model: HeteroskedasticModel, acquisition_function_str: str, beta: float): 
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
        return acquisition_function