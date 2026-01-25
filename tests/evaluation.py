import sys, os

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)
from typing import Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
from botorch.test_functions import Ackley, Branin
import numpy as np
import torch
import pandas as pd

from heteroskedastic_model_test import HeteroskedasticModel


class Evaluation2D:
    def __init__(self):
        self._f = Ackley(dim=2, negate=True)


    def evaluate(self, dataset: pd.DataFrame, no_of_trials: int) -> np.ndarray:
        true_opt = self._f.optimal_value
        max_so_far = -1e9
        how_far = []
        for t in range(no_of_trials):

            # if (t <= 5):
            #     beta = 4.0
            #     acquisition_function_str = 'ucb'
            # elif (t > 5 and t < 8):
            #     beta = 1.0
            #     acquisition_function_str = 'ucb'
            # else:
            #     beta = 1.0
            #     acquisition_function_str = 'nei'

            new_point_model, mu, std, model = self.init_model_and_get_new_point(
                dataset=dataset,
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                nu_mean=1.5,
                nu_noise=2.5,
                n_restarts=40,
                num_samples=256,
                warmup_steps=512,
                thinning=16,
                max_tree_depth=6,
                raw_samples=512,
                max_iter=200,
                acquisition_function_str='nei',
                beta=0.4,
                warp_inputs=True,
                warp_outputs=False,
                csv_output_file_name='function_1'
            )
            x1, x2 = (float(x) for x in new_point_model.split("-"))
            new_point_t = torch.tensor([[x1, x2]], dtype=torch.float64)
            y_new = self._f(new_point_t).item()
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame({"x1": [x1], "x2": [x2], "y": [y_new]})
                ],
                ignore_index=True
            )
            if y_new > max_so_far:
                max_so_far = y_new
            
            diff = np.abs(max_so_far - true_opt)
            breakpoint()
            how_far.append(diff)

        print(how_far)

        

    
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
        beta: float = 1.0, 
        warp_inputs: bool = False,
        warp_outputs: bool = False
    ) -> Tuple[str, float, float]:
        model = HeteroskedasticModel(dataset=dataset, nu_mean=nu_mean, nu_noise = nu_noise, n_restarts=n_restarts, warp_inputs=warp_inputs,warp_outputs=warp_outputs)
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
            csv_output_file_name=csv_output_file_name
        )
        return new_point_model, mu, std, model