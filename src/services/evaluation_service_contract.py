
from typing import Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
import pandas as pd
from abc import ABC, abstractmethod

import torch

from models.heteroskedastic_contract import HeteroskedasticContract
from utils.plot_utils import PerformancePlotUtils


class EvaluationServiceContract(ABC):

    @abstractmethod
    def run_suggest(self, function_identifier: int) -> Tuple[str, HeteroskedasticContract]:
        pass


    def plot_new_point(self, dataset: pd.DataFrame, new_point_model: str, model: HeteroskedasticContract):
        if (len(dataset.copy().drop('y', axis=1).columns) == 2):
            PerformancePlotUtils.plot_3d_with_candidate_botorch(dataset, new_point_model, model=model._model)
        elif (len(dataset.copy().drop('y', axis=1).columns) == 3):
            PerformancePlotUtils.plot_pairwise_slices_botorch_3d(dataset, new_point_model, model=model._model)
        elif(len(dataset.copy().drop('y', axis=1).columns) == 4):
            PerformancePlotUtils.plot_pairwise_slices_botorch_4d(dataset, new_point_model, model=model._model)

    
    def get_acquisition_function(self, model: HeteroskedasticContract, acquisition_function_str: str, beta: float): 
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
