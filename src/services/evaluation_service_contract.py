
import pandas as pd
from abc import ABC, abstractmethod

from models.heteroskedastic_contract import HeteroskedasticContract
from src.utils.plot_utils import PerformancePlotUtils


class EvaluationServiceContract(ABC):

    @abstractmethod
    def run_suggest(self, csv_output_file_name: str, function_number: int):
        pass


    def plot_new_point(self, dataset: pd.DataFrame, new_point_model: str, model: HeteroskedasticContract):
        if (len(dataset.copy().drop('y', axis=1).columns) == 2):
            PerformancePlotUtils.plot_3d_with_candidate_botorch(dataset, new_point_model, model=model._model)
        elif (len(dataset.copy().drop('y', axis=1).columns) == 3):
            PerformancePlotUtils.plot_pairwise_slices_botorch_3d(dataset, new_point_model, model=model._model)
        elif(len(dataset.copy().drop('y', axis=1).columns) == 4):
            PerformancePlotUtils.plot_pairwise_slices_botorch_4d(dataset, new_point_model, model=model._model)
