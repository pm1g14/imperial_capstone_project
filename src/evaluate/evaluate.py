

import math
from typing import List

from domain.domain_models import EvaluationMetricsModel
import numpy as np
import pandas as pd
from utils.csv_utils import CsvUtils

class ModelEvaluator:


    @staticmethod
    def point_metrics(y_true, mu: float, var: float) -> EvaluationMetricsModel:
        var = max(var, 1e-18)
        abs_err = abs(y_true - mu)
        sq_err = (y_true - mu)**2
        z = (y_true - mu) / math.sqrt(var)
        nlpd = 0.5*(math.log(2*math.pi*var) + sq_err/var) #negative log predictive density shows how probable the observed value ytrue is under the model's predictive distribution
        inside95 = abs(z) <= 1.96
        return EvaluationMetricsModel(
            abs_err=abs_err,
            sq_err=sq_err,
            z=z,
            nlpd=nlpd,
            inside95=inside95
        )

    
    @staticmethod
    def calculate_standardized_residual(mu:float, y_last: float, sigma:float) -> float:
       return (y_last - mu) / sigma

    @staticmethod
    def calculate_performance_change_for_all_functions()-> pd.DataFrame:
        performance_change = []
        columns =[]
        for i in range(1, 9):
            inputs, outputs = CsvUtils.get_inputs_and_outputs(i)
            submissions, sub_outputs = CsvUtils.get_submission_values(i)
            improvement = ModelEvaluator._calculate_performance_change_for_function(inputs, outputs, sub_outputs)
            performance_change.append(improvement)
            columns.append(f'f{i}')
        df_performance = pd.DataFrame(index=np.arange(1, 13), columns=columns)
        for i in range(len(performance_change[0])):
            df_performance.loc[i+1] = [
                performance_change[0][i], 
                performance_change[1][i], 
                performance_change[2][i],
                performance_change[3][i], 
                performance_change[4][i], 
                performance_change[5][i], 
                performance_change[6][i],     
                performance_change[7][i]
            ]

        return df_performance

    @staticmethod
    def _calculate_performance_change_for_function(inputs, outputs, sub_outputs)-> np.ndarray:
        columns = []
        for i in range(1,inputs.shape[1]+1):
            columns.append(f'x{i}')
        columns.append('y')    
        all_values = np.concatenate((inputs, outputs), axis=1)
        df_before_submissions = pd.DataFrame(all_values, columns=columns)
        return ModelEvaluator._calculate_improvement_over_max(df_before_submissions, sub_outputs, len(sub_outputs))

    @staticmethod
    def _calculate_improvement_over_max(df:pd.DataFrame, outputs:np.ndarray, trial_no: int) -> List[float]: 
        improvement = []
        for i in range(trial_no):
            current_submission = outputs[i][0]
            previous_outputs = outputs[:i].flatten()
            
            combined_ys = df['y'].to_numpy() if len(previous_outputs)== 0 else np.concatenate((df['y'].to_numpy(), previous_outputs))
            max_so_far = np.max(combined_ys.reshape(-1))
            improvement.append((current_submission - max_so_far)/np.abs(max_so_far))
        return improvement



    
