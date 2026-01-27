

import math
from typing import List

import torch

from domain.domain_models import EvaluationMetricsModel
import numpy as np
import pandas as pd
from models.heteroskedastic_contract import HeteroskedasticContract
from utils.csv_utils import CsvUtils
from scipy.stats import norm

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


    @staticmethod
    def should_switch_to_trust_region(trial_no: int, total_budget: int, dataframe: pd.DataFrame) -> bool:
        consecutive_success_trials_threshold = 3
        ys = dataframe['y'].to_numpy()
        top_y_so_far = np.max(ys[:-1])
        last_y = ys[-1]
        remaining_budget = 1 - trial_no/total_budget
    
        improvement = (last_y - top_y_so_far) / abs(top_y_so_far)
        if improvement > 0.15 or remaining_budget <= 0.45:
            return True

        if len(ys) >= consecutive_success_trials_threshold:
            recent_trials = ys[-consecutive_success_trials_threshold:]

            if all(recent_trials[i] > recent_trials[i-1] for i in range(1, len(recent_trials))):
                return True
        return False
    

    @staticmethod
    def normality_score_qq_l2(residuals: np.ndarray) -> float:
        r = np.asarray(residuals, float).reshape(-1)
        r = r[np.isfinite(r)]
        n = r.size
        if n < 3:
            return np.inf
        r_sorted = np.sort(r)
        # plotting positions
        p = (np.arange(1, n+1) - 0.5) / n
        z = norm.ppf(p)
        # optimal linear fit? If residuals are standardized already, compare directly:
        return float(np.mean((r_sorted - z)**2))

    @staticmethod
    def residuals_over_trials(sub_outputs_f1:np.array, submissions_f1: np.array, model: HeteroskedasticContract) -> np.array:
        residuals = []
        xis = torch.tensor(submissions_f1, dtype=torch.float)
        for i in range(xis.shape[0]):
            xi = xis[i]
            post_f = model.get_model().posterior(xi.reshape(1, -1), observation_noise=True)
            mu_i = post_f.mean.item()
            sigma_i = post_f.variance.sqrt().item()
            y_actual = sub_outputs_f1[i, 0]
            residual_for_trial = ModelEvaluator.calculate_standardized_residual( 
                mu_i, 
                y_actual,
                sigma_i
            )
            residuals.append(residual_for_trial)
        return np.array(residuals)