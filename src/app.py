#!/usr/bin/env python3
import itertools
import logging
from pathlib import Path
import sys
from typing import List, Tuple
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
import torch
from evaluate.evaluate import ModelEvaluator
from models.function1 import FieldContaminationModel as saas
from models.function1_matern import FieldContaminationModel as matern
from models.heteroskedastic_model import HeteroskedasticModel
from models.nn_classifier import NNClassifierModel
from models.nn_surrogate import NNSurrogateModel
from domain.domain_models import ModelOutput, TurboState
from models.heteroskedastic_turbo_model import HeteroskedasticTurboModel
from services.turbo_service import TurboService
from models.heteroskedastic_contract import HeteroskedasticContract
from src.services.evaluation_service import EvaluationService
from utils.json_utils import JsonUtils
from tune.tune_warp import TuneWarp
from utils.csv_utils import CsvUtils
from utils.min_max_scaler import MinMaxScaler
from utils.plot_utils import PerformancePlotUtils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import argparse

def plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Function 1')
    plt.colorbar(scatter, label='Raw Contamination Reading (y)')
    plt.show()



def main() -> None:    
   
    parser = argparse.ArgumentParser(description='BO with TuRBO')
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['evaluate', 'update'],
        help='Mode to run the BO in. Use evaluate to get a new evaluation point. Use update to update the turbo state.'
    )
    parser.add_argument(
        '--input_dataset_path', 
        type=str, 
        required=True, 
        help='The path to the input dataset. Must be an .npy file.'
    )
    parser.add_argument(
        '--dimensions', 
        type=int, 
        required=True, 
        help='The dimensionality of the input dataset.'
    )
    parser.add_argument(
        '--total_budget', 
        type=int, 
        required=True, 
        help='The total budget for the BO.'
    )
    parser.add_argument(
        '--trial_no', 
        type=int, 
        required=True, 
        help='The trial number for the BO.'
    )
    parser.add_argument(
        '--y_new', 
        type=float, 
        default=None,
        required=False, 
        help='In case of update mode, the new y value is used to compare against the max so far.'
    )
    parser.add_argument(
        '--function_number', 
        type=int, 
        required=True, 
        help='The function number for which we are optimizing or updating state.'
    )
    args = parser.parse_args()

    if args.mode == "evaluate":
        run_suggest()
    elif args.mode == "update":
        if args.y is None:
            raise ValueError("In update mode you must provide --y <value>")
        run_update(args.y_new, args.function_number)


def run_update(y_new: float, function_number:int) -> None:
    turbo_service = TurboService()
    turbo_state = turbo_service.load_turbo_state(function_number)

    if not turbo_state:
        turbo_state = TurboState(dim=5)

    turbo_service.update_turbo_state(
        turbo_state=turbo_state,
        function_number=function_number,
        y_new = y_new
    )
    logging.info(f"Turbo state updated for function {function_number}")
    sys.exit(0)



def calculate_max_acquisition_value(
    combinations: list[tuple[float, float]], 
    dataframe: pd.DataFrame, 
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
    max_acqs = []
    for new_bounds in combinations:
        new_point_model, mu, std, model, max_acq = init_model_and_get_new_point(
            dataset=dataframe,
            bounds=new_bounds,
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
        max_acqs.append((new_point_model, max_acq))
    
    max_acq_value = -np.inf
    best_new_X = None
    for new_X, acq_value in max_acqs:
        if acq_value > max_acq_value:
            max_acq_value = acq_value
            best_new_X = new_X
    return best_new_X, max_acq_value

def initialize_turbo_state(function_number: int, inputs: np.array) -> TurboState:
    turbo_service = TurboService()
    turbo_state = turbo_service.load_turbo_state(function_number)
    if not turbo_state:
        turbo_state = TurboState(dim=len(inputs))
    return turbo_state


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


def convert_df_y_to_classification(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    quantile_80 = df['y'].quantile(0.8)
    df['y'] = df['y'].apply(lambda x: 1 if x >= quantile_80  else 0)
    return df


def predict_y_at_evaluation_point(x:str, df: pd.DataFrame, hidden_activation: str, output_activation:str):
    new_point_model_arr = tf.convert_to_tensor(np.array(x.split('-')), dtype=tf.float32)
    model = NNSurrogateModel(df, hidden_activation=hidden_activation, output_activation = output_activation)
    y_pred = model.fit_and_predict(learning_rate=1e-2, epochs=1000)
    new_y =  model._model.predict(tf.expand_dims(new_point_model_arr, axis=0), verbose=0).squeeze(-1)
    feat_influence, rank, point_steepness = model.summarize_gradients(tf.expand_dims(new_point_model_arr, axis=0))
    return new_y, feat_influence, rank, point_steepness



def residuals_over_trials(sub_outputs_f1:np.array, submissions_f1: np.array, model: HeteroskedasticContract):
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


def init_model_and_get_new_point(
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
        new_point_model, mu, std, max_acq = model.suggest_next_point_to_evaluate(
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

        warped_dataset = dataset.copy()
        if (warp_outputs):
            warped_dataset['y'] = model._Y_train.detach().cpu().numpy()

        if (warp_inputs):
            X_warped = model._X_train.detach().cpu().numpy()
            for i, col in enumerate(dataset.copy().drop('y', axis=1).columns):
                warped_dataset[col] = X_warped[:, i]
    
        if (display_plots):
            plot_new_point(warped_dataset, new_point_model, model)

        return new_point_model, mu, std, model, max_acq


def get_new_bounds(dimensions: int, splits: int, bounds: list[tuple[float, float]]):
    if dimensions < 5:
        segment_list = []

        for (low, high) in bounds:
            width = (high - low) / splits

            segments= []
            for i in range(splits):
                l_bound = low + i*width
                h_bound = low + (i+1) * width
                segments.append((l_bound, h_bound))
            
            segment_list.append(segments)

        return list(itertools.product(*segment_list))
    return bounds

def init_model_and_get_new_point_turbo(
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
    turbo_state: TurboState,
    beta: float = 1.0, 
    warp_inputs: bool = False,
    warp_outputs: bool = False,
    display_plots: bool = False,
    
) -> Tuple[str, float, float]:
        
        model = HeteroskedasticTurboModel(
            dataset=dataset, 
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

        warped_dataset = dataset.copy()
        if (warp_outputs):
            warped_dataset['y'] = model.get_Y().detach().cpu().numpy()

        if (warp_inputs):
            X_warped = model._X_train.detach().cpu().numpy()
            for i, col in enumerate(dataset.copy().drop('y', axis=1).columns):
                warped_dataset[col] = X_warped[:, i]
    
        if (display_plots):
            plot_new_point(warped_dataset, new_point_model, model)

        return new_point_model, mu, std, model


def plot_new_point(dataset: pd.DataFrame, new_point_model: str, model: HeteroskedasticModel):
    if (len(dataset.copy().drop('y', axis=1).columns) == 2):
        PerformancePlotUtils.plot_3d_with_candidate_botorch(dataset, new_point_model, model=model._model)
    elif (len(dataset.copy().drop('y', axis=1).columns) == 3):
        PerformancePlotUtils.plot_pairwise_slices_botorch_3d(dataset, new_point_model, model=model._model)
    elif(len(dataset.copy().drop('y', axis=1).columns) == 4):
        PerformancePlotUtils.plot_pairwise_slices_botorch_4d(dataset, new_point_model, model=model._model)


if __name__ == "__main__":
    main()
