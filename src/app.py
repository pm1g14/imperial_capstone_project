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

def plot_4d_data(df: pd.DataFrame) -> None:
    # 1. Create a figure and add a 3D subplot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 2. Plot the data using ax.scatter
    # x1, x2, and x3 define the 3D coordinates
    # c=df['y'] defines the color based on the 'y' value
    scatter = ax.scatter(
        df['x1'], 
        df['x2'], 
        df['x3'], 
        c=df['y'], 
        cmap='viridis',
        s=100 # Optional: increase size for better visibility in 3D
    )
    
    # 3. Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3') # Crucial: setting the third axis label
    ax.set_title('Function 3: Contamination Readings in 3D Space')
    
    # 4. Add the color bar
    fig.colorbar(scatter, label='Raw Contamination Reading (y)')
    
    plt.show()
    


    
def plot_4d_data_with_candidate(df, *,
                 title="Experiments (color = y)",
                 candidate=None,            # tuple/list/ndarray like (x1,x2,x3)
                 candidate_str=None,        # e.g. "1.000000-0.000000-0.717419"
                 gp=None,                   # fitted sklearn GPR (optional)
                 cmap="viridis"):
    """
    Plots 3D inputs (x1,x2,x3) colored by y. Optionally overlays a candidate point.
    If gp is provided, the candidate is colored by predicted mean and shows ±σ.
    """
    X = df[["x1","x2","x3"]].to_numpy(float)
    y = df["y"].to_numpy(float)

    # Resolve candidate
    if candidate is None and candidate_str is not None:
        candidate = np.array([float(p) for p in candidate_str.split("-")], dtype=float)
    if candidate is not None:
        candidate = np.asarray(candidate, dtype=float).reshape(3,)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # base scatter
    vmin, vmax = np.min(y), np.max(y)
    sc = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, s=40, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax, pad=0.10)
    cb.set_label("y (your plotted target scale)")

    # overlay candidate
    if candidate is not None:
        if gp is not None:
            mu, s = gp.predict(candidate.reshape(1,-1), return_std=True)
            ax.scatter(candidate[0], candidate[1], candidate[2],
                       c=[float(mu)], cmap=cmap, vmin=vmin, vmax=vmax,
                       marker="*", s=300, edgecolor="k", linewidths=0.8,
                       label=f"candidate μ={float(mu):.3f} ± {float(s):.3f}")
        else:
            # no GP: just outline the star
            ax.scatter(candidate[0], candidate[1], candidate[2],
                       marker="*", s=300, edgecolor="k", facecolor="none",
                       linewidths=0.9, label="candidate (no prediction)")

    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title)
    if candidate is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def plot_3d_data_with_candidate(df, *,
                 title="Experiments (color = y)",
                 candidate=None,            # tuple/list/ndarray like (x1,x2,x3)
                 candidate_str=None,        # e.g. "1.000000-0.000000-0.717419"
                 gp=None,                   # fitted sklearn GPR (optional)
                 cmap="viridis"):
    """
    Plots 3D inputs (x1,x2) colored by y. Optionally overlays a candidate point.
    If gp is provided, the candidate is colored by predicted mean and shows ±σ.
    """
    X = df[["x1","x2"]].to_numpy(float)
    y = df["y"].to_numpy(float)

    # Resolve candidate
    if candidate is None and candidate_str is not None:
        candidate = np.array([float(p) for p in candidate_str.split("-")], dtype=float)
    if candidate is not None:
        candidate = np.asarray(candidate, dtype=float).reshape(3,)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # base scatter
    vmin, vmax = np.min(y), np.max(y)
    sc = ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax, pad=0.10)
    cb.set_label("y (your plotted target scale)")

    # overlay candidate
    if candidate is not None:
        if gp is not None:
            mu, s = gp.predict(candidate.reshape(1,-1), return_std=True)
            ax.scatter(candidate[0], candidate[1],
                       c=[float(mu)], cmap=cmap, vmin=vmin, vmax=vmax,
                       marker="*", s=300, edgecolor="k", linewidths=0.8,
                       label=f"candidate μ={float(mu):.3f} ± {float(s):.3f}")
        else:
            # no GP: just outline the star
            ax.scatter(candidate[0], candidate[1],
                       marker="*", s=300, edgecolor="k", facecolor="none",
                       linewidths=0.9, label="candidate (no prediction)")

    ax.set_xlabel("x1"); ax.set_ylabel("x2");
    ax.set_title(title)
    if candidate is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def plot_gp_and_acq(model, df, bounds=[(0.0, 1.0), (0.0, 1.0)],
                    acq="ucb", kappa=2.0, xi=0.0, grid=150,
                    candidate_str=None):
    # grid in [low, high]^2
    (lx, ux), (ly, uy) = bounds
    xs = np.linspace(lx, ux, grid)
    ys = np.linspace(ly, uy, grid)
    XX, YY = np.meshgrid(xs, ys)
    Xgrid = np.c_[XX.ravel(), YY.ravel()]

    # GP predictions on grid
    mu, std = model.predict(Xgrid, return_std=True)
    MU = mu.reshape(grid, grid)
    STD = std.reshape(grid, grid)

    # Acquisition
    if acq.lower() == "ucb":
        A = (MU + kappa * STD)
        acq_title = f"UCB (kappa={kappa})"
    elif acq.lower() == "ei":
        best_f = float(df["y"].max())
        s = np.maximum(STD, 1e-12)
        imp = MU - best_f - xi
        Z = imp / s
        A = imp * norm.cdf(Z) + s * norm.pdf(Z)
        A = np.maximum(A, 0.0)
        acq_title = f"EI (best_f={best_f:.4f}, xi={xi})"
    else:
        raise ValueError("acq must be 'ucb' or 'ei'.")

    # Optional candidate
    x_next = None
    if candidate_str is not None:
        x_next = np.array([float(s) for s in candidate_str.split("-")], dtype=float)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Posterior mean
    im0 = axes[0].imshow(MU, origin="lower", extent=[lx, ux, ly, uy], aspect="auto", cmap="viridis")
    axes[0].scatter(df["x1"], df["x2"], c="k", s=20, label="Obs")
    if x_next is not None:
        axes[0].scatter([x_next[0]], [x_next[1]], s=120, marker="*", c="red", edgecolor="white", linewidths=1.2)
    axes[0].set_title("GP posterior mean")
    axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
    fig.colorbar(im0, ax=axes[0], shrink=0.9, label="mean")

    # Acquisition
    im1 = axes[1].imshow(A, origin="lower", extent=[lx, ux, ly, uy], aspect="auto", cmap="magma")
    axes[1].scatter(df["x1"], df["x2"], c="w", s=10, alpha=0.7)
    if x_next is not None:
        axes[1].scatter([x_next[0]], [x_next[1]], s=120, marker="*", c="cyan", edgecolor="k", linewidths=1.0)
    axes[1].set_title(acq_title)
    axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2")
    fig.colorbar(im1, ax=axes[1], shrink=0.9, label="acq")

    plt.show()



def main() -> None:    
    run_suggest()
    # parser = argparse.ArgumentParser(description='BO with TuRBO')
    # parser.add_argument(
    #     '--mode', 
    #     type=str, 
    #     required=True, 
    #     choices=['evaluate', 'update'],
    #     help='Mode to run the BO in. Use evaluate to get a new evaluation point. Use update to update the turbo state.'
    # )
    # parser.add_argument(
    #     '--y_new', 
    #     type=float, 
    #     default=None,
    #     required=False, 
    #     help='In case of update mode, the new y value is used to compare against the max so far.'
    # )
    # parser.add_argument(
    #     '--function_number', 
    #     type=int, 
    #     required=True, 
    #     help='The function number for which we are optimizing or updating state.'
    # )
    # args = parser.parse_args()

    # if args.mode == "evaluate":
    #     run_suggest()
    # elif args.mode == "update":
    #     if args.y is None:
    #         raise ValueError("In update mode you must provide --y <value>")
    #     run_update(args.y_new, args.function_number)


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

def run_suggest() -> None:
    # initial inputs/outputs
    inputs_f1, outputs_f1 = CsvUtils.get_inputs_and_outputs(1)
    inputs_f2, outputs_f2 = CsvUtils.get_inputs_and_outputs(2)
    inputs_f3, outputs_f3 = CsvUtils.get_inputs_and_outputs(3)
    inputs_f4, outputs_f4 = CsvUtils.get_inputs_and_outputs(4)
    inputs_f5, outputs_f5 = CsvUtils.get_inputs_and_outputs(5)
    inputs_f6, outputs_f6 = CsvUtils.get_inputs_and_outputs(6)
    inputs_f7, outputs_f7 = CsvUtils.get_inputs_and_outputs(7)
    inputs_f8, outputs_f8 = CsvUtils.get_inputs_and_outputs(8)
 

    # values from submissions
    submissions_f1, sub_outputs_f1 = CsvUtils.get_submission_values(1)
    submissions_f2, sub_outputs_f2 = CsvUtils.get_submission_values(2)
    submissions_f3, sub_outputs_f3 = CsvUtils.get_submission_values(3)
    submissions_f4, sub_outputs_f4 = CsvUtils.get_submission_values(4)
    submissions_f5, sub_outputs_f5 = CsvUtils.get_submission_values(5)
    submissions_f6, sub_outputs_f6 = CsvUtils.get_submission_values(6)
    submissions_f7, sub_outputs_f7 = CsvUtils.get_submission_values(7)
    submissions_f8, sub_outputs_f8 = CsvUtils.get_submission_values(8)
    

    PerformancePlotUtils.plot_performance(
        ModelEvaluator.calculate_performance_change_for_all_functions().dropna(axis=0)
    )

    # concatenated inputs
    inputs_f1 = np.concatenate((inputs_f1, submissions_f1), axis=0)
    inputs_f2 = np.concatenate((inputs_f2, submissions_f2), axis=0)
    inputs_f3 = np.concatenate((inputs_f3, submissions_f3), axis=0)
    inputs_f4 = np.concatenate((inputs_f4, submissions_f4), axis=0)
    inputs_f5 = np.concatenate((inputs_f5, submissions_f5), axis=0)
    inputs_f6 = np.concatenate((inputs_f6, submissions_f6), axis=0)
    inputs_f7 = np.concatenate((inputs_f7, submissions_f7), axis=0)
    inputs_f8 = np.concatenate((inputs_f8, submissions_f8), axis=0)

    # # concatenated outputs
    outputs_f1 = np.concatenate((outputs_f1, sub_outputs_f1), axis=0)
    outputs_f2 = np.concatenate((outputs_f2, sub_outputs_f2), axis=0)
    outputs_f3 = np.concatenate((outputs_f3, sub_outputs_f3), axis=0)
    outputs_f4 = np.concatenate((outputs_f4, sub_outputs_f4), axis=0)
    outputs_f5 = np.concatenate((outputs_f5, sub_outputs_f5), axis=0)
    outputs_f6 = np.concatenate((outputs_f6, sub_outputs_f6), axis=0)
    outputs_f7 = np.concatenate((outputs_f7, sub_outputs_f7), axis=0)
    outputs_f8 = np.concatenate((outputs_f8, sub_outputs_f8), axis=0)
    
    # combined results
    all_values_f1 = np.concatenate((inputs_f1, outputs_f1), axis=1)
    all_values_f2 = np.concatenate((inputs_f2, outputs_f2), axis=1)
    all_values_f3 = np.concatenate((inputs_f3, outputs_f3), axis=1)
    all_values_f4 = np.concatenate((inputs_f4, outputs_f4), axis=1)
    all_values_f5 = np.concatenate((inputs_f5, outputs_f5), axis=1)
    all_values_f6 = np.concatenate((inputs_f6, outputs_f6), axis=1)
    all_values_f7 = np.concatenate((inputs_f7, outputs_f7), axis=1)
    all_values_f8 = np.concatenate((inputs_f8, outputs_f8), axis=1)

    df1 = pd.DataFrame(all_values_f1, columns=['x1', 'x2', 'y'])
    df2 = pd.DataFrame(all_values_f2, columns=['x1', 'x2', 'y'])
    df3 = pd.DataFrame(all_values_f3, columns=['x1', 'x2', 'x3', 'y'])
    df4 = pd.DataFrame(all_values_f4, columns=['x1', 'x2', 'x3', 'x4', 'y'])
    df5 = pd.DataFrame(all_values_f5, columns=['x1', 'x2', 'x3', 'x4', 'y'])
    df6 = pd.DataFrame(all_values_f6, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
    df7 = pd.DataFrame(all_values_f7, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y'])
    df8 = pd.DataFrame(all_values_f8, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y'])


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

    

    # model = NNSurrogateModel(df6, hidden_activation='gelu', output_activation = None)
    # y_pred = model.fit_and_predict(learning_rate=1e-2, epochs=1000)
    # y_true = df6['y'].to_numpy()
    # mse = np.mean((y_true - y_pred)**2)
    # print(mse)
    # mse6 = predict_with_nn_model(df6, hidden_activation='gelu', output_activation = None)
    # print(mse6)
    # model = NNSurrogateModel(df8, hidden_activation='silu', output_activation = None)
    
    # y_pred = model.fit_and_predict(learning_rate=1e-2, epochs=1000)
    # feat_influence, rank, point_steepness= model.summarize_gradients(df8.copy().drop('y', axis=1).to_numpy())

    # print(feat_influence)
    # print(rank)
    # print(point_steepness)
    

    # mse6 = predict_with_nn_model(df6, hidden_activation='silu', output_activation = None)
    # print(mse6)
    # mse5 = predict_with_nn_model(df5, hidden_activation='silu', output_activation = None)
    # print(mse5)
    # mse4 = predict_with_nn_model(df4, hidden_activation='silu', output_activation = None)
    # print(mse4)
    # mse3 = predict_with_nn_model(df3, hidden_activation='silu', output_activation = None)
    # print(mse3)
    
    # model7 = HeteroskedasticModel(dataset=df7,   nu_mean = 1.5,
    #     nu_noise = 2.5, n_restarts=40)
    
    # model5 = HeteroskedasticModel(dataset=df5,   nu_mean = 1.5,
    #     nu_noise = 2.5, n_restarts=40)
    
    # model4 = HeteroskedasticModel(dataset=df4, nu_mean=0.5, nu_noise = 1.5, n_restarts=40)
    # model3 = HeteroskedasticModel(dataset=df3, nu_mean=1.5, nu_noise = 2.5, n_restarts=40)
    # model2 = HeteroskedasticModel(dataset=df2, nu_mean=0.5, nu_noise = 0.5, n_restarts=40)
    # model1 = HeteroskedasticModel(dataset=df1, nu_mean=1.5, nu_noise = 2.5, n_restarts=40)
    # model1 = matern(dataset=df1, n_restarts=100)
    # new_point_model_1 = model1.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0)]
    # )
    # print(new_point_model_1)
    # model_eval_matern_1 = matern(length_scale=1.0, length_scale_bounds=[(1e-2, 0.5)] * 2, dataset=df1, n_restarts=40)
    # model_eval_saas_1 = saas(dataset=df1, n_restarts=40)
    # model_eval_matern_1.suggest_next_point_to_evaluate(bounds=[(0.0, 1.0), (0.0, 1.0)])
    # model_eval_saas_1.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0)],
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=1024,
    #     max_iter=200
    # )
    # eval_matern_1 =model_eval_matern_1.predict(submissions_f1)
    # eval_saas_1 = model_eval_saas_1.predict(torch.from_numpy(submissions_f1.astype(np.float32, copy=False)))
    # y_actual = sub_outputs_f1.reshape(-1, 1)
    # metrics_matern_1 = ModelEvaluator.point_metrics(y_actual, eval_matern_1.mean_noise, eval_matern_1.variance_noise)
    # metrics_saas_1 = ModelEvaluator.point_metrics(y_actual, eval_saas_1.mean_noise, eval_saas_1.variance_noise)
    # print(metrics_matern_1.__dict__)
    # print(metrics_saas_1.__dict__)

    # model1 = FieldContaminationModel(length_scale=1.0, length_scale_bounds=[(1e-2, 0.5)] * 2, dataset=df1, n_restarts=40)
    # model2 = MysteryMLModel(dataset=df2, n_restarts=50)
    # new_point_model_2 = model2.suggest_next_point_to_evaluate_ucb(bounds=[(0.0, 1.0), (0.0, 1.0)])
    # model3 = DrugDiscoveryModel(dataset=df3, n_restarts=40)
    # model4 = ProductsAcrossWarehouseModel(dataset=df4, n_restarts=40)
    # model5 = ChemicalProcessModel(dataset=df5, n_restarts=40)
    # plot_2D_output_values(df2)
    # model6 = CakeRecipeModel(dataset=df6, n_restarts=40)
    # model7 = HyperparameterTuningModel(dataset=df7, n_restarts=40)
    # model8 = BlackBoxOptimizerModel(dataset=df8, n_restarts=40)
    
    # new_point_model_2 = model2.suggest_next_point_to_evaluate(bounds=[(0.0, 1.0), (0.0, 1.0)])
    # new_point_model_3 = model3.suggest_next_point_to_evaluate(bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    # new_point_model_4 = model4.suggest_next_point_to_evaluate(bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    # new_point_model_5 = model5.suggest_next_point_to_evaluate(bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    # new_point_model_6 = model6.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    # )
    # new_point_model_4 = model4.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=8,
    #     raw_samples=512,
    #     max_iter=200
    # )
    # new_point_model_3 = model3.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], 
    #     num_samples=256,
    #     warmup_steps=512, 
    #     thinning=16, 
    #     max_tree_depth=6, 
    #     raw_samples=512, 
    #     max_iter=200
    # )
    
    # new_point_model_2 = model2.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0)], 
    #     num_samples=256,
    #     warmup_steps=512, 
    #     thinning=16, 
    #     max_tree_depth=6, 
    #     raw_samples=512, 
    #     max_iter=200,
    #     acquisition_function=acquisition_function_nei
    # )
    # new_point_model_3, _, _, model = init_model_and_get_new_point_turbo(
    #     dataset=df3,
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     turbo_state=TurboState(dim=3),
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     acquisition_function_str='ucb',
    #     beta=2.0,
    #     csv_output_file_name='function_3',
    # )
  
    # new_point_model_3, _, _, model, _ = init_model_and_get_new_point(
    #     dataset=df3,
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     acquisition_function_str='nei',
    #     beta=2.0,
    #     csv_output_file_name='function_3',
    #     display_plots=True
    # )
    # new_point_model_4, _, _, model,_ = init_model_and_get_new_point(
    #     dataset=df4,
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     nu_mean=0.5,
    #     nu_noise=1.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     acquisition_function_str='nei',
    #     beta=2.0,
    #     csv_output_file_name='function_4',
    #     display_plots=True
    # )
    # new_point_model_5, _, _, model, _ = init_model_and_get_new_point(
    #     dataset=df5,
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     warp_inputs=False,
    #     warp_outputs=False,
    #     max_iter=200,
    #     acquisition_function_str='nei',
    #     beta=5.0,
    #     csv_output_file_name='function_5',
    #     display_plots=True
    # )
    
    # new_point_model_6, _, _, model,_ = init_model_and_get_new_point(
    #     dataset=df6,
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     warp_outputs=True,
    #     warp_inputs=True,
    #     raw_samples=512,
    #     max_iter=200,
    #     acquisition_function_str='nei',
    #     beta=5.0,
    #     csv_output_file_name='function_6'
    # )
    # new_point_model_7, _, _, model,_ = init_model_and_get_new_point(
    #     dataset=df7,
    #     bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),(0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=51,
    #     max_iter=200,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     acquisition_function_str='nei',
    #     beta=5.0,
    #     csv_output_file_name='function_7'
    # )

    
    # plot_new_point(df1, new_point_model_1, model)
    # data_scaled = StandardScaler().fit_transform(df8)
    # data_scaled = pd.DataFrame(data_scaled, columns=df8.columns)
    # pca = PCA(n_components=8).fit(data_scaled)
    # exp_var = pca.explained_variance_ratio_
    # cum_var = np.cumsum(exp_var)

    # # 4. Create the Plot
    # plt.figure(figsize=(10, 6))

    # # Individual Variance (Bars)
    # plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.6, color='b', label='Individual Variance')

    # # Cumulative Variance (Line)
    # plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', color='r', label='Cumulative Variance')

    # plt.ylabel('Explained Variance Ratio')
    # plt.xlabel('Principal Component Index')
    # plt.title('Scree Plot for Portfolio Features')
    # plt.legend(loc='best')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    
    # new_point_model_8, _, _, model, _ = init_model_and_get_new_point(
    #     dataset=df8,
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=60,
    #     num_samples=512,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=2048,
    #     max_iter=200,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     acquisition_function_str='nei',
    #     beta=3.0,
    #     csv_output_file_name='function_8'
    # )

    new_point_model_2, _, _, model, _ = init_model_and_get_new_point(
        dataset=df2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
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
    # new_point_model_1, mu, std, model, _ = init_model_and_get_new_point(
    #     dataset=df1,
    #     bounds=[(0.0, 1.0), (0.0, 1.0)],
    #     nu_mean=1.5,
    #     nu_noise=2.5,
    #     n_restarts=40,
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200,
    #     acquisition_function_str=None,
    #     beta=5.0,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     csv_output_file_name='function_1',
    #     display_plots=True
    # )
    print(new_point_model_2)
    residuals = residuals_over_trials(
        model.get_Y().detach().cpu().numpy(), 
        model.get_X().detach().cpu().numpy(), 
        model
    )

    PerformancePlotUtils.plot_outputs_vs_distributions(np.array(residuals))
    # y_pred_new, feat_influence, rank, steepness = predict_y_at_evaluation_point(new_point_model_8,df8, 'silu', None)
    # print(y_pred_new)
    # print(feat_influence)
    # print(rank)
    # print(steepness)
    # mse6 = predict_with_nn_model(df6, hidden_activation='silu', output_activation = None)
    # print(mse6)
    # print(new_point_model_1)
    # new_point_model_5 = model5.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200
    # )
    # new_point_model_6 = model6.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200
    # )
    # new_point_model_7 = model7.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=1024,
    #     max_iter=200
    # )
    # print(new_point_model_5)
    # new_point_model_8 = model8.suggest_next_point_to_evaluate(
    #     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    #     num_samples=256,
    #     warmup_steps=512,
    #     thinning=20,
    #     max_tree_depth=10,
    #     raw_samples=1024,
    #     max_iter=200
    # )
    # print(new_point_model_3)
    # logging.info(f"New point: {new_point_model_1}")


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


def predict_with_nn_classifier_model(dataset: pd.DataFrame, hidden_activation: str, output_activation: str, threshold: float) -> Tuple[float, float]:
    cross_v_tries = dataset.shape[0]
    y_test_preds = []
    y_test_trues = []
    for i in range(cross_v_tries):
        train_df = dataset.drop(i)
        test_df = dataset.iloc[[i]]
        model = NNClassifierModel(train_df, hidden_activation=hidden_activation, output_activation = output_activation)
        y_pred = model.fit_and_predict(learning_rate=1e-2, epochs=1000)
        y_test_pred = model._model.predict(test_df.drop('y', axis=1).to_numpy(), verbose=0)
        y_test_true = test_df['y'].values[0]
        y_test_trues.append(y_test_true)
        y_test_preds.append(1 if y_test_pred.squeeze(-1)[0] >= threshold else 0)
        
    acc = accuracy_score(y_test_trues, y_test_preds)
    cm  = confusion_matrix(y_test_trues, y_test_preds, labels=[0,1])
    return acc, cm

def predict_with_nn_model(dataset: pd.DataFrame, hidden_activation: str, output_activation: str) -> Tuple[float, float]:
    cross_v_tries = dataset.shape[0]
    y_test_preds = []
    y_test_trues = []
    for i in range(cross_v_tries):
        train_df = dataset.drop(i)
        test_df = dataset.iloc[[i]]
        model = NNSurrogateModel(train_df, hidden_activation=hidden_activation, output_activation = output_activation)
        y_pred = model.fit_and_predict(learning_rate=1e-2, epochs=1000)
        y_test_pred = model._model.predict(test_df.drop('y', axis=1).to_numpy(), verbose=0)
        y_test_true = test_df['y'].values[0]
        y_test_trues.append(y_test_true)
        y_test_preds.append(y_test_pred)
    mse = np.mean((np.array(y_test_trues) - np.array(y_test_preds))**2)
    return mse

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
