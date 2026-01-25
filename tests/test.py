from dataclasses import dataclass
import os, sys



ROOT = os.path.dirname(os.path.dirname(__file__))
# SRC = os.path.join(ROOT, "src")

# if SRC not in sys.path:
#     sys.path.insert(0, SRC)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import optuna
from typing import List, Tuple
from test_utils.input_utils import InputUtils
import torch
from tests.heteroskedastic_model_test import HeteroskedasticModel
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance
from hpo_bench_eval import HpoBenchEval
from evaluation import Evaluation2D
from src.utils.csv_utils import CsvUtils
import numpy as np
import pandas as pd

def run_test():
    # inputs_f1, outputs_f1 = CsvUtils.get_inputs_and_outputs(1)
    # submissions_f1, sub_outputs_f1 = CsvUtils.get_submission_values(1)
    # inputs_f1 = np.concatenate((inputs_f1, submissions_f1), axis=0)
    # outputs_f1 = np.concatenate((outputs_f1, sub_outputs_f1), axis=0)
    # all_values_f1 = np.concatenate((inputs_f1, outputs_f1), axis=1)
    # df1 = pd.DataFrame(all_values_f1, columns=['x1', 'x2', 'y'])

    # # evaluation2D = Evaluation2D()
    # # evaluation2D.evaluate(dataset=df1, no_of_trials=13)
    # hp_names = ['init_lr', 'n_units_1']
    # hpo_bench_eval = HpoBenchEval(hp_names=hp_names)
    # hpo_bench_eval.evaluate(dataset=df1, no_of_trials=13, output_file_name="bo_results_6_trial_f1.png")
    # hp_names = ['init_lr', 'n_units_1']
    # inputs_f2, outputs_f2 = CsvUtils.get_inputs_and_outputs(2)
    # submissions_f2, sub_outputs_f2 = CsvUtils.get_submission_values(2)
    # inputs_f2 = np.concatenate((inputs_f2, submissions_f2), axis=0)
    # outputs_f2 = np.concatenate((outputs_f2, sub_outputs_f2), axis=0)
    # all_values_f2 = np.concatenate((inputs_f2, outputs_f2), axis=1)
    # df2 = pd.DataFrame(all_values_f2, columns=['x1', 'x2', 'y'])
    # hpo_bench_eval = HpoBenchEval(hp_names=hp_names)
    # hpo_bench_eval.evaluate(dataset=df2, no_of_trials=13, output_file_name="bo_results_6_trial_f2.png")
    # hp_names = ['init_lr', 'n_units_1', 'n_units_2']
    # inputs_f3, outputs_f3 = CsvUtils.get_inputs_and_outputs(3)
    # submissions_f3, sub_outputs_f3 = CsvUtils.get_submission_values(3)
    # inputs_f3 = np.concatenate((inputs_f3, submissions_f3), axis=0)
    # outputs_f3 = np.concatenate((outputs_f3, sub_outputs_f3), axis=0)
    # all_values_f3 = np.concatenate((inputs_f3, outputs_f3), axis=1)
    # df3 = pd.DataFrame(all_values_f3, columns=['x1', 'x2', 'x3', 'y'])
    # hpo_bench_eval = HpoBenchEval(hp_names=hp_names)
    # hpo_bench_eval.evaluate(dataset=df3, no_of_trials=13, output_file_name="bo_results_6_trial_f3.png")

    # hp_names = ['init_lr', 'n_units_1', 'n_units_2', 'batch_size']
    # inputs_f4, outputs_f4 = CsvUtils.get_inputs_and_outputs(4)
    # submissions_f4, sub_outputs_f4 = CsvUtils.get_submission_values(4)
    # inputs_f4 = np.concatenate((inputs_f4, submissions_f4), axis=0)
   
    # outputs_f4 = np.concatenate((outputs_f4, sub_outputs_f4), axis=0)
    # all_values_f4 = np.concatenate((inputs_f4, outputs_f4), axis=1)
    # df4 = pd.DataFrame(all_values_f4, columns=['x1', 'x2', 'x3', 'x4', 'y'])

    # hpo_bench_eval = HpoBenchEval(hp_names=hp_names)
    # hpo_bench_eval.evaluate(
    #     dataset=df4, 
    #     no_of_trials=13, 
    #     output_file_name="bo_results_6_trial_f4.png", 
    #     nu_mean=0.5,
    #     nu_noise=1.5,
    #     n_restarts=40,
    #     num_samples=512,
    #     warmup_steps=512,
    #     thinning=16,
    #     max_tree_depth=6,
    #     raw_samples=512,
    #     max_iter=200,
    #     warp_inputs=True,
    #     warp_outputs=True,
    #     csv_output_file_name='function_4'
    # )
    hp_names = ['init_lr', 'n_units_1', 'n_units_2', 'batch_size']

    hpo_bench_eval = HpoBenchEval(hp_names=hp_names)
    dataset = InputUtils.get_dataframe_for_function(5, ['x1', 'x2', 'x3', 'x4', 'y'])
    breakpoint()
    Xs = dataset[['x1', 'x2', 'x3', 'x4']].to_numpy()
    Ys = np.zeros(len(Xs))
    for i, X in enumerate(Xs):
        y = hpo_bench_eval.evaluate_at_point(X)
        Ys[i] = y

    
    new_dataset = pd.DataFrame(np.concatenate((Xs, Ys.reshape(-1, 1)), axis=1), columns=['x1', 'x2', 'x3', 'x4', 'y'])    
    breakpoint()
    best_y_value = hpo_bench_eval.evaluate(
        dataset=new_dataset, 
        no_of_trials=13, 
        output_file_name="bo_results_6_trial_f5_turbo.png", 
        nu_mean=1.5,
        nu_noise=2.5,
        n_restarts=40,
        num_samples=512,
        warmup_steps=512,
        thinning=16,
        max_tree_depth=6,
        raw_samples=512,
        max_iter=200,
        warp_inputs=True,
        warp_outputs=True,
        csv_output_file_name='function_5'
    )



def run_tuning():
    pass
    # tune_bo = TuneBO(function_number=5, hp_names=['init_lr', 'n_units_1', 'n_units_2', 'batch_size'])
    # study = optuna.create_study(direction="maximize", study_name="function_5")
    # study.optimize(tune_bo.optuna_objective, n_trials=20)

    # print("Number of trials: ", len(study.trials))
    # print("Best trial: ", study.best_value)
    # print("Best hyperparameters:")
    # for k, v in study.best_trial.params.items():
    #     print(f"  {k}: {v}")


if __name__ == "__main__":
    run_test()
    # run_tuning()
