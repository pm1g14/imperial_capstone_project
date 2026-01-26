#!/usr/bin/env python3
from typing import Annotated
import typer
import itertools
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
import torch
from evaluate.evaluate import ModelEvaluator
from models.function1 import FieldContaminationModel as saas
from models.function1_matern import FieldContaminationModel as matern
from domain.domain_models import TurboState
from services.turbo_service import TurboService
from models.heteroskedastic_contract import HeteroskedasticContract
import numpy as np
from scipy.stats import norm
from utils.input_utils import InputUtils
from utils.csv_utils import CsvUtils
import argparse

from services.evaluation_service_factory import EvaluationServiceFactory

app = typer.Typer()

PathArg = Annotated[str, typer.Option(help="The path to the input dataset. Must be an .npy file.")]
OutPathArg = Annotated[str, typer.Option(help="The path to the evaluation outputs. Must be an .npy file.")]
SubmissionPathArg = Annotated[str, typer.Option(help="The path to the dataset with the trial submissions. Must be an .csv file.")]
TrialArg = Annotated[int, typer.Option(help="The trial number for the BO.")]

@app.command()
def evaluate(
    input_dataset_path: PathArg,
    trial_no: TrialArg,
    output_dataset_path: OutPathArg,
    submission_path: SubmissionPathArg,
    function_number: int,
    dimensions: int,
    total_budget: int =13
):
    columns = [f'x{i}' for i in range(1, dimensions+1)] + ['y']
    dataframe = InputUtils.convert_initial_data_to_dataframe(function_number, columns, input_dataset_path, output_dataset_path, submission_path)

    trust_region_flg = ModelEvaluator.should_switch_to_trust_region(trial_no, total_budget, dataframe)

    evaluation_service = EvaluationServiceFactory.get_evaluation_service(trust_region_flg, dimensions, dataframe, total_budget, trial_no)
    best_new_X = evaluation_service.run_suggest(output_dataset_path, function_number)
    print(f"Next suggested evaluation point is: {best_new_X}")
    sys.exit(0)


@app.command()
def update(
    y_new: float,
    function_number: int,
    dimensions: int,
    trial_no: TrialArg,
    total_budget: int =13
):
    turbo_service = TurboService()
    turbo_state = turbo_service.load_turbo_state(function_number)

    if not turbo_state:
        turbo_state = TurboState(dim=dimensions)

    turbo_service.update_turbo_state(
        turbo_state=turbo_state,
        function_number=function_number,
        y_new = y_new
    )
    logging.info(f"Turbo state updated for function {function_number}")
    sys.exit(0)





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


if __name__ == "__main__":
    app()
