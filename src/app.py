#!/usr/bin/env python3
from typing import Annotated
import typer
import itertools
import logging
import sys
import numpy as np
import pandas as pd
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
from utils.input_utils import InputUtils
from utils.csv_utils import CsvUtils
import argparse

from services.evaluation_service_factory import EvaluationServiceFactory

app = typer.Typer()

PathArg = Annotated[str | None, typer.Option(help="The path to the input dataset. Must be an .npy file.")]
OutPathArg = Annotated[str | None, typer.Option(help="The path to the evaluation outputs. Must be an .npy file.")]
SubmissionPathArg = Annotated[str | None, typer.Option(help="The path to the dataset with the trial submissions. Must be an .csv file.")]
TrialArg = Annotated[int, typer.Option(help="The trial number for the BO.")]

@app.command()
def evaluate(
    function_number: int,
    dimensions: int,
    total_budget: int,
    trial_no: TrialArg,
    input_dataset_path: PathArg = None,
    output_dataset_path: OutPathArg = None,
    submission_path: SubmissionPathArg = None
):
    columns = [f'x{i}' for i in range(1, dimensions+1)] + ['y']
    dataframe = InputUtils.convert_initial_data_to_dataframe(function_number, columns, input_dataset_path, output_dataset_path, submission_path)

    trust_region_flg = ModelEvaluator.should_switch_to_trust_region(trial_no, total_budget, dataframe)

    evaluation_service = EvaluationServiceFactory.get_evaluation_service(trust_region_flg, dimensions, dataframe, total_budget, trial_no, function_number)
    best_new_X = evaluation_service.run_suggest(function_number)
    print(f"Next suggested evaluation point is: {best_new_X}")
    sys.exit(0)


@app.command()
def update(
    y_new: float,
    function_number: int,
    dimensions: int
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




if __name__ == "__main__":
    app()
