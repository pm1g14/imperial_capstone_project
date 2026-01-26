import numpy as np
import pandas as pd
from typing import List
from src.utils.csv_utils import CsvUtils


class InputUtils:


    @staticmethod
    def get_dataframe_for_function(function_number: int, columns: List[str], input_path: str | None = None, output_path: str | None = None, submission_path: str | None = None) -> pd.DataFrame:
        inputs_f, outputs_f = CsvUtils.get_inputs_and_outputs(function_number, input_path, output_path)
        submissions_f, sub_outputs_f = CsvUtils.get_submission_values(function_number, submission_path)
        inputs_f = np.concatenate((inputs_f, submissions_f), axis=0)
       
        outputs_f = np.concatenate((outputs_f, sub_outputs_f), axis=0)
        all_values_f = np.concatenate((inputs_f, outputs_f), axis=1)
        return pd.DataFrame(all_values_f, columns=columns)