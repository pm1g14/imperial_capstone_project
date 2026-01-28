import json
import pandas as pd
import os
import numpy as np
from typing import Any, Dict, List
from pathlib import Path
from typing import Tuple
class CsvUtils:

    @staticmethod
    def to_csv(headers: List[str], function_identifier: str, *args):
        project_root = "/home/pmavrothalassitis/development/capstone_imperial/resources"
        input_path = f"{project_root}/{function_identifier}/historical_results.csv"

        file_exists = os.path.exists(input_path)
        file_is_empty = not file_exists or os.path.getsize(input_path) == 0
        write_mode = "w" if file_is_empty else "a"

        if file_is_empty:
            df = pd.DataFrame(columns=headers, data=np.array([args]))
            df.to_csv(input_path, index=False, header=True, mode=write_mode)
        else:
            df = pd.DataFrame(columns=headers, data=np.array([args]))
            df.to_csv(input_path, index=False, header=False, mode=write_mode)


    @staticmethod
    def get_inputs_and_outputs(function_number: int, input_p: str | None = None, output_p: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
        project_root = Path(__file__).resolve().parent.parent.parent

        input_path = input_p or project_root / "resources" / f"function_{function_number}" / "initial_inputs.npy"
        output_path = output_p or project_root / "resources" / f"function_{function_number}" / "initial_outputs.npy"
        inputs = np.load(input_path)
        outputs = np.load(output_path).reshape(-1, 1)
        return inputs, outputs

    @staticmethod
    def get_submission_values(function_number: int, submission_path: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = submission_path or project_root / "resources" / f"function_{function_number}" / "submissions.csv"
        df = pd.read_csv(path)
        outputs = df['y'].to_numpy().reshape(-1, 1)
        inputs = df.drop(columns=['y']).to_numpy()
        return inputs, outputs

