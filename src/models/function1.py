from decimal import Decimal, DecimalException
from typing import List, Tuple
from botorch.fit import SaasFullyBayesianSingleTaskGP
from botorch.models.multitask import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize
import torch
from models.acquisition.acquisition_functions import AcquisitionFunctions
from models.homoskedastic_multi_d_bb_model import HomoskedasticMultiDimensionalBlackBoxModel
from utils.csv_utils import CsvUtils

class FieldContaminationModel(HomoskedasticMultiDimensionalBlackBoxModel):


    def __init__(
        self, 
        dataset: pd.DataFrame,
        n_restarts: int = 20
    ):
        self._dataset = dataset    
        self._X_train, self._Y_train = self._get_tensors_from_dataframe()
        self._model = SaasFullyBayesianSingleTaskGP(
            self._X_train,
            self._Y_train,
            outcome_transform=None
        )
        self._sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=30)
        self._num_restarts = n_restarts
    


    def _get_tensors_from_dataframe(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = np.ascontiguousarray(self._dataset[['x1', 'x2']].to_numpy(), dtype=np.double)
        Y_np = np.ascontiguousarray(self._dataset['y'].to_numpy().reshape(-1, 1), dtype=np.double)
        train_X = torch.from_numpy(X_np)
        train_Y = torch.from_numpy(Y_np)
        return train_X, train_Y



    def _print_to_csv(
        self, 
        num_samples: int, 
        warmup_steps: int, 
        thinning: int, 
        max_tree_depth: int, 
        raw_samples: int, 
        max_iter: int, 
        best_x: np.ndarray, 
        max_acq: float,
        csv_output_file_name: str
    ) -> None:
        CsvUtils.to_csv(
            ['num_samples', 'warmup_steps', 'thinning', 'max_tree_depth', 'num_restarts', 'raw_samples', 'max_iter', 'max_acquisition', 'best_x'], 
            csv_output_file_name,
            num_samples, 
            warmup_steps, 
            thinning, 
            max_tree_depth,
            self._num_restarts,
            raw_samples,
            max_iter, 
            max_acq,
            best_x
        )

    
   