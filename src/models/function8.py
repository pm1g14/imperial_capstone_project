
from gc import disable
from typing import Tuple
import logging
from typing import Callable, List
from botorch.models.multitask import Standardize
from botorch.optim import optimize_acqf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.base import clone
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim.initializers import gen_batch_initial_conditions

from botorch.sampling.normal import SobolQMCNormalSampler
import torch

from models.homoskedastic_multi_d_bb_model import HomoskedasticMultiDimensionalBlackBoxModel
from utils.csv_utils import CsvUtils

torch.manual_seed(30)

class BlackBoxOptimizerModel(HomoskedasticMultiDimensionalBlackBoxModel):
    
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
            outcome_transform=Standardize(m=1)
        )
        self._sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=30)
        self._num_restarts = n_restarts
    



    def _get_tensors_from_dataframe(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = np.ascontiguousarray(self._dataset[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']].to_numpy(), dtype=np.double)
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

    