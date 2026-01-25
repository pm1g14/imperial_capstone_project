
from abc import ABC, abstractmethod
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

from domain.domain_models import PredictedValuesModel


class HomoskedasticMultiDimensionalBlackBoxModel(ABC):

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

    def get_correlation_matrix(self) -> np.ndarray:
        X = self._dataset[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']].to_numpy()
        y = self._dataset['y'].to_numpy()
        return np.corrcoef(X, rowvar=False)

    
    def suggest_next_point_to_evaluate(self, bounds: List[Tuple[float, float]], num_samples: int, warmup_steps: int, thinning: int, max_tree_depth: int, raw_samples: int, max_iter: int, csv_output_file_name: str) -> str: 
        fit_fully_bayesian_model_nuts(
            self._model, 
            num_samples=num_samples, 
            warmup_steps=warmup_steps, 
            thinning=thinning, 
            max_tree_depth=max_tree_depth, 
            disable_progbar=False
        )
        tensor_bounds = torch.from_numpy(np.array(bounds, dtype=np.double).transpose())
        best_x, max_acq = self._maximize_acquisition_function(bounds=tensor_bounds, raw_samples=raw_samples, max_iter=max_iter)
        logging.info(f"Best new exploration point found with acquisition maxing at: {max_acq}")

        formatter:str = lambda arr: "-".join(f"{x:.6f}" for x in arr)
        self._print_to_csv(num_samples, warmup_steps, thinning, max_tree_depth, raw_samples, max_iter, formatter(best_x), max_acq, csv_output_file_name)
        return formatter(best_x)


    def _maximize_acquisition_function(self, bounds: torch.Tensor, raw_samples: int, max_iter: int) -> int:
        
        acquisition_function = qLogNoisyExpectedImprovement(
            model=self._model, 
            X_baseline = self._X_train, 
            sampler=self._sampler
        )
        Xinit = gen_batch_initial_conditions(
            acq_function=acquisition_function,
            bounds=bounds,
            q=1,
            num_restarts=self._num_restarts,
            raw_samples=raw_samples,
            options={"seed":30},
        )
        candidate, acq_value = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=1,
            num_restarts=self._num_restarts,
            raw_samples=raw_samples,
            batch_initial_conditions=Xinit,
            options={"maxiter":max_iter}
        )
        print(f"acq_value: {acq_value}")
        logging.info(f"Found max acquisition value {acq_value}")
        return candidate.squeeze(0), acq_value.numpy().item()

    
    def predict(self, X: torch.Tensor) -> PredictedValuesModel:
        with torch.no_grad():
            # ensure shape (1, d)
            if X.dim() == 1:
                X = X.unsqueeze(0)

            post_f = self._model.posterior(X, observation_noise=False)
            post_y = self._model.posterior(X, observation_noise=True)
        

            # Collapse the MCMC sample batch (S, q, m) -> (1,1), else pass through
            def _collapse(post):
                mu_all, var_all = post.mean, post.variance
                if mu_all.dim() == 3:  # SAAS FB: (S, q, m)
                    mu_mix  = mu_all.mean(dim=0)                                    # (1,1)
                    var_mix = var_all.mean(dim=0) + mu_all.var(dim=0, unbiased=False)  # (1,1)
                else:  # non-FB: (q, m)
                    mu_mix, var_mix = mu_all, var_all
                mu = float(mu_mix.squeeze())
                var = float(var_mix.squeeze())
                return mu, var

            mu_f, var_f   = _collapse(post_f)  # latent (noise-free)
            mu_y, var_y   = _collapse(post_y)  # observed (includes noise)

            return PredictedValuesModel(
                mean=mu_f,               # latent mean
                variance=var_f,          # latent variance
                mean_noise=mu_y,         # observed mean (== latent mean)
                variance_noise=var_y,    # observed variance = latent var + noise var
            )

    @abstractmethod
    def _get_tensors_from_dataframe(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _print_to_csv(self, num_samples: int, warmup_steps: int, thinning: int, max_tree_depth: int, raw_samples: int, max_iter: int, best_x: np.ndarray, max_acq: float, csv_output_file_name: str) -> None:
        pass

