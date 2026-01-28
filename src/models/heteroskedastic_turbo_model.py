

from abc import ABC
import logging
import random
from typing import List, Tuple
from botorch import fit_gpytorch_mll
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound, qLogNoisyExpectedImprovement
from botorch.fit import SingleTaskGP
from botorch.optim import gen_batch_initial_conditions, optimize_acqf
from botorch.posteriors import GPyTorchPosterior, Posterior
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
import pandas as pd
from botorch.models.transforms.outcome import Standardize
from pyro.distributions import MultivariateNormal
from scipy.stats import mvn
from gpytorch.constraints import Interval
import torch
import numpy as np
from typing import Callable
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from torch.nn import Parameter

from domain.domain_models import TurboState, WarpConfig
from models.heteroskedastic_contract import HeteroskedasticContract
from utils.csv_utils import CsvUtils
from utils.matern_bounds_calculator_utils import MaternBoundsCalculatorUtils
from utils.warp_utils import WarpUtils

# torch.manual_seed(30)
# random.seed(30)
# np.random.seed(30)


class HeteroskedasticTurboModel(HeteroskedasticContract):

    def __init__(
        self, 
        dataset: pd.DataFrame, 
        nu_mean: float, 
        nu_noise: float, 
        turbo_state: TurboState,
        n_restarts: int = 20, 
        device: str = 'cpu', 
        dtype=torch.float64, 
        warp_inputs:bool = False, 
        warp_outputs:bool = False,
        warp_config: WarpConfig | None = None
    ):
        self._dataset = dataset
        self._X_train, self._Y_train = self._get_tensors_from_dataframe()
        
        self._warp_outputs = warp_outputs
        self._y_warp = None

        if warp_outputs:
            self._y_warp = WarpedOutputTransformation(y=self._Y_train, lam_low=warp_config.lam_low, lam_high=warp_config.lam_high)

        self._model_mean = self._build_mean_model(nu_mean=nu_mean, warp_inputs=warp_inputs)
        self._model = self._fit_two_models(nu_noise = nu_noise, eps=1e-8, num_iters=2)
        # self._sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=30)
        self._sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self._num_restarts = n_restarts
        self._turbo_state = turbo_state

    def get_model(self) -> Model:
        return self._model

    def get_X(self) -> torch.Tensor:
        return self._X_train

    def get_Y(self) -> torch.Tensor:
        return self._y_warp(self._Y_train) if self._warp_outputs else self._Y_train

    def get_sampler(self) -> SobolQMCNormalSampler:
        return self._sampler


    def _build_mean_model(self, nu_mean: float, warp_inputs: bool) -> SingleTaskGP:
            Y_warped = self._y_warp(self._Y_train) if self._warp_outputs else self._Y_train
            lik = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            )
            if warp_inputs:
                alpha = Parameter(torch.ones(self._X_train.shape[1]))
                beta = Parameter(torch.ones(self._X_train.shape[1]))
                
                return WarpedTaskGP(
                    self._X_train, 
                    Y_warped, 
                    alpha, 
                    beta,
                    likelihood=lik,
                    outcome_transform=Standardize(m=1),
                    covar_module=self._ls_make_matern_covar(d=self._X_train.shape[1], nu=nu_mean)
                ).double()
            else:
                return SingleTaskGP(
                    self._X_train, 
                    Y_warped, 
                    likelihood=lik,
                    outcome_transform=Standardize(m=1),
                    covar_module=self._ls_make_matern_covar(d=self._X_train.shape[1], nu=nu_mean)
                ).double()

    def _ls_make_matern_covar(self, d: int, nu: float) -> ScaleKernel:
            # ls_prior =None
            # os_prior = None
            ls_init, ls_bounds = self._ls_init_and_bounds_from_spacing(self._dataset.iloc[:, :-1].to_numpy(), nu=nu)
            lo_tensor = torch.tensor([b[0] for b in ls_bounds], dtype=torch.double)
            hi_tensor = torch.tensor([b[1] for b in ls_bounds], dtype=torch.double)

            ls_constraint= Interval(lower_bound=lo_tensor, upper_bound=hi_tensor)
            base = MaternKernel(
                nu=nu, 
                # lengthscale_prior=ls_prior, 
                lengthscale_constraint=ls_constraint, 
                ard_num_dims=d
            )
            with torch.no_grad():
                ls_init_t = torch.tensor(ls_init, dtype=torch.double)
                # ensure init lies inside the bounds
                ls_init_t = torch.max(torch.min(ls_init_t, hi_tensor), lo_tensor)
                base.lengthscale = ls_init_t.view(1, 1, -1)
            return ScaleKernel(
                base_kernel=base,
                # outputscale_prior=os_prior,
            )

            

    def _fit_two_models(self, nu_noise:float, eps: float, num_iters: int = 1, warp_inputs: bool = False):
            self._fit_mean_model_with_optional_warp()
            print(f"lengthscale: {self._model_mean.covar_module.base_kernel.lengthscale.detach().numpy()}")

            for i in range(num_iters):
                with torch.no_grad():
                    r = self._calculate_residuals(eps=eps)
                lik = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
                )
                noise_model = SingleTaskGP(
                    self._X_train, 
                    r, 
                    likelihood=lik,
                    outcome_transform=Standardize(m=1),
                    covar_module=self._ls_make_matern_covar(d=self._X_train.shape[1], nu=nu_noise)
                ).double() if warp_inputs is False else WarpedTaskGP(
                    self._X_train, 
                    r, 
                    self._model_mean._alpha, 
                    self._model_mean._beta,
                    outcome_transform=Standardize(m=1),
                    covar_module=self._ls_make_matern_covar(d=self._X_train.shape[1], nu=nu_noise)
                ).double()
                mll_noise = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
                noise_model.train(); noise_model.likelihood.train()
                fit_gpytorch_mll(mll_noise)
                noise_model.eval(); noise_model.likelihood.eval()
            
            return WrapperHeteroskedasticModel(mean_model=self._model_mean, noise_model=noise_model)


    def _fit_mean_model_with_optional_warp(self, num_steps=300, lr=0.05):
        mll_mean = ExactMarginalLogLikelihood(self._model_mean.likelihood, self._model_mean)
        self._model_mean.train(); self._model_mean.likelihood.train()

        params = [{"params": self._model_mean.parameters()}]
        if self._warp_outputs:
            params.append({"params": self._y_warp.parameters()})

        opt = torch.optim.Adam(params, lr=lr)

        Y_t = self._y_warp(self._Y_train) if self._warp_outputs else self._Y_train

        for _ in range(num_steps):
            opt.zero_grad()
            out = self._model_mean(self._X_train)
            with gpytorch.settings.cholesky_jitter(1e-3):
                base = mll_mean(out, Y_t.squeeze(-1))
        

            if self._warp_outputs:
                jac = self._y_warp.log_jacobian(self._Y_train).sum()
                loss = -(base + jac)
            else:
                loss = -base

            loss.backward()
            opt.step()

        self._model_mean.eval(); self._model_mean.likelihood.eval()

    

    def _describe_kernel(self, k, indent=0):
            pad = "  " * indent
            print(f"{pad}{k.__class__.__name__}")
            # Recursively descend into wrappers if they exist
            if hasattr(k, "base_kernel"):
                self._describe_kernel(k.base_kernel, indent + 1)
            if hasattr(k, "kernels"):  # e.g., AdditiveKernel/ProductKernel
                for child in k.kernels:
                    self._describe_kernel(child, indent + 1)

    def _calculate_residuals(self, eps: float):
            Y_warped = self._y_warp(self._Y_train) if self._warp_outputs else self._Y_train

            f_post = self._model_mean.posterior(self._X_train, observation_noise=False)
            resid = (Y_warped - f_post.mean).squeeze(-1)
            return torch.log(resid.pow(2) + eps).unsqueeze(-1)


    def suggest_next_point_to_evaluate(
            self, bounds: List[Tuple[float, float]], 
            num_samples: int, 
            warmup_steps: int, 
            thinning: int, 
            max_tree_depth: int, 
            raw_samples: int, 
            max_iter: int, 
            acquisition_function_str, 
            acquisition_function: qLogNoisyExpectedImprovement | UpperConfidenceBound | qNegIntegratedPosteriorVariance, 
            csv_output_file_name: str
        ) -> str: 
            tensor_bounds = torch.from_numpy(np.array(bounds, dtype=np.double).transpose())
            best_x, max_acq = self._maximize_acquisition_function(bounds=tensor_bounds, raw_samples=raw_samples, max_iter=max_iter, acquisition_function=acquisition_function)
            logging.info(f"Best new exploration point found with acquisition maxing at: {max_acq}")

            formatter:str = lambda arr: "-".join(f"{x:.6f}" for x in arr)
            formatter_lengthscales:str = lambda arr: "-".join(f"{x}" for x in arr)
            posterior = self._model.posterior(best_x.reshape(1, -1), observation_noise=True)
            mu = posterior.mean.view(-1).detach().numpy()
            std = posterior.variance.clamp_min(0).sqrt().view(-1).detach().numpy()
            self._print_to_csv(
                num_samples, 
                warmup_steps, 
                thinning, 
                max_tree_depth, 
                raw_samples,
                max_iter, 
                formatter_lengthscales(self._model.mean_gp.covar_module.base_kernel.lengthscale.detach().numpy()), 
                acquisition_function_str,
                formatter(best_x),
                mu,
                std,
                csv_output_file_name
            )
            return formatter(best_x), mu, std


    def _maximize_acquisition_function(self, bounds: torch.Tensor, raw_samples: int, max_iter: int, acquisition_function) -> int:
            turbo_bounds = self._get_trust_region_bounds(
                turbo_state=self._turbo_state, 
                global_bounds=bounds,
                warp_outputs=self._warp_outputs
            )
            print(f"turbo_bounds: {turbo_bounds}")
            Xinit = gen_batch_initial_conditions(
                acq_function=acquisition_function,
                bounds=turbo_bounds,
                q=1,
                num_restarts=self._num_restarts,
                raw_samples=raw_samples,
                # options={"seed":30},
            )
            candidate, acq_value = optimize_acqf(
                acq_function=acquisition_function,
                bounds=turbo_bounds,
                q=1,
                num_restarts=self._num_restarts,
                raw_samples=raw_samples,
                batch_initial_conditions=Xinit,
                options={"maxiter":max_iter}
            )
            print(f"acq_value: {acq_value}")
            logging.info(f"Found max acquisition value {acq_value}")
            return candidate.squeeze(0), acq_value.numpy().item()


    def _get_tensors_from_dataframe(self) -> Tuple[torch.Tensor, torch.Tensor]:
            df = self._dataset.copy()
            Xs = df.drop('y', axis=1).to_numpy()
            X_np = np.ascontiguousarray(Xs, dtype=np.double)
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
            lengthscales: str,
            acquisition_function: str,
            best_x: str, 
            posterior_mu: np.ndarray,
            posterior_sigma: np.ndarray,
            csv_output_file_name: str
        ) -> None:
            CsvUtils.to_csv(
                ['num_samples', 'warmup_steps', 'thinning', 'max_tree_depth', 'raw_samples', 'max_iter', 'lengthscales', 'acquisition_function', 'best_x', 'posterior_mu', 'posterior_sigma'],
                csv_output_file_name,
                num_samples, 
                warmup_steps, 
                thinning, 
                max_tree_depth,
                raw_samples,
                max_iter, 
                lengthscales,
                acquisition_function,
                best_x,
                posterior_mu[0],
                posterior_sigma[0]
            )

    def _ls_init_and_bounds_from_spacing(
            self,
            X,
            *,
            # Global bounds for lengthscales in *input units*
            global_lo: float = 1e-2,
            global_hi: float = 10.0,
            # How wide around the "typical" spacing to set per-dim bounds
            span_factor: float = 3.0,
            nu: float = 1.5,      # kept for signature compatibility, not used
            **kwargs,             # ignore rho_star, k_low, k_high, min_log_span, etc.
    ):
    
            X = np.asarray(X, float)
            if X.ndim != 2:
                raise ValueError("X must be 2D (n x d).")

            n, d = X.shape

            lowers = []
            uppers = []

            for j in range(d):
                xj = np.unique(np.sort(X[:, j]))
                if xj.size <= 1:
                    # Degenerate dimension: fall back to very generic bounds
                    lo_j = global_lo
                    hi_j = global_hi
                else:
                    diffs = np.diff(xj)
                    diffs = diffs[diffs > 0]
                    # if diffs.size == 0:
                    lo_j = global_lo
                    hi_j = global_hi
                    # else:
                    #     med = float(np.median(diffs))

                    #     # Center a "local" window around the median spacing
                    #     lo_j = med / span_factor
                    #     hi_j = med * span_factor

                    #     # Intersect with global bounds
                    #     lo_j = max(global_lo, lo_j)
                    #     hi_j = min(global_hi, hi_j)

                    #     # Ensure non-degenerate interval
                    #     if hi_j <= lo_j:
                    #         hi_j = min(global_hi, lo_j * 1.5)

                lowers.append(lo_j)
                uppers.append(hi_j)

            lowers = np.array(lowers, dtype=float)
            uppers = np.array(uppers, dtype=float)

            # Geometric mean init inside the (lo, hi) interval
            ls_init = np.sqrt(lowers * uppers).astype(float)
            ls_bounds = [(float(lo), float(hi)) for lo, hi in zip(lowers, uppers)]
            print(f"ls bounds: {ls_bounds}")
            return ls_init, ls_bounds

    def _get_trust_region_bounds(
            self,
            turbo_state: TurboState,
            global_bounds: torch.Tensor, 
            warp_outputs: bool = False
        ) -> torch.Tensor:
        
            Y = self._y_warp(self._Y_train).squeeze(-1) if warp_outputs else self._Y_train.squeeze(-1) 

            d = self._X_train.shape[1]

            best_idx = torch.argmax(Y)
            x_center = self._X_train[best_idx].detach() 

            ls = (
                self._model.mean_gp.covar_module.base_kernel.lengthscale
                .detach()
                .view(-1)            
            )

            weights = ls / ls.mean()        
            weights = weights / torch.prod(weights.pow(1.0 / d))

            half_side = 0.5 * turbo_state.length * weights 

            lb = x_center - half_side
            ub = x_center + half_side

            lb = torch.max(lb, global_bounds[0])
            ub = torch.min(ub, global_bounds[1])

            return torch.stack([lb, ub])

class WarpedOutputTransformation(torch.nn.Module):

    def __init__(self, y:torch.Tensor, lam_low: float, lam_high: float, eps=1e-8):
        super().__init__()
        self._lam_low = lam_low
        self._lam_high = lam_high
        self._eps = eps
        self._shift = (1.0 - y.min()).clamp_min(self._eps)
        self._lam_value = torch.nn.Parameter(torch.zeros(1))

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        z = (Y +self._shift).clamp_min(self._eps)
        
        if torch.isclose(self.lam, torch.zeros_like(self.lam), atol=1e-6).all():
            return torch.log(z)
        return (z**self.lam - 1.0)/self.lam
        
    @property
    def lam(self):
        s = torch.sigmoid(self._lam_value)
        return self._lam_low + (self._lam_high - self._lam_low)*s

    def log_jacobian(self, y: torch.Tensor) -> torch.Tensor:
        z = (y +self._shift).clamp_min(self._eps)
        return (self.lam - 1.0)*torch.log(z)


class WrapperHeteroskedasticModel(Model):
            
        def __init__(self, mean_model: SingleTaskGP, noise_model: SingleTaskGP):
            super().__init__()
            self.mean_gp = mean_model
            self.noise_gp = noise_model


        @property
        def num_outputs(self) -> int:
            return 1

        def posterior(
            self,
            X: torch.Tensor,
            output_indices: list[int] | None = None,
            observation_noise: bool | torch.Tensor = True,
            posterior_transform: PosteriorTransform | None = None,
        ) -> Posterior:

            f_post = self.mean_gp.posterior(X, observation_noise=False)
            if not observation_noise:
                return f_post

            logv_post = self.noise_gp.posterior(X, observation_noise=False)
            # noise_var = torch.exp(logv_post.mean+0.5*logv_post.variance)
            noise_var = torch.exp(logv_post.mean)
            noise_var = noise_var.clamp_min(1e-12).squeeze(-1)

            
            mvn = f_post.mvn
            cov = mvn.covariance_matrix
            cov_obs = cov + torch.diag_embed(noise_var)
            mvn_obs = gpytorch.distributions.MultivariateNormal(mvn.mean, cov_obs)
            return GPyTorchPosterior(mvn_obs)


class WarpedTaskGP(SingleTaskGP):
        def __init__(self, X: torch.Tensor, Y: torch.Tensor, alpha: Parameter, beta: Parameter, **kwargs):
            super().__init__(X, Y, **kwargs)
            self._alpha = alpha
            self._beta = beta

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            X_warped = WarpUtils.warp(X, self._alpha, self._beta)
            return super().forward(X_warped)


        