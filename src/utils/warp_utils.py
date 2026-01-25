
import torch
from torch.nn import Parameter
from scipy.stats import boxcox, boxcox_llf, boxcox_normmax, yeojohnson, yeojohnson_normmax
import numpy as np
import torch.nn.functional as F

class WarpUtils:

    @staticmethod
    def warp(X: torch.Tensor, alpha: Parameter, beta: Parameter, eps: float = 1e-6) -> torch.Tensor:
        X_clamped = X.clamp(eps, 1.0 - eps)

        # Make sure alpha, beta are broadcastable and positive
        a = F.softplus(alpha)  # shape (d,)
        b = F.softplus(beta)   # shape (d,)

        # Reshape for broadcasting: (1, d) or (..., d) is fine
        while a.dim() < X_clamped.dim():
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)

        # Warp: 1 - (1 - x^a)^b
        X_pow = X_clamped.pow(a)
        inner = (1.0 - X_pow).clamp(eps, 1.0)  # avoid exact 0
        warped = 1.0 - inner.pow(b)
        return warped

    @staticmethod
    def warp_asynh(Y: torch.Tensor):
        return np.arcsinh(Y)


    @staticmethod
    def inv_warp_asinh(Y: torch.Tensor):
        return np.sinh(Y)

    @staticmethod
    def yeojohnson_normmax(Y: torch.Tensor, lam: float | None = None):
        Y_np = Y.detach().cpu().numpy().astype(float).ravel()

        if not np.all(np.isfinite(Y_np)):
            raise ValueError("Y contains NaN or inf values; cannot apply Yeo-Johnson.")

        if lam is None:
            lam = yeojohnson_normmax(Y_np)
        Y_warp_np = yeojohnson(Y_np, lam)
        Y_warp = torch.from_numpy(Y_warp_np).to(Y.device, dtype=Y.dtype)
        return Y_warp.view_as(Y), lam

    @staticmethod
    def warp_boxcox(Y: torch.Tensor, e: float = 1e-6, lam: float | None = None):
        Y_np = Y.detach().cpu().numpy().astype(float).flatten()
        shift = (- Y.min().item() + e) if Y.min() <= 0 else e
        Y_shifted = Y_np + shift
        
        if lam is None:
            lam = boxcox_normmax(Y_shifted)
    
        y_warp = torch.tensor(boxcox(Y_shifted, lam).reshape(-1, 1), dtype=torch.float64)
        ll_bc = boxcox_llf(lam, Y_shifted)
        return y_warp, ll_bc

    @staticmethod
    def warp_logit_y(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y_clamped = y.clamp(eps, 1 - eps)
        return torch.log(y_clamped / (1 - y_clamped))



