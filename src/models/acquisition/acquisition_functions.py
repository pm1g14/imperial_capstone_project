import numpy as np
from scipy.stats import norm


class AcquisitionFunctions:

    @staticmethod
    def ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float = 5.0) -> np.ndarray:
        mu = np.asarray(mu).reshape(-1, 1)
        sigma = np.maximum(np.asarray(sigma).reshape(-1, 1), 0.0)
        return mu + kappa * sigma


    @staticmethod
    def ei(mu, sigma, best_f, xi=0.01):
        s = max(float(sigma), 1e-12)
        imp = float(mu) - best_f - xi
        Z = imp / s
        return imp * norm.cdf(Z) + s * norm.pdf(Z)

