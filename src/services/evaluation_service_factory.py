

from services.evaluation_service_contract import EvaluationServiceContract
from services.global_evaluation_service import GlobalEvaluationService
from services.trust_region_evaluation_service import TrustRegionEvaluationService
import pandas as pd

from domain.domain_models import ModelInputConfig, WarpConfig
from evaluate.evaluate import ModelEvaluator


class EvaluationServiceFactory:

    @staticmethod
    def get_evaluation_service(trust_region_flg: bool, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int, function_number: int) -> EvaluationServiceContract:
        warp_config = WarpConfig(lam_low=-1.5, lam_high=1.5)

        if trust_region_flg:
            model_input_config = ModelInputConfig(
                bounds=[(0.0, 1.0) for _ in range(dims)],
                nu_mean=1.5,
                nu_noise=2.5,
                n_restarts=40,
                num_samples=256,
                warmup_steps=512,
                thinning=16,
                max_tree_depth=6,
                raw_samples=512,
                max_iter=200,
                acquisition_function_str='nei',
                beta=None,
                warp_inputs=True,
                warp_outputs=True,
                lam_config=warp_config
            )
            return TrustRegionEvaluationService(dims, dataframe, total_budget, trial_no, function_number, model_input_config)
        else:
            model_input_config = ModelInputConfig(
                bounds=[(0.0, 1.0) for _ in range(dims)],
                nu_mean=1.5,
                nu_noise=2.5,
                n_restarts=40,
                num_samples=256,
                warmup_steps=512,
                thinning=16,
                max_tree_depth=6,
                raw_samples=512,
                max_iter=200,
                acquisition_function_str=None,
                beta=None,
                warp_inputs=True,
                warp_outputs=True,
                lam_config=warp_config
            )
            return GlobalEvaluationService(dims, dataframe, total_budget, trial_no, model_input_config)
        