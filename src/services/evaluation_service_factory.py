

from services.evaluation_service_contract import EvaluationServiceContract
from services.global_evaluation_service import GlobalEvaluationService
from services.trust_region_evaluation_service import TrustRegionEvaluationService
import pandas as pd


class EvaluationServiceFactory:

    @staticmethod
    def get_evaluation_service(trust_region_flg: bool, dims: int, dataframe: pd.DataFrame, total_budget: int, trial_no: int) -> EvaluationServiceContract:
        if trust_region_flg:
            return TrustRegionEvaluationService(dims, dataframe, total_budget, trial_no)
        else:
            return GlobalEvaluationService(dims, dataframe, total_budget, trial_no)
        