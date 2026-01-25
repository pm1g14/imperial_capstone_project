from typing import List
import optuna
from hpo_bench_eval import HpoBenchEval
from test_utils.input_utils import InputUtils

class TuneBO:
    def __init__(self, function_number: int, hp_names: List[str]):
        self._function_number = function_number
        self._hp_names = hp_names


    def optuna_objective(self, trial: optuna.trial.Trial, csv_output_file_name: str, output_file_name: str) -> float:
        nu_candidates = [0.5, 1.5, 2.5]

        nu_mean = trial.suggest_categorical("nu_mean", nu_candidates)
        nu_noise = trial.suggest_categorical("nu_noise", nu_candidates)
        n_restarts = trial.suggest_int("n_restarts", 10, 100)
        num_samples = trial.suggest_int("num_samples", 64, 1024, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 256, 2048, log=True)
        thinning = trial.suggest_int("thinning", 4, 64)
        max_tree_depth = trial.suggest_int("max_tree_depth", 3, 10)
        raw_samples = trial.suggest_int("raw_samples", 128, 2048, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 1000)
        warp_inputs = trial.suggest_categorical("warp_inputs", [True, False])
        warp_outputs = trial.suggest_categorical("warp_outputs", [True, False])

        hpo_bench_eval = HpoBenchEval(hp_names=self._hp_names)
        return hpo_bench_eval.evaluate(
            dataset=InputUtils.get_dataframe_for_function(self._function_number, ['x1', 'x2', 'x3', 'x4', 'y']), 
            no_of_trials=13, 
            output_file_name="bo_results_6_trial_f5.png", 
            nu_mean=nu_mean,
            nu_noise=nu_noise,
            n_restarts=n_restarts,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            thinning=thinning,
            max_tree_depth=max_tree_depth,
            raw_samples=raw_samples,
            max_iter=max_iter,
            warp_inputs=warp_inputs,
            warp_outputs=warp_outputs,
            csv_output_file_name=csv_output_file_name
        )