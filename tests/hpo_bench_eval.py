import json
import logging
import sys
from typing import Any, Dict, List, Tuple, Union

from pathlib import Path
from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark

import numpy as np
import pandas as pd
from heteroskedastic_model_test import HeteroskedasticModel
from tests.helper.calc_max_objective_f import compute_true_max
import torch
import matplotlib.pyplot as plt
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    OrdinalHyperparameter,
)
from botorch.acquisition import UpperConfidenceBound, qLogNoisyExpectedImprovement, qNegIntegratedPosteriorVariance

from tests.heteroskedastic_turbo_model_test import HeteroskedasticTurboModel, TurboState


class HpoBenchEval:

    def __init__(self, hp_names: List[str], rng:int=42):
        
        base_bench = SliceLocalizationBenchmark(rng=42)
        self._bench_wrapper = ContinuousHPObenchWrapper(
            bench=base_bench,
            hp_names=hp_names,
            rng=rng,
        )
        
        meta = self._bench_wrapper.bench.get_meta_information()
        self._true_opt_max, true_config = compute_true_max()  # value to MINIMIZE
        self._dim = len(hp_names)

    def evaluate_at_point(self, X: np.ndarray) -> float:
        res, config = self._bench_wrapper.evaluate(X)
        return res.get("function_value", None)

    def evaluate(
        self, 
        dataset: pd.DataFrame, 
        no_of_trials: int, 
        output_file_name: str, 
        nu_mean: float,
        nu_noise: float,
        n_restarts: int,
        num_samples: int,
        warmup_steps: int,
        thinning: int,
        max_tree_depth: int,
        raw_samples: int,
        max_iter: int,
        warp_inputs: bool,
        warp_outputs: bool,
        csv_output_file_name: str
    ) -> float:
    
        max_so_far = -1e9
        how_far = []
        max_so_far_list = []
        bounds = [(0.0, 1.0) for _ in range(self._dim)]
        y_new_list =[]
        for t in range(no_of_trials):

            if (t <= 5):
                beta = 4.0
                acquisition_function_str = 'ucb'
            elif (t > 5 and t < 8):
                beta = 1.0
                acquisition_function_str = 'ucb'
            else:
                beta = 1.0
                acquisition_function_str = 'nei'

            new_point_model, mu, std, model = self.init_model_and_get_new_point_turbo(
                dataset=dataset,
                bounds=bounds,
                nu_mean=nu_mean,
                nu_noise=nu_noise,
                n_restarts=n_restarts,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                thinning=thinning,
                max_tree_depth=max_tree_depth,
                raw_samples=raw_samples,
                max_iter=max_iter,
                acquisition_function_str=acquisition_function_str,
                beta=beta,
                warp_inputs=warp_inputs,
                warp_outputs=warp_outputs,
                csv_output_file_name=csv_output_file_name,
                turbo_state=TurboState(dim=5)
            )
            x1, x2, x3, x4 = (float(x) for x in new_point_model.split("-"))
            x_unit = np.array([x1, x2, x3, x4])

            res, config = self._bench_wrapper.evaluate(x_unit)
            y_new_min = res.get("function_value", None)
            y_new = - y_new_min  
            row = {}
            for i in range(self._dim):
                row[f"x{i+1}"] = float(x_unit[i])
            row["y"] = float(y_new)

            dataset = pd.concat(
                [dataset, pd.DataFrame([row])],
                ignore_index=True
            )

            if y_new > max_so_far:
                max_so_far = y_new
            y_new_list.append(y_new)
            max_so_far_list.append(max_so_far)
            diff = np.abs(max_so_far - self._true_opt_max)
            how_far.append(diff)
            print(f"y_new is {y_new} after trial no: {t}. Diff to max: {diff}")
        
            self.run_update(y_new=y_new, function_number=5)
        self._plot_evaluation_performance(max_so_far_list=max_so_far_list, y_new_list=y_new_list, how_far=how_far, output_file_name=output_file_name)
        return max_so_far

    def print_hyperparameters(self):
        cs = self.base_bench.get_configuration_space(seed=0)
        for hp in cs.get_hyperparameters():
            print(hp.name, type(hp).__name__)

    def _plot_evaluation_performance(self, how_far, max_so_far_list, y_new_list, output_file_name):
        T = len(how_far)
        x = np.arange(T)

        plt.figure(figsize=(10, 6))

        plt.plot(x, how_far, label="Regret (|best_so_far - f*|)")
        plt.plot(x, max_so_far_list, label="Best-so-far value")
        plt.plot(x, y_new_list, label="New evaluation y_new", alpha=0.3)

        plt.yscale("log")   # log-scale often reveals BO behaviour better
        plt.xlabel("Trial")
        plt.ylabel("Error / Function Value")
        plt.title("Bayesian Optimization Performance on SliceLocalization")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file_name, dpi=200, bbox_inches="tight")
        print("Saved plot to bo_results.png")

    

    def run_update(self, y_new: float, function_number:int) -> None:
        def save_turbo_state(turbo_state: Dict[str, Any], function_number: int) -> bool:
            breakpoint()
            if turbo_state:
                project_root = Path(__file__).resolve().parent
                try:
                    input_path = project_root / "turbo_state.json"
                    with open(input_path, 'w') as f:
                        json.dump(turbo_state, f)
                    return True
                except Exception as e:
                    print(f"Error: {e}")
                    return False
            
            return False
        
        def update_turbo_state(turbo_state: TurboState, y_new: float, function_number: int) -> TurboState:
        
            if turbo_state:
                improved = y_new > turbo_state.best_value

                if improved:
                    turbo_state.best_value = y_new
                    turbo_state.success_counter += turbo_state.batch_size
                    turbo_state.failure_counter = 0
                else:
                    turbo_state.failure_counter += turbo_state.batch_size
                    turbo_state.success_counter = 0

                if turbo_state.success_counter >= turbo_state.success_tolerance:
                    turbo_state.length = min(2.0 * turbo_state.length, turbo_state.length_max)
                    turbo_state.success_counter = 0

                # Shrink on consecutive failures
                if turbo_state.failure_counter >= turbo_state.failure_tolerance:
                    turbo_state.length = max(0.5 * turbo_state.length, turbo_state.length_min)
                    turbo_state.failure_counter = 0

                # Trigger restart if TR has become too small
                turbo_state.restart_triggered = turbo_state.length <= turbo_state.length_min + 1e-12
                
                if turbo_state.restart_triggered:
                    turbo_state.length = 0.8
                    turbo_state.success_counter = 0
                    turbo_state.failure_counter = 0
                    turbo_state.restart_triggered = False
                save_turbo_state(turbo_state.__dict__, function_number)
                return turbo_state

        def get_last_turbo_state() -> Tuple[np.ndarray, np.ndarray]:
            breakpoint()
            project_root = Path(__file__).resolve().parent
            input_path = project_root / "turbo_state.json"
            try:
                with open(input_path, 'r') as f:
                    # Load the entire content, which is expected to be a list of dictionaries
                    data: Dict[str, Any] = json.load(f)
                
                    if not data:
                        # Handle empty file/list
                        print(f"JSON file at {input_path} is empty.")
                        return None
                    
                    return data
            except json.JSONDecodeError:
                    print(f"Error: JSON file at {input_path} is not valid JSON.")
                    return None

        def _map_to_turbo_state(turbo_state_dict: Dict[str, Any]) -> Union[TurboState, None]:
            try:
                if turbo_state_dict:
                    turbo_state = TurboState(
                        dim=turbo_state_dict['dim'],
                        batch_size=turbo_state_dict['batch_size'],
                        length=turbo_state_dict['length'],
                        length_min=turbo_state_dict['length_min'],
                        length_max=turbo_state_dict['length_max'],
                        success_counter=turbo_state_dict['success_counter'],
                        failure_counter=turbo_state_dict['failure_counter'],
                        success_tolerance=turbo_state_dict['success_tolerance'],
                        failure_tolerance=turbo_state_dict['failure_tolerance'],
                        best_value=turbo_state_dict['best_value'],
                        restart_triggered=turbo_state_dict['restart_triggered']
                    )
                    return turbo_state
            except Exception as e:
                print(f"Error: {e}")
                return None            
        breakpoint()
        turbo_state = get_last_turbo_state()
        if not turbo_state:
            turbo_state = TurboState(dim=5)

        update_turbo_state(
            turbo_state=_map_to_turbo_state(turbo_state),
            function_number=function_number,
            y_new = y_new
        )
        logging.info(f"Turbo state updated for function {function_number}")

        

    def init_model_and_get_new_point(
        self,
        dataset, 
        bounds, 
        nu_mean, 
        nu_noise, 
        n_restarts, 
        num_samples, 
        warmup_steps, 
        thinning, 
        max_tree_depth, 
        raw_samples, 
        max_iter, 
        csv_output_file_name,
        acquisition_function_str, 
        beta: float = 1.0, 
        warp_inputs: bool = False,
        warp_outputs: bool = False
    ) -> Tuple[str, float, float]:
        model = HeteroskedasticModel(dataset=dataset, nu_mean=nu_mean, nu_noise = nu_noise, n_restarts=n_restarts, warp_inputs=warp_inputs,warp_outputs=warp_outputs)
        if (acquisition_function_str == 'nei'):
            acquisition_function=qLogNoisyExpectedImprovement(
                model=model.get_model(), 
                X_baseline=model.get_X(), 
                sampler=model.get_sampler()
            )
        elif (acquisition_function_str == 'ucb'):
            acquisition_function=UpperConfidenceBound(
                model=model.get_model(), 
                beta=beta
            )
        else:
            d = model.get_X().shape[1]
            mc_points = torch.rand(256, d, dtype=torch.double)
            acquisition_function=qNegIntegratedPosteriorVariance(
                model=model.get_model().mean_gp,
                mc_points=mc_points
            )
        new_point_model, mu, std = model.suggest_next_point_to_evaluate(
            bounds=bounds, 
            num_samples=num_samples, 
            warmup_steps=warmup_steps, 
            thinning=thinning, 
            max_tree_depth=max_tree_depth, 
            raw_samples=raw_samples, 
            max_iter=max_iter, 
            acquisition_function_str=acquisition_function_str,
            acquisition_function=acquisition_function,
            csv_output_file_name=csv_output_file_name
        )
        return new_point_model, mu, std, model

    def init_model_and_get_new_point_turbo(
        self,
        dataset, 
        bounds, 
        nu_mean, 
        nu_noise, 
        n_restarts, 
        num_samples, 
        warmup_steps, 
        thinning, 
        max_tree_depth, 
        raw_samples, 
        max_iter, 
        csv_output_file_name,
        acquisition_function_str,
        turbo_state: TurboState,
        beta: float = 1.0, 
        warp_inputs: bool = False,
        warp_outputs: bool = False,
        
        ) -> Tuple[str, float, float, HeteroskedasticTurboModel]:
            
            model = HeteroskedasticTurboModel(
                dataset=dataset, 
                nu_mean=nu_mean, 
                nu_noise = nu_noise, 
                n_restarts=n_restarts, 
                warp_inputs=warp_inputs,
                warp_outputs=warp_outputs, 
                turbo_state=turbo_state
            )
            if (acquisition_function_str == 'nei'):
                acquisition_function=qLogNoisyExpectedImprovement(
                    model=model.get_model(), 
                    X_baseline=model.get_X(), 
                    sampler=model.get_sampler()
                )
            elif (acquisition_function_str == 'ucb'):
                acquisition_function=UpperConfidenceBound(
                    model=model.get_model(), 
                    beta=beta
                )
            else:
                d = model.get_X().shape[1]
                mc_points = torch.rand(256, d, dtype=torch.double)
                acquisition_function=qNegIntegratedPosteriorVariance(
                    model=model.get_model().mean_gp,
                    mc_points=mc_points
                )
            new_point_model, mu, std = model.suggest_next_point_to_evaluate(
                bounds=bounds, 
                num_samples=num_samples, 
                warmup_steps=warmup_steps, 
                thinning=thinning, 
                max_tree_depth=max_tree_depth, 
                raw_samples=raw_samples, 
                max_iter=max_iter, 
                acquisition_function_str=acquisition_function_str,
                acquisition_function=acquisition_function,
                csv_output_file_name=csv_output_file_name,
            )

            return new_point_model, mu, std, model

        
class ContinuousHPObenchWrapper:
    """
    Wraps an HPOBench benchmark so you can query it with a vector in [0,1]^D,
    just like you did with Branin. Assumes the chosen hyperparameters are
    UniformFloat or UniformInteger.
    """
    def __init__(self, bench, hp_names, budget=None, rng=42):
        self.bench = bench
        self.cs = bench.get_configuration_space(seed=rng)
        self.hp_names = hp_names  # list of strings, e.g. 5 names
        self.hps = [self.cs.get_hyperparameter(name) for name in hp_names]
        self.budget = budget      # e.g. 100 for multi-fidelity benchmarks

        # basic sanity check: we only handle uniform float/int here
        for hp in self.hps:
            if not isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter, OrdinalHyperparameter)):
                raise ValueError(f"Hyperparameter {hp.name} is not uniform float/int; got {type(hp)}")

    def x_unit_to_config(self, x_unit: np.ndarray):
        """
        Map x_unit in [0,1]^D to a ConfigSpace.Configuration.
        """
        assert len(x_unit) == len(self.hp_names)
        values = {}
        for u, hp in zip(x_unit, self.hps):
            # clamp for safety
            u = float(np.clip(u, 0.0, 1.0))

            if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                low, high = hp.lower, hp.upper

                val = low + u * (high - low)
                if isinstance(hp, UniformIntegerHyperparameter):
                    val = int(round(val))
                    # make sure we don't step out of bounds after rounding
                    val = max(low, min(high, val))

                values[hp.name] = val
            elif isinstance(hp, OrdinalHyperparameter):
                seq = list(hp.sequence)
                idx = int(np.floor(u*len(seq)))
                idx = max(0, min(len(seq)-1, idx))
                val = seq[idx]
            
            values[hp.name] = val

        config = self.cs.get_default_configuration()
        # overwrite selected hps with our mapped values
        for name, v in values.items():
            config[name] = v
        return config

    def evaluate(self, x_unit: np.ndarray):
        """
        Evaluate the benchmark at x_unit in [0,1]^D.

        Returns the dict from objective_function, e.g. with key 'function_value'.
        """
        config = self.x_unit_to_config(x_unit)

        kwargs = {"configuration": config}
        if self.budget is not None:
            kwargs["fidelity"] = {"budget": self.budget}

        res = self.bench.objective_function(**kwargs)
        return res, config