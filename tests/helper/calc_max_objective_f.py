from functools import cache
from itertools import product
from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
from ConfigSpace.configuration import Configuration


def compute_true_max():
    bench = SliceLocalizationBenchmark(rng=0)
    cs = bench.get_configuration_space()

    # Collect lists of values for each hp
    hp_names = []
    hp_values = []

    for hp in cs.get_hyperparameters():
        hp_names.append(hp.name)

        if hasattr(hp, "sequence"):
            hp_values.append(list(hp.sequence))
        elif hasattr(hp, "choices"):
            hp_values.append(list(hp.choices))
        else:
            raise RuntimeError(f"Unsupported hp type: {hp}")

    best_val = float("-inf")
    best_config = None

    for vals in product(*hp_values):
        cfg_dict = dict(zip(hp_names, vals))
        config = Configuration(cs, values=cfg_dict)

        res = bench.objective_function(config)
        y = -res["function_value"]   # convert minimization → maximization

        if y > best_val:
            best_val = y
            best_config = cfg_dict
    print(f"MAX VALUE IS: {best_val}")
    return best_val, best_config