
import numpy as np
import time
from itertools import repeat
from multiprocessing import Pool

from src.data import generate_dataset
from src.cross_validation import cross_validate
from src.fit import fit_by_lstsq
import src.configs as configs


def run_single_trial(
    rng: np.random._generator.Generator,
    beta: np.ndarray,
    data_parameters: dict = configs.DEFAULT_DATA_PARAMETERS,
    method: str = configs.DEFAULT_DATA_PARAMETERS,
    params: list = [],
    block_kind: str = configs.DEFAULT_BLOCK_KIND,
    algorithm: str = configs.DEFAULT_ALGORITHM,
):
    X, Y = generate_dataset(rng, beta, data_parameters=data_parameters)
    ts = time.time()
    beta_hat = fit_by_lstsq(X, Y)[0][0][0]
    beta_m = beta_hat * rng.uniform(size=beta.size)
    beta_M = beta_hat * rng.uniform(size=beta.size)
    if method == "OLS":
        return [
            {"cv_strategy": "min_loss", "beta_strategy": "best", "best_param": "-", "beta_hat": beta_hat},
            {"cv_strategy": "max_slope", "beta_strategy": "best", "best_param": "-", "beta_hat": beta_hat},
            {"cv_strategy": "min_loss", "beta_strategy": "last", "best_param": "-", "beta_hat": beta_hat},
            {"cv_strategy": "max_slope", "beta_strategy": "last", "best_param": "-", "beta_hat": beta_hat}
        ], time.time() - ts
    else:
        strategies_results = cross_validate(
            X,
            Y,
            beta_m,
            beta_M,
            rng,
            n_folds = 5,
            method = method,
            params = params,
            block_kind = block_kind,
            algorithm = algorithm,
        )
        return strategies_results, time.time() - ts


def run_trials(
    beta: np.ndarray,
    data_parameters: dict = configs.DEFAULT_DATA_PARAMETERS,
    method: str = configs.DEFAULT_METHOD,
    params: list = [],
    block_kind: str = configs.DEFAULT_BLOCK_KIND,
    algorithm: str = configs.DEFAULT_ALGORITHM,
    n_trials: int = configs.DEFAULT_N_TRIALS,
    n_jobs: int = configs.DEFAULT_N_JOBS,
    start_random_seed: int = configs.DEFAULT_START_RANDOM_SEED,
):
    rngs = [np.random.default_rng(start_random_seed + trial) for trial in range(n_trials)]
    with Pool(n_jobs) as pool:
        results = pool.starmap(
            run_single_trial,
            zip(
                rngs,
                repeat(beta),
                repeat(data_parameters),
                repeat(method),
                repeat(params),
                repeat(block_kind),
                repeat(algorithm),
            ),
        )

        df = []
        for i, (strategies_results, time) in enumerate(results):
            for strategy_results in strategies_results:
                beta_hat = strategy_results["beta_hat"]
                df.append({
                    "seed": i + start_random_seed,
                    "method": method,
                    "params": params,
                    "beta_hat": beta_hat.tolist(),
                    "L2_dist": np.linalg.norm(beta - beta_hat),
                    "best_param": strategy_results["best_param"],
                    "block_kind": block_kind,
                    "algorithm": algorithm,
                    "cv_strategy": strategy_results["cv_strategy"],
                    "beta_strategy": strategy_results["beta_strategy"],
                    "time": time
                } | data_parameters)
        return df