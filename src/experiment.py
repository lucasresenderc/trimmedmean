
import numpy as np
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
    selection_strategy = configs.DEFAULT_SELECTION_STRATEGY,
    fold_K = configs.DEFAULT_FOLD_K,
):
    X, Y = generate_dataset(rng, beta, data_parameters=data_parameters)
    beta_m = rng.uniform(size=beta.size)
    beta_M = rng.uniform(size=beta.size)
    if method == "OLS":
        beta_hat = fit_by_lstsq(X, Y)
        return beta_hat, np.array([]), None
    else:
        return cross_validate(
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
            selection_strategy = selection_strategy,
            fold_K = fold_K,
        )


def run_trials(
    beta: np.ndarray,
    data_parameters: dict = configs.DEFAULT_DATA_PARAMETERS,
    method: str = configs.DEFAULT_METHOD,
    params: list = [],
    block_kind: str = configs.DEFAULT_BLOCK_KIND,
    algorithm: str = configs.DEFAULT_ALGORITHM,
    selection_strategy = configs.DEFAULT_SELECTION_STRATEGY,
    fold_K = configs.DEFAULT_FOLD_K,
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
                repeat(selection_strategy),
                repeat(fold_K),
            ),
        )

        df = []
        for i, (beta_hat, params_losses, best_param) in enumerate(results):
            df.append({
                "seed": i + start_random_seed,
                "method": method,
                "params": params,
                "params_losses": params_losses.tolist(),
                "beta_hat": beta_hat.tolist(),
                "L2_dist": np.linalg.norm(beta - beta_hat),
                "best_param": best_param,
                "block_kind": block_kind,
                "algorithm": algorithm,
                "selection_strategy": selection_strategy,
                "fold_K": fold_K
            } | data_parameters)
        return df