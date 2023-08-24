import numpy as np

from src.fit import fit_by_gd, fit_by_plugin, MOMBlockGenerator
import src.configs as configs

def cross_validate(
    X,
    Y,
    beta_m,
    beta_M,
    rng,
    n_folds: int = configs.DEFAULT_N_FOLDS,
    method: str = configs.DEFAULT_METHOD,
    params: list = [],
    algorithm: str = configs.DEFAULT_ALGORITHM,
    block_kind = configs.DEFAULT_BLOCK_KIND,
    selection_strategy = configs.DEFAULT_SELECTION_STRATEGY,
    fold_K = configs.DEFAULT_FOLD_K,
):
    n = X.shape[0]
    fold_size = n//n_folds

    # shuffle
    new_order = rng.permutation(n)
    X = X[new_order, :]
    Y = Y[new_order]

    # set fit method
    if algorithm == "gd":
        fit = fit_by_gd
    else:
        fit = fit_by_plugin
    
    params_losses = []
    for param in params:
        block_generator = None
        if method == "MOM":
            k = param
            block_generator = MOMBlockGenerator(block_kind, rng, n - fold_size, k)
        elif method == "TM":
            k = int(param*(n-fold_size))

        fold_losses = []
        for fold in range(n_folds):
            # get fold indexes
            fold_indexes = np.zeros(n)
            fold_indexes[fold*fold_size:(fold+1)*fold_size] += 1

            # fit on complementary of the fold
            beta_hat = fit(
                X[fold_indexes == 0], 
                Y[fold_indexes == 0],
                beta_m, beta_M, k, block_generator = block_generator, method = method
            )

            # eval at fold
            point_losses = (X[fold_indexes == 1] @ beta_hat - Y[fold_indexes == 1])**2
            
            # return the correct estimator
            if method == "TM":
                k_prime = int(param*fold_size)
                if k_prime == 0:
                    fold_losses.append(np.mean(point_losses))
                else:
                    fold_losses.append(np.mean(np.sort(point_losses)[k_prime:-k_prime]))
            elif method == "MOM":
                # Lerasle and Lecue used K' = max(grid_K)/V
                if fold_K == "maxK/V":
                    K_prime = int(np.max(params)/n_folds)
                elif fold_K == "K/V":
                    K_prime = int(k/n_folds)
                if K_prime < 1:
                    K_prime = 1
                fold_losses.append(np.median([
                    np.mean(point_losses[int(fold_size/K_prime*i):int(fold_size/K_prime*(i+1))]) for i in range(K_prime)
                ]))
            
        params_losses.append(np.median(fold_losses))
    
    # select best param
    params_losses = np.array(params_losses)
    if selection_strategy == "min_loss":
        best_param = params[np.argmin(params_losses)]
    elif selection_strategy == "max_slope":
        best_param = params[np.argmax(params_losses[:-1] / params_losses[1:]) + 1]
        if params_losses[0] > params_losses[-1]:
            best_param = params[0]

    # fit all data using the best param
    if method == "TM":
        beta_hat = fit(X,Y, beta_m, beta_M, int(best_param*n), block_generator = None, method = method)
    elif method == "MOM":
        block_generator = MOMBlockGenerator(block_kind, rng, n, best_param)
        beta_hat = fit(X,Y, beta_m, beta_M, best_param, block_generator = block_generator, method = method)
    
    return beta_hat, params_losses, best_param
    