import numpy as np

from src.fit import fit_by_aasd, fit_by_admm, fit_by_plugin, fit_huber, MOMBlockGenerator
import src.configs as configs


def get_beta_hat(betas, costs, beta_strategy):
    if beta_strategy == "last":
        return betas[-1][-1]
    elif beta_strategy == "best":
        i,j = np.unravel_index(np.argmin(costs), (len(costs), 2))
        return betas[i][j]


def find_best_param(X, Y, histories, method, cv_strategy, beta_strategy, n_folds, fold_size):
    param_losses = []
    params = [p for p in histories]

    for param in histories:
        fold_losses = []
        for fold in histories[param]:
            # evaluate the loss
            fold_indexes = histories[param][fold]["indexes"]
            if method == "HUBER":
                beta_hat = histories[param][fold]["beta"]
                M = param * histories[param][fold]["sigma"]
            else:
                beta_hat = get_beta_hat(
                    histories[param][fold]["betas"],
                    histories[param][fold]["costs"],
                    beta_strategy
                )
            
            # eval at fold
            if method == "HUBER":
                abs_diff = np.abs( Y - X @ beta_hat )
                fold_losses.append(
                    np.mean(abs_diff * (abs_diff > M) + abs_diff**2 * (abs_diff <= M))
                )
            else:
                point_losses = (X[fold_indexes == 1] @ beta_hat - Y[fold_indexes == 1])**2
            
            # return the correct estimator
            if method == "TM":
                k_prime = int(param*fold_size)
                if k_prime == 0:
                    fold_losses.append(np.mean(point_losses))
                else:
                    fold_losses.append(np.mean(np.sort(point_losses)[k_prime:-k_prime]))
            elif method == "MOM":
                K_prime = int(param/n_folds)
                if K_prime < 1:
                    K_prime = 1
                fold_losses.append(np.median([
                    np.mean(point_losses[int(fold_size/K_prime*i):int(fold_size/K_prime*(i+1))]) for i in range(K_prime)
                ]))
        param_losses.append(np.median(fold_losses))

    # select best param
    param_losses = np.array(param_losses)
    if cv_strategy == "min_loss":
        best_param = params[np.argmin(param_losses)]
    elif cv_strategy == "max_slope":
        best_param = params[np.argmax(param_losses[:-1] / param_losses[1:]) + 1]
        if param_losses[0] < param_losses[-1]:
            best_param = params[0]
    
    return best_param
                

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
):
    n = X.shape[0]
    fold_size = n//n_folds

    # shuffle
    new_order = rng.permutation(n)
    X = X[new_order, :]
    Y = Y[new_order]

    # set fit method
    if algorithm == "aasd":
        fit = fit_by_aasd
    elif algorithm == "admm":
        fit = fit_by_admm
    elif algorithm == "plugin":
        fit = fit_by_plugin
    
    histories = {}
    if len(params) > 1:
        for param in params:
            if method == "MOM":
                k = param
                block_generator = MOMBlockGenerator(block_kind, rng, n - fold_size, k)
            elif method == "TM":
                k = int(param*(n-fold_size))
                block_generator = None

            histories[param] = {}
            for fold in range(n_folds):
                histories[param][fold] = {}

                # get fold indexes
                fold_indexes = np.zeros(n)
                fold_indexes[fold*fold_size:(fold+1)*fold_size] += 1

                # fit on complementary of the fold
                if method == "HUBER":
                    beta, sigma = fit_huber(X, Y, param)
                    histories[param][fold]["indexes"] = fold_indexes
                    histories[param][fold]["beta"] = beta
                    histories[param][fold]["sigma"] = sigma
                else:
                    beta_history, cost_history = fit(
                        X[fold_indexes == 0], 
                        Y[fold_indexes == 0],
                        beta_m, beta_M, k, block_generator = block_generator, method = method
                    )
                    histories[param][fold]["indexes"] = fold_indexes
                    histories[param][fold]["betas"] = beta_history
                    histories[param][fold]["costs"] = cost_history
    
    returns = []
    for cv_strategy in ["min_loss", "max_slope"]:
        for beta_strategy in ["best", "last"]:
            if len(params) > 1:
                best_param = find_best_param(
                    X, Y, histories, method, cv_strategy, beta_strategy, n_folds, fold_size
                )
            else:
                best_param = params[0]

            # fit all data using the best param
            if method == "TM":
                betas, costs = fit(X,Y, beta_m, beta_M, int(best_param*n), block_generator = None, method = method)
            elif method == "MOM":
                block_generator = MOMBlockGenerator(block_kind, rng, n, best_param)
                betas, costs = fit(X,Y, beta_m, beta_M, best_param, block_generator = block_generator, method = method)
            elif method == "HUBER":
                beta, _ = fit_huber(X, Y, best_param)

            if method == "HUBER":
                returns.append({
                    "cv_strategy": cv_strategy,
                    "beta_strategy": beta_strategy,
                    "beta_hat": beta,
                    "best_param": best_param
                })
            else:
                returns.append({
                    "cv_strategy": cv_strategy,
                    "beta_strategy": beta_strategy,
                    "beta_hat": get_beta_hat(betas, costs, beta_strategy),
                    "best_param": best_param
                })
    
    return returns
    