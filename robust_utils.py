import numpy as np
from multiprocessing import Pool
from itertools import repeat
import pathlib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

"""

    FIT FUNCTIONS

    sort_blocks: define new blocks at random for MOM
    get_active_indexes: discover the active indexes at a point
        if method == "MOM" it will generate a random partition
        of [n] with k sets and return the set realizing the
        median of means;
        if method == "TM" it will return the set of active indexes
        where the trimmed mean is taken, i.e., the set {1,...,n}
        minus the k bigger and k larger loss values.
    update_by_armijo: update the objetive using the armijo rule
    fit_by_gd: fits data using the gradient descent algorithm
    fit_by_plugin: fits data using the plugin method

"""

class MOMBlockGenerator():
    def __init__(self, kind, rng, n, K):
        self.kind = kind
        self.rng = rng
        self.n = n
        self.K = K
        self.l = n // K
        if kind == "fixed":
            self.B = self.sort_blocks()

    def sort_blocks(self):
        p = np.arange(self.n)
        self.rng.shuffle(p)
        B = [p[m*self.l : m*self.l + self.l] for m in range(self.K)]
        B[-1] = p[self.K*self.l - self.l :]
        return B
    
    def get_blocks(self):
        if self.kind == "fixed":
            return self.B
        else:
            return self.sort_blocks()

def get_active_indexes(k, loss_m, loss_M, method = "TM", block_generator = None):
    diff = loss_m - loss_M
    if method == "MOM":
        B = block_generator.get_blocks()
        means = [ np.mean(diff[b]) for b in B ]
        return B[ np.argsort(means)[len(means)//2] ]
    if method == "TM":
        return np.argsort(diff)[k:-k]

def update_by_armijo(X, Y, b, beta, armijo_rate, eta):
    YXb = Y[b] - X[b, :] @ beta
    direction = 2*X[b, :].T @ YXb
    Xd = X[b, :] @ direction
    now = np.sum(np.power(YXb, 2))
    while np.sum(np.power(YXb - eta*Xd, 2)) > now :
        eta *= armijo_rate
    return beta + eta*direction, eta/armijo_rate

def fit_by_gd(
    X,
    Y,
    beta_m,
    beta_M,
    k : int,
    armijo_rate : float = .9,
    max_iter = 1000,
    method = "TM",
    block_generator = None,
    verbose = 0,
):
    loss_m = np.power( Y - X @ beta_m, 2 )
    loss_M = np.power( Y - X @ beta_M, 2 )
    b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)

    eta = 10
    gamma = 10
    for i in range(1, max_iter):
        beta_m, eta = update_by_armijo(X, Y, b, beta_m, armijo_rate, eta)
        loss_m = np.power( Y - X @ beta_m, 2 )
        b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)

        beta_M, gamma = update_by_armijo(X, Y, b, beta_M, armijo_rate, gamma)
        loss_M = np.power( Y - X @ beta_M, 2 )
        b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)

        if verbose:
            print(f"Iteration {i}\n\t lr at beta_m: {eta}, lr at beta_M: {gamma}")
    
    return beta_M

def fit_by_lstsq(X, Y):
    beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
    return beta_hat 

def evaluate_cost(X, Y, loss, beta, k):
    b = np.argsort(loss)[k:-k]
    return np.mean(np.power(Y[b] - X[b, :] @ beta, 2))

def fit_by_plugin(
    X,
    Y,
    beta_m,
    beta_M,
    k : int,
    max_iter = 100,
    method = "TM",
    block_generator = None,
):
    loss_m = np.power( Y - X @ beta_m, 2 )
    loss_M = np.power( Y - X @ beta_M, 2 )

    cost_m = evaluate_cost(X, Y, loss_m, beta_m, k)
    cost_M = evaluate_cost(X, Y, loss_M, beta_M, k)
    costs = [(cost_m, cost_M)]
    if cost_m < cost_M:
        beta_hat = np.copy(beta_m)
        cost_hat = cost_m
    else:
        beta_hat = np.copy(beta_M)
        cost_hat = cost_M
    
    for i in range(max_iter):
        b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)
        beta_m = np.linalg.lstsq(X[b, :], Y[b], rcond=None)[0]
        loss_m = np.power( Y - X @ beta_m, 2 )

        b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)
        beta_M = np.linalg.lstsq(X[b, :], Y[b], rcond=None)[0]
        loss_M = np.power( Y - X @ beta_M, 2 )

        cost_m = evaluate_cost(X, Y, loss_m, beta_m, k)
        cost_M = evaluate_cost(X, Y, loss_M, beta_M, k)
        if (cost_m, cost_M) in costs:
            break
        else:
            costs.append((cost_m, cost_M))
        
        if cost_m < cost_hat:
            beta_hat = np.copy(beta_m)
            cost_hat = cost_m
        if cost_M < cost_hat:
            beta_hat = np.copy(beta_M)
            cost_hat = cost_M

    return beta_hat


"""

    DATA GENERATION

    generate_dataset: generates a dataset with 5 kinds of
    data, as described in Lecué's paper.

"""

def generate_dataset(
    rng : np.random._generator.Generator,
    beta : np.ndarray,
    quantities : list = [200, 10, 10, 10, 10],
    student_degrees : float = 4.0,
    correlation_rate : float = .5
):
    n = np.sum(quantities)
    ns = np.cumsum(quantities)
    d = beta.size

    X = np.zeros(shape=(n,d))
    X[:ns[0], :] = rng.normal(size=(quantities[0],d))
    X[ns[0]:ns[2], :] += 1
    X[ns[2]:ns[3], :] = rng.uniform(size=(quantities[3],d))
    cov = np.power( correlation_rate, np.abs( np.array([range(d),]*d) - np.array([range(d),]*d).T ) )
    X[ns[3]:, :] = rng.multivariate_normal(np.zeros(d), cov, size=quantities[4])

    Y = np.zeros(n)
    Y[:ns[0]] =  X[:ns[0], :] @ beta + rng.normal(size=quantities[0])
    Y[ns[0]:ns[1]] += 1
    Y[ns[1]:ns[2]] += 10000
    Y[ns[2]:ns[3]] = 1.*(rng.uniform(size=quantities[3]) > .5)
    Y[ns[3]:] =  X[ns[3]:, :] @ beta + rng.standard_t(student_degrees, size=quantities[4])

    return X, Y

"""

    EXPERIMENT UTILS

"""

def run_single_trial(
    rng: np.random._generator.Generator,
    beta : np.ndarray,
    k : int,
    quantities : list = [200, 10, 10, 10, 10],
    method : str = "TM",
    block_kind : str = "fixed",
    student_degrees : float = 4.0,
    correlation_rate : float = .5,
    algorithm : str = "gd",
    return_time : bool = False,
):
    X,Y = generate_dataset(rng, beta, quantities=quantities, student_degrees=student_degrees, correlation_rate=correlation_rate)
    beta_m = rng.uniform(size=beta.size)
    beta_M = rng.uniform(size=beta.size)
    t = time.time()
    if method == "ERM":
        beta_hat = fit_by_lstsq(X,Y)
    else:
        block_generator = None
        if method == "MOM":
            block_generator = MOMBlockGenerator(block_kind, rng, np.sum(quantities), k)
        if algorithm == "gd":
            beta_hat = fit_by_gd(X, Y, beta_m, beta_M, k, method=method, block_generator = block_generator, max_iter=1000)
        else:
            beta_hat = fit_by_plugin(X, Y, beta_m, beta_M, k, method=method, block_generator = block_generator, max_iter=1)
    
    if return_time:
        return beta_hat, time.time()-t
    else:
        return beta_hat

def run_trials(
    beta : np.ndarray,
    k : int,
    quantities : list = [200, 10, 10, 10, 10],
    method : str = "TM",
    block_kind : str = "fixed",
    student_degrees : float = 4.0,
    correlation_rate : float = .5,
    algorithm : str = "gd",
    return_time : bool = False,
    n_trials : int = 70,
    n_jobs : int = 1,
    random_seed : int = 1
):
    if method == "ERM":
        print(f"Evaluating {method}...")
    else:
        print(f"Evaluating {method} experiment with parameter {k}...")
    rngs = [np.random.default_rng(random_seed + trial) for trial in range(n_trials)]

    with Pool(n_jobs) as pool:
        results = pool.starmap(
            run_single_trial,
            zip(
                rngs,
                repeat(beta),
                repeat(k),
                repeat(quantities),
                repeat(method),
                repeat(block_kind),
                repeat(student_degrees),
                repeat(correlation_rate),
                repeat(algorithm),
                repeat(return_time)
            )
        )
        return results

def divisors(n):
    divs = [1,n]
    for i in range(2,int(np.sqrt(n))+1):
        if n%i == 0:
            divs += [i,n//i]
    return list(set(divs))

def get_file_name_prefix(d, quantities, correlation_rate = 0, student_degrees = 4.0, algorithm : str = "gd"):
    return f"{d}-{'-'.join([str(q) for q in quantities])}-{correlation_rate}-{student_degrees}-{algorithm}"

def L2_error_gaussian(beta, beta_hat, correlation_rate = 0):
    d = beta.size
    cov = np.power( correlation_rate, np.abs( np.array([range(d),]*d) - np.array([range(d),]*d).T ) )
    return np.sqrt(np.dot( beta - beta_hat, cov @ (beta - beta_hat) ))

def run_combination_with_gaussian_data(
    OUT_DIR,
    beta,
    quantities,
    k,
    MOM_Ks,
    correlation_rate : float = 0,
    student_degrees : float = 4.0,
    algorithm : str = "gd",
    n_trials : int = 70,
    n_jobs : int = 1
):
    filename = get_file_name_prefix(beta.size, quantities, correlation_rate = correlation_rate, student_degrees = student_degrees, algorithm = algorithm) + ".json"
    if not (OUT_DIR / filename).is_file():
        
        ERM_errors = [L2_error_gaussian(beta, beta_hat, correlation_rate = correlation_rate) for beta_hat in run_trials(beta, k, quantities=quantities, method="ERM", n_trials=n_jobs, n_jobs=n_jobs, correlation_rate = correlation_rate, student_degrees = student_degrees, algorithm = algorithm)]
        TM_errors = [L2_error_gaussian(beta, beta_hat, correlation_rate = correlation_rate) for beta_hat in run_trials(beta, k, quantities=quantities, method="TM", n_trials=n_jobs, n_jobs=n_jobs, correlation_rate = correlation_rate, student_degrees = student_degrees, algorithm = algorithm)]
        MOM_errors = [[L2_error_gaussian(beta, beta_hat, correlation_rate = correlation_rate) for beta_hat in run_trials(beta, K, quantities=quantities, method="MOM", n_trials=n_jobs, n_jobs=n_jobs, correlation_rate = correlation_rate, student_degrees = student_degrees, algorithm = algorithm)] for K in MOM_Ks]

        data = {
            "ERM_errors" : ERM_errors,
            "TM_errors" : TM_errors,
            "MOM_errors" : MOM_errors,
            "k" : k,
            "Ks" : MOM_Ks
        }

        json.dump(data, open(OUT_DIR / filename, 'w'))
    return filename

def plot_combination_with_gaussian_data(
    DATA_DIR,
    OUT_DIR,
    d,
    quantities_list,
    x_values,
    x_label,
    experiment_name,
    experiment_invariant,
    correlation_rate : float = 0,
    student_degrees : float = 4.0,
    algorithm : str = "gd",
    methods : list = ["TM", "MOM", "ERM"],
):
    dfs = []
    methods.sort(reverse=True)
    for quantities, x_value in zip(quantities_list, x_values):
        filename = get_file_name_prefix(d, quantities, correlation_rate = correlation_rate, student_degrees = student_degrees, algorithm = algorithm) + ".json"
        data = json.load(open(DATA_DIR / filename, 'r'))

        ERM_errors = data["ERM_errors"]
        TM_errors = data["TM_errors"]
        MOM_errors = np.min(data["MOM_errors"], axis=0).tolist()

        #print(quantities[2], np.mean([data["Ks"][i] for i in np.argmin(data["MOM_errors"], axis=0)]) )

        distances = []
        if "TM" in methods:
            distances += TM_errors
        if "MOM" in methods:
            distances += MOM_errors
        if "ERM" in methods:
            distances += ERM_errors


        dfs.append(pd.DataFrame({
            "Method": sum([[m for i in range(len(ERM_errors))] for m in methods], []),
            r"$\left|\left| \hat{\beta}_n - \beta^* \right|\right|_{L^2}$": distances,
            x_label: [x_value for i in range(len(ERM_errors)*len(methods))],
        }))

    df = pd.concat(dfs)
    sns.set(rc={'figure.figsize':(5,4)})
    g = sns.boxplot(x=x_label, y=r"$\left|\left| \hat{\beta}_n - \beta^* \right|\right|_{L^2}$", hue="Method", data=df, linewidth=.1)
    g.set_yscale("log")
    if quantities[0] > 0:
        filename = f"{experiment_name}-{experiment_invariant}-gaussian-error-{d}-{correlation_rate}-{student_degrees}.pdf"
    else:
        filename = f"{experiment_name}-{experiment_invariant}-student-error-{d}-{correlation_rate}-{student_degrees}.pdf"
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 90)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.clf()