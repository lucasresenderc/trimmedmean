import json
import time
from itertools import repeat
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from tqdm import tqdm

"""

    FIT FUNCTIONS

    MOMBlockGenerator: class that controls the generation of blocks for MOM
    MOMBlockGenerator::sort_blocks: define new blocks at random for MOM
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


class MOMBlockGenerator:
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
        B = [p[m * self.l : m * self.l + self.l] for m in range(self.K)]
        B[-1] = p[self.K * self.l - self.l :]
        return B

    def get_blocks(self):
        if self.kind == "fixed":
            return self.B
        else:
            return self.sort_blocks()


def get_active_indexes(k, loss_m, loss_M, method="TM", block_generator=None):
    diff = loss_m - loss_M
    if method == "MOM":
        B = block_generator.get_blocks()
        means = [np.mean(diff[b]) for b in B]
        return B[np.argsort(means)[len(means) // 2]]
    if method == "TM":
        return np.argsort(diff)[k:-k]


def update_by_armijo(X, Y, b, beta, armijo_rate, eta):
    YXb = Y[b] - X[b, :] @ beta
    direction = 2 * X[b, :].T @ YXb
    Xd = X[b, :] @ direction
    now = np.sum(np.power(YXb, 2))
    while np.sum(np.power(YXb - eta * Xd, 2)) > now:
        eta *= armijo_rate
    return beta + eta * direction, eta / armijo_rate


def fit_by_gd(
    X,
    Y,
    beta_m,
    beta_M,
    k: int,
    armijo_rate: float = 0.9,
    max_iter=1000,
    method="TM",
    block_generator=None,
    verbose=0,
):
    loss_m = np.power(Y - X @ beta_m, 2)
    loss_M = np.power(Y - X @ beta_M, 2)
    b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)

    eta = 10
    gamma = 10
    for i in range(1, max_iter):
        beta_m, eta = update_by_armijo(X, Y, b, beta_m, armijo_rate, eta)
        loss_m = np.power(Y - X @ beta_m, 2)
        b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)

        beta_M, gamma = update_by_armijo(X, Y, b, beta_M, armijo_rate, gamma)
        loss_M = np.power(Y - X @ beta_M, 2)
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
    k: int,
    max_iter=100,
    method="TM",
    block_generator=None,
):
    loss_m = np.power(Y - X @ beta_m, 2)
    loss_M = np.power(Y - X @ beta_M, 2)

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
        loss_m = np.power(Y - X @ beta_m, 2)

        b = get_active_indexes(k, loss_m, loss_M, method=method, block_generator=block_generator)
        beta_M = np.linalg.lstsq(X[b, :], Y[b], rcond=None)[0]
        loss_M = np.power(Y - X @ beta_M, 2)

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
    rng: np.random._generator.Generator,
    beta: np.ndarray,
    data_parameters: dict = {
        "type": "LerasleLecue",
        "quantities": [200, 10, 10, 10, 10],
        "student_degrees": 4.0,
        "correlation_rate": 0.5,
    },
):
    if data_parameters["type"] == "LerasleLecue":
        quantities = data_parameters["quantities"]
        student_degrees = data_parameters["student_degrees"]
        correlation_rate = data_parameters["correlation_rate"]

        n = np.sum(quantities)
        ns = np.cumsum(quantities)
        d = beta.size

        X = np.zeros(shape=(n, d))
        X[: ns[0], :] = rng.normal(size=(quantities[0], d))
        X[ns[0] : ns[2], :] += 1
        X[ns[2] : ns[3], :] = rng.uniform(size=(quantities[3], d))
        cov = np.power(
            correlation_rate,
            np.abs(
                np.array(
                    [
                        range(d),
                    ]
                    * d
                )
                - np.array(
                    [
                        range(d),
                    ]
                    * d
                ).T
            ),
        )
        X[ns[3] :, :] = rng.multivariate_normal(np.zeros(d), cov, size=quantities[4])

        Y = np.zeros(n)
        Y[: ns[0]] = X[: ns[0], :] @ beta + rng.normal(size=quantities[0])
        Y[ns[0] : ns[1]] += 1
        Y[ns[1] : ns[2]] += 10000
        Y[ns[2] : ns[3]] = 1.0 * (rng.uniform(size=quantities[3]) > 0.5)
        Y[ns[3] :] = X[ns[3] :, :] @ beta + rng.standard_t(student_degrees, size=quantities[4])

    elif data_parameters["type"] == "BernoulliNormal":
        n = data_parameters["sample_size"]
        nc = data_parameters["sample_contaminated"]
        p = data_parameters["p"]
        d = beta.size

        # generate zero entries with bernoulli
        n_entries_to_zero = n - rng.binomial(n, p)
        # add more zeros with corruption
        n_entries_to_zero += nc
        if n_entries_to_zero > n:
            n_entries_to_zero = n

        X = rng.normal(size=(n, d)) / np.sqrt(p)
        X[:n_entries_to_zero, :] *= 0
        Y = X @ beta + rng.normal(size=n)

    return X, Y


"""

    EXPERIMENT UTILS

"""


def run_single_trial(
    rng: np.random._generator.Generator,
    beta: np.ndarray,
    k: int,
    data_parameters: dict = {
        "type": "LerasleLecue",
        "quantities": [200, 10, 10, 10, 10],
        "student_degrees": 4.0,
        "correlation_rate": 0.5,
    },
    method: str = "TM",
    block_kind: str = "fixed",
    algorithm: str = "gd",
    return_time: bool = False,
):
    X, Y = generate_dataset(rng, beta, data_parameters=data_parameters)
    beta_m = rng.uniform(size=beta.size)
    beta_M = rng.uniform(size=beta.size)
    t = time.time()
    if method == "ERM":
        beta_hat = fit_by_lstsq(X, Y)
    else:
        block_generator = None
        if method == "MOM":
            block_generator = MOMBlockGenerator(block_kind, rng, len(Y), k)
        if algorithm == "gd":
            beta_hat = fit_by_gd(
                X,
                Y,
                beta_m,
                beta_M,
                k,
                method=method,
                block_generator=block_generator,
                max_iter=1000,
            )
        else:
            beta_hat = fit_by_plugin(
                X,
                Y,
                beta_m,
                beta_M,
                k,
                method=method,
                block_generator=block_generator,
                max_iter=2,
            )

    if return_time:
        return beta_hat, time.time() - t
    else:
        return beta_hat


def run_trials(
    beta: np.ndarray,
    k: int,
    data_parameters: dict = {
        "type": "LerasleLecue",
        "quantities": [200, 10, 10, 10, 10],
        "student_degrees": 4.0,
        "correlation_rate": 0.5,
    },
    method: str = "TM",
    block_kind: str = "fixed",
    algorithm: str = "gd",
    return_time: bool = False,
    n_trials: int = 70,
    n_jobs: int = 1,
    random_seed: int = 1,
):
    rngs = [np.random.default_rng(random_seed + trial) for trial in range(n_trials)]
    with Pool(n_jobs) as pool:
        results = pool.starmap(
            run_single_trial,
            zip(
                rngs,
                repeat(beta),
                repeat(k),
                repeat(data_parameters),
                repeat(method),
                repeat(block_kind),
                repeat(algorithm),
                repeat(return_time),
            ),
        )
        return results


def divisors(n):
    divs = [1, n]
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divs += [i, n // i]
    return list(set(divs))


def get_file_name_prefix(d, data_parameters, algorithm: str = "gd"):
    data_string = ""
    for key in data_parameters:
        data_string += key
        if type(data_parameters[key]) == list:
            data_string += "-" + "-".join([str(a) for a in data_parameters[key]])
        else:
            data_string += "-" + str(data_parameters[key])
    return f"{d}-{data_string}-{algorithm}"


def L2_error_gaussian(beta, beta_hat, correlation_rate=0):
    d = beta.size
    cov = np.power(
        correlation_rate,
        np.abs(
            np.array(
                [
                    range(d),
                ]
                * d
            )
            - np.array(
                [
                    range(d),
                ]
                * d
            ).T
        ),
    )
    return np.sqrt(np.dot(beta - beta_hat, cov @ (beta - beta_hat)))


def run_combination(
    OUT_DIR,
    beta,
    k,
    MOM_Ks,
    data_parameters: dict = {
        "type": "LerasleLecue",
        "quantities": [200, 10, 10, 10, 10],
        "student_degrees": 4.0,
        "correlation_rate": 0.5,
    },
    algorithm: str = "gd",
    n_trials: int = 70,
    n_jobs: int = 1,
):
    if data_parameters["type"] == "BernoulliNormal":
        correlation_rate = 0
    else:
        correlation_rate = data_parameters["correlation_rate"]

    filename = get_file_name_prefix(beta.size, data_parameters, algorithm=algorithm) + ".json"
    if not (OUT_DIR / filename).is_file():
        print("Evaluating ERM...")
        ERM_errors = [
            L2_error_gaussian(beta, beta_hat, correlation_rate=correlation_rate)
            for beta_hat in run_trials(
                beta,
                k,
                data_parameters=data_parameters,
                method="ERM",
                n_trials=n_trials,
                n_jobs=n_jobs,
                algorithm=algorithm,
            )
        ]
        print(f"Evaluating TM with parameter {k}...")
        TM_errors = [
            L2_error_gaussian(beta, beta_hat, correlation_rate=correlation_rate)
            for beta_hat in run_trials(
                beta,
                k,
                data_parameters=data_parameters,
                method="TM",
                n_trials=n_trials,
                n_jobs=n_jobs,
                algorithm=algorithm,
            )
        ]
        print(f"Evaluating MOM with parameters {','.join([str(K) for K in MOM_Ks])}...")
        MOM_errors = [
            [
                L2_error_gaussian(beta, beta_hat, correlation_rate=correlation_rate)
                for beta_hat in run_trials(
                    beta,
                    K,
                    data_parameters=data_parameters,
                    method="MOM",
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    algorithm=algorithm,
                )
            ]
            for K in tqdm(MOM_Ks)
        ]

        data = {
            "ERM_errors": ERM_errors,
            "TM_errors": TM_errors,
            "MOM_errors": MOM_errors,
            "k": k,
            "Ks": MOM_Ks,
        }

        json.dump(data, open(OUT_DIR / filename, "w"))
    return filename


def plot_combination(
    DATA_DIR,
    OUT_DIR,
    d,
    x_values,
    x_label,
    experiment_name,
    data_parameters_space,
    algorithm: str = "gd",
    methods: list = ["TM", "MOM", "ERM"],
):
    dfs = []
    methods.sort(reverse=True)
    for data_parameters, x_value in zip(data_parameters_space, x_values):
        filename = get_file_name_prefix(d, data_parameters, algorithm=algorithm) + ".json"
        data = json.load(open(DATA_DIR / filename, "r"))

        ERM_errors = data["ERM_errors"]
        TM_errors = data["TM_errors"]
        MOM_errors = np.min(data["MOM_errors"], axis=0).tolist()

        distances = []
        if "TM" in methods:
            distances += TM_errors
        if "MOM" in methods:
            distances += MOM_errors
        if "ERM" in methods:
            distances += ERM_errors

        correct_names = {"TM": "TM", "MOM": "Best-MoM", "ERM": "OLS"}

        dfs.append(
            pd.DataFrame(
                {
                    "Method": sum(
                        [[correct_names[m] for i in range(len(ERM_errors))] for m in methods],
                        [],
                    ),
                    r"$\left\| \hat{\beta}_n - \beta^\star \right\|_{L^2}$": distances,
                    x_label: [x_value for i in range(len(ERM_errors) * len(methods))],
                }
            )
        )

    df = pd.concat(dfs)
    sns.set(rc={"figure.figsize": (5, 4)})
    rc("text", usetex=True)
    g = sns.boxplot(
        x=x_label,
        y=r"$\left\| \hat{\beta}_n - \beta^\star \right\|_{L^2}$",
        hue="Method",
        data=df,
        linewidth=0.1,
    )
    g.set_yscale("log")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    filename = f"{experiment_name}.pdf"
    plt.xticks(rotation=45)
    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, pad_inches=0, bbox_inches='tight')
    plt.clf()
