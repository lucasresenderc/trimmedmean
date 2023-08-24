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

import numpy as np

import src.configs as configs

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


def fit_by_lstsq(X, Y):
    beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
    return beta_hat


def get_active_indexes(k, loss_m, loss_M, method, block_generator):
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
    armijo_rate: float = configs.DEFAULT_ARMIJO_RATE,
    max_iter = configs.DEFAULT_GD_MAX_ITER,
    method = configs.DEFAULT_METHOD,
    block_generator = configs.DEFAULT_BLOCK_GENERATOR,
):
    if (method == "TM" and k == 0) or (method == "MOM" and k == 1):
        return fit_by_lstsq(X,Y)

    loss_m = np.power(Y - X @ beta_m, 2)
    loss_M = np.power(Y - X @ beta_M, 2)
    b = get_active_indexes(k, loss_m, loss_M, method, block_generator)

    eta = 10
    gamma = 10
    for i in range(1, max_iter):
        beta_m, eta = update_by_armijo(X, Y, b, beta_m, armijo_rate, eta)
        loss_m = np.power(Y - X @ beta_m, 2)
        b = get_active_indexes(k, loss_m, loss_M, method, block_generator)

        beta_M, gamma = update_by_armijo(X, Y, b, beta_M, armijo_rate, gamma)
        loss_M = np.power(Y - X @ beta_M, 2)
        b = get_active_indexes(k, loss_m, loss_M, method, block_generator)

    return beta_M


def evaluate_cost(X, Y, loss, beta, k):
    b = np.argsort(loss)[k:-k]
    return np.mean(np.power(Y[b] - X[b, :] @ beta, 2))


def fit_by_plugin(
    X,
    Y,
    beta_m,
    beta_M,
    k: int,
    max_iter = configs.DEFAULT_PLUGIN_MAX_ITER,
    method = configs.DEFAULT_METHOD,
    block_generator = configs.DEFAULT_BLOCK_GENERATOR,
):
    if (method == "TM" and k == 0) or (method == "MOM" and k == 1):
        return fit_by_lstsq(X,Y)

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
        b = get_active_indexes(k, loss_m, loss_M, method, block_generator)
        beta_m = np.linalg.lstsq(X[b, :], Y[b], rcond=None)[0]
        loss_m = np.power(Y - X @ beta_m, 2)

        b = get_active_indexes(k, loss_m, loss_M, method, block_generator)
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