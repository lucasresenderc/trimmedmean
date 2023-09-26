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
        if self.K > self.n:
            self.K = self.n
        self.l = self.n // self.K

        self.I = [[i + m*self.K for m in range(self.l)] for i in range(self.K)]
        for i in range(self.n % self.K):
            self.I[i].append(self.K * self.l + i)

        if kind == "fixed":
            self.B = self.sort_blocks()

    def sort_blocks(self):
        p = np.arange(self.n)
        self.rng.shuffle(p)
        return [p[i] for i in self.I]

    def get_blocks(self):
        if self.kind == "fixed":
            return self.B
        else:
            return self.sort_blocks()


def fit_by_lstsq(X, Y):
    beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
    cost = np.mean(np.power(Y - X @ beta_hat, 2))
    return [[beta_hat, beta_hat]], [[cost, cost]]


def get_active_indexes(k, loss, method, block_generator):
    if method == "MOM":
        B = block_generator.get_blocks()
        means = [np.mean(loss[b]) for b in B]
        return B[np.argsort(means)[len(means) // 2]]
    if method == "TM":
        return np.argsort(loss)[k:-k]


def update_by_armijo(X, Y, b, beta, armijo_rate, eta):
    YXb = Y[b] - X[b, :] @ beta
    direction = 2 * X[b, :].T @ YXb
    Xd = X[b, :] @ direction
    now = np.sum(np.power(YXb, 2))
    while np.sum(np.power(YXb - eta * Xd, 2)) > now:
        eta *= armijo_rate
    return beta + eta * direction, eta / armijo_rate


def fit_by_aasd(
    X,
    Y,
    beta_m,
    beta_M,
    k: int,
    armijo_rate: float = configs.DEFAULT_ARMIJO_RATE,
    max_iter: int = configs.DEFAULT_GD_MAX_ITER,
    method: str = configs.DEFAULT_METHOD,
    block_generator = configs.DEFAULT_BLOCK_GENERATOR,
):
    if (method == "TM" and k == 0) or (method == "MOM" and k == 1):
        return fit_by_lstsq(X,Y)

    eta = 10
    gamma = 10

    loss_m = np.power(Y - X @ beta_m, 2)
    loss_M = np.power(Y - X @ beta_M, 2)
    b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

    beta_history = [[np.copy(beta_m), np.copy(beta_M)]]
    cost_history = [[
        np.mean(loss_m[get_active_indexes(k, loss_m, method, block_generator)]),
        np.mean(loss_M[get_active_indexes(k, loss_M, method, block_generator)])
    ]]
    for _ in range(1, max_iter):
        beta_m, eta = update_by_armijo(X, Y, b, beta_m, armijo_rate, eta)
        loss_m = np.power(Y - X @ beta_m, 2)
        b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

        beta_M, gamma = update_by_armijo(X, Y, b, beta_M, armijo_rate, gamma)
        loss_M = np.power(Y - X @ beta_M, 2)
        b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

        beta_history.append([np.copy(beta_m), np.copy(beta_M)])
        cost_history.append([
            np.mean(loss_m[get_active_indexes(k, loss_m, method, block_generator)]),
            np.mean(loss_M[get_active_indexes(k, loss_M, method, block_generator)])
        ])

    return beta_history, cost_history 
    

def fit_by_admm(
    X,
    Y,
    beta_m,
    beta_M,
    k: int,
    max_iter: int = configs.DEFAULT_GD_MAX_ITER,
    method: str = configs.DEFAULT_METHOD,
    block_generator = configs.DEFAULT_BLOCK_GENERATOR
):
    if (method == "TM" and k == 0) or (method == "MOM" and k == 1):
        return fit_by_lstsq(X,Y)

    d = beta_m.size
    rho = 5.0
    
    loss_m = np.power(Y - X @ beta_m, 2)
    loss_M = np.power(Y - X @ beta_M, 2)
    b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

    beta_history = [[np.copy(beta_m), np.copy(beta_M)]]
    cost_history = [[
        np.mean(loss_m[get_active_indexes(k, loss_m, method, block_generator)]),
        np.mean(loss_M[get_active_indexes(k, loss_M, method, block_generator)])
    ]]
    for _ in range(1, max_iter):
        beta_m = np.linalg.solve((X[b, :].T)@X[b, :] + rho*np.identity(d), (X[b, :].T)@Y[b] + rho*beta_m)
        loss_m = np.power(Y - X @ beta_m, 2)
        b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

        beta_M = np.linalg.solve((X[b, :].T)@X[b, :] + rho*np.identity(d), (X[b, :].T)@Y[b] + rho*beta_M)
        loss_M = np.power(Y - X @ beta_M, 2)
        b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

        beta_history.append([np.copy(beta_m), np.copy(beta_M)])
        cost_history.append([
            np.mean(loss_m[get_active_indexes(k, loss_m, method, block_generator)]),
            np.mean(loss_M[get_active_indexes(k, loss_M, method, block_generator)])
        ])

    return beta_history, cost_history 


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
    b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

    beta_history = [[np.copy(beta_m), np.copy(beta_M)]]
    cost_history = [[
        np.mean(loss_m[get_active_indexes(k, loss_m, method, block_generator)]),
        np.mean(loss_M[get_active_indexes(k, loss_M, method, block_generator)])
    ]]
    for _ in range(max_iter):
        beta_m = np.linalg.lstsq(X[b, :], Y[b], rcond=None)[0]
        loss_m = np.power(Y - X @ beta_m, 2)
        b = get_active_indexes(k, loss_m - loss_M, method, block_generator)
        
        beta_M = np.linalg.lstsq(X[b, :], Y[b], rcond=None)[0]
        loss_M = np.power(Y - X @ beta_M, 2)
        b = get_active_indexes(k, loss_m - loss_M, method, block_generator)

        beta_history.append([np.copy(beta_m), np.copy(beta_M)])
        cost_history.append([
            np.mean(loss_m[get_active_indexes(k, loss_m, method, block_generator)]),
            np.mean(loss_M[get_active_indexes(k, loss_M, method, block_generator)])
        ])

    return beta_history, cost_history
