"""

    DATA GENERATION

    generate_dataset: ...

"""

import numpy as np

import src.configs as configs 
from src.distributions import SkeGTD


def generate_dataset(
    rng: np.random._generator.Generator,
    beta: np.ndarray,
    data_parameters: dict = configs.DEFAULT_DATA_PARAMETERS,
):
    if data_parameters["type"] == "SkeGTD":
        n = data_parameters["sample_size"]
        nc = data_parameters["sample_contaminated"]
        alpha = data_parameters["alpha"]
        skew = data_parameters["skew"]
        d = beta.size

        if alpha == "inf":
            alpha = np.inf

        X = rng.normal(size=(n, d))

        dist = SkeGTD(alpha, skew, rng)
        xi = dist.rvs(shape=n)
        if alpha > .5:
            xi -= dist.mean()

        if data_parameters["heteroscedasticity"]:
            xi = xi * np.exp(np.sum(X**2, axis=1)/(2*d))

        Y = X @ beta + xi
        Y[:nc] = 1000

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