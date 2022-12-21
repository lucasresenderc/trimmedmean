import numpy as np
import robust_utils as ru

n = 100
d = 2
p = 0.25
eps = 0.1
k = int(eps*n) + 1

print(f"k = {k}")

beta = np.ones(d)
X, Y = ru.generate_dataset(
    np.random.default_rng(1),
    beta,
    data_parameters = {
                    "type": "BernoulliNormal",
                    "sample_size": n,
                    "sample_contaminated": int(n*eps),
                    "p" : p
                }
)

# set rng seed to reproducibility
# and generate initial values for beta_m and beta_M
rng = np.random.default_rng(42)
beta_m = rng.uniform(size=d)
beta_M = rng.uniform(size=d)

# return the estimator
beta_hat = ru.fit_by_gd(X, Y, beta_m, beta_M, k, method="TM", block_generator = None, max_iter=10)
