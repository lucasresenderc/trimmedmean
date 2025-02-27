# import numpy as np
# from scipy.stats import norm, t
# from scipy.integrate import quad

DEFAULT_DATA_PARAMETERS = {
    "type": "SkeSGD",
    "sample_size": 1000,
    "sample_contaminated": 200,
    "alpha": "inf", # inf = normal, n/2 = st with n df, 1/2 is cauchy
    "skew": 0,
    "heteroscedasticity": False,
}

DEFAULT_METHOD = "TM"
DEFAULT_BLOCK_KIND = "random"
DEFAULT_ALGORITHM = "plugin"

DEFAULT_N_TRIALS = 8
DEFAULT_N_JOBS = 1
DEFAULT_START_RANDOM_SEED = 1

DEFAULT_N_FOLDS = 5

DEFAULT_BLOCK_GENERATOR = None
DEFAULT_ARMIJO_RATE = 0.9
DEFAULT_GD_MAX_ITER = 20
DEFAULT_PLUGIN_MAX_ITER = 20