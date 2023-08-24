DEFAULT_DATA_PARAMETERS = {
    "type": "NormalContaminated",
    "sample_size": 1000,
    "sample_contaminated": 200,
    "error_type": 0, #0 = Normal, > 0 is the df of students
    "skew": False,
    "heteroscedasticity": False,
}

DEFAULT_METHOD = "TM"
DEFAULT_BLOCK_KIND = "random"
DEFAULT_ALGORITHM = "plugin"

DEFAULT_N_TRIALS = 8
DEFAULT_N_JOBS = 1
DEFAULT_START_RANDOM_SEED = 42

DEFAULT_N_FOLDS = 5
DEFAULT_SELECTION_STRATEGY = "max_slope"
DEFAULT_FOLD_K = "maxK/V"

DEFAULT_BLOCK_GENERATOR = None
DEFAULT_ARMIJO_RATE = 0.9
DEFAULT_GD_MAX_ITER = 1000
DEFAULT_PLUGIN_MAX_ITER = 10