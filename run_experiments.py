import numpy as np
from tqdm.auto import tqdm
import itertools
import json

from src.experiment import run_trials

N_JOBS = 32
N_TRIALS = 128
SAMPLE_SIZE = 1000
DIMENSION = 5

# '''

# EXPERIMENTS FOR METHOD COMPARISON

# '''

# print("Running experiments comparing different algorithms...")

# d = DIMENSION
# beta = np.ones(d)/d

# n = SAMPLE_SIZE

# r = n // 100
# n_contaminateds = [2 * r, 6 * r, 10 * r, 14 * r]
# error_types = [0,1]
# skews = [False, True]
# heteroscedasticitys = [False, True]

# mom_params = [2*nc+21 for nc in n_contaminateds]
# tm_params = [(p-1)/(2*n) for p in mom_params]

# algorithms = ["plugin", "gd"]
# selection_strategies = ["max_slope", "min_loss"]
# fold_Ks = ["maxK/V", "K/V"]

# results = []
# loop_list = list(itertools.product(
#     error_types, skews, heteroscedasticitys, n_contaminateds, algorithms, selection_strategies, fold_Ks
# ))
# for error_type, skew, heteroscedasticity, n_contaminated, algorithm, selection_strategy, fold_K in tqdm(loop_list, desc="For MoM"):
#     results += run_trials(
#         beta,
#         data_parameters = {
#             "type": "NormalContaminated",
#             "sample_size": n,
#             "sample_contaminated": n_contaminated,
#             "error_type": error_type,
#             "skew": skew,
#             "heteroscedasticity": heteroscedasticity,
#         },
#         method = "MOM",
#         params = mom_params,
#         algorithm = algorithm,
#         selection_strategy = selection_strategy,
#         fold_K = fold_K,
#         n_trials = N_TRIALS,
#         n_jobs = N_JOBS,
#         start_random_seed = 1
#     )
# loop_list = list(itertools.product(
#     error_types, skews, heteroscedasticitys, n_contaminateds, algorithms, selection_strategies
# ))
# for error_type, skew, heteroscedasticity, n_contaminated, algorithm, selection_strategy in tqdm(loop_list, desc="For TM"):
#     results += run_trials(
#         beta,
#         data_parameters = {
#             "type": "NormalContaminated",
#             "sample_size": n,
#             "sample_contaminated": n_contaminated,
#             "error_type": error_type,
#             "skew": skew,
#             "heteroscedasticity": heteroscedasticity,
#         },
#         method = "TM",
#         params = tm_params,
#         algorithm = algorithm,
#         selection_strategy = selection_strategy,
#         n_trials = N_TRIALS,
#         n_jobs = N_JOBS,
#         start_random_seed = 1
#     )
# json.dump(results, open("results/comparison_algorithms_strategies.json", "w"), indent=4)


# '''

# EXPERIMENTS FOR SETUP A

# '''

# print("Running experiments for Setup A...")

# d = DIMENSION
# beta = np.ones(d)/d

# n = SAMPLE_SIZE

# r = n // 100
# n_contaminateds = [0, 2 * r, 4 * r, 6 * r, 8 * r, 10 * r, 12 * r, 14 * r]
# error_types = [0, 1, 2]
# skews = [True, False]
# heteroscedasticitys = [True, False]

# mom_params = [2*nc+1 + 20 for nc in n_contaminateds]
# tm_params = [(p-1)/(2*n) for p in mom_params]

# results = []
# loop_list = list(itertools.product(
#     error_types, skews, heteroscedasticitys, n_contaminateds,
#     zip(["TM", "MOM","OLS"], [tm_params, mom_params, []])
# ))
# for error_type, skew, heteroscedasticity, n_contaminated, (method, params) in tqdm(loop_list):
#     results += run_trials(
#         beta,
#         data_parameters = {
#             "type": "NormalContaminated",
#             "sample_size": n,
#             "sample_contaminated": n_contaminated,
#             "error_type": error_type,
#             "skew": skew,
#             "heteroscedasticity": heteroscedasticity,
#         },
#         method = method,
#         params = params,
#         selection_strategy = "max_slope",
#         fold_K = "K/V",
#         n_trials = N_TRIALS,
#         n_jobs = N_JOBS,
#         start_random_seed = 1
#     )
# json.dump(results, open("results/setupA.json", "w"), indent=4)



d = DIMENSION
beta = np.ones(d)/d

n = SAMPLE_SIZE

r = n // 100
n_contaminateds = [0, 2 * r, 4 * r, 6 * r, 8 * r, 10 * r, 12 * r, 14 * r]
ps = [.01, .03, .05, .07, .09, .11, .13]

mom_params = [2*nc+1 + 20 for nc in n_contaminateds]
tm_params = [(p-1)/(2*n) for p in mom_params]

results = []
loop_list = list(itertools.product(
    ps, n_contaminateds,
    zip(["TM", "MOM","OLS"], [tm_params, mom_params, []])
))
for p, n_contaminated, (method, params) in tqdm(loop_list):
    results += run_trials(
        beta,
        data_parameters = {
            "type": "BernoulliNormal",
            "sample_size": n,
            "sample_contaminated": n_contaminated,
            "p": p
        },
        method = method,
        params = params,
        selection_strategy = "max_slope",
        fold_K = "K/V",
        n_trials = N_TRIALS,
        n_jobs = N_JOBS,
        start_random_seed = 1
    )
json.dump(results, open("results/setupB.json", "w"), indent=4)