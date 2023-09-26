import numpy as np
from tqdm.auto import tqdm
import itertools
import json

from src.experiment import run_trials

N_JOBS = 8
N_TRIALS = 96

'''

EXPERIMENTS FOR SETUP A

'''

print("Running Setup A (it may take a few hours)...")

n = 300
d = 20
beta = np.ones(d)/np.sqrt(d)

r = n // 100
n_contaminateds = [0 * r, 2 * r, 4 * r, 6 * r, 8 * r, 10 * r, 15 * r, 20 * r, 30 * r, 40 * r]
error_types = [0,1,2,4]
skews = [False, True]
heteroscedasticitys = [False, True]

mom_params = [2*nc+21 for nc in n_contaminateds]
tm_params = [(p-1)/(2*n) for p in mom_params]

results = []
loop_list = list(itertools.product(
    error_types, skews, heteroscedasticitys, n_contaminateds
))
for alg in ["plugin", "admm"]:
    print(f"Using the {alg} algorithm:")
    for error_type, skew, heteroscedasticity, n_contaminated in tqdm(loop_list):
        for method, params in zip(["TM", "MOM", "OLS"], [tm_params, mom_params, []]):
            results += run_trials(
                beta,
                data_parameters = {
                    "type": "NormalContaminated",
                    "sample_size": n,
                    "sample_contaminated": n_contaminated,
                    "error_type": error_type,
                    "skew": skew,
                    "heteroscedasticity": heteroscedasticity,
                },
                method = method,
                block_kind = "random",
                params = params,
                algorithm = alg,
                n_trials = N_TRIALS,
                n_jobs = N_JOBS,
                start_random_seed = 1
            )
    json.dump(results, open(f"results/setupA_{alg}.json", "w"), indent=4)


'''

EXPERIMENTS FOR SETUP B

'''

print("Running Setup B (it may take a few hours)...")

n = 1000
d = 5
beta = np.ones(d)/d

r = n // 100
n_contaminateds = [0, 2 * r, 4 * r, 6 * r, 8 * r, 10 * r, 12 * r, 14 * r]
ps = [.03, .07, .11]

mom_params = [2*nc+1 + 20 for nc in n_contaminateds]
tm_params = [(p-1)/(2*n) for p in mom_params]

results = []
loop_list = list(itertools.product(ps, n_contaminateds))
for alg in ["admm", "plugin"]:
    print(f"Using the {alg} algorithm:")
    for p, n_contaminated in tqdm(loop_list):
        for method, params in zip(["TM", "MOM","OLS"], [tm_params, mom_params, []]):
            results += run_trials(
                beta,
                data_parameters = {
                    "type": "BernoulliNormal",
                    "sample_size": n,
                    "sample_contaminated": n_contaminated,
                    "p": p
                },
                method = method,
                block_kind = "random",
                params = params,
                algorithm = alg,
                n_trials = N_TRIALS,
                n_jobs = N_JOBS,
                start_random_seed = 1
            )
    json.dump(results, open(f"results/setupB_{alg}.json", "w"), indent=4)