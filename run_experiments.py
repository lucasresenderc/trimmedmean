import numpy as np
from tqdm.auto import tqdm
import itertools
import json

from src.experiment import run_trials

N_JOBS = 8
N_TRIALS = 96

'''

EXPERIMENTS FOR SETUP A
WITH UNKNOWN CONTAMINATION
(REQUIRES CROSSVALIDATION)

'''

print("Running Setup A with contamination (it may take a few hours)...")

n = 300
d = 20
beta = np.ones(d)/np.sqrt(d)

r = n // 100
n_contaminateds = [0 * r, 2 * r, 4 * r, 6 * r, 8 * r, 10 * r, 15 * r, 20 * r, 30 * r, 40 * r]
alphas = ["inf", .5, 1, 2]
skews = [0, .3, .6, .9]
heteroscedasticitys = [False, True]

mom_params = [2*nc+21 for nc in n_contaminateds]
tm_params = [(p-1)/(2*n) for p in mom_params]
huber_params = np.linspace(1.0, 1.35, 10).tolist()

results = []
loop_list = list(itertools.product(
    alphas, skews, heteroscedasticitys, n_contaminateds
))
for alg in ["admm", "plugin"]: # add admm
    print(f"Using the {alg} algorithm:")
    for alpha, skew, heteroscedasticity, n_contaminated in tqdm(loop_list):
        for method, params in zip(["TM", "MOM", "OLS", "QUANTILE", "HUBER"], [tm_params, mom_params, [], [], huber_params]):
            results += run_trials(
                beta,
                data_parameters = {
                    "type": "SkeGTD",
                    "sample_size": n,
                    "sample_contaminated": n_contaminated,
                    "alpha": alpha,
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

EXPERIMENTS FOR SETUP A
WITH NO CONTAMINATION

'''

print("Running Setup A without contamination (it is faster)...")

n = 300
d = 20
beta = np.ones(d)/np.sqrt(d)

r = n // 100
alphas = ["inf", .5, 1, 2]
skews = [0, .15, .3, .45, .6, .75, .9]
heteroscedasticitys = [False, True]

results = []
loop_list = list(itertools.product(
    alphas, skews, heteroscedasticitys
))
for alg in ["plugin"]:
    print(f"Using the {alg} algorithm:")
    for alpha, skew, heteroscedasticity in tqdm(loop_list):
        for method, params in zip(["TM", "MOM", "OLS", "QUANTILE", "HUBER"], [[5/n], [5], [], [], [1.35]]):
            results += run_trials(
                beta,
                data_parameters = {
                    "type": "SkeGTD",
                    "sample_size": n,
                    "sample_contaminated": 0,
                    "alpha": alpha,
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
    json.dump(results, open(f"results/setupA_{alg}_eps_zero.json", "w"), indent=4)


'''

EXPERIMENTS FOR SETUP B

'''

print("Running Setup B (it may take a few minutes)...")

n = 1000
d = 5
beta = np.ones(d)/d

r = n // 100
n_contaminateds = [0, 2 * r, 4 * r, 6 * r, 8 * r, 10 * r, 12 * r, 14 * r]
ps = [.05, .09]

mom_params = [2*nc+1 + 20 for nc in n_contaminateds]
tm_params = [(p-1)/(2*n) for p in mom_params]
huber_params = np.linspace(1.0, 1.35, 10).tolist()

results = []
loop_list = list(itertools.product(ps, n_contaminateds))
for alg in ["plugin"]:
    for p, n_contaminated in tqdm(loop_list):
        for method, params in zip(["TM", "MOM","OLS", "HUBER", "QUANTILE"], [tm_params, mom_params, [], huber_params, []]):
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
    json.dump(results, open(f"results/setupB.json", "w"), indent=4)