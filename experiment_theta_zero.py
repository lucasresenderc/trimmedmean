import pathlib
import numpy as np
import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# number of jobs
n_jobs = 240

# number of trails per experiment
n_trials = n_jobs

# dimension of the feature space
d = 20

# beta
beta = np.ones(d)

# list of ps and eps_lists
ps = [0.05, 0.1, 0.2, 0.3]
eps_lists = [
    [.005, .01, .015, .02, .03, .04, .05, .06, .07],
    [.01, .02, .03, .04, .06, .08, .1, .12, .14],
    [.02, .04, .06, .08, .12, .16, .2, .24, .28],
    [.03, .06, .09, .12, .18, .24, .3, .36, .42]
]

for n in [100, 1000]:
    for p, eps_list in zip(ps, eps_lists):
        for eps in eps_list:
            print(f"Running n={n}, eps={eps} and p={p}")

            n_contaminated = int(n*eps)

            # trimmed mean parameter
            k = n_contaminated + 5

            # mom parameter family to vary
            MOM_Ks = [2*n_contaminated+1]

            ru.run_combination(
                OUT_DIR,
                beta,
                k,
                MOM_Ks,
                data_parameters = {
                    "type": "BernoulliNormal",
                    "sample_size": n,
                    "sample_contaminated": n_contaminated,
                    "p" : p
                },
                n_trials=n_trials,
                n_jobs=n_jobs
            )    