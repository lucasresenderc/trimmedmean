import pathlib
import numpy as np
import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("results")

# dimension of the feature space
d = 20

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
        ru.plot_combination(
                DATA_DIR,
                OUT_DIR,
                d,
                eps_list,
                r"$\varepsilon$",
                f"theta_zero_p={p}_n={n}",
                [
                    {
                        "type": "BernoulliNormal",
                        "sample_size": n,
                        "sample_contaminated": int(n*eps),
                        "p" : p
                    }
                    for eps in eps_list
                ]
            )