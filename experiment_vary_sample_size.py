import pathlib
import numpy as np
import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# number of jobs
n_jobs = 200

# number of trails per experiment
n_trials = n_jobs

# parameters to loop
for eps in [.05, .2]:
    if eps == .05:
        n_range = [20,40,60,80,100,200]
        d = 5
    elif eps == .2:
        n_range = [n for n in range(90,220,30)]
        d = 20
    # beta
    beta = np.ones(d)

    for n in n_range:
        print(f"Running eps={eps} and n={n}")
        n_contaminated = int(eps*n)

        # trimmed mean parameter
        k = 2*n_contaminated + 5

        # mom parameter family to vary
        MOM_Ks = [2*n_contaminated+1] #list(set([i for i in range(np.max([1,2*n_contaminated-10]),np.min([2*n_contaminated+10,n])+1,1)] + ru.divisors(n)))

        # with normal error
        quantities = [n - n_contaminated, 0, n_contaminated, 0, 0]
        ru.run_combination(
            OUT_DIR,
            beta,
            k,
            MOM_Ks,
            data_parameters = {
                "type": "LerasleLecue",
                "quantities": quantities,
                "student_degrees": 4.0,
                "correlation_rate" : 0.0
            },
            n_trials=n_trials,
            n_jobs=n_jobs
        )       

        # with student error
        quantities = [0, 0, n_contaminated, 0, n - n_contaminated]
        for correlation_rate in [0, .5]:
            for student_degrees in [1.0, 2.0, 4.0]:
                ru.run_combination(
                    OUT_DIR,
                    beta,
                    k,
                    MOM_Ks,
                    data_parameters = {
                        "type": "LerasleLecue",
                        "quantities": quantities,
                        "student_degrees": student_degrees,
                        "correlation_rate" : correlation_rate
                    },
                    n_trials=n_trials,
                    n_jobs=n_jobs
                )