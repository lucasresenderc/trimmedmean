import pathlib
import numpy as np
import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("personal/trimmed_mean/results")

# number of jobs
n_jobs = 200

# number of trails per experiment
n_trials = n_jobs

# dimension of the feature space
d = 20

# beta
beta = np.ones(d)

# parameters to loop
for n in [5*d, 10*d, 20*d]:

    for n_contaminated in [ i for i in range(0, int(.1*n), int(.01*n) ) ] + [ i for i in range(int(.1*n), int(.5*n), int(.1*n) ) ]:
        print(f"Running n={n} and n_contaminated={n_contaminated}")

        # trimmed mean parameter
        k = 2*n_contaminated + 5

        # mom parameter family to vary
        MOM_Ks = list(set([i for i in range(np.max([1,2*n_contaminated-10]),np.min([2*n_contaminated+10,n])+1,1)] + ru.divisors(n)))

        # with normal error
        quantities = [n - n_contaminated, 0, n_contaminated, 0, 0]
        ru.run_combination_with_gaussian_data(
            OUT_DIR,
            beta,
            quantities,
            k,
            MOM_Ks,
            correlation_rate = 0,
            student_degrees = 4.0,
            n_trials=n_trials,
            n_jobs=n_jobs
        )    

        # with student error
        quantities = [0, 0, n_contaminated, 0, n - n_contaminated]
        for correlation_rate in [0, .5]:
            for student_degrees in [1.0, 2.0, 4.0]:
                ru.run_combination_with_gaussian_data(
                    OUT_DIR,
                    beta,
                    quantities,
                    k,
                    MOM_Ks,
                    correlation_rate = correlation_rate,
                    student_degrees = student_degrees,
                    n_trials=n_trials,
                    n_jobs=n_jobs
                ) 