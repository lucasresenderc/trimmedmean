import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import robust_utils as ru

def L2_error_gaussian(beta, beta_hat):
    return np.sqrt(np.dot( beta - beta_hat, beta - beta_hat ))

n_jobs = 200

d = 20
beta = np.ones(d)
quantities = [495, 0, 5, 0, 0]
k = 10

ERM_trials = [ ru.run_gd_trials(beta, k, quantities=quantities, method="ERM", n_trials=n_jobs, n_jobs=n_jobs) ]
TM_trials = [ ru.run_gd_trials(beta, k, quantities=quantities, method="TM", n_trials=n_jobs, n_jobs=n_jobs) ]
MOM_trials = [ ru.run_gd_trials(beta, K, quantities=quantities, method="MOM", n_trials=n_jobs, n_jobs=n_jobs) for K in range(2,21,1) ]

ERM_errors = [ L2_error_gaussian(beta, beta_hat) for beta_hat in ERM_trials[0] ]
TM_errors = [ L2_error_gaussian(beta, beta_hat) for beta_hat in TM_trials[0] ]
MOM_errors = np.min([ [L2_error_gaussian(beta, beta_hat) for beta_hat in trial] for trial in MOM_trials ], axis=0).tolist()

MOM_Ks = (np.argmin([ [L2_error_gaussian(beta, beta_hat) for beta_hat in trial] for trial in MOM_trials ], axis=0) + 2).tolist()

df = pd.DataFrame({
    "Method": ["TM" for i in TM_errors] + ["MOM" for i in MOM_errors] + ["ERM" for i in ERM_errors],
    "Distance": TM_errors + MOM_errors + ERM_errors,
    "Parameter": [k for i in TM_errors] + MOM_Ks + [0 for i in ERM_errors]
})

df.to_csv(f"personal/trimmed_mean/{'-'.join([str(q) for q in quantities])}.csv")