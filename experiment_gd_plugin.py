import pathlib
import numpy as np
import robust_utils as ru
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# output directory
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# number of jobs
n_jobs = 200

# number of trails per experiment
n_trials = n_jobs

# dimension of the feature space
d = 20
n = 100
n_contaminated = 10
beta = np.ones(d)
quantities = [n - n_contaminated, 0, n_contaminated, 0, 0]

k = 2*n_contaminated + 5

gd_trials = ru.run_trials(
    beta,
    k,
    quantities = quantities,
    method = "TM",
    algorithm = "gd",
    n_trials = n_trials,
    n_jobs = n_jobs,
    return_time = True
)
gd_errors = [ru.L2_error_gaussian(beta, trial[0], correlation_rate = 0) for trial in gd_trials]
gd_times = [trial[1] for trial in gd_trials]


plugin_trials = ru.run_trials(
    beta,
    k,
    quantities = quantities,
    method = "TM",
    algorithm = "plugin",
    n_trials = n_trials,
    n_jobs = n_jobs,
    return_time = True
)
plugin_errors = [ru.L2_error_gaussian(beta, trial[0], correlation_rate = 0) for trial in plugin_trials]
plugin_times = [trial[1] for trial in plugin_trials]

df = pd.DataFrame({
    "Solution error ratio (%)": np.array(plugin_errors)/np.array(gd_errors)*100,
    "Time ratio (%)": np.array(plugin_times)/np.array(gd_times)*100
})

print(df.describe())

sns.set(rc={'figure.figsize':(10,6)})
sns.histplot(df, x="Solution error ratio (%)", color="grey")
plt.tight_layout()
plt.savefig(OUT_DIR / "plugin_gd_err_diff.pdf")
plt.clf()

sns.histplot(df, x="Time ratio (%)", color="grey")
plt.tight_layout()
plt.savefig(OUT_DIR / "plugin_gd_time_diff.pdf")
plt.clf()