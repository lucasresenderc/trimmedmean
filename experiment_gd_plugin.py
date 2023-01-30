import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import robust_utils as ru

# helper function
def process_results(n, eps, config, plugin_errors, gd_errors):
    df.append(pd.DataFrame(
        {
            "Error": plugin_errors + gd_errors,
            "Method": ["Plug-in"]*n_trials + ["AASD"]*n_trials,
            "Config": [config]*(2*n_trials)
        }
    ))
    pmu, pstd, pm = np.mean(plugin_errors),np.std(plugin_errors),np.median(plugin_errors)
    gmu, gstd, gm = np.mean(gd_errors),np.std(gd_errors),np.median(gd_errors)

    print(f"{n} & {eps} & {config} & {gmu:0.3f} & {gstd:0.3f} & {pmu:0.3f} & {pstd:0.3f} & {int((pmu - gmu)/gmu*100)} & {int((pstd - gstd)/gstd*100)} \\\\")

# output directory
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# number of jobs
n_jobs = 24

# number of trails per experiment
n_trials = 240

# dimension of the feature space
d = 20

# beta
beta = np.ones(d)

# sample size
ns = [120, 360]

# contamination levels
epss = [.05, 0.2]

for eps in epss:
    for n in ns:
        df = []

        # gaussian
        # number of contaminated samples
        n_contaminated = int(eps*n)

        # sample parameters
        quantities = [n - n_contaminated, 0, n_contaminated, 0, 0]

        # TM parameters
        k = 2 * n_contaminated + 5

        gd_trials = ru.run_trials(
            beta,
            k,
            data_parameters={
                "type": "LerasleLecue",
                "quantities": quantities,
                "student_degrees": 1.0,
                "correlation_rate": 0.0,
            },
            method="TM",
            algorithm="gd",
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        gd_errors = [ru.L2_error_gaussian(beta, trial[0], correlation_rate=0) for trial in gd_trials]

        plugin_trials = ru.run_trials(
            beta,
            k,
            data_parameters={
                "type": "LerasleLecue",
                "quantities": quantities,
                "student_degrees": 1.0,
                "correlation_rate": 0.0,
            },
            method="TM",
            algorithm="plugin",
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        plugin_errors = [ru.L2_error_gaussian(beta, trial, correlation_rate=0) for trial in plugin_trials]

        process_results(n, eps, "gaussian", plugin_errors, gd_errors)
        

        for student_degrees in [1.0, 2.0, 4.0]:
            # sample parameters
            quantities = [0, 0, n_contaminated, 0, n - n_contaminated]

            # TM parameters
            k = 2 * n_contaminated + 5

            gd_trials = ru.run_trials(
                beta,
                k,
                data_parameters={
                    "type": "LerasleLecue",
                    "quantities": quantities,
                    "student_degrees": student_degrees,
                    "correlation_rate": 0.0,
                },
                method="TM",
                algorithm="gd",
                n_trials=n_trials,
                n_jobs=n_jobs
            )
            gd_errors = [ru.L2_error_gaussian(beta, trial, correlation_rate=0) for trial in gd_trials]

            plugin_trials = ru.run_trials(
                beta,
                k,
                data_parameters={
                    "type": "LerasleLecue",
                    "quantities": quantities,
                    "student_degrees": student_degrees,
                    "correlation_rate": 0.0,
                },
                method="TM",
                algorithm="plugin",
                n_trials=n_trials,
                n_jobs=n_jobs
            )
            plugin_errors = [ru.L2_error_gaussian(beta, trial, correlation_rate=0) for trial in plugin_trials]

            process_results(n, eps, student_degrees, plugin_errors, gd_errors)

        df = pd.concat(df)
        sns.set(rc={"figure.figsize": (5, 4)})
        g = sns.boxplot(
            x="Config",
            y="Error",
            hue="Method",
            data=df,
            linewidth=0.1,
            palette="Set2"
        )
        g.set_yscale("log")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        filename = f"two_algs_n={n}_eps={eps}.pdf"
        plt.xticks(rotation=45)
        plt.yticks(rotation=90)
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, pad_inches=0, bbox_inches='tight')
        plt.clf()
