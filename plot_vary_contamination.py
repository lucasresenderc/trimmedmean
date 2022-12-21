import pathlib

import numpy as np

import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("results")

# dimension of the feature space
d = 20

# parameters to loop
for n in [120, 240, 360]:
    r = int(n / 120)
    n_contaminated_list = [0, 3 * r, 6 * r, 9 * r, 12 * r, 24 * r, 36 * r, 48 * r]

    # with normal error
    quantities_list = [[n - n_contaminated, 0, n_contaminated, 0, 0] for n_contaminated in n_contaminated_list]
    for correlation_rate in [0, 0.5]:
        ru.plot_combination(
            DATA_DIR,
            OUT_DIR,
            d,
            [np.around(q / n, decimals=3) for q in n_contaminated_list],
            r"$\varepsilon$",
            f"vary_contamination_LL_gaussian_error_cr={correlation_rate}_n={n}",
            [
                {
                    "type": "LerasleLecue",
                    "quantities": quantities,
                    "student_degrees": 4.0,
                    "correlation_rate": correlation_rate,
                }
                for quantities in quantities_list
            ],
        )

    # with student error
    quantities_list = [[0, 0, n_contaminated, 0, n - n_contaminated] for n_contaminated in n_contaminated_list]
    for correlation_rate in [0, 0.5]:
        for student_degrees in [1.0, 2.0, 4.0]:
            ru.plot_combination(
                DATA_DIR,
                OUT_DIR,
                d,
                [np.around(q / n, decimals=3) for q in n_contaminated_list],
                r"$\varepsilon$",
                f"vary_contamination_LL_students_sd={student_degrees}_cr={correlation_rate}_n={n}",
                [
                    {
                        "type": "LerasleLecue",
                        "quantities": quantities,
                        "student_degrees": student_degrees,
                        "correlation_rate": correlation_rate,
                    }
                    for quantities in quantities_list
                ],
            )
