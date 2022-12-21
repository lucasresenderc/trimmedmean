import pathlib

import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("results")

# dimension of the feature space
d = 20

# sample size range
n_range = [60, 120, 180, 240, 360, 720]

# with normal error
eps = 0.4
quantities_list = [[n - int(n * eps), 0, int(n * eps), 0, 0] for n in n_range]
for correlation_rate in [0, 0.5]:
    ru.plot_combination(
        DATA_DIR,
        OUT_DIR,
        d,
        n_range,
        r"$n$",
        f"vary_sample_size_LL_gaussian_error_cr={correlation_rate}_eps={eps}",
        [
            {
                "type": "LerasleLecue",
                "quantities": quantities,
                "student_degrees": 4.0,
                "correlation_rate": correlation_rate,
            }
            for quantities in quantities_list
        ],
        methods=["TM", "MOM"],
    )

# with student error
eps = 0.2
quantities_list = [[0, 0, int(n * eps), 0, n - int(n * eps)] for n in n_range]
for correlation_rate in [0, 0.5]:
    for student_degrees in [1.0, 2.0, 4.0]:
        ru.plot_combination(
            DATA_DIR,
            OUT_DIR,
            d,
            n_range,
            r"$n$",
            f"vary_sample_size_LL_students_sd={student_degrees}_cr={correlation_rate}_eps={eps}",
            [
                {
                    "type": "LerasleLecue",
                    "quantities": quantities,
                    "student_degrees": student_degrees,
                    "correlation_rate": correlation_rate,
                }
                for quantities in quantities_list
            ],
            methods=["TM", "MOM"],
        )
