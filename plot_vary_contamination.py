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
for n in [5*d, 10*d, 20*d]:
    n_contaminated_list = [ i for i in range(0, int(.1*n), int(.01*n) ) ] + [ i for i in range(int(.1*n), int(.5*n), int(.1*n) ) ]
    
    # with normal error
    quantities_list = [[n - n_contaminated, 0, n_contaminated, 0, 0] for n_contaminated in n_contaminated_list]
    ru.plot_combination(
        DATA_DIR,
        OUT_DIR,
        d,
        [q/n for q in n_contaminated_list],
        r"$\varepsilon$",
        f"vary_contamination_LL_gaussian_error_n={n}",
        [
            {
                "type": "LerasleLecue",
                "quantities": quantities,
                "student_degrees": 4.0,
                "correlation_rate" : 0.0
            }
            for quantities in quantities_list
        ]
    )

    # with student error
    quantities_list = [[0, 0, n_contaminated, 0, n - n_contaminated] for n_contaminated in n_contaminated_list]
    for correlation_rate in [0, .5]:
        for student_degrees in [1.0, 2.0, 4.0]:
            ru.plot_combination(
                DATA_DIR,
                OUT_DIR,
                d,
                [q/n for q in n_contaminated_list],
                r"$\varepsilon$",
                f"vary_contamination_LL_students_sd={student_degrees}_cr={correlation_rate}_n={n}",
                [
                    {
                        "type": "LerasleLecue",
                        "quantities": quantities,
                        "student_degrees": student_degrees,
                        "correlation_rate" : correlation_rate
                    }
                    for quantities in quantities_list
                ]
            )