import pathlib
import numpy as np
import robust_utils as ru

# output directory
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("results")

# parameters to loop
for eps in [.05, .2]:
    if eps == .05:
        n_range = [20,40,60,80,100,200]
        d = 5
    elif eps == .2:
        n_range = [n for n in range(90,220,30)]
        d = 20

    # with normal error
    quantities_list = [[n - int(n*eps), 0, int(n*eps), 0, 0] for n in n_range]
    ru.plot_combination_with_gaussian_data(
        DATA_DIR,
        OUT_DIR,
        d,
        quantities_list,
        n_range,
        r"$n$",
        "vary_sample_size",
        f"eps={eps}",
        correlation_rate = 0,
        student_degrees = 4.0,
        methods=["TM", "MOM"]
    )
    
    # with student error
    quantities_list = [[0, 0, int(n*eps), 0, n - int(n*eps)] for n in n_range]
    for correlation_rate in [0, .5]:
        for student_degrees in [1.0, 2.0, 4.0]:
            ru.plot_combination_with_gaussian_data(
                DATA_DIR,
                OUT_DIR,
                d,
                quantities_list,
                n_range,
                r"$n$",
                "vary_sample_size",
                f"eps={eps}",
                correlation_rate = correlation_rate,
                student_degrees = student_degrees,
                methods=["TM", "MOM"]
            )