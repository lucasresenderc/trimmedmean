# trimmedmean

This repository is a companion to the text "Trimmed sample means for robust uniform mean estimation and regression".

## How is it structured?

All code required to perform robust regression and cross-validation is available at <code>src/</code>.

To run the experiments simply execute the <code> run_experiments.py </code> file (it may take several hours to finish).

The images and tables appearing in the article and some more insights can be obtained running the notebook <code>Results.ipynb</code>.

## What if I want to use it to fit some data of my own?

The most basic usage of our code is shown below:
```python
import numpy as np
import robust_utils as ru

X, Y = load_our_data(...)

cv_params = [...] # The grid of parameters for the cross-validation procedure

# set rng seed to reproducibility
# and generate initial values for beta_m and beta_M
rng = np.random.default_rng(42)
beta_m = rng.uniform(size=d)
beta_M = rng.uniform(size=d)

results = cross_validate(
    X,
    Y,
    beta_m,
    beta_M,
    rng,
    n_folds = 5,
    method = "TM",
    params = cv_params,
    algorithm = "plugin",
)
# return a list with four choices for beta_hat,
# with each line in the following form
# {
# "cv_strategy": "max_slope",
# "beta_strategy": "best",
# "best_param": "...",
# "beta_hat": [...]
# }
# we recommend selecting the line with
# "cv_strategy": "max_slope" and
# "beta_strategy": "best"

```
