# trimmedmean

This repository is a companion to the text "Trimmed sample means for robust uniform mean estimation and regression".

## How is it structured?

This repository has 6 main scripts:
- **robust_utils.py:** contains the implementation of the adapted gradient descent for the trimmed mean and for the median of means, it also contains functions to realize parallel computation, generate artificial data and plot experiments;

- **experiment_gd_plugin.py:** realizes an experiment comparing the gradient descent method with the plug-in method for the linear regression with quadratic error. It generates two plots, both saved on <code>/plots</code>, one with information about the computational time between methods and another with their performance;

- **experiment_vary_contamination:** realizes an experiment with every variable fixed but the contamination. Its outputs are saved as <code>.json</code> files on <code>/results</code>;

- **plot_vary_contamination.py:** reads the output files of the last script and generate a plot saved on <code>/plots</code>. The name of the file will correspond to the arguments used on the script;

- **exoeriment_vary_sample_size and plot_vary_sample_size:** those files are analogous to the former two.

## What if I want to use it to fit some data of my own?

The most basic usage of our code is shown below:
<pre>
import numpy as np
import robust_utils as ru

X, Y = load_our_data(...)
eps = 0.1 # an upper bound for the contamination level
d = X.shape[1] # dimension of the features

# set rng seed to reproducibility
# and generate initial values for beta_m and beta_M
rng = np.random.default_rng(42)
beta_m = rng.uniform(size=d)
beta_M = rng.uniform(size=d)

# return the estimator
beta_hat = ru.fit_by_gd(X, Y, beta_m, beta_M, k, method="TM", block_generator = None, max_iter=1000)
</pre>

## How to generate the same results as in the article?

Just change the parameter <code>n_jobs</code> on the experiment scripts to fit your machine specs and run the following:
<pre>
python experiment_gd_plugin.py
python experiment_vary_contamination.py
python experiment_vary_sample_size.py
python plot_vary_contamination.py
python plot_vary_sample_size.py
</pre>

Now, take a look at <code>/plots</code>.