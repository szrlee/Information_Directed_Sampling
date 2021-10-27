# %%
""" Packages import """
import os
import expe as exp
import numpy as np

# import jax.numpy as np
import pickle as pkl
import utils
import time

import jax
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=RuntimeWarning)
#     mean = np.mean([])
#     print(mean)

# %%
# Global flag to set a specific platform, must be used at startup.
jax.config.update("jax_platform_name", "cpu")
# random number generation setup
np.random.seed(46)

# configurations
from datetime import datetime

Game = "movieLens"
now = datetime.now()
dir = now.strftime("%Y_%m%d_%H%M_%S")
path = os.path.join("./storage/", Game, dir)
os.makedirs(path, exist_ok=True)


param = {
    "UCB1": {"rho": np.sqrt(2)},
    "LinUCB": {"lbda": 10e-4, "alpha": 10e-1},
    "BayesUCB": {"p1": 1, "p2": 1, "c": 0},
    "MOSS": {"rho": 0.2},
    "ExploreCommit": {"m": 10},
    "Tuned_GPUCB": {"c": 0.9},
    "IDS": {"M": 1000},
    "IDS_approx": {"N": 1000, "display_results": False},
    "IDS_sample": {"M": 10000, "VIDS": False}, # The parameter VIDS is only reserved for Bernoulli MAB
    "VIDS_approx": {"rg": 10.0, "N": 1000},
    "VIDS_sample": {"M": 10000, "VIDS": True}, # The parameter VIDS is only reserved for Bernoulli MAB
    "FGTS": {"fg_lambda": 1},
    "VIDS_sample_sgmcmc": {"M": 10000},
    "VIDS_sample_sgmcmc_fg": {"M": 10000, "fg_lambda": 1},
    "VIDS_sample_sgmcmc_fg01": {"M": 10000},
}

# linear_methods = ["FGTS", "TS_SGMCMC", "TS", "LinUCB", "BayesUCB", "GPUCB", "Tuned_GPUCB", "VIDS_sample"]
linear_methods = [
    "TS",
    "TS_SGMCMC",
    "FGTS",
    "FGTS01",
    "VIDS_sample",
    "VIDS_sample_sgmcmc",
    "VIDS_sample_sgmcmc_fg",
    "VIDS_sample_sgmcmc_fg01",
]


"""Kind of Bandit problem"""
check_Linear = True
store = True  # if you want to store the results
check_time = False


# %%
# Regret
labels, colors = utils.labelColor(linear_methods)
lin = exp.LinMAB_expe(
    n_expe=20,
    n_features=30,
    n_arms=207,
    T=200,
    methods=linear_methods,
    param_dic=param,
    labels=labels,
    colors=colors,
    path=path,
    problem=Game,  # choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
