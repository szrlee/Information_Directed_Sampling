# %%
""" Packages import """
import torch
import json
import argparse
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

def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--game', type=str, default='Zhang')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--noise_dim', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--update-num', type=int, default=100)
    parser.add_argument('--repeat-num', type=int, default=500)
    parser.add_argument('--time-period', type=int, default=200)
    parser.add_argument('--n-expe', type=int, default=20)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--norm-noise', type=int, default=0, choices=[0, 1])
    args = parser.parse_known_args()[0]
    return args

args = get_args()
game = args.game
now = datetime.now()
dir = f"{game.lower()}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join("~/results/vids/", game, dir))
os.makedirs(path, exist_ok=True)

hyper_params = {'noise_dim': args.noise_dim, 'lr': args.lr, 'batch_size': args.batch_size, 'optim': args.optim, 'update_num': args.update_num}
hyper_reset_params = {'noise_dim': args.noise_dim, 'lr': args.lr, 'batch_size': args.batch_size, 'optim': args.optim, 'update_num': args.repeat_num}
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
    "TS_hyper:0": {'fg_lambda':0.0, **hyper_params},
    "TS_hyper:1": {'fg_lambda':1.0, **hyper_params},
    "TS_hyper:2": {'fg_lambda':0.1, **hyper_params},
    "TS_hyper_reset": hyper_reset_params,
    "VIDS_approx": {"rg": 10.0, "N": 1000},
    "VIDS_sample": {"M": 10000, "VIDS": True}, # The parameter VIDS is only reserved for Bernoulli MAB
    "VIDS_sample_hyper:0": {"M": 10000, 'fg_lambda':0.0, **hyper_params},
    "VIDS_sample_hyper:1": {"M": 10000, 'fg_lambda':1.0, **hyper_params},
    "VIDS_sample_hyper:2": {"M": 10000, 'fg_lambda':0.1, **hyper_params},
    "VIDS_sample_hyper_reset": {"M": 10000, **hyper_reset_params},
    "VIDS_sample_solution": {"M": 10000, "VIDS": True},
    "VIDS_sample_solution_hyper:0": {"M": 10000, 'fg_lambda':0.0, **hyper_params},
    "VIDS_sample_solution_hyper:1": {"M": 10000, 'fg_lambda':1.0, **hyper_params},
    "VIDS_sample_solution_hyper:2": {"M": 10000, 'fg_lambda':0.1, **hyper_params},
    "VIDS_sample_solution_hyper_reset": {"M": 10000, **hyper_reset_params},
    "FGTS": {"fg_lambda": 1},
    "VIDS_sample_sgmcmc": {"M": 10000},
    "VIDS_sample_sgmcmc_fg": {"M": 10000, "fg_lambda": 1},
    "VIDS_sample_sgmcmc_fg01": {"M": 10000},
}

linear_methods = [
    "TS",
    "TS_hyper:0",
    # "TS_hyper:1",
    # "TS_hyper:2",
    "VIDS_sample",
    "VIDS_sample_hyper:0",
    # "VIDS_sample_hyper:1",
    # "VIDS_sample_hyper:2",
    "VIDS_sample_solution",
    "VIDS_sample_solution_hyper:0",
    # "VIDS_sample_solution_hyper:1",
    # "VIDS_sample_solution_hyper:2",
]

game_config = {
    'FreqRusso': {'n_features': 5, 'n_arms': 30, 'T': args.time_period},
    'movieLens': {'n_features': 30, 'n_arms': 207, 'T': args.time_period},
    'Russo': {'n_features': 5, 'n_arms': 30, 'T': args.time_period},
    'Zhang': {'n_features': 100, 'n_arms': 10, 'T': args.time_period},
}

with open(os.path.join(path, "config.json"), "wt") as f:
    methods_param = {method : param.get(method, '') for method in linear_methods}
    f.write(json.dumps(
        {
            'methods_param': methods_param, 'game_config': game_config[game], 'user_config': vars(args),
            'methods': linear_methods, 'labels': utils.mapping_methods_labels, 'colors': utils.mapping_methods_colors,
        }, indent=4) + '\n')
    f.flush()
    f.close()

"""Kind of Bandit problem"""
check_Linear = True
store = True  # if you want to store the results
check_time = False


# %%
# Regret
labels, colors = utils.labelColor(linear_methods)
lin = exp.LinMAB_expe(
    n_expe=args.n_expe,
    # n_features=100,
    # n_arms=10,
    # T=200,
    methods=linear_methods,
    param_dic=param,
    labels=labels,
    colors=colors,
    path=path,
    problem=game,  # choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    **game_config[game]
)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
