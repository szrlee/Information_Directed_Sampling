# %%
""" Packages import """
import os, sys

sys.path.append(os.getcwd())
import json
import argparse
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
    parser.add_argument("--game", type=str, default="Russo")
    parser.add_argument("--time-period", type=int, default=50)
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--logdir", type=str, default="~/results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
game = args.game
now = datetime.now()
dir = f"{game.lower()}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.logdir, game, dir))
os.makedirs(path, exist_ok=True)


param = {
    "TS": {},
    "LinUCB": {"lbda": 10e-4, "alpha": 10e-1},
    "BayesUCB": {},
    "GPUCB": {},
    "Tuned_GPUCB": {"c": 0.9},
    "VIDS_sample": {"M": 10000},
}

methods = [
    "TS",
    "LinUCB",
    "BayesUCB",
    "GPUCB",
    "Tuned_GPUCB",
    "VIDS_sample",
]

game_config = {
    "FreqRusso": {"n_features": 5, "n_arms": 30, "T": args.time_period},
    "movieLens": {"n_features": 30, "n_arms": 207, "T": args.time_period},
    "Russo": {"n_features": 5, "n_arms": 30, "T": args.time_period},
    "Zhang": {"n_features": 100, "n_arms": 10, "T": args.time_period},
}

with open(os.path.join(path, "config.json"), "wt") as f:
    methods_param = {method: param.get(method, "") for method in methods}
    f.write(
        json.dumps(
            {
                "methods_param": methods_param,
                "game_config": game_config[game],
                "user_config": vars(args),
                "methods": methods,
                "labels": utils.mapping_methods_labels,
                "colors": utils.mapping_methods_colors,
            },
            indent=4,
        )
        + "\n"
    )
    f.flush()
    f.close()

"""Kind of Bandit problem"""
check_Linear = True
store = True  # if you want to store the results
check_time = False


# %%
# Regret
labels, colors = utils.labelColor(methods)
expe_params = {
    "n_expe": args.n_expe,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "path": path,
    "problem": game,
    **game_config[game],
}
lin = exp.LinMAB_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
