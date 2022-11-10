# %%
""" Packages import """
import os, sys

sys.path.append(os.getcwd())
import json
import argparse
import expe as exp
import numpy as np

import pickle as pkl
import utils
import time

# random number generation setup
np.random.seed(46)

# configurations
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument(
        "--game",
        type=str,
        default="Bernoulli",
        choices=["Gaussian", "FreqGaussian", "Bernoulli", "FreqBernoulli"],
    )
    parser.add_argument("--time-period", type=int, default=50)
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--n-arms", type=int, default=10)
    parser.add_argument("--d-index", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="./results/bandit")
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
    "ES": {"M": args.d_index},
    "BayesUCB": {"p1": 0.01, "p2": 0.1, "c": 0},
    "IDS_approx": {"N": 1000, "display_results": False},
    "KG": {},
    "Approx_KG_star": {},
    "IDS_sample": {"M": 1000, "VIDS": False},
    "VIDS_sample": {"M": 1000, "VIDS": True},
}

methods = [
    "TS",
    "ES",
    # "BayesUCB",
    # "IDS_approx",
    # "KG",
    # "Approx_KG_star",
    # "IDS_sample",
    # "VIDS_sample",
]

with open(os.path.join(path, "config.json"), "wt") as f:
    methods_param = {method: param.get(method, "") for method in methods}
    f.write(
        json.dumps(
            {
                "methods_param": methods_param,
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
    "T": args.time_period,
    "n_expe": args.n_expe,
    "n_arms": args.n_arms,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "path": path,
    "problem": game,
}
lin = exp.bernoulli_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
