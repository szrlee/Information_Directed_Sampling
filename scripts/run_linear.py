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
        default="Russo",
        choices=[
            "Russo",
            "FreqRusso",
            "Zhang",
            "movieLens",
            "ChangingRusso",
            "FreqChangingRusso",
        ],
    )
    parser.add_argument("--time-period", type=int, default=50)
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--d-index", type=int, default=10)
    parser.add_argument("--n-arms", type=int, default=30)
    parser.add_argument("--d-theta", type=int, default=10)
    parser.add_argument("--scheme", type=str, default="ts", choices={"ts", "ots"})
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
    "TS": {
        "scheme": args.scheme,
    },
    # "ES": {"M": args.d_index},
    # "LinUCB": {"lbda": 10e-4, "alpha": 10e-1},
    # "BayesUCB": {},
    # "GPUCB": {},
    # "Tuned_GPUCB": {"c": 0.9},
    # "VIDS_sample": {"M": 10000},
}

# done
# index_done = []
# noise_done = []

index_done = [
    "Normal",
    "PMCoord",
    "Sphere",
    "UnifCube",
]

noise_done = [
    "Gaussian",
    "Sphere",
    "PMCoord",
    "UnifCube",
]

# all

index_candidates = [
    "Normal",
    "Sparse",
    "SparseConsistent",
    "PMCoord",
    "Sphere",
    "UnifCube",
]

noise_candidates = [
    "Gaussian",
    "Sphere",
    "PMCoord",
    "UnifCube",
    "Sparse",
    "SparseConsistent",
]

index_noise_candidates = []

for index in index_candidates:
    for noise in noise_candidates:
        if index in index_done and noise in noise_done:
            continue
        param["IS:{}_{}".format(index, noise)] = {
            "M": args.d_index,
            "haar": False,
            "index": index,
            "perturbed_noise": noise,
            "scheme": args.scheme,
        }
        index_noise_candidates.append("IS:{}_{}".format(index, noise))

methods = [
    # "TS",
    ## "ES",
    # "IS:Haar",
    # "LinUCB",
    # "BayesUCB",
    # "GPUCB",
    # "Tuned_GPUCB",
    # "VIDS_sample",
]

methods.extend(index_noise_candidates)

game_config = {
    "FreqRusso": {
        "n_features": args.d_theta,
        "n_arms": args.n_arms,
        "T": args.time_period,
    },
    "Russo": {"n_features": args.d_theta, "n_arms": args.n_arms, "T": args.time_period},
    "ChangingFreqRusso": {
        "n_features": args.d_theta,
        "n_arms": args.n_arms,
        "T": args.time_period,
    },
    "ChangingRusso": {
        "n_features": args.d_theta,
        "n_arms": args.n_arms,
        "T": args.time_period,
    },
    "movieLens": {"n_features": 30, "n_arms": 207, "T": args.time_period},
    "Zhang": {"n_features": args.d_theta, "n_arms": args.n_arms, "T": args.time_period},
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
labels, colors, markers = utils.labelColor(methods)
expe_params = {
    "n_expe": args.n_expe,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "markers": markers,
    "path": path,
    "problem": game,
    **game_config[game],
}
lin = exp.LinMAB_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
