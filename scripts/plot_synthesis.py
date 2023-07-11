# %%
import sys
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# %%
cmap = {
    0: "black",
    1: "blue",
    2: "yellow",
    3: "green",
    4: "red",
    5: "grey",
    6: "purple",
    7: "brown",
    8: "pink",
    9: "cyan",
}

GMAE_NAMES = {
    "Zhang": "Zhang",
    "FreqRusso": "FreqRusso",
    "movieLens": "Movie",
    "Russo": "BayesRusso",
    "Bernoulli": "Bernoulli",
    "Gaussian": "Gaussian",
    "synthetic-v1": "Synthetic h1",
    "synthetic-v2": "Synthetic h2",
}

# %%


def labelColor(methods, methods_labels, methods_colors):
    """
    Map methods to labels and colors for regret curves
    :param methods: list, list of methods
    :return: lists, labels and vectors
    """
    labels = {m: methods_labels[m] for m in methods}
    colors = {m: methods_colors[m] for m in methods}
    return labels, colors


# %%
path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# %%
path

# %%
game_name = "Russo"

# %%
time_tag = "20230702"

# %%
load_path = f"{path}/results/bandit/{game_name}/{time_tag}"

# %%
load_path

# %%
cum_quantities = {}
all_quantities = {}
titles = []
for root, dirs, files in os.walk(load_path):
    if len(files) != 0 and len(dirs) == 0:
        config = json.load(open(os.path.join(root, "..", "config.json"), "r"))
        user_config = config["user_config"]
        T = user_config["time_period"]
        methods_param = config["methods_param"]
        methods = dict()  # list(methods_param.keys())
        for key, val in methods_param.items():
            if "TS" in key:
                methods[key.lower()] = key
            else:
                methods[key.lower().replace(":", "_")] = f"{key}: M={val['M']}"
            if "scheme" in val.keys():
                methods[key.lower().replace(":", "_")] = (
                    methods[key.lower().replace(":", "_")] + f"; scheme={val['scheme']}"
                )
        if game_name in ["Bernoulli", "Gaussian"]:
            title = f"{GMAE_NAMES[game_name]} \n n_arms: {user_config['n_arms']}"
        else:
            game_config = config["game_config"]
            title = f"{GMAE_NAMES[game_name]} \n n_arms: {game_config['n_arms']} - n_features: {game_config['n_features']}"
        if title not in cum_quantities.keys():
            cum_quantities[title] = {}
            all_quantities[title] = {}
        titles.append(title)
        for file in files:
            try:
                with open(os.path.join(root, file), "rb") as f:
                    loaded_dict = pickle.load(f)
                if file.replace(".pkl", "") in methods.keys():
                    cum_quantities[title][
                        methods[file.replace(".pkl", "")]
                    ] = loaded_dict["cum"]
                    all_quantities[title][
                        methods[file.replace(".pkl", "")]
                    ] = loaded_dict["all"]
            except:
                print(os.path.join(root, file))

# %%


def plotall(ax, dict, set_y_label, title, k, syn="expected_regret", is_cum=False):
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.set_title(f"{title}", size=20)
    ax.set_xlabel("Time period", fontsize=20)
    if is_cum:
        y_label = f"cum_{syn}"
    else:
        y_label = syn
    if set_y_label:
        ax.set_ylabel(y_label, fontsize=20)
    if game_name == "Russo":
        # pass
        if "n_arms: 100 - n_features: 50" in title:
            ax.set(ylim=(1000, 6000))
        elif "n_arms: 30 - n_features: 10" in title:
            ax.set(ylim=(1000, 5000))
        elif "n_arms: 30 - n_features: 50" in title:
            ax.set(ylim=(1000, 3500))
    elif game_name == "FreqRusso":
        ax.set(ylim=(100, 3000))
        # ax.set(ylim=(100, 300))
    elif game_name == "movieLens":
        ax.set(ylim=(600, 1500))
    elif game_name == "Zhang":
        ax.set(ylim=(0, 10))
        ax.set(xlim=(0, 2000))
    elif game_name == "Bernoulli":
        if "n_arms: 160" in title:
            ax.set(ylim=(100, 500))
        elif "n_arms: 40" in title:
            ax.set(ylim=(50, 200))
        elif "n_arms: 10" in title:
            ax.set(ylim=(0, 100))
        elif "n_arms: 20" in title:
            ax.set(ylim=(0, 200))
        elif "n_arms: 80" in title:
            ax.set(ylim=(100, 300))
    elif game_name == "Gaussian":
        if "n_arms: 20" in title:
            ax.set(ylim=(100, 300))
        elif "n_arms: 40" in title:
            ax.set(ylim=(200, 500))
        elif "n_arms: 10" in title:
            ax.set(ylim=(50, 200))
        elif "n_arms: 80" in title:
            ax.set(ylim=(300, 800))
        elif "n_arms: 160" in title:
            ax.set(ylim=(600, 1600))
    for key in dict.keys():
        if k not in key and "TS" not in key:
            continue
        if syn not in dict[key].keys():
            continue
        x = np.arange(dict[key][syn]["mean"].shape[-1])
        low_CI_bound, high_CI_bound = (
            dict[key][syn]["low_CI"],
            dict[key][syn]["high_CI"],
        )
        ax.plot(x, dict[key][syn]["mean"], label=key)
        ax.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.2)
        ax.legend(loc="lower right")


# %%
tag = 0
for title in cum_quantities.keys():
    tag += 1
    for syn in [
        "est_regret",
        "pess_regret",
        "expected_regret",
        "approx_potential",
        "potential",
    ]:
        n_row, n_col = 2, 3
        fig, axes = plt.subplots(
            n_row, n_col, figsize=(10 * n_col, 12 * n_row + 0.6)
        )  # 3.6
        fig.subplots_adjust(
            left=0.05, right=0.98, bottom=0.03, top=0.97, hspace=0.15, wspace=0.15
        )

        plotall(
            axes[0][0],
            cum_quantities[title],
            True,
            title,
            "IS:Normal_Sphere",
            syn,
            True,
        )
        plotall(
            axes[0][1],
            cum_quantities[title],
            False,
            title,
            "IS:Normal_Gaussian",
            syn,
            True,
        )
        plotall(
            axes[0][2],
            cum_quantities[title],
            False,
            title,
            "IS:Normal_UnifGrid",
            syn,
            True,
        )
        plotall(
            axes[1][0],
            cum_quantities[title],
            True,
            title,
            "IS:PMCoord_Sphere",
            syn,
            True,
        )
        plotall(
            axes[1][1],
            cum_quantities[title],
            False,
            title,
            "IS:PMCoord_Gaussian",
            syn,
            True,
        )
        plotall(
            axes[1][2],
            cum_quantities[title],
            False,
            title,
            "IS:PMCoord_UnifGrid",
            syn,
            True,
        )
        # plotall(axes[2][0], all_regs, mean_reg, True, title, "IS:Sphere_Gaussian")
        # plotall(axes[2][1], all_regs, mean_reg, False, title, "IS:Sphere_PM")
        # plotall(axes[2][2], all_regs, mean_reg, False, title, "IS:Sphere_Sphere")

        plt.savefig(
            f"{path}/results/bandit/{game_name}/{time_tag}/{time_tag}_{game_name}_{tag}_cum_{syn}"
        )
        plt.savefig(
            f"{path}/results/bandit/{game_name}/{time_tag}/{time_tag}_{game_name}_{tag}_cum_{syn}.pdf"
        )
        print(
            f"save to {path}/results/bandit/{game_name}/{time_tag}/{time_tag}_{game_name}_{tag}_cum_{syn}"
        )
        plt.close()

# %%
tag = 0
for title in all_quantities.keys():
    tag += 1

    for syn in [
        # "reward",
        # "expected_regret",
        "pess_regret",
        "est_regret",
        "lmax",
        "lmin",
        "lmax_inv",
        "lmin_inv",
        "kappa_inv",
        # "potential",
        # "approx_potential",
        "up_norm_err",
        "tilde_up_norm_err",
        "low_norm_err",
        "tilde_low_norm_err",
        "up_set_err",
        "tilde_up_set_err",
        "low_set_err",
        "tilde_low_set_err",
        "a_t_err",
        "tilde_a_t_err",
        "a_star_err",
        "tilde_a_star_err",
    ]:
        n_row, n_col = 2, 3
        fig, axes = plt.subplots(
            n_row, n_col, figsize=(10 * n_col, 12 * n_row + 0.6)
        )  # 3.6
        fig.subplots_adjust(
            left=0.05, right=0.98, bottom=0.03, top=0.97, hspace=0.15, wspace=0.15
        )

        plotall(axes[0][0], all_quantities[title], True, title, "IS:Normal_Sphere", syn)
        plotall(
            axes[0][1],
            all_quantities[title],
            False,
            title,
            "IS:Normal_Gaussian",
            syn,
        )
        plotall(
            axes[0][2],
            all_quantities[title],
            False,
            title,
            "IS:Normal_UnifGrid",
            syn,
        )
        plotall(
            axes[1][0], all_quantities[title], True, title, "IS:PMCoord_Sphere", syn
        )
        plotall(
            axes[1][1],
            all_quantities[title],
            False,
            title,
            "IS:PMCoord_Gaussian",
            syn,
        )
        plotall(
            axes[1][2],
            all_quantities[title],
            False,
            title,
            "IS:PMCoord_UnifGrid",
            syn,
        )
        # plotall(axes[2][0], all_regs, mean_reg, True, title, "IS:Sphere_Gaussian")
        # plotall(axes[2][1], all_regs, mean_reg, False, title, "IS:Sphere_PM")
        # plotall(axes[2][2], all_regs, mean_reg, False, title, "IS:Sphere_Sphere")

        plt.savefig(
            f"{path}/results/bandit/{game_name}/{time_tag}/png/{time_tag}_{game_name}_{tag}_{syn}"
        )
        plt.savefig(
            f"{path}/results/bandit/{game_name}/{time_tag}/pdf/{time_tag}_{game_name}_{tag}_{syn}.pdf"
        )
        print(
            f"save to {path}/results/bandit/{game_name}/{time_tag}/{time_tag}_{game_name}_{tag}_{syn}"
        )

# %%
