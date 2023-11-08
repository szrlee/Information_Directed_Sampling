""" Packages import """
import numpy as np
import os
import csv
import numba as nb

import random as rd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import sqrtm
from tqdm import tqdm
import inspect
import pickle as pkl
import time

rng = None

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


mapping_methods_labels = {
    "LinUCB": "LinUCB",
    "LinUCB:test": "red-test",
    "BayesUCB": "BayesUCB",
    "GPUCB": "GPUCB",
    "Tuned_GPUCB": "Tuned_GPUCB",
    "TS": "TS - Conjugacy",
    "TS:G": "TS - Gaussian Conjugacy",
    "TS:B": "TS - Beta Conjugacy",
    "ES": "Ensemble Sampling",
    "ES:G": "Ensemble Sampling - Gaussian",
    "ES:E": "Ensemble Sampling - Exponential",
    "IS:Sphere": "Index Sampling: Sphere",
    "IS:Haar": "Index Sampling: Haar",
    # "IS:Normal_Sphere": "Index Sampling: Normal index, Sphere noise",
    # "IS:Normal_Gaussian": "Index Sampling: Normal index, Gaussian noise",
    # "IS:Normal_PMCoord": "Index Sampling: Normal index, PMCoord noise",
    # "IS:Sphere_Sphere": "Index Sampling: Sphere index, Sphere noise",
    # "IS:Sphere_Gaussian": "Index Sampling: Sphere index, Gaussian noise",
    # "IS:Sphere_PMCoord": "Index Sampling: Sphere index, PMCoord noise",
    # "IS:PMCoord_Gaussian": "Index Sampling: PMCoord index, Gaussian noise",
    # "IS:PMCoord_Sphere": "Index Sampling: PMCoord index, Sphere noise",
    # "IS:PMCoord_PMCoord": "Index Sampling: PMCoord index, PMCoord noise",
    "TS_hyper": "TS - HyperModel",
    "TS_hyper:Reset": "TS - HyperModel Reset",
    "TS_hyper:FG": "TS - HyperModel FG",
    "TS_hyper:FG Decay": "TS - HyperModel FG Decay",
    "VIDS_action": "VIDS-action - Conjugacy",
    "VIDS_action_hyper": "VIDS-action - HyperModel",
    "VIDS_action_hyper:Reset": "VIDS-action - HyperModel Reset",
    "VIDS_action_hyper:FG": "VIDS-action - HyperModel FG",
    "VIDS_action_hyper:FG Decay": "VIDS-action - HyperModel FG Decay",
    "VIDS_action:theta": "VIDS-action - Conjugacy theta",
    "VIDS_action_hyper:theta": "VIDS-action - HyperModel theta",
    "VIDS_policy": "VIDS-policy - Conjugacy",
    "VIDS_policy_hyper": "VIDS-policy - HyperModel",
    "VIDS_policy_hyper:Reset": "VIDS-policy - HyperModel Reset",
    "VIDS_policy_hyper:FG": "VIDS-policy - HyperModel FG",
    "VIDS_policy_hyper:FG Decay": "VIDS-policy - HyperModel FG Decay",
    "VIDS_policy:theta": "VIDS-policy - Conjugacy theta",
    "VIDS_policy_hyper:theta": "VIDS-policy - HyperModel theta",
}


mapping_methods_colors = {
    "LinUCB": "green",
    "LinUCB:test": "red",
    "BayesUCB": "purple",
    "GPUCB": "violet",
    "Tuned_GPUCB": "blue",
    "TS": "black",
    "TS:G": "black",
    "TS:B": "black",
    "ES": "green",
    "ES:G": "green",
    "ES:E": "green",
    # "IS:Normal_Sphere": "red",
    # "IS:Normal_Gaussian": "blue",
    # "IS:Normal_PMCoord": "green",
    # "IS:Normal_UnifGrid": "brown",
    # "IS:Sphere_Sphere": "red",
    # "IS:Sphere_Gaussian": "blue",
    # "IS:Sphere_PMCoord": "green",
    # "IS:Sphere_UnifGrid": "brown",
    # "IS:PMCoord_Gaussian": "red",
    # "IS:PMCoord_Sphere": "blue",
    # "IS:PMCoord_PMCoord": "green",
    # "IS:PMCoord_UnifGrid": "brown",
    # "IS:UnifGrid_Gaussian": "red",
    # "IS:UnifGrid_Sphere": "blue",
    # "IS:UnifGrid_PMCoord": "green",
    # "IS:UnifGrid_UnifGrid": "brown",
    "IS:Sphere": "red",
    "IS:Haar": "blue",
    "TS_hyper": "green",
    "TS_hyper:Reset": "purple",
    "TS_hyper:FG": "violet",
    "TS_hyper:FG Decay": "darkpurple",
    "VIDS_action": "blue",
    "VIDS_action_hyper": "red",
    "VIDS_action_hyper:Reset": "brown",
    "VIDS_action_hyper:FG": "salmon",
    "VIDS_action_hyper:FG Decay": "darksalmon",
    "VIDS_action:theta": "brown",
    "VIDS_action_hyper:theta": "salmon",
    "VIDS_policy": "yellow",
    "VIDS_policy_hyper": "grey",
    "VIDS_policy_hyper:Reset": "pink",
    "VIDS_policy_hyper:FG": "cyan",
    "VIDS_policy_hyper:FG Decay": "darkcyan",
    "VIDS_policy:theta": "pink",
    "VIDS_policy_hyper:theta": "cyan",
}


mapping_methods_markers = {
    "IS:Normal_Sphere": ".",
    "IS:Normal_Gaussian": ".",
    "IS:Normal_PMCoord": ".",
    "IS:Normal_UnifGrid": ".",
    "IS:Sphere_Sphere": "v",
    "IS:Sphere_Gaussian": "v",
    "IS:Sphere_PMCoord": "v",
    "IS:Sphere_UnifGrid": "v",
    "IS:PMCoord_Gaussian": "s",
    "IS:PMCoord_Sphere": "s",
    "IS:PMCoord_PMCoord": "s",
    "IS:PMCoord_UnifGrid": "s",
    "IS:UnifGrid_Gaussian": "x",
    "IS:UnifGrid_Sphere": "x",
    "IS:UnifGrid_PMCoord": "x",
    "IS:UnifGrid_UnifGrid": "x",
}


def labelColor(methods):
    """
    Map methods to labels and colors for regret curves
    :param methods: list, list of methods
    :return: lists, labels and vectors
    """
    labels = [
        mapping_methods_labels[m] if m in mapping_methods_labels.keys() else m
        for m in methods
    ]
    colors = [
        mapping_methods_colors[m] if m in mapping_methods_colors.keys() else None
        for m in methods
    ]
    markers = [
        mapping_methods_markers[m] if m in mapping_methods_markers.keys() else None
        for m in methods
    ]
    return labels, colors, markers


@nb.njit
def rankone_update(f, Sigma, eta):
    ffT = np.outer(f, f)
    return Sigma - (Sigma @ ffT @ Sigma) / (eta**2 + f @ Sigma @ f)


@nb.njit
def updatePosterior(sigma, p, f, r, eta):
    """
    Update posterior mean and covariance matrix incrementally
    without matrix inversion
    :param arm: int, arm chose
    :param sigma: np.array, posterior covariance matrix
    :param p: np.array, sigma_inv mu
    :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
    """
    sigma_ = rankone_update(f, sigma, eta)
    p_ = p + ((r * f) / eta**2)
    mu_ = sigma_ @ p_
    return mu_, sigma_, p_


@nb.njit
def update_inv_sigma(inv_sigma, f, eta):
    ffT = np.outer(f, f)
    return inv_sigma + (1 / eta**2) * ffT


# @nb.njit
def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    # vector += 1e-20 * np.random.randn(len(vector))
    # return np.argmax(vector)
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)


# @nb.njit
def haar_matrix(M):
    """
    Haar random matrix generation
    """
    # z = np.random.randn(M, M)
    z = rng.standard_normal((M, M), dtype=np.float32)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return np.multiply(q, ph)


# @nb.njit
def sphere_matrix(N, M):
    # v = np.random.randn(N, M)
    v = rng.standard_normal((N, M), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# # @nb.njit
# def sphere_matrix(N, M):
#     v = np.random.randn(N, M)
#     v /= np.sqrt((v**2).sum(axis=1)).reshape(-1, 1)
#     return v


# @nb.njit
def random_choice_noreplace(m, n, axis=-1):
    # m, n are the number of rows, cols of output
    return rng.random((m, n)).argsort(axis=axis)


# @nb.njit
def random_sign(N=None):
    if (N is None) or (N == 1):
        return rng.integers(2, size=1) * 2 - 1
    elif N > 1:
        return rng.integers(2, size=N) * 2 - 1


def sample_noise(noise_type, M, dim=1, sparsity=2):
    # ensure the sampled vector is isotropic
    assert M > 0
    if noise_type == "Sphere":
        return sphere_matrix(dim, M)
    elif noise_type == "Gaussian" or noise_type == "Normal":
        return rng.standard_normal((dim, M), dtype=np.float32) / np.sqrt(M)
    elif noise_type == "PMCoord":
        i = rng.choice(M, dim)
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.arange(dim), i] = random_sign(dim)
        return B
    elif noise_type == "Sparse":
        i = random_choice_noreplace(dim, M)[:, :sparsity]
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.expand_dims(np.arange(dim), axis=1), i] = random_sign(
            dim * sparsity
        ).reshape(dim, sparsity) / np.sqrt(sparsity)
        return B
    elif noise_type == "SparseConsistent":
        i = random_choice_noreplace(dim, M)[:, :sparsity]
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.expand_dims(np.arange(dim), axis=1), i] = random_sign(dim).reshape(
            dim, 1
        ) / np.sqrt(sparsity)
        return B
    elif noise_type == "UnifCube":
        return (
            2 * rng.binomial(1, 0.5, (dim, M)).astype(dtype=np.float32) - 1
        ) / np.sqrt(M)
    else:
        raise NotImplementedError


def multi_haar_matrix(N, M):
    v = np.zeros(((N // M + 1) * M, M))
    for _ in range(N // M + 1):
        v[np.arange(M), :] = haar_matrix(M)
    return v[np.arange(N), :]


def display_results(delta, g, ratio, p_star):
    """
    Display quantities of interest in IDS algorithm
    """
    print("delta {}".format(delta))
    print("g {}".format(g))
    print("ratio : {}".format(ratio))
    print("p_star {}".format(p_star))


def plotRegret(labels, regret, colors, title, path, log=False, markers=None):
    """
    Plot Bayesian regret
    :param labels: list, list of labels for the different curves
    :param mean_regret: np.array, averaged regrets from t=1 to T for all methods
    :param colors: list, list of colors for the different curves
    :param title: string, plot's title
    """

    low_CI_bound = regret["low_CI_bound"]
    high_CI_bound = regret["high_CI_bound"]
    mean_regret = regret["mean_regret"]
    plt.figure(figsize=(10, 8), dpi=80)
    # plt.rcParams["figure.figsize"] = (16, 9)

    T = mean_regret.shape[1]
    # sort out how many points to drop: max_points = 25
    ratio = max(T // 25, 1)
    for i, l in enumerate(labels):
        # if 'TS' not in l:
        #     continue
        c = cmap[i] if not colors else colors[i]
        m = None if not markers else markers[i]
        x = np.arange(T)
        # low_CI_bound, high_CI_bound = st.t.interval(
        # 0.95, T - 1, loc=mean_regret[i], scale=st.sem(cum_regrets[i])
        # )
        # low_CI_bound = np.quantile(cum_regrets[i], 0.05, axis=0)
        # high_CI_bound = np.quantile(cum_regrets[i], 0.95, axis=0)
        plt.plot(x, mean_regret[i], c=c, label=l, marker=m, markevery=ratio)
        plt.fill_between(x, low_CI_bound[i], high_CI_bound[i], color=c, alpha=0.2)
        if log:
            plt.yscale("log")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.ylabel("Cumulative regret")
    plt.xlabel("Time period")
    plt.legend(loc="best")
    plt.savefig(path + "/regret.pdf")

    # mean_regret = regret["mean_regret"]
    # plt.rcParams["figure.figsize"] = (8, 6)
    # for i, l in enumerate(labels):
    #     c = colors[i] or cmap[i]
    #     plt.plot(mean_regret[i], c=c, label=l)
    #     if log:
    #         plt.yscale("log")
    # plt.grid(color="grey", linestyle="--", linewidth=0.5)
    # plt.title(title)
    # plt.ylabel("Cumulative regret")
    # plt.xlabel("Time period")
    # plt.legend(loc="best")
    # plt.savefig(path + "/regret.pdf")


def storeRegret(models, methods, param_dic, n_expe, T, path, use_torch=False):
    """
    Compute the experiment for all specified models and methods
    :param models: list of MAB
    :param methods: list of algorithms
    :param param_dic: parameters for all the methods
    :param n_expe: number of trials
    :param T: Time horizon
    :return: Dictionnary with results from the experiments
    """
    mean_regret = np.zeros((len(methods), T), dtype=np.float32)
    low_CI_bound = np.zeros((len(methods), T), dtype=np.float32)
    high_CI_bound = np.zeros((len(methods), T), dtype=np.float32)
    # cum_regrets = np.zeros((n_expe, T), dtype=np.float32)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for i, m in enumerate(methods):
        set_seed(2023, use_torch=use_torch)
        alg_name = m.split(":")[0]
        all_dic = {}
        file_name = m.replace(":", "_").replace(" ", "_").lower()
        file_path = os.path.join(path, "data", f"{file_name}.pkl")
        # Loop for n_expe repeated experiments
        for j in tqdm(range(n_expe)):
            model = models[j]
            alg = model.__getattribute__(alg_name)
            # args = inspect.getfullargspec(alg)[0][2:]
            # kwargs = {key: param_dic[m][key] for key in args and param_dic[m].keys()}
            kwargs = param_dic[m]
            return_dic = alg(T, **kwargs)
            if len(all_dic) == 0:
                all_dic = return_dic
                for key in all_dic.keys():
                    all_dic[key] = np.expand_dims(all_dic[key], axis=0)
            else:
                for key in all_dic.keys():
                    all_dic[key] = np.vstack((all_dic[key], return_dic[key]))

        # Summary for one method (i, m)
        cum_list = [
            "expected_regret",
            "pess_regret",
            "est_regret",
            "potential",
            "approx_potential",
        ]
        cum_dic = {
            key: np.cumsum(all_dic[key], axis=1)
            for key in set(all_dic.keys()) & set(cum_list)
        }

        print(cum_dic.keys())

        cum_dic = {
            key: {
                "mean": mean,
                "high_CI": high_CI,
                "low_CI": low_CI,
            }
            for (key, mean, (low_CI, high_CI)) in [
                (
                    key,
                    mean,
                    st.t.interval(
                        0.95,
                        T - 1,
                        loc=mean,
                        scale=st.sem(cum, nan_policy="omit") + 1e-20,
                    ),
                )
                for key, mean, cum in [
                    (key, np.nanmean(value, axis=0), value)
                    for key, value in cum_dic.items()
                ]
            ]
        }
        all_dic = {
            key: {
                "mean": mean,
                "low_CI": low_CI,
                "high_CI": high_CI,
            }
            for (key, mean, (low_CI, high_CI)) in [
                (
                    key,
                    mean,
                    st.t.interval(
                        0.95,
                        T - 1,
                        loc=mean,
                        scale=st.sem(cum, nan_policy="omit") + 1e-20,
                    ),
                )
                for key, mean, cum in [
                    (key, np.nanmean(value, axis=0), value)
                    for key, value in all_dic.items()
                ]
            ]
        }

        cum_regrets = cum_dic["expected_regret"]
        mean_regret[i], low_CI_bound[i], high_CI_bound[i] = (
            cum_regrets["mean"],
            cum_regrets["low_CI"],
            cum_regrets["high_CI"],
        )
        print(f"{m}: {mean_regret[i][-1]}")

        pkl.dump({"all": all_dic, "cum": cum_dic}, open(file_path, "wb"))

    results = {
        "mean_regret": mean_regret,
        "low_CI_bound": low_CI_bound,
        "high_CI_bound": high_CI_bound,
    }
    if models[0].store_IDS:
        IDS_res = [m.IDS_results for m in models]
        results["IDS_results"] = IDS_res
    return results


def plot_IDS_results(T, n_expe, results):
    """
    Plot the evolution of delta, g and IR with 90% quantiles for the experiments
    :param T: Time horizon
    :param n_expe: Number of experiments
    :param results: Results of the experiments as stored in StoreRegret(..)['IDS_results']
    :return:
    """
    delta = np.empty((T, n_expe))
    g = np.empty((T, n_expe))
    IR = np.empty((T, n_expe))
    for i, r in enumerate(results):
        r["policy"] = np.asarray(r["policy"])
        delta[: r["policy"].shape[0], i] = (np.asarray(r["delta"]) * r["policy"]).sum(
            axis=1
        )
        g[: r["policy"].shape[0], i] = (np.asarray(r["g"]) * r["policy"]).sum(axis=1)
        IR[: r["policy"].shape[0], i] = np.asarray(r["IR"])
    x = np.arange(1, T + 1)
    f, (ax1, ax2, ax3) = plt.subplots(3)
    plt.xlabel("Time horizon")
    ax1.semilogy(x, delta.mean(axis=1))
    ax1.fill_between(
        x,
        np.quantile(delta, 0.05, axis=1),
        np.quantile(delta, 0.95, axis=1),
        color="lightblue",
    )
    ax2.semilogy(x, g.mean(axis=1))
    ax2.fill_between(
        x, np.quantile(g, 0.05, axis=1), np.quantile(g, 0.95, axis=1), color="lightblue"
    )
    ax3.plot(x, IR.mean(axis=1), color="r")
    ax3.fill_between(
        x, np.quantile(IR, 0.05, axis=1), np.quantile(IR, 0.95, axis=1), color="#FFA07A"
    )
    ax1.set_title("Average Delta")
    ax2.set_title("Average Information Gain")
    ax3.set_title("Average Information Ratio")
    plt.show()


def build_finite(L, K, N):
    """
    Build automatically a finite bandit environment
    :param L: int, number of possible values for theta
    :param K: int, number of arms
    :param N: int, number of possible rewards
    :return: np.arrays, parameters required for launching an experiment with a finite bandit
    (prior, q values and R function)
    """
    R = np.linspace(0.0, 1.0, N)
    q = np.random.uniform(size=(L, K, N))
    for i in range(q.shape[0]):
        q[i] = np.apply_along_axis(lambda x: x / x.sum(), 1, q[i])
    p = np.random.uniform(0, 1, L)
    p = p / p.sum()
    return p, q, R


def build_bernoulli_finite_set(L, K):
    r = np.array([0, 1])
    p = np.ones(L) / L
    q = np.empty((L, K, 2))
    q[:, :, 0] = np.random.uniform(size=L * K).reshape((L, K))
    q[:, :, 1] = 1 - q[:, :, 0]
    return p, q, r


def set_seed(seed, use_torch=False):
    np.random.seed(seed)
    rd.seed(seed)
    global rng
    rng = np.random.default_rng(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)


@nb.njit
def approx_err(a, action_set, P, Q, Q_inv):
    # Compute the approximation error of P. Q is the matrix to be approximated.
    # Q_inv = np.linalg.inv(Q)
    up_norm_err = np.linalg.norm(Q_inv @ P, 2) - 1  # eps_1 = (1+ eps_1 -1)
    low_norm_err = 1 - np.linalg.norm(Q_inv @ P, -2)  # eps_2 = (1- (1-eps_2))
    # norm_err = max(up_norm_err, low_norm_err)
    # t = time.time()
    xPx = np.zeros(len(action_set), dtype=np.float32)
    xQx = np.zeros(len(action_set), dtype=np.float32)
    for i in range(len(action_set)):
        xPx[i] = np.dot(action_set[i], np.dot(P, action_set[i]))
        xQx[i] = np.dot(action_set[i], np.dot(Q, action_set[i]))
    # xPx = np.zeros(len(action_set))
    # print(np.round_(time.time() - t, 3), "sec elapsed")
    # t = time.time()
    # xPxt = np.diag(action_set @ P @ action_set.T)
    # print(np.round_(time.time() - t, 3), "sec elapsed")
    # print(np.allclose(xPx, xPxt))
    # xQx = np.einsum("ij,jk,ki->i", action_set, Q, action_set.T)
    # xQx = np.zeros(len(action_set))
    # pot = xQx[a]
    # approx_pot = xPx[a]

    return up_norm_err, low_norm_err, (xPx - xQx) / xQx, xQx[a], xPx[a]
