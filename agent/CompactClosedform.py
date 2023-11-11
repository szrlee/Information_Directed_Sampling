""" Packages import """
import numpy as np
from utils import (
    sample_noise,
    approx_err_compact,
    rankone_update,
    update_inv_sigma,
    updatePosterior,
    index_sampling,
    posterior_sampling,
    init_prior,
)
from scipy.stats import norm
from scipy.linalg import sqrtm


class LinCompact:
    def __init__(self, env):
        """
        :param env: ArmGaussianLinear object
        """
        self.env = env
        (self.expect_regret, self.d) = (env.expect_regret, env.n_features)
        self.reward, self.eta = env.reward, env.eta
        self.prior_sigma = env.alg_prior_sigma
        self.threshold = 0.999
        self.store_IDS = False
        # For synthesis / analysis
        self.pess_regret = env.pessimism_regret
        self.optimal_action = env.optimal_action
        self.updatePosterior_ = updatePosterior
        self.rankone_update = rankone_update
        self.update_inv_sigma = update_inv_sigma

        self.data_groups = {
            "reward": 1,
            "expected_regret": 1,
            "pess_regret": 1,
            "est_regret": 1,
            "potential": 1,
            # immediate quantity for synthesis
            "lmax": 20,
            "lmin": 20,
            "lmax_inv": 20,
            "lmin_inv": 20,
            "kappa_inv": 20,
        }
        self.data_groups_IS = {
            # Only for IS
            "approx_potential": 1,
            "up_norm_err": 20,
            "low_norm_err": 20,
            "a_t_err": 20,
            "a_star_err": 20,
            # Only for IS - tilde for independent z
            "tilde_up_norm_err": 20,
            "tilde_low_norm_err": 20,
            "tilde_a_t_err": 20,
            "tilde_a_star_err": 20,
        }

    def data_init(self, total_steps, data_groups):
        data = {}
        for key, value in data_groups.items():
            data[key] = np.full((total_steps), np.nan)
        return data

    def set_context(self):
        self.env.set_context()

    def initPrior(self, dtype=np.float64):
        return init_prior(0, self.prior_sigma, self.d, dtype=dtype)

    def action_selection(self, scheme, theta):
        if scheme == "ts":
            norm = np.linalg.norm(theta)
            if norm == 0:
                return sample_noise("Sphere", self.d)[0]
            else:
                return theta.reshape(-1) / norm
        elif scheme == "ots":
            norm = np.linalg.norm(theta, axis=0)
            i = np.argmax(norm)
            if norm[i] == 0:
                return sample_noise("Sphere", self.d)[0]
            else:
                return theta[:, i] / norm[i]

    def TS(self, T, scheme="ts"):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        dic = self.data_init(T, self.data_groups)

        # Algorithm
        mu_t, sigma_t = self.initPrior(dtype=np.float64)
        inv_sigma_t = np.linalg.inv(sigma_t)
        p_t = inv_sigma_t @ mu_t
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # scheme for posterior sampling
            if scheme == "ts":
                theta_t = posterior_sampling(mu_t, sigma_t)
            elif scheme == "ots":
                theta_t = posterior_sampling(mu_t, sigma_t, 10)
            # action selection
            f_t = self.action_selection(scheme, theta_t)
            # selected feature and reward feedback
            r_t = self.reward(f_t)[0]
            # compute regret
            dic["reward"][t], dic["expected_regret"][t] = r_t, self.expect_regret(f_t)
            # Synthesis: lmax, lmin, pess_regret, est_regret, potential: pot; lmax_inv, lmin_inv, kappa_inv
            dic["potential"][t] = f_t.T @ sigma_t @ f_t
            dic["pess_regret"][t] = self.pess_regret(f_t, theta_t, scheme)
            dic["est_regret"][t] = dic["expected_regret"][t] - dic["pess_regret"][t]
            if t % self.data_groups["lmax"] == 0:
                dic["lmax"][t] = np.linalg.norm(sigma_t, 2)
                dic["lmin"][t] = np.linalg.norm(sigma_t, -2)
                dic["lmax_inv"][t] = np.linalg.norm(inv_sigma_t, 2)
                dic["lmin_inv"][t] = np.linalg.norm(inv_sigma_t, -2)
                dic["kappa_inv"][t] = np.linalg.cond(inv_sigma_t)

            # Update posterior
            mu_t, sigma_t, p_t = self.updatePosterior_(sigma_t, p_t, f_t, r_t, self.eta)
            inv_sigma_t = self.update_inv_sigma(inv_sigma_t, f_t, self.eta)

        return dic

    def IS(
        self,
        T,
        M=10,
        haar=False,
        index="gaussian",
        perturbed_noise="sphere",
        scheme="ts",
    ):
        """
        Index Sampling
        Protocol:
        1. E[ norm(perturbed_noise) ] = 1
        2. E[ norm(index) ] = sqrt(M)
        """

        dic = self.data_init(T, {**self.data_groups, **self.data_groups_IS})

        # Algorithm initialization
        mu_t, Sigma_t = self.initPrior(dtype=np.float64)
        S_inv = np.linalg.inv(Sigma_t)
        p_t = S_inv @ mu_t
        sqrt_Sigma_t = sqrtm(Sigma_t)

        # Initialization for factor A: Sample matrix B with size d x M
        B = sample_noise(perturbed_noise, M, self.d)
        A_t = sqrt_Sigma_t @ B
        P_t = S_inv @ A_t
        # for synthesis
        tilde_B = sample_noise(perturbed_noise, M, self.d)
        tilde_A_t = sqrt_Sigma_t @ tilde_B
        tilde_P_t = S_inv @ tilde_A_t

        # interaction
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # index sampling
            if scheme == "ts":
                img_theta = index_sampling(A_t, mu_t, index)
            elif scheme == "ots":
                img_theta = index_sampling(A_t, mu_t, index, 10)
            # action selection
            f_t = self.action_selection(scheme, img_theta)
            # selected feature and reward feedback
            r_t = self.reward(f_t)[0]

            # compute regret
            dic["reward"][t], dic["expected_regret"][t] = r_t, self.expect_regret(f_t)

            # Synthesis: lmax, lmin, pess_regret, est_regret, potential: pot; lmax_inv, lmin_inv, kappa_inv
            P = A_t @ A_t.T
            dic["approx_potential"][t] = f_t.T @ P @ f_t
            dic["potential"][t] = f_t.T @ Sigma_t @ f_t
            dic["pess_regret"][t] = self.pess_regret(f_t, img_theta, scheme)
            dic["est_regret"][t] = dic["expected_regret"][t] - dic["pess_regret"][t]
            if t % self.data_groups["lmax"] == 0:
                # Synthesis only for index sampling (not every step)
                a_star = self.optimal_action
                aQa = a_star @ Sigma_t @ a_star
                #
                dic["up_norm_err"][t], dic["low_norm_err"][t] = approx_err_compact(
                    P, S_inv
                )
                dic["a_t_err"][t] = (
                    dic["approx_potential"][t] - dic["potential"][t]
                ) / dic["potential"][t]
                dic["a_star_err"][t] = (a_star @ P @ a_star - aQa) / aQa
                # tilde
                tilde_P = tilde_A_t @ tilde_A_t.T
                (
                    dic["tilde_up_norm_err"][t],
                    dic["tilde_low_norm_err"][t],
                ) = approx_err_compact(tilde_P, S_inv)
                dic["tilde_a_t_err"][t] = (
                    f_t @ tilde_P @ f_t - dic["potential"][t]
                ) / dic["potential"][t]
                dic["tilde_a_star_err"][t] = (a_star @ tilde_P @ a_star - aQa) / aQa

                # Synthesis: lmax, lmin, pess_regret, est_regret, potential: pot, approx_potential: approx_pot
                dic["lmax"][t] = np.linalg.norm(Sigma_t, 2)
                dic["lmin"][t] = np.linalg.norm(Sigma_t, -2)
                dic["lmax_inv"][t] = np.linalg.norm(S_inv, 2)
                dic["lmin_inv"][t] = np.linalg.norm(S_inv, -2)
                dic["kappa_inv"][t] = np.linalg.cond(S_inv)

            # incremental update on Sigma and mu
            mu_t, Sigma_t, p_t = self.updatePosterior_(Sigma_t, p_t, f_t, r_t, self.eta)
            S_inv = self.update_inv_sigma(S_inv, f_t, self.eta)

            # Update covariance factor A
            b_t = sample_noise(perturbed_noise, M)[0]
            P_t += np.outer(f_t, b_t) / self.eta
            A_t = Sigma_t @ P_t

            # for synthesis
            tilde_b_t = sample_noise(perturbed_noise, M)[0]
            tilde_P_t += np.outer(f_t, tilde_b_t) / self.eta
            tilde_A_t = Sigma_t @ tilde_P_t

        return dic
