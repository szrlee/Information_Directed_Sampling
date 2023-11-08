""" Packages import """
import numpy as np
from utils import (
    rd_argmax,
    sample_noise,
    approx_err,
    rankone_update,
    update_inv_sigma,
    updatePosterior,
)
from scipy.stats import norm
from scipy.linalg import sqrtm


class LinMAB:
    def __init__(self, env):
        """
        :param env: ArmGaussianLinear object
        """
        self.env = env
        (
            self.expect_regret,
            self.n_a,
            self.d,
            self.features,
        ) = (
            env.expect_regret,
            env.n_actions,
            env.n_features,
            env.features,
        )
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
            "up_set_err": 20,
            "low_set_err": 20,
            "a_t_err": 20,
            "a_star_err": 20,
            # Only for IS - tilde for independent z
            "tilde_up_norm_err": 20,
            "tilde_low_norm_err": 20,
            "tilde_up_set_err": 20,
            "tilde_low_set_err": 20,
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
        self.features = self.env.features

    def initPrior(self, dtype=np.float64):
        a0 = 0
        s0 = self.prior_sigma
        mu_0 = a0 * np.ones(self.d, dtype=dtype)
        sigma_0 = s0 * np.eye(
            self.d, dtype=dtype
        )  # to adapt according to the true distribution of theta
        return mu_0, sigma_0

    # def updatePosterior(self, a, mu, sigma):
    #     """
    #     Update posterior mean and covariance matrix
    #     :param arm: int, arm chose
    #     :param mu: np.array, posterior mean vector
    #     :param sigma: np.array, posterior covariance matrix
    #     :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
    #     """
    #     f, r = self.features[a], self.reward(a)[0]
    #     s_inv = np.linalg.inv(sigma)
    #     ffT = np.outer(f, f)
    #     mu_ = np.dot(
    #         np.linalg.inv(s_inv + ffT / self.eta**2),
    #         np.dot(s_inv, mu) + r * f / self.eta**2,
    #     )
    #     sigma_ = np.linalg.inv(s_inv + ffT / self.eta**2)
    #     return r, mu_, sigma_

    def TS(self, T, scheme="ts"):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        def posterior_sampling(mu, sigma, n_samples=1):
            try:
                theta = np.random.multivariate_normal(mu, sigma, n_samples).T
            except:
                try:
                    z = np.random.normal(0, 1, size=(self.d, n_samples))
                    theta = sqrtm(sigma) @ z + np.expand_dims(mu, axis=1)
                except:
                    print(np.isnan(sigma).any(), np.isinf(sigma).any())
            return theta

        dic = self.data_init(T, self.data_groups)

        # Algorithm
        mu_t, sigma_t = self.initPrior(dtype=np.float64)
        inv_sigma_t = np.linalg.inv(sigma_t)
        p_t = inv_sigma_t @ mu_t
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # print("features: {}".format(self.features[:5]))
            # input()
            # scheme for img_reward
            if scheme == "ts":
                theta_t = posterior_sampling(mu_t, sigma_t)
                img_reward = np.dot(self.features, theta_t)
            elif scheme == "ots":
                theta_t = posterior_sampling(mu_t, sigma_t, 10)
                img_reward = np.max(np.dot(self.features, theta_t), axis=1)
            # action selection
            a_t = rd_argmax(img_reward)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]

            dic["reward"][t], dic["expected_regret"][t] = r_t, self.expect_regret(
                a_t, self.features
            )
            # Synthesis: lmax, lmin, pess_regret, est_regret, potential: pot; lmax_inv, lmin_inv, kappa_inv
            dic["potential"][t] = f_t.T @ sigma_t @ f_t
            dic["pess_regret"][t] = self.pess_regret(a_t, self.features, img_reward)
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

    # @nb.njit
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

        def index_sampling(A, mu, index, scheme="ts"):
            M = A.shape[1]
            if scheme == "ts":
                z = sample_noise(index, M)[0] * np.sqrt(M)
                img_theta = A @ z + mu
                return np.dot(self.features, img_theta)
            elif scheme == "ots":
                z = sample_noise(index, M, 10).T * np.sqrt(M)
                img_theta = A @ z + np.expand_dims(mu, axis=1)
                return np.max(np.dot(self.features, img_theta), axis=1)
            else:
                raise NotImplementedError

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
            img_reward = index_sampling(A_t, mu_t, index, scheme=scheme)
            # action selection
            a_t = rd_argmax(img_reward)
            # selected feature and reward feedback
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            # compute regret
            dic["reward"][t], dic["expected_regret"][t] = r_t, self.expect_regret(
                a_t, self.features
            )

            #
            P = A_t @ A_t.T
            dic["approx_potential"][t] = f_t.T @ P @ f_t
            dic["potential"][t] = f_t.T @ Sigma_t @ f_t
            dic["pess_regret"][t] = self.pess_regret(a_t, self.features, img_reward)
            dic["est_regret"][t] = dic["expected_regret"][t] - dic["pess_regret"][t]
            if t % self.data_groups["lmax"] == 0:
                # Synthesis only for index sampling (not every step)
                a_star = self.optimal_action(self.features)
                (
                    dic["up_norm_err"][t],
                    dic["low_norm_err"][t],
                    all_err,
                    _,
                    _,
                ) = approx_err(a_t, self.features, P, Sigma_t, S_inv)
                dic["a_t_err"][t] = all_err[a_t]
                dic["a_star_err"][t] = all_err[a_star]
                dic["up_set_err"][t] = np.max((all_err))
                dic["low_set_err"][t] = np.max(-(all_err))
                tilde_P = tilde_A_t @ tilde_A_t.T
                (
                    dic["tilde_up_norm_err"][t],
                    dic["tilde_low_norm_err"][t],
                    tilde_all_err,
                    _,
                    _,
                ) = approx_err(a_t, self.features, tilde_P, Sigma_t, S_inv)
                dic["tilde_a_t_err"][t] = tilde_all_err[a_t]
                dic["tilde_a_star_err"][t] = tilde_all_err[a_star]
                dic["tilde_up_set_err"][t] = np.max((tilde_all_err))
                dic["tilde_low_set_err"][t] = np.max(-(tilde_all_err))

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

    def ES(self, T, M=10):
        """
        Ensemble sampling
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, Sigma_t = self.initPrior()
        # A_t = np.zeros((self.d, M))
        # P_t = np.zeros((self.d, M))
        B = np.random.normal(0, 1, (self.d, M))
        # print(B.shape, np.linalg.norm(B, axis=1))
        A_t = mu_t.reshape((self.d, 1)) + sqrtm(Sigma_t) @ B
        P_t = np.linalg.inv(Sigma_t) @ A_t
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # Ensemble sampling
            i = np.random.choice(M, 1)[0]
            theta_t = A_t[:, [i]]
            a_t = rd_argmax(np.dot(self.features, theta_t))
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            Sigma_t = self.rankone_update(f_t, Sigma_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)
            # Update M models with algorithmic random perturbation
            P_t += np.outer(
                f_t, (r_t + np.random.normal(0, self.eta, M)) / self.eta**2
            )
            A_t = Sigma_t @ P_t

        return reward, expected_regret

    def LinUCB(self, T, lbda=10e-4, alpha=10e-1):
        """
        Implementation of Linear UCB algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param lbda: float, regression regularization parameter
        :param alpha: float, tunable parameter to control between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        a_t, A_t, b_t = (
            np.random.randint(0, self.n_a - 1, 1)[0],
            lbda * np.eye(self.d),
            np.zeros(self.d),
        )
        r_t = self.reward(a_t)
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # Algorithm
            A_t += np.outer(self.features[a_t, :], self.features[a_t, :])
            b_t += r_t * self.features[a_t, :]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha * np.sqrt(
                np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T))
            )
            a_t = rd_argmax(np.dot(self.features, theta_t) + beta_t)
            r_t = self.reward(a_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def BayesUCB(self, T):
        """
        Implementation of Bayesian Upper Confidence Bounds (BayesUCB) algorithm for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # Algorithm
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + norm.ppf(t / (t + 1))
                * np.sqrt(
                    np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))
                )
            )
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            mu_t, sigma_t, p_t = self.updatePosterior_(sigma_t, p_t, f_t, r_t, self.eta)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def GPUCB(self, T):
        """
        Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and
        Experimental Design' for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # Algorithm
            beta_t = 2 * np.log(self.n_a * ((t + 1) * np.pi) ** 2 / 6 / 0.1)
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + np.sqrt(
                    beta_t
                    * np.diagonal(
                        np.dot(np.dot(self.features, sigma_t), self.features.T)
                    )
                )
            )
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            mu_t, sigma_t, p_t = self.updatePosterior_(sigma_t, p_t, f_t, r_t, self.eta)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def Tuned_GPUCB(self, T, c=0.9):
        """
        Implementation of Tuned GPUCB described in Russo & Van Roy's paper of study for Linear Bandits with
        multivariate normal prior
        :param T: int, time horizon
        :param c: float, tunable parameter. Default 0.9
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # Algorithm
            beta_t = c * np.log(t + 1)
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + np.sqrt(
                    beta_t
                    * np.diagonal(
                        np.dot(np.dot(self.features, sigma_t), self.features.T)
                    )
                )
            )
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            mu_t, sigma_t, p_t = self.updatePosterior_(sigma_t, p_t, f_t, r_t, self.eta)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def computeVIDS(self, thetas):
        """
        Implementation of linearSampleVIR (algorithm 6 in Russo & Van Roy, p. 244) applied for Linear  Bandits with
        multivariate normal prior. Here integrals are approximated in sampling thetas according to their respective
        posterior distributions.
        :param thetas: np.array, posterior samples
        :return: int, np.array, arm chose and p*
        """
        # print(thetas.shape)
        M = thetas.shape[0]
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        # print("theta_hat shape: {}".format(theta_hat.shape))
        theta_hat_ = [thetas[np.where(theta_hat == a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)]) / M
        # if np.max(p_a) >= self.threshold:
        #     # Stop learning policy
        #     self.optimal_arm = np.argmax(p_a)
        #     arm = self.optimal_arm
        # else:
        # print("theta_hat_[0]: {}, theta_hat_[0] length: {}".format(theta_hat_[0], len(theta_hat_[0])))
        mu_a = np.nan_to_num(
            np.array(
                [np.mean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]
            )
        )
        L_hat = np.sum(
            np.array(
                [p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu) for a in range(self.n_a)]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [p_a[a] * np.dot(self.features[a], mu_a[a]) for a in range(self.n_a)]
            ),
            axis=0,
        )
        v = np.array(
            [
                np.dot(np.dot(self.features[a], L_hat), self.features[a].T)
                for a in range(self.n_a)
            ]
        )
        delta = np.array(
            [rho_star - np.dot(self.features[a], mu) for a in range(self.n_a)]
        )
        arm = rd_argmax(-(delta**2) / (v + 1e-20))
        return arm, p_a

    def VIDS_sample(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            # Changing action set (If env applicable)
            self.set_context()
            # Algorithm
            thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
            a_t, p_a = self.computeVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            mu_t, sigma_t, p_t = self.updatePosterior_(sigma_t, p_t, f_t, r_t, self.eta)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret
