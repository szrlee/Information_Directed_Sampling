""" Packages import """
import numpy as np

# import jax.numpy as np
from utils import rd_argmax
from scipy.stats import norm

from math import sqrt

import jax.numpy as jnp
import jax.random as random
from sgmcmcjax.samplers import build_sgld_sampler

from hypermodel import HyperModel


class ArmGaussianLinear(object):
    def __init__(self, prior_random_state=2021, reward_random_state=2022):
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following N(0, eta)
        :param arm: int
        :return: float
        """
        return np.dot(self.features[arm], self.real_theta) + self.reward_random.normal(
            0, self.eta, 1
        )

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]

    def set_context(self):
        feature_num = self.n_features // self.n_actions
        feature = self.prior_random.uniform(-self.u, self.u, (feature_num, ))
        for i in range(self.n_actions):
            self.features[i][feature_num * i: feature_num * (i+1)] = feature

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        best_arm_reward = np.max(np.dot(self.features, self.real_theta))
        return best_arm_reward * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, arm_sequence_action_sets, T):
        """
        Compute the regret of a single experiment
        :param arm_sequence_action_sets: (arm: (T,) context: (T, n_arm, n_feature))
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        arm_sequence, action_sets = arm_sequence_action_sets # (arm_sequence: (T,) action_sets: (T, n_arm, n_feature))
        print(arm_sequence)
        feature = action_sets[np.arange(T), arm_sequence] # (T, n_feature)
        expect_reward = np.dot(feature, self.real_theta) # (T, )
        best_arm_reward = np.max(np.dot(action_sets, self.real_theta), axis=-1) # (T, )
        return np.cumsum(best_arm_reward) - np.cumsum(expect_reward)


class InfiniteContextPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=1, sigma=10):
        """
        Initialization of the arms, features and theta in
        Russo, Daniel, and Benjamin Van Roy. "Learning to optimize via information-directed sampling." Operations Research 66.1 (2018): 230-252.
        :param u: float, features are drawn from a uniform Unif(-u, u)
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward N(a^T.theta, eta)
        :param sigma: float, multiplicative factor for the covariance matrix of theta which is drawn from a
        multivariate distribution N(0, sigma*I)
        """
        super(InfiniteContextPaperLinModel, self).__init__(
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.u = u
        self.eta = eta
        self.features = np.zeros((n_actions, n_features*n_actions))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features*n_actions), sigma * np.eye(n_features*n_actions)
        )
        self.alg_prior_sigma = sigma
        self.set_context()


class InfiniteContextFreqPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=1, sigma=10):
        """
        (Frequentist modification: use fixed random seed to sample arms, features and theta.)
        Initialization of the arms, features and theta in
        Russo, Daniel, and Benjamin Van Roy. "Learning to optimize via information-directed sampling." Operations Research 66.1 (2018): 230-252.
        :param u: float, features are drawn from a uniform Unif(-u, u)
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward N(a^T.theta, eta)
        :param sigma: float, multiplicative factor for the covariance matrix of theta which is drawn from a
        multivariate distribution N(0, sigma*I)
        """
        super(InfiniteContextFreqPaperLinModel, self).__init__(
            prior_random_state=0,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.u = u
        self.eta = eta
        self.features = np.zeros((n_actions, n_features*n_actions))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features*n_actions), sigma * np.eye(n_features*n_actions)
        )
        self.alg_prior_sigma = sigma
        self.set_context()


class InfiniteContextLinMAB:
    def __init__(self, model):
        """
        :param model: ArmGaussianLinear object
        """
        self.model = model
        self.regret, self.expect_regret, self.n_a, self.d, self.features = (
            model.regret,
            model.expect_regret,
            model.n_actions,
            model.n_features,
            model.features,
        )
        self.reward, self.eta = model.reward, model.eta
        self.prior_sigma = model.alg_prior_sigma
        # self.flag = False
        # self.optimal_arm = None
        self.threshold = 0.999
        self.store_IDS = False

    def set_context(self):
        self.model.set_context()
        self.features = self.model.features

    def initPrior(self):
        a0 = 0
        s0 = self.prior_sigma
        mu_0 = a0 * np.ones(self.d)
        sigma_0 = s0 * np.eye(
            self.d
        )  # to adapt according to the true distribution of theta
        return mu_0, sigma_0

    def updatePosterior(self, a, mu, sigma):
        """
        Update posterior mean and covariance matrix
        :param arm: int, arm chose
        :param mu: np.array, posterior mean vector
        :param sigma: np.array, posterior covariance matrix
        :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
        """
        f, r = self.features[a], self.reward(a)[0]
        s_inv = np.linalg.inv(sigma)
        ffT = np.outer(f, f)
        mu_ = np.dot(
            np.linalg.inv(s_inv + ffT / self.eta ** 2),
            np.dot(s_inv, mu) + r * f / self.eta ** 2,
        )
        sigma_ = np.linalg.inv(s_inv + ffT / self.eta ** 2)
        return r, mu_, sigma_

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            self.set_context()
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
        return reward, (arm_sequence, action_sets)

    def TS_hyper(self, T, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=2):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = model.sample_theta(1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def TS_hyper_reset(self, T, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=1):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef, reset=True
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = model.sample_theta(1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            model.reset()
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def LinUCB(self, T, lbda=10e-4, alpha=10e-1):
        """
        Implementation of Linear UCB algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param lbda: float, regression regularization parameter
        :param alpha: float, tunable parameter to control between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        a_t, A_t, b_t = (
            np.random.randint(0, self.n_a - 1, 1)[0],
            lbda * np.eye(self.d),
            np.zeros(self.d),
        )
        r_t = self.reward(a_t)
        for t in range(T):
            A_t += np.outer(self.features[a_t, :], self.features[a_t, :])
            b_t += r_t * self.features[a_t, :]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha * np.sqrt(
                np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T))
            )
            a_t = rd_argmax(np.dot(self.features, theta_t) + beta_t)
            r_t = self.reward(a_t)
            arm_sequence[t], reward[t] = a_t, r_t
        return reward, arm_sequence

    def BayesUCB(self, T):
        """
        Implementation of Bayesian Upper Confidence Bounds (BayesUCB) algorithm for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + norm.ppf(t / (t + 1))
                * np.sqrt(
                    np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))
                )
            )
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

    def GPUCB(self, T):
        """
        Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and
        Experimental Design' for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
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
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

    def Tuned_GPUCB(self, T, c=0.9):
        """
        Implementation of Tuned GPUCB described in Russo & Van Roy's paper of study for Linear Bandits with
        multivariate normal prior
        :param T: int, time horizon
        :param c: float, tunable parameter. Default 0.9
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
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
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

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
                [
                    np.mean([theta_hat_[a]], axis=1).squeeze()
                    for a in range(self.n_a)
                ]
            )
        )
        L_hat = np.sum(
            np.array(
                [
                    p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu)
                    for a in range(self.n_a)
                ]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [
                    p_a[a] * np.dot(self.features[a], mu_a[a])
                    for a in range(self.n_a)
                ]
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
        arm = rd_argmax(-(delta ** 2) / (v + 1e-20))
        return arm, p_a

    def solveVIDS(self, thetas):
        M = thetas.shape[0]
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        theta_hat_ = [thetas[np.where(theta_hat == a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)]) / M
        mu_a = np.nan_to_num(
            np.array(
                [
                    np.mean([theta_hat_[a]], axis=1).squeeze()
                    for a in range(self.n_a)
                ]
            )
        )
        L_hat = np.sum(
            np.array(
                [
                    p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu)
                    for a in range(self.n_a)
                ]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [
                    p_a[a] * np.dot(self.features[a], mu_a[a])
                    for a in range(self.n_a)
                ]
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
        prob = np.zeros(shape=(self.n_a, self.n_a))
        psi = np.ones(shape=(self.n_a, self.n_a)) * np.inf
        for i in range(self.n_a-1):
            for j in range(i+1, self.n_a):
                if delta[j] < delta[i]:
                    D1, D2, I1, I2, flip = delta[j], delta[i], v[j], v[i], True
                else:
                    D1, D2, I1, I2, flip = delta[i], delta[j], v[i], v[j], False
                p = np.clip((D1 / (D2 - D1)) - (2 * I1 / (I2 - I1)), 0., 1.) if I1 < I2 else 0.
                psi[i][j] = ((1 - p) * D1 + p * D2)**2 / ((1 - p) * I1 + p * I2 + 1e-20)
                prob[i][j] = 1 - p if flip else p
        psi = psi.flatten()
        optim_indexes = np.nonzero(psi == psi.min())[0].tolist()
        optim_index = np.random.choice(optim_indexes)
        optim_index = [optim_index // self.n_a, optim_index % self.n_a]
        optim_prob = prob[optim_index[0], optim_index[1]]
        arm = np.random.choice(optim_index, p=[1 - optim_prob, optim_prob])
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
        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
            a_t, p_a = self.computeVIDS(thetas)
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_hyper(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=2):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = model.sample_theta(M)
            a_t, p_a = self.computeVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_hyper_reset(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=1):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef, reset=True
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = model.sample_theta(M)
            a_t, p_a = self.computeVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            model.reset()
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_solution(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        mu_t, sigma_t = self.initPrior()
        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
            a_t, p_a = self.solveVIDS(thetas)
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_solution_hyper(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=2):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = model.sample_theta(M)
            a_t, p_a = self.solveVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_solution_hyper_reset(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=1):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef, reset=True
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = model.sample_theta(M)
            a_t, p_a = self.solveVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            model.reset()
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def SGLD_Sampler(self, X, y, n_samples, n_iters, fg_lambda):
        assert n_iters >= n_samples + 99
        # define model in JAX
        def loglikelihood(theta, x, y):
            return -(
                1 / (2 * (self.eta ** 2)) * (jnp.dot(x, theta) - y) ** 2
            ) + fg_lambda * jnp.max(jnp.dot(self.features, theta))

        def logprior(theta):
            return -0.5 * jnp.dot(theta, theta) * (1 / self.prior_sigma)

        # generate random key in jax
        key = random.PRNGKey(np.random.randint(1, 312414))
        # print("fg_lambda={}".format(fg_lambda), key)

        dt = 1e-5
        N = X.shape[0]
        # print(N)
        batch_size = int(0.1*(N+9))
        # print(batch_size)
        my_sampler = build_sgld_sampler(
            dt, loglikelihood, logprior, (X, y), batch_size, pbar=False
        )
        # run sampler
        key, subkey = random.split(key)
        thetas = my_sampler(subkey, n_iters, jnp.zeros(self.d))
        # if jnp.isnan(thetas).any():
        #     print(jnp.where(jnp.isnan(thetas)))
        sample_idx = 100 * jnp.arange(n_samples) + 10000
        # print(sample_idx)
        return thetas[sample_idx]

    def FGTS(self, T, fg_lambda=1.0):
        """
        Implementation of Feel-Good Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param fg_lambda: float, coefficient for feel good term
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)

        for t in range(T):
            if t == 0:
                mu_t, sigma_t = self.initPrior()
                theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            else:
                X = jnp.asarray(self.features[arm_sequence[:t]])
                y = jnp.asarray(reward[:t])
                theta_t = self.SGLD_Sampler(
                    X, y, n_samples=1, n_iters=max(100 * 1 + 10000, t), fg_lambda=fg_lambda
                ).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            reward[t], arm_sequence[t] = self.reward(a_t)[0], a_t
        return reward, arm_sequence

    def FGTS01(self, T):
        """
        Implementation of Feel-Good Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior and posterios sampling via SGMCMC
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.FGTS(T, fg_lambda=10)

    def TS_SGMCMC(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior and posterios sampling via SGMCMC
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.FGTS(T, fg_lambda=0)

    def VIDS_sample_sgmcmc_fg(self, T, M=10000, fg_lambda=1):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # print("stop")
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                if t == 0:
                    mu_t, sigma_t = self.initPrior()
                    thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
                else:
                    X = jnp.asarray(self.features[arm_sequence[:t]])
                    y = jnp.asarray(reward[:t])
                    thetas = self.SGLD_Sampler(X, y, M, max(100 * M + 10000, t), fg_lambda)
                a_t, p_a = self.computeVIDS(thetas)
            reward[t], arm_sequence[t] = self.reward(a_t)[0], a_t
        return reward, arm_sequence

    def VIDS_sample_sgmcmc(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.VIDS_sample_sgmcmc_fg(T, M, 0)

    def VIDS_sample_sgmcmc_fg01(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.VIDS_sample_sgmcmc_fg(T, M, 0.1)
