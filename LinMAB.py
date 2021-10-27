""" Packages import """
import numpy as np

# import jax.numpy as np
from utils import rd_argmax
from scipy.stats import norm

from math import sqrt

import jax.numpy as jnp
import jax.random as random
from sgmcmcjax.samplers import build_sgld_sampler


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

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        best_arm_reward = np.max(np.dot(self.features, self.real_theta))
        return best_arm_reward * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, arm_sequence, T):
        """
        Compute the regret of a single experiment
        :param arm_sequence: np.array, sequence of chosen arms obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        print(arm_sequence)
        expect_reward = np.dot(self.features[arm_sequence], self.real_theta)
        best_arm_reward = np.max(np.dot(self.features, self.real_theta))
        return best_arm_reward * np.arange(1, T + 1) - np.cumsum(expect_reward)


class PaperLinModel(ArmGaussianLinear):
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
        super(PaperLinModel, self).__init__(
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = self.prior_random.uniform(-u, u, (n_actions, n_features))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma

class FreqPaperLinModel(ArmGaussianLinear):
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
        super(PaperLinModel, self).__init__(
            prior_random_state=0,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = self.prior_random.uniform(-u, u, (n_actions, n_features))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma


class FGTSLinModel(ArmGaussianLinear):
    def __init__(self, n_features=100, n_actions=2, eta=sqrt(0.5)):
        """
        Initialization of the arms, features and theta in
        Zhang, Tong. "Feel-Good Thompson Sampling for Contextual Bandits and Reinforcement Learning." arXiv preprint arXiv:2110.00871 (2021).
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward likelihood model N(a^T.theta, eta)
        """
        super(FGTSLinModel, self).__init__(
            prior_random_state=0,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = np.zeros((n_actions, n_features))
        self.features[0, 0] = 1.0
        self.features[1, 1] = 0.2
        if n_actions > 2:
            self.features[2:] = self.prior_random.uniform(-1, 1, (n_actions-2, n_features)) 
            self.features[2:] = 0.2 * self.features[2:] / np.expand_dims(np.linalg.norm(self.features[2:], axis=1), axis=1)
            # print(self.features)
            # print(np.linalg.norm(self.features, axis=1))
        # print(self.features)
        self.real_theta = np.zeros(n_features)
        self.real_theta[0:2] = 1.0
        self.alg_prior_sigma = 0.01

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following Unif(-0.5, 0.5)
        :param arm: int
        :return: float
        """
        return np.dot(self.features[arm], self.real_theta) + self.reward_random.uniform(
            -0.5, 0.5, 1
        )


class ColdStartMovieLensModel(ArmGaussianLinear):
    def __init__(self, n_features=30, n_actions=207, eta=1, sigma=10):
        super(ColdStartMovieLensModel, self).__init__(
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = np.loadtxt("Data/Vt.csv", delimiter=",").T
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma


class LinMAB:
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
        self.flag = False
        self.optimal_arm = None
        self.threshold = 0.9
        self.store_IDS = False

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
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

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
        if np.max(p_a) >= self.threshold:
            # Stop learning policy
            self.optimal_arm = np.argmax(p_a)
            arm = self.optimal_arm
        else:
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
            arm = rd_argmax(-(delta ** 2) / v)
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
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            if not self.flag:
                if np.max(p_a) >= self.threshold:
                    # Stop learning policy
                    self.flag = True
                    a_t = self.optimal_arm
                else:
                    thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
                    a_t, p_a = self.computeVIDS(thetas)
            else:
                a_t = self.optimal_arm
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

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
        my_sampler = build_sgld_sampler(
            dt, loglikelihood, logprior, (X, y), 1, pbar=False
        )
        # run sampler
        key, subkey = random.split(key)
        thetas = my_sampler(subkey, n_iters, jnp.zeros(self.d))
        # if jnp.isnan(thetas).any():
        #     print(jnp.where(jnp.isnan(thetas)))
        return thetas[-n_samples:]

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
                    X, y, n_samples=1, n_iters=max(t, 10000 + 100), fg_lambda=fg_lambda
                ).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            reward[t], arm_sequence[t] = self.reward(a_t)[0], a_t
        return reward, arm_sequence

    def FGTS10(self, T):
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
            if not self.flag:
                if np.max(p_a) >= self.threshold:
                    # Stop learning policy
                    self.flag = True
                    a_t = self.optimal_arm
                else:
                    if t == 0:
                        mu_t, sigma_t = self.initPrior()
                        thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
                    else:
                        X = jnp.asarray(self.features[arm_sequence[:t]])
                        y = jnp.asarray(reward[:t])
                        thetas = self.SGLD_Sampler(X, y, M, max(M + 100, t), fg_lambda)
                    a_t, p_a = self.computeVIDS(thetas)
            else:
                a_t = self.optimal_arm
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

    def VIDS_sample_sgmcmc_fg10(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.VIDS_sample_sgmcmc_fg(T, M, 10)
