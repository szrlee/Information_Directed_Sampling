import numpy as np
from utils import rd_argmax
import time
import jax.numpy as jnp
from jax import random as jrd


class CompactGaussianLinear(object):
    # Gaussian Linear bandit with compact action set (i.e. action set is the unit ball in R^d)
    def __init__(self, prior_random_seed=2022, reward_random_seed=2023):
        self.prior_random = np.random.default_rng(seed=prior_random_seed)
        self.reward_random = np.random.default_rng(seed=reward_random_seed)
        self.prior_key = jrd.PRNGKey(prior_random_seed)

    @property
    def real_theta(self):
        return self._real_theta

    @real_theta.setter
    def real_theta(self, value):
        self._real_theta = value
        self._optimal_action = self._real_theta / np.linalg.norm(self._real_theta)
        self._best_arm_reward = np.dot(self._optimal_action, self._real_theta)

    @property
    def optimal_action(self):
        return self._optimal_action

    @property
    def best_arm_reward(self):
        return self._best_arm_reward

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following N(0, eta)
        :param arm: int
        :return: float
        """
        return np.dot(arm, self.real_theta) + self.reward_random.normal(0, self.eta, 1)

    def set_context(self):
        pass

    def expect_regret(self, arm):
        """
        Compute the regret of a single step
        """
        expect_reward = np.dot(arm, self.real_theta)
        # best_arm_reward = np.dot(self.optimal_action, self.real_theta)
        return self.best_arm_reward - expect_reward

    def pessimism_regret(self, arm, img_theta, scheme):
        """
        Compute the pessimism regret of a single step
        """
        if scheme == "ts":
            img_reward = np.dot(arm, img_theta)
        elif scheme == "ots":
            img_reward = np.max(np.dot(arm, img_theta))

        return self.best_arm_reward - img_reward


class CompactLinModel(CompactGaussianLinear):
    def __init__(self, n_features=100, eta=1, sigma=10):
        """
        Initialization of Linear Gaussian bandit with compact action set (i.e. action set is the unit ball in R^d)
        :param n_features: int, dimension of the feature vectors
        :param eta: float, std from the reward likelihood model N(a^T.theta, eta)
        """
        super(CompactLinModel, self).__init__(
            prior_random_seed=np.random.randint(1, 312414),
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.alg_prior_sigma = sigma
        self.n_features = n_features
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features),
            sigma * np.eye(n_features),
        )


# Creat a frequentist version of CompactLinModel
class FreqCompactLinModel(CompactGaussianLinear):
    def __init__(self, n_features=100, eta=1, sigma=10):
        """
        (Frequentist version) Initialization of Linear Gaussian bandit with compact action set (i.e. action set is the unit ball in R^d)
        :param n_features: int, dimension of the feature vectors
        :param eta: float, std from the reward likelihood model N(a^T.theta, eta)
        """
        super(FreqCompactLinModel, self).__init__(
            prior_random_seed=2022,
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.alg_prior_sigma = sigma
        self.n_features = n_features
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features),
            sigma * np.eye(n_features),
        )


class ArmGaussianLinear(object):
    def __init__(self, prior_random_seed=2022, reward_random_seed=2023):
        self.prior_random = np.random.default_rng(seed=prior_random_seed)
        self.reward_random = np.random.default_rng(seed=reward_random_seed)
        self.prior_key = jrd.PRNGKey(prior_random_seed)

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
        pass

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        best_arm_reward = np.max(np.dot(self.features, self.real_theta))
        return best_arm_reward * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, arm, action_set):
        """
        Compute the regret of a single step
        """
        arm_reward = np.dot(action_set, self.real_theta)
        expect_reward = arm_reward[arm]
        best_arm_reward = arm_reward.max()
        return best_arm_reward - expect_reward

    def optimal_action(self, action_set):
        return rd_argmax(np.dot(action_set, self.real_theta))

    def pessimism_regret(self, arm, action_set, img_reward):
        """
        Compute the pessimism regret of a single step
        """
        arm_reward = np.dot(action_set, self.real_theta)
        best_arm_reward = arm_reward.max()

        # arm_reward = np.dot(action_set, img_theta)
        # img_reward = arm_reward[arm]

        return best_arm_reward - img_reward[arm]


class ChangingLinModel(ArmGaussianLinear):
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
        super(ChangingLinModel, self).__init__(
            prior_random_seed=np.random.randint(1, 312414),
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.u = u
        self.eta = eta
        self.features = np.zeros((n_actions, n_features), dtype=np.float64)
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features, dtype=np.float64),
            sigma * np.eye(n_features, dtype=np.float64),
        )
        self.alg_prior_sigma = sigma
        # for changing action seet
        self.t = 0
        self.n_buffer = 10
        # self.x = jnp.zeros((self.n_steps * n_actions, n_features))

    def set_features(self, n_buffer, n_actions, n_features):
        # self.x[:] = rng.standard_normal((n_steps * n_actions, n_features))
        # self.x /= np.linalg.norm(self.x, axis=1, keepdims=True)
        self.prior_key, subkey = jrd.split(self.prior_key)
        self.x = jrd.normal(subkey, (n_buffer * n_actions, n_features))
        self.x /= jnp.linalg.norm(self.x, axis=1, keepdims=True)

    def set_context(self):
        if self.t % self.n_buffer == 0:
            self.set_features(self.n_buffer, self.n_actions, self.n_features)
        self.features[:] = self.x[
            self.t * self.n_actions : (self.t + 1) * self.n_actions
        ]
        self.t = (self.t + 1) % self.n_buffer


class FreqChangingLinModel(ArmGaussianLinear):
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
        super(FreqChangingLinModel, self).__init__(
            prior_random_seed=0,
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.u = u
        self.eta = eta
        self.features = np.zeros((n_actions, n_features), dtype=np.float64)
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features, dtype=np.float64),
            sigma * np.eye(n_features, dtype=np.float64),
        )
        self.alg_prior_sigma = sigma
        # for changing action seet
        self.t = 0
        self.n_buffer = 10
        # self.x = jnp.zeros((self.n_steps * n_actions, n_features))

    def set_features(self, n_buffer, n_actions, n_features):
        # self.x[:] = rng.standard_normal((n_steps * n_actions, n_features))
        # self.x /= np.linalg.norm(self.x, axis=1, keepdims=True)
        self.prior_key, subkey = jrd.split(self.prior_key)
        self.x = jrd.normal(subkey, (n_buffer * n_actions, n_features))
        self.x /= jnp.linalg.norm(self.x, axis=1, keepdims=True)

    def set_context(self):
        if self.t % self.n_buffer == 0:
            self.set_features(self.n_buffer, self.n_actions, self.n_features)
        self.features[:] = self.x[
            self.t * self.n_actions : (self.t + 1) * self.n_actions
        ]
        self.t = (self.t + 1) % self.n_buffer


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
            prior_random_seed=np.random.randint(1, 312414),
            reward_random_seed=np.random.randint(1, 312414),
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
        super(FreqPaperLinModel, self).__init__(
            prior_random_seed=2022,
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = self.prior_random.uniform(-u, u, (n_actions, n_features))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma


class FGTSLinModel(ArmGaussianLinear):
    def __init__(self, n_features=100, n_actions=2, eta=np.sqrt(0.5)):
        """
        Initialization of the arms, features and theta in
        Zhang, Tong. "Feel-Good Thompson Sampling for Contextual Bandits and Reinforcement Learning." arXiv preprint arXiv:2110.00871 (2021).
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward likelihood model N(a^T.theta, eta)
        """
        super(FGTSLinModel, self).__init__(
            prior_random_seed=2022,
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = np.zeros((n_actions, n_features))
        self.features[0, 0] = 1.0
        self.features[1, 1] = 0.2
        if n_actions > 2:
            self.features[2:] = self.prior_random.uniform(
                -1, 1, (n_actions - 2, n_features)
            )
            self.features[2:] = (
                0.2
                * self.features[2:]
                / np.expand_dims(np.linalg.norm(self.features[2:], axis=1), axis=1)
            )
            # print(self.features)
            # print(np.linalg.norm(self.features, axis=1))
        self.features[np.where(self.features == 0)] = 1e-6
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
            prior_random_seed=np.random.randint(1, 312414),
            reward_random_seed=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = np.loadtxt("Data/Vt.csv", delimiter=",").T
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma
