""" Packages import """
import numpy as np


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
        feature = self.prior_random.uniform(-self.u, self.u, (feature_num,))
        for i in range(self.n_actions):
            self.features[i][feature_num * i : feature_num * (i + 1)] = feature

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
        self.features = np.zeros((n_actions, n_features * n_actions))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features * n_actions), sigma * np.eye(n_features * n_actions)
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
        self.features = np.zeros((n_actions, n_features * n_actions))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features * n_actions), sigma * np.eye(n_features * n_actions)
        )
        self.alg_prior_sigma = sigma
        self.set_context()
