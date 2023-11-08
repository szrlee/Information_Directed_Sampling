""" Packages import """
import numpy as np
from env.linear import ArmGaussianLinear as ChangingArmGaussianLinear


class InfiniteContextPaperLinModel(ChangingArmGaussianLinear):
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

    def set_context(self):
        feature_num = self.n_features // self.n_actions
        feature = self.prior_random.uniform(-self.u, self.u, (feature_num,))
        for i in range(self.n_actions):
            self.features[i][feature_num * i : feature_num * (i + 1)] = feature


class InfiniteContextFreqPaperLinModel(ChangingArmGaussianLinear):
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

    def set_context(self):
        feature_num = self.n_features // self.n_actions
        feature = self.prior_random.uniform(-self.u, self.u, (feature_num,))
        for i in range(self.n_actions):
            self.features[i][feature_num * i : feature_num * (i + 1)] = feature
